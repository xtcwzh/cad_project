"""Runtime metadata for SDPA calls, decoder passes, aten backends, and OOM allocs.

Records shapes/dtypes/devices and operator names only. Never copies activations.
SDPA wrapping uses torch.compiler.disable and a TorchDispatchMode; memory peaks
from an instrumented run are not comparable to uninstrumented probes.
Omitted SDPA arguments stay absent/None; defaults are never filled in.
"""
import inspect
import json
import re
import traceback
from collections import deque
from contextlib import contextmanager
from pathlib import Path

_SESSION = None
_RECOMPUTE_FRAME_NAMES = {'backward', 'recompute_fn', 'unpack_hook'}
_CHECKPOINT_FRAME_NAMES = {'checkpoint', 'CheckpointFunction'}
_SDPA_POS = ('attn_mask', 'dropout_p', 'is_causal')
_OOM_UNITS = {
    'b': 1, 'byte': 1, 'bytes': 1,
    'kib': 1024, 'kb': 1000,
    'mib': 1024 ** 2, 'mb': 1000 ** 2,
    'gib': 1024 ** 3, 'gb': 1000 ** 3,
    'tib': 1024 ** 4, 'tb': 1000 ** 4,
}


def tensor_meta(obj):
    import torch
    if obj is None:
        return None
    if not torch.is_tensor(obj):
        return {'kind': 'non_tensor', 'type': type(obj).__name__}
    return {
        'shape': list(obj.shape),
        'dtype': str(obj.dtype).replace('torch.', ''),
        'device': str(obj.device),
        'requires_grad': bool(obj.requires_grad),
        'numel': int(obj.numel()),
        'element_size': int(obj.element_size()),
    }


def reject_tensors(obj, path='$'):
    import torch
    if torch.is_tensor(obj):
        raise TypeError('tensor leaked into diagnostics at ' + path)
    if isinstance(obj, dict):
        for key, value in obj.items():
            reject_tensors(value, path + '.' + str(key))
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            reject_tensors(value, path + '[' + str(i) + ']')


def checkpoint_frames(limit=80):
    frames = []
    for frame in traceback.extract_stack(limit=limit):
        filename = frame.filename or ''
        name = frame.name or ''
        if ('checkpoint' not in filename.lower()
                and name not in _CHECKPOINT_FRAME_NAMES
                and name not in {'recompute_fn', 'unpack_hook'}):
            continue
        frames.append({
            'file': filename.replace('\\', '/').rsplit('/', 1)[-1],
            'line': frame.lineno,
            'name': name,
        })
        if len(frames) >= 12:
            break
    return frames


def classify_pass():
    import torch
    grad = bool(torch.is_grad_enabled())
    frames = checkpoint_frames()
    recompute = any(frame['name'] in _RECOMPUTE_FRAME_NAMES for frame in frames)
    if recompute:
        checkpoint_phase = 'checkpoint_recompute'
    elif frames:
        checkpoint_phase = 'checkpoint_forward'
    else:
        checkpoint_phase = None
    if not grad:
        label = 'no_grad'
    elif recompute:
        label = 'checkpoint_recompute'
    else:
        label = 'grad'
    return {
        'pass': label,
        'grad_enabled': grad,
        'checkpoint_phase': checkpoint_phase,
        'checkpoint_frames': frames,
    }


def parse_oom_message(message):
    text = str(message)
    tried = re.search(
        r'Tried to allocate\s+([0-9]+(?:\.[0-9]+)?)\s*(KiB|MiB|GiB|TiB|KB|MB|GB|TB|bytes?)\b',
        text, re.I)
    gpu = re.search(r'\bGPU\s+(\d+)\b', text)
    amount = unit = raw = nbytes = None
    if tried:
        raw = tried.group(0)
        amount = float(tried.group(1))
        unit = tried.group(2)
        nbytes = int(round(amount * _OOM_UNITS[unit.lower()]))
    return {
        'tried_to_allocate_raw': raw,
        'tried_to_allocate_amount': amount,
        'tried_to_allocate_unit': unit,
        'tried_to_allocate_bytes': nbytes,
        'gpu_index_from_message': int(gpu.group(1)) if gpu else None,
    }


def _is_alloc_like(action):
    if action is None:
        return True
    name = str(action).lower()
    return 'alloc' in name or name in ('oom', 'fail')


def failed_allocation_from_history(trimmed, tried_bytes, gpu_index):
    recent = []
    size_equal = []
    if not isinstance(trimmed, dict):
        return {
            'recent_alloc_events': [],
            'alloc_events_with_size_equal_to_tried_allocate': [],
            'alloc_events_with_size_and_gpu_equal_to_message': [],
            'tried_to_allocate_bytes': tried_bytes,
            'match_rule': 'event.size == tried_to_allocate_bytes; no nearest-size fallback',
        }
    for trace in trimmed.get('device_traces') or []:
        for event in trace.get('events') or []:
            item = {
                'device_index': trace.get('device_index'),
                'action': event.get('action'),
                'size': event.get('size'),
                'stream': event.get('stream'),
                'frames': event.get('frames'),
            }
            recent.append(item)
            if tried_bytes is None or event.get('size') != tried_bytes:
                continue
            if not _is_alloc_like(event.get('action')):
                continue
            size_equal.append({
                **item,
                'gpu_index_matched': gpu_index is None or trace.get('device_index') == gpu_index,
            })
    return {
        'recent_alloc_events': recent[-32:],
        'alloc_events_with_size_equal_to_tried_allocate': size_equal,
        'alloc_events_with_size_and_gpu_equal_to_message': [
            item for item in size_equal if item.get('gpu_index_matched')
        ],
        'tried_to_allocate_bytes': tried_bytes,
        'match_rule': 'event.size == tried_to_allocate_bytes; no nearest-size fallback',
    }


def jsonable_arg(obj):
    if obj is None:
        return None
    import torch
    if torch.is_tensor(obj):
        return tensor_meta(obj)
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float, str)):
        return obj
    return {'kind': 'non_tensor', 'type': type(obj).__name__}


def parse_sdpa_call(original, query, key, value, args, kwargs):
    kwargs = dict(kwargs or {})
    bound = {}
    for i, name in enumerate(_SDPA_POS):
        if i < len(args):
            bound[name] = args[i]
    bound.update(kwargs)
    extra_positional = max(0, len(args) - len(_SDPA_POS))
    try:
        sig = inspect.signature(original)
        positional = [p.name for p in sig.parameters.values()
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if positional[:3] not in (['query', 'key', 'value'], ['q', 'k', 'v']):
            return bound, False, extra_positional
        arguments = sig.bind_partial(query, key, value, *args, **kwargs).arguments
        parsed = {k: v for k, v in arguments.items()
                  if k not in ('query', 'key', 'value', 'q', 'k', 'v', 'args', 'kwargs')}
        extra = arguments.get('args')
        extra_positional = len(extra) if isinstance(extra, tuple) else 0
        return parsed, True, extra_positional
    except (TypeError, ValueError):
        return bound, False, extra_positional


def find_decoder_layers(model):
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    try:
        return base.model.language_model.layers
    except AttributeError:
        raise RuntimeError('Decoder layers not found at model.language_model.layers; refusing to guess')


def trim_cuda_snapshot(snapshot, max_trace_events=256, max_frames=20, max_segments=64):
    if snapshot is None:
        return None
    if isinstance(snapshot, dict):
        traces = snapshot.get('device_traces')
        segments = snapshot.get('segments')
    elif isinstance(snapshot, (list, tuple)):
        traces, segments = snapshot, None
    else:
        return {'unparsed_type': type(snapshot).__name__}
    out = {'device_traces': [], 'segments_summary': []}
    if isinstance(traces, (list, tuple)):
        for device_index, trace in enumerate(traces):
            if not isinstance(trace, (list, tuple)):
                continue
            events = []
            for entry in trace[-max_trace_events:]:
                events.append(_trim_trace_entry(entry, max_frames, device_index))
            out['device_traces'].append({
                'device_index': device_index,
                'trace_len': len(trace),
                'events_kept': len(events),
                'events': events,
            })
    if isinstance(segments, list):
        for segment in segments[-max_segments:]:
            if not isinstance(segment, dict):
                continue
            out['segments_summary'].append({
                'device': segment.get('device'),
                'total_size': segment.get('total_size') or segment.get('size'),
                'allocated_size': segment.get('allocated_size') or segment.get('active_size'),
                'stream': segment.get('stream'),
                'segment_type': segment.get('segment_type') or segment.get('kind'),
            })
    return out


def _trim_trace_entry(entry, max_frames, device_index):
    if not isinstance(entry, dict):
        return {'unparsed_type': type(entry).__name__}
    frames = entry.get('frames') or entry.get('user_frames') or []
    trimmed = []
    if isinstance(frames, (list, tuple)):
        for frame in frames[:max_frames]:
            if isinstance(frame, dict):
                trimmed.append({
                    'filename': frame.get('filename') or frame.get('file'),
                    'line': frame.get('line') or frame.get('lineno'),
                    'name': frame.get('name'),
                })
            else:
                trimmed.append({'repr': repr(frame)[:200]})
    return {
        'action': entry.get('action') or entry.get('kind'),
        'size': entry.get('size'),
        'stream': entry.get('stream'),
        'device': entry.get('device', device_index),
        'frames': trimmed,
    }


def _op_name(func):
    name = getattr(func, 'name', None)
    if callable(name):
        return name()
    if isinstance(name, str):
        return name
    return str(func)


def _json_dump(path, value):
    reject_tensors(value)
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


class BoundedLog:
    def __init__(self, max_events):
        self.max_events = max_events
        self.events = deque()
        self.dropped = 0
        self.total = 0

    def append(self, event):
        self.total += 1
        if len(self.events) >= self.max_events:
            self.events.popleft()
            self.dropped += 1
        self.events.append(event)

    def as_list(self):
        return list(self.events)


class SdpaOpTracker:
    def __init__(self, session):
        from torch.utils._python_dispatch import TorchDispatchMode
        self._mode_cls = TorchDispatchMode
        self.session = session
        self._mode = None

    def __torch_dispatch__(self, func, types, args, kwargs):
        name = _op_name(func)
        if 'scaled_dot_product' in name:
            kwargs = kwargs or {}
            payload = {
                'op': name,
                'kind': 'backward' if 'backward' in name else 'forward',
                **classify_pass(),
                'arg_metas': [tensor_meta(arg) for arg in args if _is_tensor(arg)][:8],
                'kwarg_keys': sorted(kwargs),
                'kwarg_tensor_metas': {key: tensor_meta(value)
                                       for key, value in kwargs.items() if _is_tensor(value)},
            }
            layer = self.session.current_layer()
            if layer is not None:
                payload['layer'] = layer
            self.session.emit('aten_sdpa', payload)
            current = self.session.current_sdpa
            if current is not None:
                current.setdefault('aten_ops_during_call', []).append({
                    'op': name, 'kind': payload['kind'],
                })
        return func(*args, **(kwargs or {}))

    def start(self):
        tracker = self

        class _Mode(self._mode_cls):
            def __torch_dispatch__(self, func, types, args, kwargs):
                return tracker.__torch_dispatch__(func, types, args, kwargs)

        self._mode = _Mode()
        self._mode.__enter__()

    def stop(self):
        if self._mode is None:
            return
        try:
            self._mode.__exit__(None, None, None)
        finally:
            self._mode = None


def _is_tensor(obj):
    import torch
    return torch.is_tensor(obj)


class RuntimeDiagnostics:
    def __init__(self, directory, max_events=4096, memory_trace_entries=2048):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_events = max_events
        self.memory_trace_entries = memory_trace_entries
        self.logs = {
            'sdpa_call': BoundedLog(max_events),
            'decoder_pass': BoundedLog(max_events),
            'aten_sdpa': BoundedLog(max_events),
        }
        self.seq = 0
        self._layer_stack = []
        self.current_sdpa = None
        self._original_sdpa = None
        self._tracker = None
        self._jsonl = (self.directory / 'events.jsonl').open('w', encoding='utf-8')
        self._closed = False
        self._finalized = False
        self._summary = None
        self._sdpa_compiler_disable = False
        self.perturbations = {
            'runtime_diagnostics': True,
            'wraps_python_sdpa': False,
            'sdpa_wrapper_compiler_disable': False,
            'wraps_decoder_layer_forward': False,
            'aten_torch_dispatch_mode': False,
            'cuda_memory_history': False,
            'graph_breaks_expected': True,
            'memory_peaks_comparable_to_uninstrumented_probe': False,
            'reason': ('Python F.scaled_dot_product_attention is wrapped with '
                       'torch.compiler.disable; aten TorchDispatchMode runs for the '
                       'whole step. Decoder forwards are wrapped in Python. Peaks '
                       'from this run must not be compared to uninstrumented probes.'),
        }

    def current_layer(self):
        if not self._layer_stack:
            return None
        rec = self._layer_stack[-1]
        return {key: rec[key] for key in
                ('layer_index', 'block_type', 'module_class', 'pass',
                 'grad_enabled', 'checkpoint_phase')
                if key in rec}

    @contextmanager
    def layer_scope(self, rec):
        self._layer_stack.append(rec)
        try:
            yield
        finally:
            self._layer_stack.pop()

    def emit(self, kind, payload):
        event = {'kind': kind, 'seq': self.seq}
        event.update(payload)
        reject_tensors(event)
        self.seq += 1
        self.logs[kind].append(event)
        self._jsonl.write(json.dumps(event, ensure_ascii=False) + '\n')
        self._jsonl.flush()
        return event

    def wrap_sdpa(self):
        import torch
        import torch.nn.functional as F
        if self._original_sdpa is not None:
            raise RuntimeError('SDPA already wrapped')
        original = F.scaled_dot_product_attention
        session = self

        def wrapped(query, key, value, *args, **kwargs):
            parsed, signature_bound, extra_positional = parse_sdpa_call(
                original, query, key, value, args, kwargs)
            record = {
                'query': tensor_meta(query),
                'key': tensor_meta(key),
                'value': tensor_meta(value),
                'attn_mask': tensor_meta(parsed['attn_mask']) if 'attn_mask' in parsed else None,
                'attn_mask_passed': 'attn_mask' in parsed,
                'is_causal': jsonable_arg(parsed['is_causal']) if 'is_causal' in parsed else None,
                'is_causal_passed': 'is_causal' in parsed,
                'enable_gqa': jsonable_arg(parsed['enable_gqa']) if 'enable_gqa' in parsed else None,
                'enable_gqa_passed': 'enable_gqa' in parsed,
                'dropout_p': jsonable_arg(parsed['dropout_p']) if 'dropout_p' in parsed else None,
                'scale': jsonable_arg(parsed['scale']) if 'scale' in parsed else None,
                'signature_bound': signature_bound,
                'passed_argument_names': sorted(parsed),
                'unnamed_positional_count': extra_positional,
                'layer': session.current_layer(),
                **classify_pass(),
            }
            session.current_sdpa = record
            try:
                out = original(query, key, value, *args, **kwargs)
                session.emit('sdpa_call', record)
                return out
            except BaseException:
                record['raised'] = True
                try:
                    session.emit('sdpa_call', record)
                except Exception:
                    pass
                raise
            finally:
                session.current_sdpa = None

        disable = getattr(getattr(torch, 'compiler', None), 'disable', None)
        if disable is None:
            self.perturbations['sdpa_wrapper_compiler_disable'] = False
            self.perturbations['sdpa_wrapper_compiler_disable_reason'] = (
                'torch.compiler.disable missing; wrapper still graph-breaks compiled attention')
        else:
            try:
                wrapped = disable(wrapped, recursive=False)
                self.perturbations['sdpa_wrapper_compiler_disable_recursive'] = False
            except TypeError:
                wrapped = disable(wrapped)
                self.perturbations['sdpa_wrapper_compiler_disable_recursive'] = (
                    'api_did_not_accept_recursive')
            self._sdpa_compiler_disable = True
            self.perturbations['sdpa_wrapper_compiler_disable'] = True
        self._original_sdpa = original
        F.scaled_dot_product_attention = wrapped
        torch.nn.functional.scaled_dot_product_attention = wrapped
        self.perturbations['wraps_python_sdpa'] = True
        self.perturbations['graph_breaks_expected'] = True

    def start_dispatch(self):
        if self._tracker is not None:
            raise RuntimeError('aten dispatch already started')
        self._tracker = SdpaOpTracker(self)
        self._tracker.start()
        self.perturbations['aten_torch_dispatch_mode'] = True

    def wrap_decoder_layers(self, layers):
        if not layers:
            raise RuntimeError('No decoder layers to wrap')
        wrapped = 0
        for index, layer in enumerate(layers):
            if hasattr(layer, '_cad_runtime_diag'):
                raise ValueError(f'Decoder layer {index} already wrapped')
            slot = '_old_forward' if hasattr(layer, '_hf_hook') else 'forward'
            original = getattr(layer, slot)
            session = self

            def forward(*args, _original=original, _index=index, _layer=layer, **kwargs):
                rec = {
                    'layer_index': _index,
                    'block_type': getattr(_layer, 'block_type', None),
                    'module_class': type(_layer).__name__,
                    **classify_pass(),
                }
                session.emit('decoder_pass', rec)
                with session.layer_scope(rec):
                    return _original(*args, **kwargs)

            setattr(layer, slot, forward)
            layer._cad_runtime_diag = True
            wrapped += 1
        self.perturbations['wraps_decoder_layer_forward'] = True
        self.perturbations['decoder_layers_wrapped'] = wrapped
        return wrapped

    def start_memory_history(self):
        import torch
        if not torch.cuda.is_available():
            self.perturbations['cuda_memory_history'] = False
            self.perturbations['cuda_memory_history_reason'] = 'cuda_unavailable'
            return
        history_args = {
            'enabled': 'all', 'context': 'all', 'stacks': 'all',
            'max_entries': self.memory_trace_entries,
        }
        try:
            torch.cuda.memory._record_memory_history(**history_args)
        except TypeError:
            try:
                history_args = {
                    'enabled': True, 'stacks': 'python',
                    'max_entries': self.memory_trace_entries,
                }
                torch.cuda.memory._record_memory_history(**history_args)
            except TypeError:
                history_args = {'enabled': True}
                torch.cuda.memory._record_memory_history(True)
        self.perturbations['cuda_memory_history'] = True
        self.perturbations['cuda_memory_history_args'] = history_args
        self.perturbations['cuda_memory_history_max_entries'] = self.memory_trace_entries
        self.perturbations['cuda_memory_history_started'] = 'after_model_and_lora_load'

    def install(self, model):
        self.wrap_sdpa()
        self.start_dispatch()
        self.wrap_decoder_layers(find_decoder_layers(model))
        self.start_memory_history()
        return self.info()

    def info(self):
        return {
            'runtime_diagnostics': True,
            'diagnostics_dir': str(self.directory),
            **self.perturbations,
        }

    def capture_oom(self, exc):
        parsed = parse_oom_message(exc)
        payload = {
            'exception_type': type(exc).__name__,
            'exception_message': str(exc),
            'python_traceback': traceback.format_exc(),
            'failed_allocation_python_stack': traceback.format_tb(exc.__traceback__),
            'oom_parse': parsed,
        }
        try:
            import torch
            if torch.cuda.is_available():
                trimmed = trim_cuda_snapshot(torch.cuda.memory._snapshot())
                payload['cuda_memory_history'] = trimmed
                payload['failed_allocation_from_cuda_history'] = failed_allocation_from_history(
                    trimmed, parsed['tried_to_allocate_bytes'],
                    parsed['gpu_index_from_message'])
                summaries = {}
                for index in range(torch.cuda.device_count()):
                    summaries[str(index)] = torch.cuda.memory_summary(index)
                payload['cuda_memory_summary'] = summaries
        except Exception as history_exc:
            payload['cuda_memory_history_error'] = repr(history_exc)
        reject_tensors(payload)
        try:
            _json_dump(self.directory / 'oom.json', payload)
        except Exception as write_exc:
            payload['oom_json_write_error'] = repr(write_exc)
        return payload

    def dump(self, oom=None):
        document = {
            'schema': 'cad.runtime_diagnostics.v1',
            'fields_are_measured_only': True,
            'unobserved_means_not_recorded': True,
            'memory_peaks_comparable_to_uninstrumented_probe': False,
            'perturbations': dict(self.perturbations),
            'counts': {kind: {'total': log.total, 'kept': len(log.events),
                              'dropped': log.dropped}
                       for kind, log in self.logs.items()},
            'sdpa_calls': self.logs['sdpa_call'].as_list(),
            'decoder_passes': self.logs['decoder_pass'].as_list(),
            'aten_sdpa_ops': self.logs['aten_sdpa'].as_list(),
            'oom': oom,
        }
        _json_dump(self.directory / 'diagnostics.json', document)
        return document

    def summarize(self, document):
        counts = document['counts']
        oom = document['oom']
        files = ['diagnostics.json', 'events.jsonl']
        if oom is not None:
            files.append('oom.json')
        observed_ops = sorted({event['op'] for event in document['aten_sdpa_ops'] if event.get('op')})
        oom_parse = (oom or {}).get('oom_parse') or {}
        return {
            'dir': str(self.directory),
            'files': files,
            'counts': counts,
            'python_sdpa_calls_seen': counts['sdpa_call']['total'],
            'aten_sdpa_ops_seen': counts['aten_sdpa']['total'],
            'observed_aten_sdpa_op_names': observed_ops,
            'decoder_passes_seen': counts['decoder_pass']['total'],
            'perturbations': document['perturbations'],
            'memory_peaks_comparable_to_uninstrumented_probe': False,
            'fields_are_measured_only': True,
            'unobserved_means_not_recorded': True,
            'oom': oom is not None,
            'oom_exception_type': None if oom is None else oom.get('exception_type'),
            'oom_tried_to_allocate_raw': oom_parse.get('tried_to_allocate_raw'),
        }

    def finalize(self, exc=None):
        if self._finalized:
            return self._summary
        try:
            oom = self.capture_oom(exc) if exc is not None else None
            document = self.dump(oom=oom)
            self._summary = self.summarize(document)
            return self._summary
        finally:
            self.close()
            self._finalized = True

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._tracker is not None:
            self._tracker.stop()
            self._tracker = None
        if self._original_sdpa is not None:
            import torch
            import torch.nn.functional as F
            F.scaled_dot_product_attention = self._original_sdpa
            torch.nn.functional.scaled_dot_product_attention = self._original_sdpa
            self._original_sdpa = None
        try:
            import torch
            if torch.cuda.is_available() and self.perturbations.get('cuda_memory_history'):
                torch.cuda.memory._record_memory_history(enabled=None)
        except Exception:
            pass
        if self._jsonl is not None:
            self._jsonl.close()
            self._jsonl = None


def install_runtime_diagnostics(model, c):
    global _SESSION
    if _SESSION is not None:
        raise RuntimeError('Runtime diagnostics already installed')
    directory = c.get('diagnostics_dir')
    if not directory:
        raise ValueError('runtime_diagnostics requires diagnostics_dir')
    session = RuntimeDiagnostics(
        directory=directory,
        max_events=int(c.get('diagnostics_max_events', 4096)),
        memory_trace_entries=int(c.get('diagnostics_memory_trace_entries', 2048)),
    )
    session.install(model)
    _SESSION = session
    return session.info()


def finalize_diagnostics(exc=None):
    global _SESSION
    session = _SESSION
    if session is None:
        return None
    try:
        return session.finalize(exc)
    finally:
        _SESSION = None


def reset_diagnostics_for_tests():
    global _SESSION
    if _SESSION is not None:
        _SESSION.close()
        _SESSION = None
