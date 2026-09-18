"""CPU metadata tests for runtime diagnostics. No model download."""
import gc
import json
import tempfile
import unittest
import weakref
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from runtime_diagnostics import (
    RuntimeDiagnostics, tensor_meta, trim_cuda_snapshot, reject_tensors,
    install_runtime_diagnostics, finalize_diagnostics, reset_diagnostics_for_tests,
    parse_oom_message, failed_allocation_from_history,
)


class DummyLayer(nn.Module):
    def __init__(self, block_type):
        super().__init__()
        self.block_type = block_type
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * self.scale + 1


class AttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_type = 'full_attention'

    def forward(self, tensors):
        query, key, value, mask = tensors
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, is_causal=False, enable_gqa=False)


class DummyModel(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.config = type('C', (), {'model_type': 'qwen3_5'})()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList(layers)

    def get_base_model(self):
        return self

    def forward(self, x):
        for layer in self.model.language_model.layers:
            x = layer(x)
        return x


class DiagnosticsTests(unittest.TestCase):
    def tearDown(self):
        reset_diagnostics_for_tests()

    def test_tensor_meta_does_not_retain(self):
        tensor = torch.randn(3, 5)
        handle = weakref.ref(tensor)
        meta = tensor_meta(tensor)
        del tensor
        gc.collect()
        self.assertIsNone(handle())
        self.assertEqual(meta['shape'], [3, 5])
        self.assertEqual(meta['dtype'], 'float32')
        self.assertNotIn('data', meta)
        reject_tensors(meta)

    def test_reject_tensors_in_payload(self):
        with self.assertRaises(TypeError):
            reject_tensors({'query': torch.zeros(2)})

    def test_sdpa_call_records_processed_arguments(self):
        with tempfile.TemporaryDirectory() as temp:
            diag = RuntimeDiagnostics(temp, max_events=16)
            diag.wrap_sdpa()
            diag.start_dispatch()
            query = torch.randn(1, 4, 6, 8)
            key = torch.randn(1, 2, 6, 8)
            value = torch.randn(1, 2, 6, 8)
            mask = torch.zeros(1, 1, 6, 6)
            torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, is_causal=False, enable_gqa=True)
            diag.close()
            calls = diag.logs['sdpa_call'].as_list()
            self.assertEqual(len(calls), 1)
            call = calls[0]
            self.assertEqual(call['query']['shape'], [1, 4, 6, 8])
            self.assertEqual(call['key']['shape'], [1, 2, 6, 8])
            self.assertEqual(call['value']['dtype'], 'float32')
            self.assertEqual(call['attn_mask']['shape'], [1, 1, 6, 6])
            self.assertEqual(call['attn_mask']['dtype'], 'float32')
            self.assertTrue(call['attn_mask_passed'])
            self.assertFalse(call['is_causal'])
            self.assertTrue(call['is_causal_passed'])
            self.assertTrue(call['enable_gqa'])
            self.assertTrue(call['enable_gqa_passed'])
            self.assertTrue(str(query.device) == call['query']['device'])
            ops = diag.logs['aten_sdpa'].as_list()
            self.assertTrue(ops)
            self.assertTrue(all('scaled_dot_product' in item['op'] for item in ops))
            self.assertTrue(diag.perturbations['sdpa_wrapper_compiler_disable'])
            self.assertFalse(diag.perturbations['memory_peaks_comparable_to_uninstrumented_probe'])

    def test_omitted_enable_gqa_not_filled_in(self):
        with tempfile.TemporaryDirectory() as temp:
            diag = RuntimeDiagnostics(temp)
            diag.wrap_sdpa()
            query = torch.randn(1, 2, 4, 8)
            torch.nn.functional.scaled_dot_product_attention(query, query, query)
            diag.close()
            call = diag.logs['sdpa_call'].as_list()[0]
            self.assertFalse(call['enable_gqa_passed'])
            self.assertIsNone(call['enable_gqa'])
            self.assertFalse(call['attn_mask_passed'])
            self.assertIsNone(call['attn_mask'])
            self.assertFalse(call['is_causal_passed'])
            self.assertIsNone(call['is_causal'])
            self.assertNotIn('is_causal', call['passed_argument_names'])
            self.assertNotIn('enable_gqa', call['passed_argument_names'])

    def test_decoder_pass_and_checkpoint_recompute(self):
        with tempfile.TemporaryDirectory() as temp:
            diag = RuntimeDiagnostics(temp)
            model = DummyModel([DummyLayer('full_attention'), DummyLayer('linear_attention')])
            try:
                diag.wrap_decoder_layers(model.model.language_model.layers)
                x = torch.randn(2, 3, requires_grad=True)
                with torch.no_grad():
                    model(x)
                no_grad = [e for e in diag.logs['decoder_pass'].as_list() if e['pass'] == 'no_grad']
                self.assertEqual({e['layer_index'] for e in no_grad}, {0, 1})
                self.assertEqual(no_grad[0]['block_type'], 'full_attention')
                self.assertEqual(no_grad[1]['block_type'], 'linear_attention')
                self.assertTrue(all(e['checkpoint_phase'] is None for e in no_grad))
                y = checkpoint(model, x, use_reentrant=False)
                y.sum().backward()
                passes = diag.logs['decoder_pass'].as_list()
                grad_first = [e for e in passes if e['pass'] == 'grad']
                recomputes = [e for e in passes if e['pass'] == 'checkpoint_recompute']
                self.assertTrue(grad_first)
                self.assertTrue(recomputes)
                self.assertTrue(all(e['grad_enabled'] for e in recomputes))
                self.assertTrue(all(e['checkpoint_phase'] == 'checkpoint_forward' for e in grad_first))
                self.assertTrue(all(e['checkpoint_phase'] == 'checkpoint_recompute' for e in recomputes))
                self.assertTrue(any(e['layer_index'] == 0 and e['block_type'] == 'full_attention'
                                    for e in recomputes))
                self.assertTrue(any(frame['name'] in ('unpack_hook', 'recompute_fn', 'backward')
                                    for e in recomputes for frame in e['checkpoint_frames']))
                x2 = torch.randn(2, 3, requires_grad=True)
                y2 = checkpoint(model, x2, use_reentrant=True)
                y2.sum().backward()
                reentrant = [e for e in diag.logs['decoder_pass'].as_list()
                             if e['checkpoint_phase'] == 'checkpoint_recompute'
                             and any(frame['name'] == 'backward' for frame in e['checkpoint_frames'])]
                self.assertTrue(reentrant)
                self.assertTrue(any(e['pass'] == 'no_grad' and e['checkpoint_phase'] == 'checkpoint_forward'
                                    for e in diag.logs['decoder_pass'].as_list()))
            finally:
                diag.close()

    def test_sdpa_layer_context_and_aten_backward(self):
        with tempfile.TemporaryDirectory() as temp:
            diag = RuntimeDiagnostics(temp)
            layer = AttentionLayer()
            model = DummyModel([layer])
            diag.wrap_sdpa()
            diag.start_dispatch()
            diag.wrap_decoder_layers(model.model.language_model.layers)
            query = torch.randn(1, 2, 4, 8, requires_grad=True)
            key = torch.randn(1, 2, 4, 8)
            value = torch.randn(1, 2, 4, 8)
            mask = torch.zeros(1, 1, 4, 4)
            out = model((query, key, value, mask))
            out.sum().backward()
            call = diag.logs['sdpa_call'].as_list()[0]
            self.assertEqual(call['layer']['layer_index'], 0)
            self.assertEqual(call['layer']['block_type'], 'full_attention')
            kinds = {item['kind'] for item in diag.logs['aten_sdpa'].as_list()}
            self.assertIn('forward', kinds)
            self.assertIn('backward', kinds)
            fwd = [item['op'] for item in diag.logs['aten_sdpa'].as_list() if item['kind'] == 'forward']
            bwd = [item['op'] for item in diag.logs['aten_sdpa'].as_list() if item['kind'] == 'backward']
            self.assertTrue(fwd and all('scaled_dot_product' in name for name in fwd))
            self.assertTrue(bwd and all('backward' in name and 'scaled_dot_product' in name for name in bwd))
            diag.close()

    def test_hooked_decoder_uses_old_forward(self):
        with tempfile.TemporaryDirectory() as temp:
            diag = RuntimeDiagnostics(temp)
            layer = DummyLayer('full_attention')
            layer._hf_hook = object()
            layer._old_forward = layer.forward
            calls = {'n': 0}

            def dispatch(x):
                calls['n'] += 1
                return layer._old_forward(x)

            layer.forward = dispatch
            model = DummyModel([layer])
            diag.wrap_decoder_layers(model.model.language_model.layers)
            model(torch.zeros(2, 2))
            self.assertGreater(calls['n'], 0)
            self.assertEqual(diag.logs['decoder_pass'].total, 1)
            diag.close()

    def test_event_cap_and_dump_has_no_activations(self):
        with tempfile.TemporaryDirectory() as temp:
            diag = RuntimeDiagnostics(temp, max_events=3)
            layer = DummyLayer('linear_attention')
            model = DummyModel([layer])
            diag.wrap_decoder_layers(model.model.language_model.layers)
            for _ in range(5):
                model(torch.zeros(1, 1))
            self.assertEqual(diag.logs['decoder_pass'].total, 5)
            self.assertEqual(diag.logs['decoder_pass'].dropped, 2)
            self.assertEqual(len(diag.logs['decoder_pass'].events), 3)
            dumped = diag.dump()
            text = Path(temp, 'diagnostics.json').read_text(encoding='utf-8')
            self.assertNotIn('tensor(', text)
            self.assertTrue(dumped['perturbations']['graph_breaks_expected'])
            self.assertFalse(dumped['memory_peaks_comparable_to_uninstrumented_probe'])
            self.assertFalse(dumped['perturbations']['memory_peaks_comparable_to_uninstrumented_probe'])
            self.assertTrue(dumped['fields_are_measured_only'])
            diag.close()

    def test_trim_snapshot_drops_blocks_and_bounds_frames(self):
        snapshot = {
            'segments': [{
                'device': 0, 'total_size': 1024, 'allocated_size': 512,
                'blocks': [{'addr': 99, 'size': 512, 'state': 'active'}],
            }],
            'device_traces': [[{
                'action': 'alloc', 'size': 768 * 1024 * 1024, 'addr': 123,
                'stream': 0,
                'frames': [{'filename': 'op.py', 'line': i, 'name': 'alloc'} for i in range(40)],
            }]],
        }
        trimmed = trim_cuda_snapshot(snapshot, max_frames=8)
        event = trimmed['device_traces'][0]['events'][0]
        self.assertEqual(len(event['frames']), 8)
        self.assertNotIn('addr', event)
        self.assertNotIn('blocks', trimmed['segments_summary'][0])
        self.assertEqual(event['size'], 768 * 1024 * 1024)

    def test_install_finalize_and_leaked_tensor_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            model = DummyModel([DummyLayer('full_attention')])
            info = install_runtime_diagnostics(model, {'diagnostics_dir': temp})
            self.assertTrue(info['runtime_diagnostics'])
            self.assertFalse(info['memory_peaks_comparable_to_uninstrumented_probe'])
            self.assertEqual(info['decoder_layers_wrapped'], 1)
            model(torch.zeros(2, 2))
            summary = finalize_diagnostics()
            self.assertEqual(summary['decoder_passes_seen'], 1)
            self.assertFalse(summary['memory_peaks_comparable_to_uninstrumented_probe'])
            self.assertTrue((Path(temp) / 'diagnostics.json').exists())
            document = json.loads((Path(temp) / 'diagnostics.json').read_text(encoding='utf-8'))
            self.assertTrue(document['fields_are_measured_only'])
            self.assertIsNone(document['oom'])
            diag = RuntimeDiagnostics(temp)
            diag.logs['sdpa_call'].append({'query': torch.zeros(1)})
            with self.assertRaises(TypeError):
                diag.dump()
            diag.close()

    def test_oom_parse_and_history_size_equality_only(self):
        message = (
            'CUDA out of memory. Tried to allocate 768.00 MiB. GPU 0; '
            'total capacity is 10.00 GiB')
        parsed = parse_oom_message(message)
        self.assertEqual(parsed['tried_to_allocate_raw'], 'Tried to allocate 768.00 MiB')
        self.assertEqual(parsed['tried_to_allocate_bytes'], 768 * 1024 * 1024)
        self.assertEqual(parsed['gpu_index_from_message'], 0)
        missing = parse_oom_message('unrelated failure')
        self.assertIsNone(missing['tried_to_allocate_bytes'])
        self.assertIsNone(missing['gpu_index_from_message'])
        snapshot = {
            'device_traces': [[
                {'action': 'alloc', 'size': 768 * 1024 * 1024,
                 'frames': [{'filename': 'sdpa.py', 'line': 10, 'name': 'forward'}]},
                {'action': 'alloc', 'size': 767 * 1024 * 1024,
                 'frames': [{'filename': 'other.py', 'line': 1, 'name': 'nearby'}]},
            ]],
            'segments': [],
        }
        trimmed = trim_cuda_snapshot(snapshot)
        matched = failed_allocation_from_history(
            trimmed, parsed['tried_to_allocate_bytes'], parsed['gpu_index_from_message'])
        self.assertEqual(len(matched['alloc_events_with_size_equal_to_tried_allocate']), 1)
        self.assertEqual(
            matched['alloc_events_with_size_equal_to_tried_allocate'][0]['size'],
            768 * 1024 * 1024)
        self.assertEqual(matched['alloc_events_with_size_and_gpu_equal_to_message'][0]['frames'][0]['name'],
                         'forward')
        unmatched = failed_allocation_from_history(trimmed, 123, 0)
        self.assertEqual(unmatched['alloc_events_with_size_equal_to_tried_allocate'], [])

    def test_capture_oom_writes_stack_without_tensors(self):
        with tempfile.TemporaryDirectory() as temp:
            diag = RuntimeDiagnostics(temp)
            try:
                raise RuntimeError('CUDA out of memory. Tried to allocate 1.00 GiB. GPU 3')
            except RuntimeError as exc:
                payload = diag.capture_oom(exc)
            self.assertEqual(payload['oom_parse']['gpu_index_from_message'], 3)
            self.assertEqual(payload['oom_parse']['tried_to_allocate_bytes'], 1024 ** 3)
            self.assertTrue(payload['failed_allocation_python_stack'])
            self.assertTrue((Path(temp) / 'oom.json').is_file())
            reject_tensors(payload)
            text = (Path(temp) / 'oom.json').read_text(encoding='utf-8')
            self.assertNotIn('tensor(', text)
            dumped = diag.dump(oom=payload)
            summary = diag.summarize(dumped)
            self.assertTrue(summary['oom'])
            self.assertEqual(summary['oom_tried_to_allocate_raw'], 'Tried to allocate 1.00 GiB')
            self.assertFalse(summary['memory_peaks_comparable_to_uninstrumented_probe'])
            diag.close()


if __name__ == '__main__':
    unittest.main()
