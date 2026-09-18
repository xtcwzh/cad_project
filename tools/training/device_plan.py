"""Explicit contiguous Qwen3.5 language-layer placement; no CUDA imports."""
import json
from pathlib import Path


def build_device_plan(model_config, counts, gpu_count):
    if model_config.get('model_type') != 'qwen3_5':
        raise ValueError('Explicit layout currently supports model_type=qwen3_5 only')
    text = model_config.get('text_config', model_config)
    n = text.get('num_hidden_layers')
    if (not isinstance(counts, list) or len(counts) != gpu_count or
            any(type(v) is not int or v < 0 for v in counts) or
            any(v == 0 for v in counts[:-1]) or sum(counts) != n):
        raise ValueError(f'layer_counts must contain {gpu_count} integers summing to {n}; '
                         'only the final head-only GPU may have zero layers')
    if model_config.get('tie_word_embeddings', False) or text.get('tie_word_embeddings', False):
        raise ValueError('Embedding/head on different GPUs requires untied weights')
    prefix = 'model.language_model'
    # Do not add a root-device fallback: Accelerate recursively attaches execution
    # hooks there before reaching the explicit layer entries, leaving child hooks
    # pointing at GPU0 even when their parameters are on another GPU.
    mapping = {'model.visual': 0, prefix + '.embed_tokens': 0,
               prefix + '.rotary_emb': 0,
               prefix + '.norm': gpu_count - 1, 'lm_head': gpu_count - 1}
    start = 0
    for gpu, count in enumerate(counts):
        for layer in range(start, start + count):
            mapping[f'{prefix}.layers.{layer}'] = gpu
        start += count
    return mapping


def plan_from_config(c):
    raw = json.loads((Path(c['model_path']) / 'config.json').read_text(encoding='utf-8'))
    return build_device_plan(raw, c['layer_counts'], len(str(c['gpu']).split(',')))


def validate_device_plan(model, mapping):
    """Check real parameter locations, including adapters, not just hf_device_map."""
    if hasattr(model, 'get_base_model'):
        model = model.get_base_model()
    modules = dict(model.named_modules())
    missing = [name for name in mapping if name and name not in modules]
    if missing:
        raise RuntimeError(f'Explicit layout module paths do not match this model: {missing}')
    prefixes = sorted(mapping, key=len, reverse=True)
    for name, param in model.named_parameters():
        prefix = next((p for p in prefixes if name == p or name.startswith(p + '.')), None)
        if prefix is None:
            raise RuntimeError(f'Parameter missing from explicit layout: {name}')
        expected = mapping[prefix]
        if param.device.type != 'cuda' or param.device.index != expected:
            raise RuntimeError(f'Placement mismatch: {name} on {param.device}, expected cuda:{expected}')
    validate_execution_hooks(model, mapping)


def validate_execution_hooks(model, mapping):
    """Reject stale descendant hooks which would move activations to another GPU."""
    def devices(hook):
        value = getattr(hook, 'execution_device', None)
        if value is not None:
            yield str(value) if not isinstance(value, int) else f'cuda:{value}'
        for child in getattr(hook, 'hooks', ()):
            yield from devices(child)

    prefixes = sorted(mapping, key=len, reverse=True)
    for name, module in model.named_modules():
        prefix = next((p for p in prefixes if name == p or name.startswith(p + '.')), None)
        if prefix is None:
            continue  # Parent I/O hooks may legitimately use the main device.
        for actual in devices(getattr(module, '_hf_hook', None)):
            if actual != f'cuda:{mapping[prefix]}':
                raise RuntimeError(f'Execution hook mismatch: {name} uses {actual}, '
                                   f'expected cuda:{mapping[prefix]}')
