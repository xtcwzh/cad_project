"""Token-wise MLP chunking, with nested non-reentrant activation checkpoints.

Never chunks attention or the training sample. No changes to parameter names.
"""


def native_checkpoint():
    import torch.utils.checkpoint as ck
    # Unsloth shims can force reentrant=True even when False was requested.
    # Use its preserved original, and fail clearly on unsupported patch stacks.
    for key in ('_unsloth_pristine_checkpoint', '_old_checkpoint', 'checkpoint'):
        candidate = getattr(ck, key, None)
        if (getattr(candidate, '__module__', '') == 'torch.utils.checkpoint' and
                getattr(candidate, '__name__', '') == 'checkpoint'):
            return candidate
    raise RuntimeError('Native PyTorch checkpoint unavailable; refusing nested Unsloth checkpoint')


def patch_mlp(module, chunk_tokens):
    import torch
    checkpoint = native_checkpoint()

    if type(chunk_tokens) is not int or chunk_tokens <= 0:
        raise ValueError('mlp_chunk_tokens must be a positive integer')
    if hasattr(module, '_cad_mlp_chunk_tokens'):
        raise ValueError('MLP already patched')
    # Preserve Accelerate's outer dispatch wrapper and its device alignment.
    slot = '_old_forward' if hasattr(module, '_hf_hook') else 'forward'
    original = getattr(module, slot)

    @torch.compiler.disable
    def forward(hidden_states):
        if hidden_states.ndim != 3:
            raise ValueError('Chunked MLP expects [batch, sequence, hidden]')
        pieces = []
        for start in range(0, hidden_states.shape[1], chunk_tokens):
            block = hidden_states[:, start:start + chunk_tokens, :]
            # Outer Unsloth checkpoint forward runs under no_grad. During its
            # backward recompute grad is enabled: checkpoint each MLP block then,
            # so their wide intermediate activations do not all survive together.
            if torch.is_grad_enabled():
                value = checkpoint(original, block, use_reentrant=False,
                                   preserve_rng_state=True)
            else:
                value = original(block)
            pieces.append(value)
        return torch.cat(pieces, dim=1)

    setattr(module, slot, forward)
    module._cad_mlp_chunk_tokens = chunk_tokens


def install_chunked_mlp(model, chunk_tokens):
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    if base.config.model_type != 'qwen3_5':
        raise ValueError('Chunked MLP currently supports dense qwen3_5 only')
    layers = base.model.language_model.layers
    targets = []
    for i, layer in enumerate(layers):
        mlp = layer.mlp
        if any(not hasattr(mlp, key) for key in ('gate_proj', 'up_proj', 'down_proj', 'act_fn')):
            raise ValueError(f'Unsupported MLP in layer {i}')
        if hasattr(mlp, '_cad_mlp_chunk_tokens'):
            raise ValueError(f'Layer {i} already patched')
        targets.append(mlp)
    for mlp in targets:
        patch_mlp(mlp, chunk_tokens)
    return {'mlp_chunk_tokens': chunk_tokens, 'chunked_mlp_layers': len(targets),
            'mlp_checkpoint': 'non_reentrant', 'mlp_chunk_loop_compile': False}
