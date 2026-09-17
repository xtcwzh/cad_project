"""Shared single-GPU QLoRA and exact chat encoding. Heavy imports stay lazy."""
import hashlib
import inspect
import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


def path(value):
    p = Path(os.path.expandvars(value)).expanduser()
    return p if p.is_absolute() else PROJECT / p


def config(filename):
    c = json.loads(path(filename).read_text(encoding='utf-8'))
    for k in ('model_path', 'sft_root', 'output_root'):
        c[k] = str(path(c[k]))
    if not Path(c['model_path']).is_dir():
        raise ValueError('Local model directory missing: ' + c['model_path'])
    if int(c['r']) < 1 or int(c['max_seq_length']) < 2:
        raise ValueError('Invalid LoRA rank or sequence length')
    return c


def gpu_indices(gpu):
    parts = str(gpu).split(',')
    if not parts or any(not p.isdigit() for p in parts):
        raise ValueError('Expected comma-separated physical GPU indices')
    indices = [int(p) for p in parts]
    if len(set(indices)) != len(indices):
        raise ValueError('Duplicate GPU indices')
    return indices


def offline(gpu):
    indices = gpu_indices(gpu)
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Use plain python, not torchrun: this is single-process model sharding')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, indices))
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['WANDB_DISABLED'] = 'true'


def write_json(filename, value):
    p = Path(filename)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(p)


def digest(filename):
    h = hashlib.sha256()
    with Path(filename).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def text_tokenizer(obj):
    return getattr(obj, 'tokenizer', obj)


def encode_record(record, tokenizer, thinking=False):
    """Render once as full conversation; exact prefix masking, never truncate."""
    messages = record['messages']
    if [m.get('role') for m in messages] != ['system', 'user', 'assistant']:
        raise ValueError('Expected single system/user/assistant sample')
    if any(not isinstance(m.get('content'), str) for m in messages):
        raise ValueError('Only text JSON samples are supported')
    if not messages[-1]['content'].strip():
        raise ValueError('Empty training answer')
    kwargs = dict(tokenize=False, enable_thinking=thinking)
    prefix = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, **kwargs)
    full = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **kwargs)
    if not isinstance(full, str) or not full.startswith(prefix):
        raise ValueError('Chat template prefix mismatch; refusing guessed assistant masking')
    ids = tokenizer.encode(full, add_special_tokens=False)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    if ids[:len(prefix_ids)] != prefix_ids or len(ids) <= len(prefix_ids):
        raise ValueError('Token boundary mismatch or empty assistant span')
    return {'input_ids': ids, 'attention_mask': [1] * len(ids),
            'labels': [-100] * len(prefix_ids) + ids[len(prefix_ids):]}


def prepare(c, tokenizer):
    rows, datasets, hashes = [], {}, {}
    seen = set()
    for split in ('train', 'val', 'test'):
        file = Path(c['sft_root']) / 'samples' / (split + '.jsonl')
        hashes[split] = digest(file)
        dataset = []
        for line in file.read_text(encoding='utf-8').splitlines():
            rec = json.loads(line)
            if rec['part'] in seen:
                raise ValueError('Duplicate/cross-split part: ' + rec['part'])
            seen.add(rec['part'])
            item = encode_record(rec, tokenizer, c['enable_thinking'])
            n = len(item['input_ids'])
            rows.append({'part': rec['part'], 'split': split, 'tokens': n,
                         'supervised_tokens': sum(v != -100 for v in item['labels']),
                         'fits': n <= c['max_seq_length']})
            dataset.append((rec['part'], item))
        if not dataset:
            raise ValueError('Empty split: ' + split)
        datasets[split] = dataset
    return datasets, {'method': 'full chat template tokenization, no truncation',
                      'max_seq_length': c['max_seq_length'], 'source_sha256': hashes,
                      'rows': rows}


def memory_snapshot(model=None):
    """Resident parameter storage (not logical dequantized size) and CUDA stats."""
    import torch
    free, total = torch.cuda.mem_get_info(0)
    result = {'free_gib': free / 2**30, 'total_gib': total / 2**30,
              'allocated_gib': torch.cuda.memory_allocated(0) / 2**30,
              'reserved_gib': torch.cuda.memory_reserved(0) / 2**30}
    physical = gpu_indices(os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    result['per_gpu'] = []
    for i in range(torch.cuda.device_count()):
        free_i, total_i = torch.cuda.mem_get_info(i)
        result['per_gpu'].append({'logical_gpu': i, 'physical_gpu': physical[i],
            'free_gib': free_i / 2**30, 'total_gib': total_i / 2**30,
            'allocated_gib': torch.cuda.memory_allocated(i) / 2**30,
            'reserved_gib': torch.cuda.memory_reserved(i) / 2**30,
            'peak_allocated_gib': torch.cuda.max_memory_allocated(i) / 2**30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved(i) / 2**30})
    if model is not None:
        groups, seen, embeddings = {}, set(), []
        for name, param in model.named_parameters():
            device = str(param.device)
            if 'embed_tokens' in name or 'lm_head' in name:
                embeddings.append({'name': name, 'device': device,
                                   'dtype': str(param.dtype), 'shape': list(param.shape)})
            if param.device.type == 'meta':
                continue
            storage = param.untyped_storage()
            identity = (device, storage.data_ptr())
            if identity in seen: continue
            seen.add(identity)
            family = 'vision' if '.visual.' in name else 'embedding' if 'embed_tokens' in name else 'lm_head' if 'lm_head' in name else 'lora' if 'lora_' in name else 'other'
            key = device + '/' + family
            groups[key] = groups.get(key, 0) + storage.nbytes() / 2**30
        result['parameter_storage_gib'] = groups
        result['embedding_and_head'] = embeddings
        result['hf_device_map'] = {k: str(v) for k, v in (getattr(model, 'hf_device_map', None) or {}).items()}
        result['note'] = 'Parameter storage excludes activations, optimizer states, buffers and some quantization metadata; device locations are observations, not proof of offload behavior during execution.'
    return result


def synchronize_gpus():
    import torch
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def validate_sharding(c, model):
    import torch
    expected = len(gpu_indices(c['gpu']))
    if expected > 1:
        devices = {p.device.index for p in model.parameters() if p.device.type == 'cuda'}
        if devices != set(range(expected)):
            raise RuntimeError(f'Model did not use all selected GPUs: actual={devices}, expected={expected}')
        if any(p.device.type != 'cuda' for p in model.parameters()):
            raise RuntimeError('Three-card recipe requires CUDA-resident parameters; unexpected CPU/meta placement')
        model.is_parallelizable = True
        model.model_parallel = True


def load_model(c, observer=None):
    # Unsloth must import before transformers/PEFT.
    from unsloth import FastModel
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    expected = len(gpu_indices(c['gpu']))
    if torch.cuda.device_count() != expected:
        raise RuntimeError('Visible CUDA device count differs from configuration')
    sharded = expected > 1
    if sharded and c.get('device_map') != 'balanced':
        raise ValueError('Multiple GPUs require explicit balanced model sharding')
    if sharded and c['offload_embedding']:
        raise ValueError('Disable embedding offload for this GPU-only sharding experiment')
    major, minor = torch.cuda.get_device_capability(0)
    use_bf16 = all(torch.cuda.get_device_capability(i)[0] >= 8 for i in range(expected)) and torch.cuda.is_bf16_supported()
    # Unsloth refuses fp16 for qwen3_5; on Turing (no bf16) Trainer fp16 autocast yields inf grad_norm.
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    placement = {'device_map': {'': 0}}
    if sharded:
        budget = c.get('weight_budget_per_gpu', '20GiB')
        placement = {'device_map': 'balanced', 'max_memory': {i: budget for i in range(expected)}}
    model, processor = FastModel.from_pretrained(
        model_name=c['model_path'], max_seq_length=c['max_seq_length'],
        dtype=dtype, load_in_4bit=True, full_finetuning=False,
        offload_embedding=c['offload_embedding'], local_files_only=True,
        **placement)
    if observer: observer('base_loaded', model)
    validate_sharding(c, model)
    model = FastModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=c['r'], lora_alpha=c['lora_alpha'], lora_dropout=0,
        bias='none', use_gradient_checkpointing='unsloth',
        random_state=c['seed'], use_rslora=False)
    if observer: observer('lora_created', model)
    validate_sharding(c, model)
    model.config.use_cache = False
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in n for n, _ in trainable):
        raise RuntimeError('Expected only LoRA parameters to be trainable')
    return model, processor, {'dtype': str(dtype), 'bf16': use_bf16, 'fp16': False,
        'compute_capability': [major, minor],
        'parallelism': 'single_process_layer_sharding' if sharded else 'single_gpu',
        'physical_gpu_indices': gpu_indices(c['gpu']),
        'trainable_parameters': sum(p.numel() for _, p in trainable)}


class Collator:
    def __call__(self, features):
        import torch
        if len(features) != 1:
            raise ValueError('This measured recipe requires micro-batch size 1')
        return {k: torch.tensor([features[0][k]], dtype=torch.long)
                for k in ('input_ids', 'attention_mask', 'labels')}


def make_trainer(c, model, tokenizer, train, val, output, max_steps=-1, probe=False):
    import torch
    from transformers import Trainer, TrainingArguments, TrainerCallback
    class FiniteLoss(TrainerCallback):
        def __init__(self):
            self.step_seconds = []
        def on_step_begin(self, args, state, control, **kwargs):
            import time
            synchronize_gpus()
            self.started = time.monotonic()
        def on_step_end(self, args, state, control, **kwargs):
            import time
            synchronize_gpus()
            self.step_seconds.append(time.monotonic() - self.started)
        def on_log(self, args, state, control, logs=None, **kwargs):
            import math
            for key in ('loss', 'eval_loss', 'grad_norm'):
                if key in (logs or {}) and not math.isfinite(float(logs[key])):
                    raise RuntimeError('Nonfinite ' + key)
    kwargs = dict(output_dir=str(output), per_device_train_batch_size=1,
        per_device_eval_batch_size=1, gradient_accumulation_steps=1 if probe else c['gradient_accumulation_steps'],
        learning_rate=c['learning_rate'], num_train_epochs=c['epochs'], max_steps=max_steps,
        optim=c['optim'], fp16=bool(c.get('fp16', False)), bf16=bool(c.get('bf16', False)),
        max_grad_norm=c.get('max_grad_norm', 1.0),
        logging_steps=1, logging_nan_inf_filter=False, report_to='none',
        save_strategy='no' if probe else 'steps', save_steps=c['save_steps'], save_total_limit=2,
        prediction_loss_only=True, remove_unused_columns=False, dataloader_num_workers=0,
        seed=c['seed'], data_seed=c['seed'], warmup_steps=0, weight_decay=0.0)
    args = TrainingArguments(**kwargs)
    monitor = FiniteLoss()
    trainer_args = dict(model=model, args=args, train_dataset=train,
                        eval_dataset=val, data_collator=Collator(), callbacks=[monitor])
    processor_key = 'processing_class' if 'processing_class' in inspect.signature(Trainer.__init__).parameters else 'tokenizer'
    trainer_args[processor_key] = tokenizer
    trainer = Trainer(**trainer_args)
    if len(gpu_indices(c['gpu'])) > 1:
        if not trainer.is_model_parallel or trainer.args.n_gpu != 1:
            raise RuntimeError('Trainer failed to select single-process model-parallel path; refusing DataParallel')
        validate_sharding(c, model)
    trainer.cad_monitor = monitor
    return trainer


def metadata(c, output, tokenizer):
    write_json(Path(output) / 'resolved_config.json', c)
    hashes = {}
    for name in ('config.json', 'tokenizer_config.json', 'chat_template.jinja'):
        p = Path(c['model_path']) / name
        if p.exists(): hashes[name] = digest(p)
    write_json(Path(output) / 'model_metadata.json', {'model_files_sha256': hashes,
        'chat_template': tokenizer.chat_template, 'model_path': c['model_path']})
    # Freeze reconstruction contract with adapter; no test answers copied.
    import shutil
    for name in ('xml_schemas.json', 'schema_guide.json', 'system_prompt.txt'):
        shutil.copy2(Path(c['sft_root']) / name, Path(output) / name)
