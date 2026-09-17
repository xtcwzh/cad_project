"""Independent processes for synthetic capacity probes and complete real samples."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from training_core import (config, offline, write_json, load_model, text_tokenizer,
                           prepare, make_trainer, metadata, memory_snapshot, synchronize_gpus)


def worker(c, length, mode, out):
    offline(c['gpu'])
    c['max_seq_length'] = length
    result = {'status': 'running', 'mode': mode, 'length': length, 'gpu': c['gpu'],
              'requested_placement': c.get('device_map', 'single_gpu'),
              'weight_budget_per_gpu': c.get('weight_budget_per_gpu'),
              'offload_embedding_requested': c['offload_embedding']}
    write_json(out / 'result.json', result)
    start = time.monotonic()
    snapshots = []
    def observe(phase, model=None):
        result['phase'] = phase
        snapshots.append({'phase': phase, **memory_snapshot(model)})
        result['memory_snapshots'] = snapshots
        write_json(out / 'result.json', result)
    try:
        result['phase'] = 'loading_model'
        write_json(out / 'result.json', result)
        model, processor, info = load_model(c, observer=observe)
        c.update(info)
        import torch
        tokenizer = text_tokenizer(processor)
        if mode == 'synthetic':
            # Capacity only: fabricated tokens never written as CAD training data.
            seed = tokenizer.encode('CAD geometry feature dimensions 1 2 3. ', add_special_tokens=False)
            if not seed: raise ValueError('Empty synthetic seed')
            ids = (seed * (length // len(seed) + 1))[:length]
            item = {'input_ids': ids, 'attention_mask': [1] * length,
                    'labels': [-100] * (length // 2) + ids[length // 2:]}
            dataset = [item]
            result['sample'] = 'synthetic; not CAD and not a model-quality test'
        else:
            datasets, report = prepare(c, tokenizer)
            write_json(out / 'exact_lengths.json', report)
            fitting = [(name, item) for name, item in datasets['train'] if len(item['input_ids']) <= length]
            if not fitting:
                result.update(status='no_fitting_real_sample', elapsed_seconds=time.monotonic() - start)
                write_json(out / 'result.json', result)
                return 2
            name, item = max(fitting, key=lambda pair: len(pair[1]['input_ids']))
            dataset = [item]
            result['sample'] = name
        result['actual_tokens'] = len(item['input_ids'])
        result['load_seconds'] = time.monotonic() - start
        # Sample every trainable tensor so a single unmodified module is not a false failure.
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        before = {n: p.detach().flatten()[:256].float().cpu().clone() for n, p in params.items()}
        trainer = make_trainer(c, model, tokenizer, dataset, None, out / 'trainer',
                               max_steps=c['probe_steps'], probe=True)
        observe('trainer_initialized', model)
        synchronize_gpus()
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
        train_start = time.monotonic()
        result['phase'] = 'training'
        write_json(out / 'result.json', result)
        trained = trainer.train()
        synchronize_gpus()
        elapsed = time.monotonic() - train_start
        training_memory = memory_snapshot()
        changed = any(not torch.equal(before[n], p.detach().flatten()[:256].float().cpu()) for n, p in params.items())
        if not all(torch.isfinite(p.detach()).all().item() for p in params.values()):
            raise RuntimeError('Nonfinite LoRA parameters')
        if not changed: raise RuntimeError('No sampled LoRA parameter changed; optimizer update unverified')
        if trainer.state.global_step != c['probe_steps']:
            raise RuntimeError('Requested optimizer steps did not complete')
        adapter = out / 'adapter'
        result['phase'] = 'saving_adapter'
        write_json(out / 'result.json', result)
        model.save_pretrained(str(adapter))
        tokenizer.save_pretrained(str(adapter))
        # Re-read the standard PEFT artifact and compare it to live adapter values.
        from peft import get_peft_model_state_dict
        from peft.utils.save_and_load import load_peft_weights
        saved = load_peft_weights(str(adapter), device='cpu')
        live = get_peft_model_state_dict(model)
        if set(saved) != set(live) or not all(torch.equal(saved[k], live[k].detach().cpu()) for k in saved):
            raise RuntimeError('Saved adapter readback mismatch')
        result.update(status='passed', runtime=info, optimizer_steps=trainer.state.global_step,
            phase='complete',
            loss=trained.training_loss, parameter_update_verified=True, adapter_readback_verified=True,
            train_seconds=elapsed, seconds_per_step=elapsed / trainer.state.global_step,
            optimizer_step_seconds=trainer.cad_monitor.step_seconds,
            training_memory=training_memory,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            elapsed_seconds=time.monotonic() - start,
            caveat='Includes first-run compilation; short probe is not long-run stability or task quality')
        metadata(c, out, tokenizer)
        write_json(out / 'result.json', result)
        return 0
    except Exception as exc:
        # A generic failure must not be mislabeled as out-of-memory.
        import torch
        status = 'oom' if isinstance(exc, torch.cuda.OutOfMemoryError) else 'error'
        if torch.cuda.is_available():
            try: result['failure_memory'] = memory_snapshot()
            except Exception: pass
        result.update(status=status, error=str(exc), traceback=traceback.format_exc(),
                      elapsed_seconds=time.monotonic() - start)
        write_json(out / 'result.json', result)
        traceback.print_exc()
        return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/qlora_probe.json')
    ap.add_argument('--lengths', nargs='+', type=int)
    ap.add_argument('--mode', choices=['synthetic', 'real'], default='synthetic')
    ap.add_argument('--run-name', default=None)
    ap.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--worker-output', help=argparse.SUPPRESS)
    args = ap.parse_args()
    c = config(args.config)
    lengths = args.lengths or c['lengths']
    if any(n < 2 for n in lengths) or c['probe_steps'] < 2:
        raise ValueError('Positive lengths and at least two optimizer steps required')
    if args.worker:
        return worker(c, lengths[0], args.mode, Path(args.worker_output))
    name = args.run_name or time.strftime('%Y%m%d-%H%M%S')
    if Path(name).name != name: raise ValueError('run-name must be one directory name')
    root = Path(c['output_root']) / ('probe-' + args.mode + '-' + name)
    root.mkdir(parents=True, exist_ok=False)
    results = []
    for length in lengths:
        out = root / str(length)
        out.mkdir()
        cmd = [sys.executable, str(Path(__file__).resolve()), '--config',
               str(Path(args.config).resolve()), '--lengths', str(length), '--mode', args.mode,
               '--worker', '--worker-output', str(out)]
        print(f'Start {args.mode} {length}; log: {out / "worker.log"}', flush=True)
        with (out / 'worker.log').open('w', encoding='utf-8') as log:
            try:
                run = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=c['timeout_seconds'])
                file = out / 'result.json'
                result = json.loads(file.read_text()) if file.exists() else {'status': 'process_error'}
                result['returncode'] = run.returncode
                if result['status'] == 'running' or (run.returncode != 0 and result['status'] == 'passed'):
                    result['status'] = 'process_error'
            except subprocess.TimeoutExpired:
                result = {'status': 'timeout', 'timeout_seconds': c['timeout_seconds']}
        result.update(length=length, mode=args.mode)
        write_json(out / 'result.json', result)
        results.append(result)
        write_json(root / 'summary.json', {'config': c, 'results': results})
        print(length, result['status'], flush=True)
        # Larger sequences won't fix OOM/config/kernel failures; real sample gaps can continue.
        if result['status'] not in ('passed', 'no_fitting_real_sample'):
            break
    print('Summary:', root / 'summary.json')
    return int(any(r['status'] not in ('passed', 'no_fitting_real_sample') for r in results))


if __name__ == '__main__': sys.exit(main())
