"""Formal single-GPU QLoRA, fail on any overlength train/validation record."""
import argparse
import json
from pathlib import Path
from training_core import (config, offline, load_model, text_tokenizer, prepare,
                           make_trainer, write_json, metadata)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/qlora_probe.json')
    ap.add_argument('--length', type=int, required=True)
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--resume', help='Explicit Trainer checkpoint directory')
    args = ap.parse_args()
    c = config(args.config)
    c['max_seq_length'] = args.length
    if args.length < 2 or Path(args.run_name).name != args.run_name:
        raise ValueError('Invalid length or run name')
    offline(c['gpu'])
    # Import Unsloth before transformers, including CPU-only preflight imports.
    import unsloth
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(c['model_path'], local_files_only=True)
    datasets, report = prepare(c, tok)
    out = Path(c['output_root']) / ('train-' + args.run_name)
    if args.resume:
        checkpoint = Path(args.resume).expanduser().resolve()
        if checkpoint.parent != out.resolve() or not (checkpoint / 'trainer_state.json').is_file():
            raise ValueError('Resume must be a Trainer checkpoint inside this run')
        old = json.loads((out / 'resolved_config.json').read_text())
        if any(old.get(k) != v for k, v in c.items()):
            raise ValueError('Resume configuration differs from saved run')
        old_report = json.loads((out / 'exact_lengths.json').read_text())
        if old_report['source_sha256'] != report['source_sha256']:
            raise ValueError('Resume dataset differs from saved run')
    else:
        out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'exact_lengths.json', report)
    bad = [r for r in report['rows'] if r['split'] in ('train', 'val') and not r['fits']]
    if bad:
        raise ValueError(f'{len(bad)} train/val records exceed length; no samples truncated or omitted. See {out / "exact_lengths.json"}')
    model, processor, info = load_model(c)
    c.update(info)
    tokenizer = text_tokenizer(processor)
    # Ensure loader did not silently change the chat template/tokenization.
    actual, checked = prepare(c, tokenizer)
    if actual != datasets:
        raise ValueError('Loaded tokenizer differs from preflight tokenizer')
    metadata(c, out, tokenizer)
    write_json(out / 'runtime.json', info)
    trainer = make_trainer(c, model, tokenizer, [v for _, v in datasets['train']],
                           [v for _, v in datasets['val']], out)
    result = trainer.train(resume_from_checkpoint=str(checkpoint) if args.resume else None)
    evaluation = trainer.evaluate()
    adapter = out / 'adapter'
    model.save_pretrained(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    trainer.save_state()
    write_json(out / 'metrics.json', {'train': result.metrics, 'validation_loss': evaluation,
               'note': 'Validation loss is not feature-tree or Creo reconstruction accuracy'})
    print('Adapter:', adapter)


if __name__ == '__main__': main()
