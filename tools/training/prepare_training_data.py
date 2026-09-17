"""Exact full-template lengths and assistant masking validation, CPU only."""
import argparse
from pathlib import Path
from training_core import config, offline, prepare, write_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/qlora_probe.json')
    ap.add_argument('--length', type=int)
    args = ap.parse_args()
    c = config(args.config)
    if args.length: c['max_seq_length'] = args.length
    offline(c['gpu'])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(c['model_path'], local_files_only=True)
    _, report = prepare(c, tok)
    out = Path(c['output_root']) / 'exact_lengths.json'
    write_json(out, report)
    for split in ('train', 'val', 'test'):
        rows = [r for r in report['rows'] if r['split'] == split]
        lengths = sorted(r['tokens'] for r in rows)
        print(split, 'min/median/max:', lengths[0], lengths[len(lengths)//2], lengths[-1],
              'fits:', sum(r['fits'] for r in rows), '/', len(rows))
    print('Report:', out)

if __name__ == '__main__': main()
