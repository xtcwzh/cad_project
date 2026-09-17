"""Measure v2 samples; exact token budgets require the deployed local tokenizer.

No truncation and no network downloads. Character counts are not token counts.
"""
import argparse
from collections.abc import Mapping
from numbers import Integral
import json
from pathlib import Path


def single_sequence_length(encoded):
    """Handle token lists, BatchEncoding mappings and single-row tensors."""
    if isinstance(encoded, Mapping):
        if 'input_ids' not in encoded:
            raise ValueError('Tokenizer result missing input_ids')
        encoded = encoded['input_ids']
    if hasattr(encoded, 'tolist'):
        encoded = encoded.tolist()
    if not isinstance(encoded, (list, tuple)):
        raise ValueError('Expected token IDs, not rendered template text')
    if encoded and isinstance(encoded[0], (list, tuple)):
        if len(encoded) != 1:
            raise ValueError('Expected one sequence, not a token batch')
        encoded = encoded[0]
    if not all(isinstance(i, Integral) and not isinstance(i, bool) for i in encoded):
        raise ValueError('Invalid token ID sequence')
    return len(encoded)


def token_budget(messages, tokenizer, context_length, max_tokens):
    from llm_call import inference_messages
    prompt = inference_messages(messages)
    prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    target_ids = tokenizer.encode(messages[-1]['content'], add_special_tokens=False)
    prompt_len = single_sequence_length(prompt_ids)
    target_len = single_sequence_length(target_ids)
    result = {'prompt_tokens': prompt_len, 'target_tokens': target_len,
              'minimum_sft_tokens': prompt_len + target_len + 8}
    result['fits'] = (result['minimum_sft_tokens'] <= context_length
                      and prompt_len + max_tokens <= context_length
                      and target_len + 8 <= max_tokens)
    return result


def load_tokenizer(path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError('Run token checks in the model environment with transformers installed') from exc
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sft-root', type=Path,
                    default=Path(__file__).resolve().parents[2] / 'data' / 'sft_v3')
    ap.add_argument('--tokenizer', type=Path)
    ap.add_argument('--context-length', type=int)
    ap.add_argument('--max-tokens', type=int)
    args = ap.parse_args()
    if args.tokenizer and (not args.context_length or not args.max_tokens):
        ap.error('Exact checks require --context-length and --max-tokens')
    tokenizer = load_tokenizer(args.tokenizer) if args.tokenizer else None
    rows = []
    for path in sorted((args.sft_root / 'samples').glob('*.jsonl')):
        for line in path.read_text(encoding='utf-8').splitlines():
            rec = json.loads(line)
            messages = rec['messages']
            row = {'part': rec['part'], 'split': path.stem,
                   'input_chars': sum(len(m['content']) for m in messages[:-1]),
                   'target_chars': len(messages[-1]['content'])}
            if tokenizer:
                row.update(token_budget(messages, tokenizer, args.context_length, args.max_tokens))
            rows.append(row)
    if not rows:
        raise ValueError('No samples found')
    report = {'length_check_version': 2,
              'sft_length_method': 'prompt plus separately encoded target plus 8-token allowance; estimate, not full training template count',
              'token_counts_available': tokenizer is not None, 'rows': rows,
              'context_length': args.context_length, 'max_tokens': args.max_tokens,
              'max_input_chars': max(r['input_chars'] for r in rows),
              'max_target_chars': max(r['target_chars'] for r in rows),
              'all_fit': all(r['fits'] for r in rows) if tokenizer else None}
    (args.sft_root / 'length_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))
    if tokenizer:
        for split in sorted({r['split'] for r in rows}):
            group = [r for r in rows if r['split'] == split]
            lengths = sorted(r['minimum_sft_tokens'] for r in group)
            print(f"{split}: n={len(group)} min={lengths[0]} median={lengths[len(lengths)//2]} max={lengths[-1]} fits={sum(r['fits'] for r in group)}")
        print('TOTAL prompt_tokens=', sum(r['prompt_tokens'] for r in rows),
              'target_tokens=', sum(r['target_tokens'] for r in rows),
              'estimated_sft_tokens=', sum(r['minimum_sft_tokens'] for r in rows))
    return int(tokenizer is not None and not report['all_fit'])


if __name__ == '__main__':
    raise SystemExit(main())
