"""Profile separately encoded sections; component counts are not additive."""
import argparse
import json
from pathlib import Path
from collections import Counter
from check_lengths import load_tokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sft-root', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    args = ap.parse_args()
    tok = load_tokenizer(args.tokenizer)
    def count(x):
        text = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, separators=(',', ':'))
        return len(tok.encode(text, add_special_tokens=False))
    rows = []
    for path in sorted((args.sft_root / 'samples').glob('*.jsonl')):
        for line in path.read_text(encoding='utf-8').splitlines():
            rec = json.loads(line)
            messages = rec['messages']
            label = json.loads(messages[-1]['content'])
            payload = json.loads(messages[1]['content'])
            transport = {}
            if label.get('label_version') in (5, 6):
                if label['label_version'] == 6:
                    from semantic_v6 import to_v4, unpack
                else:
                    from semantic_v5 import to_v4, unpack
                transport = {'target_pool_tokens': count(label['pool']),
                             'target_body_tokens': count(label['body']),
                             'input_pool_tokens': count(payload['pool']),
                             'input_body_tokens': count(payload['body'])}
                label, payload = to_v4(label), unpack(payload)
            sections = Counter()
            for f in label['feats']:
                for key, value in f.items():
                    if key == 'data':
                        for field, data in value.items():
                            sections['feature.' + field] += count(data)
                    else:
                        sections[key] += count(value)
            evidence = {k: count(v) for k, v in payload.get('evidence', {}).items()}
            rows.append({'part': rec['part'], 'split': path.stem,
                         'transport_sections_separately_encoded': transport,
                         'system_tokens': count(messages[0]['content']),
                         'user_tokens': count(messages[1]['content']),
                         'target_tokens': count(messages[-1]['content']),
                         'feature_sections_separately_encoded': dict(sections.most_common()),
                         'evidence_sections_separately_encoded': evidence})
    out = args.sft_root / 'token_profile.json'
    out.write_text(json.dumps({'note': 'Component encodings are diagnostic, not additive sequence lengths', 'rows': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    totals = Counter()
    for row in rows:
        totals.update(row['feature_sections_separately_encoded'])
    print('Expanded feature token contributors (v5 pooling excluded):', totals.most_common(10))
    print('Actual serialized target tokens:', sum(r['target_tokens'] for r in rows))
    print('Actual serialized user tokens:', sum(r['user_tokens'] for r in rows))
    print('Report:', out)


if __name__ == '__main__':
    main()
