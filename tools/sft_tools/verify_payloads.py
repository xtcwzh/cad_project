"""Audit model data, including JSON serialized inside JSONL assistant messages."""
import argparse
import json
from pathlib import Path
from payload_guard import assert_no_payload, scan_sample


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2] / 'data')
    args = ap.parse_args()
    counts, errors = {}, []
    for version in ('sft_v2', 'sft_v3'):
        folder = args.root / version
        count = 0
        for sub in ('clean', 'simplified', 'context', 'restore', 'samples'):
            for path in sorted((folder / sub).glob('*')):
                if path.suffix not in ('.json', '.jsonl'):
                    continue
                try:
                    if path.suffix == '.jsonl':
                        for line in path.read_text(encoding='utf-8').splitlines():
                            rec = json.loads(line)
                            scan_sample(rec)
                            # v3 user is JSON; v2 user has one descriptive prefix line.
                            user = rec['messages'][1]['content']
                            assert_no_payload(json.loads(user if user.startswith('{') else user.split('\n', 1)[1]))
                    else:
                        assert_no_payload(json.loads(path.read_text(encoding='utf-8')))
                    count += 1
                except (ValueError, KeyError, TypeError) as exc:
                    errors.append({'file': str(path), 'error': str(exc)})
        if count == 0:
            errors.append({'file': str(folder), 'error': 'No artifacts found'})
        counts[version] = count
    result = {'checked_files': counts, 'payload_hits': len(errors), 'errors': errors,
              'scope': 'Known payload keys/encoding, data URIs and long base64-like strings; includes nested sample JSON',
              'raw_exports': 'Untouched, original snapshots retained intentionally'}
    (args.root / 'payload_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
