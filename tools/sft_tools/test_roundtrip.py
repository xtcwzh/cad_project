"""Compare restored trees directly to raw exports, not only compact labels.

Only snapshot removal and import_mode=feature_rebuild are allowed differences.
No Creo execution. No source dataset or template mutation.
"""
import argparse
import json
from pathlib import Path
from prepare_data import strip_snapshot, simplify_tree
from restore import restore_tree

def canon_simple(s):
    """规范化简化特征, 便于对比"""
    d = dict(s)
    for k in ("refs", "par"):
        if k in d and not d[k]:
            d.pop(k)
    return d


def compare_labels(a, b, diffs):
    ka, kb = set(a.keys()), set(b.keys())
    for k in ka - kb:
        diffs.append(f"root.{k} only in A")
    for k in kb - ka:
        diffs.append(f"root.{k} only in B")
    for k in ka & kb:
        if k == "feats":
            continue
        if a[k] != b[k]:
            diffs.append(f"root.{k}: {a[k]!r} != {b[k]!r}")
    fa, fb = a.get("feats", []), b.get("feats", [])
    if len(fa) != len(fb):
        diffs.append(f"feat count {len(fa)} != {len(fb)}")
        return
    for i, (x, y) in enumerate(zip(fa, fb)):
        cx, cy = canon_simple(x), canon_simple(y)
        if cx != cy:
            for k in set(cx) | set(cy):
                if cx.get(k) != cy.get(k):
                    diffs.append(
                        f"feat[{i}]({x.get('type')}).{k}: "
                        f"{str(cx.get(k))[:60]} != {str(cy.get(k))[:60]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(r'D:\pilot_dataset'))
    ap.add_argument('--sft-root', type=Path,
                    default=Path(__file__).resolve().parents[2] / 'data' / 'sft_v2')
    args = ap.parse_args()
    result = []
    sources = sorted((args.root / 'parts').glob('*.prt'))
    if not sources:
        raise ValueError('No source parts found')
    rest_dir = args.sft_root / 'restore'
    rest_dir.mkdir(parents=True, exist_ok=True)
    for part in sources:
        name = part.stem
        try:
            raw = json.loads((args.root / 'feature_trees' / f'{name}_feature_tree.json').read_text(encoding='utf-8'))
            label = json.loads((args.sft_root / 'simplified' / f'{name}.label.json').read_text(encoding='utf-8'))
            restored, pending = restore_tree(label)
            expected = strip_snapshot(raw)
            expected['import_mode'] = 'feature_rebuild'
            if restored != expected:
                raise ValueError('Restored tree differs from original export')
            if simplify_tree(restored, name) != label:
                raise ValueError('v2 encoding is not idempotent')
            (rest_dir / f'{name}.restored.json').write_text(json.dumps(restored, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
            result.append({'part': name, 'pass': True, 'source_empty_xml': len(pending)})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            result.append({'part': name, 'pass': False, 'error': str(exc)})
    report = {'check': 'raw export equality excluding snapshot and import policy',
              'creo_rebuild_verified': False, 'parts': result,
              'passed': sum(r['pass'] for r in result), 'total': len(result)}
    (args.sft_root / 'roundtrip_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"[roundtrip] raw fidelity PASS {report['passed']}/{report['total']}")
    for r in result:
        if not r['pass']:
            print(r)
    return int(report['passed'] != report['total'])


if __name__ == '__main__':
    raise SystemExit(main())
