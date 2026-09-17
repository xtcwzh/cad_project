"""Restore v2/v3; v3 uses explicit caller context and frozen structural schemas.

Never load a source tree or copy another part's parameter values.
"""
import argparse
import json
from pathlib import Path
from artifact_io import restore_label


def restore_tree(label, templates=None, *, context=None, registry=None):
    # Obsolete argument accepted for audit compatibility; never consulted.
    tree = restore_label(label, context, registry)
    pending = [f['feat_id'] for f in tree['features'] if not f.get('params_xml')]
    return tree, pending


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('label')
    ap.add_argument('out')
    ap.add_argument('--context', type=Path, help='Declared caller metadata for v3')
    ap.add_argument('--schemas', type=Path, help='Frozen train-only registry for v3')
    args = ap.parse_args()
    context = json.loads(args.context.read_text(encoding='utf-8')) if args.context else None
    registry = json.loads(args.schemas.read_text(encoding='utf-8')) if args.schemas else None
    tree, pending = restore_tree(json.loads(Path(args.label).read_text(encoding='utf-8')),
                                 context=context, registry=registry)
    Path(args.out).write_text(json.dumps(tree, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(f"[restore] {len(tree['features'])} features; empty source XML: {len(pending)}")


if __name__ == '__main__':
    main()
