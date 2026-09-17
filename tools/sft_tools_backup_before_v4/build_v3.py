"""Build and verify v3 against immutable raw exports using train-only schemas.

Batch/reproducible by design: one manifest, one frozen registry, no per-part
manual edits or source-tree lookup during restore.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from label_codec import encode as encode_v2
from semantic_v3 import (build_registry, encode, decode, equivalent, slots,
                         materialize, default_text, compact)
from payload_guard import assert_no_payload, scan_sample


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def registry_guide(registry):
    # The same frozen guide is provided to EVERY split and sample.
    # Never choose a schema shortlist using the answer for a test part.
    return {sid: {'root': shape[0], 'slots': [
        [path, node.get('type'), default_text(node)]
        for path, node in slots(materialize(shape))]}
        for sid, shape in registry['schemas'].items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(r'D:\pilot_dataset'))
    ap.add_argument('--v2-root', type=Path,
                    default=Path(__file__).resolve().parents[2] / 'data' / 'sft_v2')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).resolve().parents[2] / 'data' / 'sft_v3')
    args = ap.parse_args()
    manifest = json.loads((args.v2_root / 'report.json').read_text(encoding='utf-8'))['parts']
    raw = {p['name']: json.loads((args.root / 'feature_trees' /
           (p['name'] + '_feature_tree.json')).read_text(encoding='utf-8')) for p in manifest}
    registry = build_registry([encode_v2(raw[p['name']], p['name'])
                               for p in manifest if p['split'] == 'train'])
    assert_no_payload(registry)
    for folder in ('simplified', 'context', 'restore', 'samples'):
        (args.out / folder).mkdir(parents=True, exist_ok=True)
    (args.out / 'xml_schemas.json').write_text(dumps(registry), encoding='utf-8')
    guide = registry_guide(registry)
    (args.out / 'schema_guide.json').write_text(dumps(guide), encoding='utf-8')
    system = (
        'Recover a Creo operation program as JSON with label_version=3, model, feats. '
        'Use the semantic_v3 contract: each feature has data, operation, pattern, optional absent. '
        'operation/pattern use op plus schema/params OR raw_xml OR empty=true. '
        'Use schema paths exactly, including array indices. Parameter values are strings or null; '
        'omitted slots use the declared lexical default. Only op values consistent with the XML '
        'are accepted. data uses v2 feature aliases; dims keep complete dimension objects. '
        'Geometry/sketch arrays may use {$table:[column names],rows:[[values]]}. '
        'Heterogeneous arrays may use {$records:[[layout columns]],rows:[[layout_index,values...]]}. '
        'Preserve IDs/references, constraints, geometric mappings and nondefault feature settings. '
        'Do not output provided_context fields in model; the caller supplies them. '
        'No snapshots or binary payloads. Unsupported XML forms use explicit raw_xml. '
        'BREP face indices are not Creo historical geometry IDs. '
        'Frozen feature defaults: ')
    from semantic_v3 import DEFAULTS
    from label_codec import ROOT_KEYS, FEAT_KEYS, SKETCH_KEYS, ENTITY_KEYS
    system += dumps(DEFAULTS)
    system += '\nField aliases (original->label): ' + dumps({
        'model': ROOT_KEYS, 'feature': FEAT_KEYS, 'sketch': SKETCH_KEYS, 'entity': ENTITY_KEYS})
    system += '\nabsent lists default keys that did not exist in source and must not be injected.'
    system += '\nFrozen XML schemas (train structures only):\n' + dumps(guide)
    (args.out / 'system_prompt.txt').write_text(system, encoding='utf-8')
    counts, xml_counts, splits, rows = Counter(), Counter(), {}, []
    for part in manifest:
        name, split = part['name'], part['split']
        label, context = encode(raw[name], name, registry)
        # Exercise serialized artifacts, not shared in-memory Python objects.
        label = json.loads(dumps(label))
        context = json.loads(dumps(context))
        restored = decode(label, context, registry)
        if not equivalent(raw[name], restored):
            raise ValueError(f'Original semantic comparison failed: {name}')
        for folder, suffix, value in [('simplified', '.label.json', label),
                                      ('context', '.context.json', context),
                                      ('restore', '.restored.json', restored)]:
            assert_no_payload(value)
            (args.out / folder / (name + suffix)).write_text(dumps(value), encoding='utf-8')
        ev = json.loads((args.root / 'evidence' / (name + '.evidence.v1.json')).read_text(encoding='utf-8'))
        # Context is explicitly an annotated caller input, NOT inferred from BREP.
        payload = {'provided_context': context,
                   'evidence': compact({k: v for k, v in ev.items() if k != 'llm_brief'})}
        assert_no_payload(payload)
        messages = [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': dumps(payload)},
                    {'role': 'assistant', 'content': dumps(label)}]
        rec = {'part': name, 'messages': messages}
        scan_sample(rec)
        splits.setdefault(split, []).append(rec)
        for f in label['feats']:
            for key in ('operation', 'pattern'):
                x = f[key]
                mode = 'schema' if 'schema' in x else 'raw_xml' if 'raw_xml' in x else 'empty'
                xml_counts[f'{x["op"]}:{mode}'] += 1
        old_size = (args.v2_root / 'simplified' / (name + '.label.json')).stat().st_size
        counts['v2_bytes'] += old_size
        counts['v3_bytes'] += len(dumps(label).encode('utf-8'))
        counts['passed'] += 1
        rows.append({'part': name, 'split': split, 'semantic_equivalence': True,
                     'input_chars': len(system) + len(messages[1]['content']),
                     'target_chars': len(messages[2]['content']),
                     'v2_bytes': old_size, 'v3_bytes': len(dumps(label).encode('utf-8'))})
    for split, records in splits.items():
        (args.out / 'samples' / (split + '.jsonl')).write_text(
            '\n'.join(dumps(r) for r in records) + '\n', encoding='utf-8')
    report = {'label_version': 3, 'training_ready': False, 'creo_rebuild_verified': False,
              'check': 'all JSON fields exact; XML tags/attributes/text/order equivalent ignoring layout whitespace',
              'base64_payload_hits': 0, 'passed': counts['passed'],
              'schemas_from_train_only': len(registry['schemas']),
              'registry_sha256': hashlib.sha256(dumps(registry).encode()).hexdigest(),
              'counts': {k: len(v) for k, v in splits.items()},
              'v2_bytes': counts['v2_bytes'], 'v3_bytes': counts['v3_bytes'],
              'label_reduction_fraction': 1 - counts['v3_bytes'] / counts['v2_bytes'],
              'shared_system_chars': len(system),
              'max_target_chars': max(r['target_chars'] for r in rows),
              'avg_target_chars': round(sum(r['target_chars'] for r in rows) / len(rows)),
              'xml_modes': dict(xml_counts), 'parts': rows,
              'limits': ['Geometry mapping tables retained until importer supports symbolic references',
                         'Unsupported/unseen XML preserved explicitly',
                         'Registry must be frozen with model checkpoint',
                         'Real tokenizer/context budget and Creo rebuild unverified',
                         'Caller context is annotated metadata, not recovered from geometry']}
    (args.out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('parts', 'xml_modes')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
