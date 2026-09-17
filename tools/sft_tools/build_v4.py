"""Build an isolated v4 candidate from v3; verify every restored field."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from semantic_v4 import encode, decode, dumps
from semantic_v3 import decode as decode_v3, equivalent
from payload_guard import assert_no_payload, scan_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parents[2] / 'data'
    parser.add_argument('--source', type=Path, default=base / 'sft_v3')
    parser.add_argument('--out', type=Path, default=base / 'sft_v4')
    args = parser.parse_args()
    if args.source.resolve() == args.out.resolve():
        raise ValueError('Output must be separate from v3')
    registry = json.loads((args.source / 'xml_schemas.json').read_text(encoding='utf-8'))
    guide = json.loads((args.source / 'schema_guide.json').read_text(encoding='utf-8'))
    # Same train-only guide for every sample; position defines the slot index.
    compact_guide = {sid: [r[0] for r in value['slots']] for sid, value in guide.items()}
    old_system = (args.source / 'system_prompt.txt').read_text(encoding='utf-8')
    system = old_system.split('\nFrozen XML schemas')[0].replace('label_version=3', 'label_version=4')
    system = system.replace('schema/params', 'schema/values').replace('semantic_v3', 'semantic_v4')
    system = system.replace('Use schema paths exactly, including array indices. Parameter values are strings or null; omitted slots use the declared lexical default.',
        'values is a list of [zero-based slot index, lexical string or null]. Omitted slots retain the frozen schema defaults (int=0, double=0.00, otherwise null).')
    system += ('\nXML may alternatively use tree=[tag,attributes,text,children]. '
               'e: expands to PRO_E_; x: expands to PRO_XML_; other tags unchanged. '
               'Leaf text is lexical string or null; compound text is null. '
               'Raw XML remains valid for unsupported structures.\nFrozen XML slot names in index order:\n') + dumps(compact_guide)
    for folder in ('samples', 'simplified', 'context', 'restore'):
        (args.out / folder).mkdir(parents=True, exist_ok=True)
    for filename, value in [('xml_schemas.json', registry), ('schema_guide.json', compact_guide)]:
        (args.out / filename).write_text(dumps(value), encoding='utf-8')
    (args.out / 'system_prompt.txt').write_text(system, encoding='utf-8')
    rows, modes = [], Counter()
    for path in sorted((args.source / 'samples').glob('*.jsonl')):
        records = []
        for line in path.read_text(encoding='utf-8').splitlines():
            sample = json.loads(line)
            name = sample['part']
            old = json.loads(sample['messages'][-1]['content'])
            context = json.loads((args.source / 'context' / (name + '.context.json')).read_text(encoding='utf-8'))
            label = json.loads(dumps(encode(old, registry)))
            restored = decode(label, context, registry)
            reference = decode_v3(old, context, registry)
            if not equivalent(reference, restored):
                raise ValueError('Semantic roundtrip failed: ' + name)
            for folder, suffix, value in [('simplified', '.label.json', label), ('context', '.context.json', context), ('restore', '.restored.json', restored)]:
                assert_no_payload(value)
                (args.out / folder / (name + suffix)).write_text(dumps(value), encoding='utf-8')
            for feature in label['feats']:
                for key in ('operation', 'pattern'):
                    modes.update(k for k in ('tree', 'raw_xml', 'schema', 'empty') if k in feature[key])
            target = dumps(label)
            old_input = sum(len(m['content']) for m in sample['messages'][:-1])
            sample['messages'][0]['content'] = system
            sample['messages'][-1]['content'] = target
            scan_sample(sample)
            records.append(dumps(sample))
            rows.append({'part': name, 'split': path.stem, 'semantic_equivalence': True,
                         'v3_target_chars': len(dumps(old)), 'v4_target_chars': len(target),
                         'v3_input_chars': old_input,
                         'v4_input_chars': sum(len(m['content']) for m in sample['messages'][:-1])})
        (args.out / 'samples' / path.name).write_text('\n'.join(records) + '\n', encoding='utf-8')
    if not rows:
        raise ValueError('No source samples')
    report = {'label_version': 4, 'training_ready': False, 'creo_rebuild_verified': False,
              'token_counts_available': False, 'passed': len(rows), 'xml_modes': dict(modes),
              'v3_system_chars': len(old_system), 'v4_system_chars': len(system),
              'target_char_reduction': 1 - sum(r['v4_target_chars'] for r in rows) / sum(r['v3_target_chars'] for r in rows),
              'registry_sha256': hashlib.sha256(dumps(registry).encode()).hexdigest(),
              'limits': ['Input evidence unchanged', 'No numeric rounding or geometry deletion',
                         'Field/XML equivalence is not proof of Creo rebuild', 'Measure actual tokens on server'],
              'parts': rows}
    (args.out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'parts'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
