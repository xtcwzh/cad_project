"""Batch v4 -> v5, exact input/label checks and semantic restored-tree checks."""
import argparse
import hashlib
import json
from pathlib import Path
from semantic_v4 import dumps, decode as decode_v4
from semantic_v5 import encode, decode, pack, unpack, to_v4
from semantic_v3 import equivalent
from payload_guard import scan_sample


def main(version=5):
    if version == 6:
        from semantic_v6 import encode, decode, pack, unpack, to_v4
    else:
        from semantic_v5 import encode, decode, pack, unpack, to_v4
    ap = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parents[2] / 'data'
    ap.add_argument('--source', type=Path, default=base / 'sft_v4')
    ap.add_argument('--out', type=Path, default=base / f'sft_v{version}')
    args = ap.parse_args()
    if args.source.resolve() == args.out.resolve():
        raise ValueError('Source and output must differ')
    registry_text = (args.source / 'xml_schemas.json').read_text(encoding='utf-8')
    registry = json.loads(registry_text)
    old_system = (args.source / 'system_prompt.txt').read_text(encoding='utf-8')
    system = ('Transport v5: output {label_version:5,pool:[ordinary JSON values],body:V4_LABEL}. '
              'Input also uses {pool,body}. Each document has an independent zero-based pool. '
              'Replace any value with {"$ref":index} to reuse an exactly equal pool value. '
              'Pool entries cannot contain $ref keys. References never cross documents. '
              'Empty pool and literal body are valid. Expand references before interpreting the following v4 contract. '
              'Do not round numbers or drop fields.\n' + old_system)
    if version == 6:
        system = system.replace('Transport v5:', 'Transport v6:').replace('label_version:5,', 'label_version:6,')
        system = ('After pool expansion, recursively expand optional forms: '
                  '{"$range":[n,start,step]} gives n integers start+i*step; '
                  '{"$sparse":[n,default,[[index,value],...]]} gives n copies with zero-based exceptions; '
                  '{"$columns":[names],"count":n,"values":[column_sequences]} expands to '
                  '{"$table":[names],"rows":[rows transposed from columns]}. '
                  'No duplicate exception indices; all columns must have count entries. '
                  '{"$groups":[tables],"order":group_index_sequence} expands to '
                  '{"$records":[each table columns],"rows":rows consumed from the indicated group in order, prefixed by group index}; consume all rows exactly once. '
                  'Literal arrays and ordinary v4 tables remain valid.\n' + system)
    for folder in ('samples', 'simplified', 'context', 'restore'):
        (args.out / folder).mkdir(parents=True, exist_ok=True)
    (args.out / 'xml_schemas.json').write_text(registry_text, encoding='utf-8')
    (args.out / 'schema_guide.json').write_text((args.source / 'schema_guide.json').read_text(encoding='utf-8'), encoding='utf-8')
    (args.out / 'system_prompt.txt').write_text(system, encoding='utf-8')
    rows = []
    source_hash = hashlib.sha256()
    for path in sorted((args.source / 'samples').glob('*.jsonl')):
        contents = path.read_bytes()
        source_hash.update(path.name.encode() + b'\0' + contents)
        records = []
        for line in contents.decode('utf-8').splitlines():
            rec = json.loads(line)
            name = rec['part']
            messages = rec['messages']
            old_label = json.loads(messages[-1]['content'])
            old_input = json.loads(messages[1]['content'])
            context = json.loads((args.source / 'context' / (name + '.context.json')).read_text(encoding='utf-8'))
            label = json.loads(dumps(encode(old_label)))
            user = json.loads(dumps(pack(old_input)))
            if dumps(to_v4(label)) != dumps(old_label) or dumps(unpack(user)) != dumps(old_input):
                raise ValueError('Exact serialized roundtrip failed: ' + name)
            restored = decode(label, context, registry)
            if not equivalent(decode_v4(old_label, context, registry), restored):
                raise ValueError('Semantic restore failed: ' + name)
            row = {'part': name, 'split': path.stem, 'exact_roundtrip': True,
                   'v4_input_chars': len(messages[0]['content']) + len(messages[1]['content']),
                   f'v{version}_input_chars': len(system) + len(dumps(user)),
                   'v4_target_chars': len(messages[-1]['content']), f'v{version}_target_chars': len(dumps(label)),
                   'input_pool_entries': len(user['pool']), 'target_pool_entries': len(label['pool'])}
            rows.append(row)
            messages[0]['content'], messages[1]['content'], messages[-1]['content'] = system, dumps(user), dumps(label)
            scan_sample(rec)
            records.append(dumps(rec))
            for folder, suffix, value in [('simplified', '.label.json', label), ('context', '.context.json', context), ('restore', '.restored.json', restored)]:
                (args.out / folder / (name + suffix)).write_text(dumps(value), encoding='utf-8')
        (args.out / 'samples' / path.name).write_text('\n'.join(records) + '\n', encoding='utf-8')
    if not rows:
        raise ValueError('No source samples')
    report = {'label_version': version, 'passed': len(rows), 'training_ready': False,
              'creo_rebuild_verified': False, 'token_counts_available': False,
              'source_samples_sha256': source_hash.hexdigest(),
              'registry_sha256': hashlib.sha256(registry_text.encode()).hexdigest(),
              'input_char_reduction': 1 - sum(r[f'v{version}_input_chars'] for r in rows) / sum(r['v4_input_chars'] for r in rows),
              'target_char_reduction': 1 - sum(r[f'v{version}_target_chars'] for r in rows) / sum(r['v4_target_chars'] for r in rows),
              'limits': ['Character savings are not token savings', 'No proof of Creo rebuild or model generation quality', 'Pool comes only from this document, never from another sample'],
              'parts': rows}
    (args.out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'parts'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
