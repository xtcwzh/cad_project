"""Read-only audit of raw export -> compact label -> restored tree.

Run with --out to save findings; never writes the source dataset or templates.
This checks field fidelity and reference integrity, NOT Creo rebuild success.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
from prepare_data import simplify_tree, build_sample
from restore import restore_tree
from test_roundtrip import compare_labels


def audit(root, templates, sft_root=None):
    counts = Counter()
    examples = {}
    affected = {}
    per_part = []
    saved = {}
    sft_root = sft_root or Path(__file__).resolve().parents[2] / 'data' / 'sft_v2'
    for path in sorted((sft_root / 'samples').glob('*.jsonl')):
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                rec = json.loads(line)
                saved[rec['part']] = rec

    def record(key, name, feat=None, detail=None):
        counts[key] += 1
        affected.setdefault(key, set()).add(name)
        examples.setdefault(key, {'part': name, 'feature_id': feat,
                                  'detail': detail})

    # Exact pairing used by prepare_data.py; excludes stray alternate exports.
    for path in sorted((root / 'parts').glob('*.prt')):
        name = path.stem
        raw = json.loads((root / 'feature_trees' /
                          f'{name}_feature_tree.json').read_text(encoding='utf-8'))
        ev = json.loads((root / 'evidence' /
                         f'{name}.evidence.v1.json').read_text(encoding='utf-8'))
        label = simplify_tree(raw, name)
        restored, _ = restore_tree(label, templates)
        counts['parts'] += 1
        diffs = []
        compare_labels(label, simplify_tree(restored, name), diffs)
        counts['compact_roundtrip_pass'] += not diffs
        expected = {k: v for k, v in raw.items() if k != 'native_snapshot'}
        expected['import_mode'] = 'feature_rebuild'
        counts['raw_fidelity_pass'] += restored == expected
        per_part.append({'part': name, 'compact_roundtrip_pass': not diffs,
                         'raw_fidelity_pass': restored == expected, 'diffs': diffs})
        sample = saved.get(name)
        if sample:
            counts['saved_labels_matching_current_simplification'] += (
                json.loads(sample['messages'][2]['content']) == label)
            counts['saved_inputs_matching_current_builder'] += (
                sample['messages'][1]['content'] ==
                build_sample(label, ev)['messages'][1]['content'])
        for f, r in zip(raw['features'], restored['features']):
            fid = f['feat_id']
            counts['features'] += 1
            if f.get('feat_type') != r.get('feat_type'):
                record('feature_type_integer_changed', name, fid,
                       [f.get('feat_type_name'), f.get('feat_type'), r.get('feat_type')])
            if not f.get('params_xml'):
                record('raw_params_xml_empty', name, fid, f.get('feat_type_name'))
            for key in ('params_xml', 'pattern_xml', 'geom_items',
                        'hole_screw_size', 'hole_thread_series'):
                if f.get(key):
                    counts[key + '_nonempty_in_source'] += 1
                    if f[key] != r.get(key):
                        record(key + '_changed', name, fid,
                               'nonempty source differs from restored field')
            for tag in ('PRO_E_FEATURE_FORM', 'PRO_E_FEATURE_TYPE',
                        'PRO_E_HLE_TYPE_NEW', 'PRO_E_EXT_DEPTH_TO_TYPE',
                        'PRO_E_STD_MATRLSIDE'):
                pattern = '<' + tag + r'\b[^>]*>(.*?)</' + tag + '>'
                a = re.findall(pattern, f.get('params_xml', ''), re.S)
                b = re.findall(pattern, r.get('params_xml', ''), re.S)
                if a and a != b:
                    record(tag + '_changed', name, fid, {'raw': a, 'restored': b})
            for ref in f.get('elem_refs', []):
                if ref.get('elem_path'):
                    record('nonempty_source_elem_path', name, fid)
            sk, rs = f.get('sketch') or {}, r.get('sketch') or {}
            if sk:
                counts['sketches'] += 1
                if sk.get('constraints') and not rs.get('constraints'):
                    record('sketch_constraints_dropped', name, fid)
                for a, b in zip(sk.get('entities', []), rs.get('entities', [])):
                    counts['sketch_entities'] += 1
                    if 'ent_type' not in b:
                        record('restored_entity_missing_ent_type', name, fid,
                               {'raw_type': a.get('ent_type'), 'restored_t': b.get('t')})
                    if a.get('json_ent_id') != b.get('json_ent_id'):
                        record('sketch_entity_id_changed', name, fid,
                               {'raw': a.get('json_ent_id'), 'restored': b.get('json_ent_id')})
                for stage, sketch in [('raw', sk), ('restored', rs)]:
                    ids = {e['json_ent_id'] for e in sketch.get('entities', [])}
                    for dim in sketch.get('dimensions', []):
                        missing = [i for i in dim.get('entity_ids', []) if i not in ids]
                        if missing:
                            record(stage + '_dangling_sketch_dimensions', name, fid,
                                   {'missing_ids': missing, 'available_ids': sorted(ids)})
        # Check exactly what build_sample passes, not what the full evidence contains.
        prompt = build_sample(label, ev)['messages'][1]['content']
        payload = json.loads(prompt.split('\n', 1)[1])
        if ev.get('faces') and payload.get('faces') != ev['faces']:
            record('parts_with_faces_omitted_from_prompt', name,
                   detail={'face_count': len(ev['faces'])})
        brief = ev.get('llm_brief', {})
        candidates = ev.get('feature_candidates', [])
        summaries = brief.get('candidate_summary', [])
        if len(candidates) == len(summaries):
            for full, summary in zip(candidates, summaries):
                omitted = sorted(set(full) - set(summary))
                if omitted and payload.get('feature_candidates') != candidates:
                    record('candidate_records_with_omitted_fields', name,
                           detail={'kind': full.get('kind'), 'omitted': omitted})
        if brief.get('candidate_summary') and 'llm_brief' in payload:
            record('parts_with_candidate_summary_repeated_in_prompt', name)

    return {'scope': 'parts directory pairing manifest, live code, read-only reconstruction',
            'limits': 'No Creo execution; differences alone do not prove geometric failure.',
            'counts': dict(counts),
            'affected_parts': {k: sorted(v) for k, v in affected.items()},
            'examples': examples, 'per_part': per_part}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(r'D:\pilot_dataset'))
    ap.add_argument('--out', type=Path)
    ap.add_argument('--sft-root', type=Path,
                    default=Path(__file__).resolve().parents[2] / 'data' / 'sft_v2')
    args = ap.parse_args()
    result = audit(args.root, {}, args.sft_root)
    if args.out:
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result['counts'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
