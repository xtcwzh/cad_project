"""Regression tests: no binary payload, no template parameter leakage, no loss."""
import copy
import json
import unittest
from unittest.mock import patch

from test_pipeline_v2 import source
from label_codec import encode as encode_v2
from semantic_v3 import (encode, decode, equivalent, build_registry, compact, expand,
                         pack_xml, unpack_xml, schema_id)
from payload_guard import assert_no_payload, scan_sample
from artifact_io import restore_label
from evaluate import eval_one


def raw_example():
    raw = source()
    raw['features'][0]['params_xml'] = (
        '<PRO_E_FEATURE_TREE type="compound">'
        '<PRO_E_FEATURE_FORM type="int">PRO_EXTRUDE</PRO_E_FEATURE_FORM>'
        '<PRO_E_DEPTH type="double">3.050</PRO_E_DEPTH>'
        '<PRO_E_DIRECTION type="int">0</PRO_E_DIRECTION>'
        '<PRO_E_REF type="selection"></PRO_E_REF>'
        '</PRO_E_FEATURE_TREE>')
    return raw


class SemanticTests(unittest.TestCase):
    def test_semantic_equivalence_no_source_lookup(self):
        raw = raw_example()
        registry = build_registry([encode_v2(raw, 'part')])
        label, context = encode(raw, 'part', registry)
        label = json.loads(json.dumps(label))
        with patch('pathlib.Path.read_text', side_effect=AssertionError('truth lookup')):
            restored = decode(label, context, registry)
        self.assertTrue(equivalent(raw, restored))
        self.assertNotIn('SECRET', json.dumps(label))
        self.assertEqual('extrude', label['feats'][0]['operation']['op'])
        self.assertEqual('3.050', label['feats'][0]['operation']['params']['depth'])
        self.assertNotIn('direction', label['feats'][0]['operation']['params'])
        self.assertNotIn('3.050', json.dumps(registry))

    def test_unknown_schema_kept_explicit_never_guessed(self):
        raw = raw_example()
        label, context = encode(raw, 'part', {'schemas': {}})
        self.assertIn('raw_xml', label['feats'][0]['operation'])
        self.assertTrue(equivalent(raw, decode(label, context, {'schemas': {}})))

    def test_operation_mismatch_and_registry_mutation_rejected(self):
        raw = raw_example()
        registry = build_registry([encode_v2(raw, 'part')])
        label, context = encode(raw, 'part', registry)
        bad = copy.deepcopy(label)
        bad['feats'][0]['operation']['params']['feature_form'] = 'PRO_SWEEP'
        with self.assertRaises(ValueError):
            decode(bad, context, registry)
        damaged = copy.deepcopy(registry)
        sid = label['feats'][0]['operation']['schema']
        damaged['schemas'][sid][0] = 'OTHER_ROOT'
        with self.assertRaises(ValueError):
            decode(label, context, damaged)

    def test_tables_preserve_missing_null_order_precision(self):
        rows = [{'a': 1.123456789123, 'b': None}, {'a': 2}, {'b': False}] * 10
        packed = compact(rows)
        self.assertEqual(rows, expand(json.loads(json.dumps(packed))))
        with self.assertRaises(ValueError):
            expand({'$table': ['a', 'b'], 'rows': [[1]]})
        with self.assertRaises(ValueError):
            expand({'$records': [['a']], 'rows': [[9, 1]]})

    def test_default_absence_distinguished_from_explicit_value(self):
        raw = raw_example()
        raw['features'][0].pop('sketch')
        raw['features'][0]['pattern_axis_angle'] = 0.0
        label, context = encode(raw, 'part', {'schemas': {}})
        restored = decode(label, context, {'schemas': {}})
        self.assertNotIn('sketch', restored['features'][0])
        self.assertIsInstance(restored['features'][0]['pattern_axis_angle'], float)
        self.assertTrue(equivalent(raw, restored))

    def test_payload_guards_include_serialized_assistant(self):
        for value in [{'native_snapshot': {}}, {'encoding': 'base64', 'data': 'x'},
                      {'nested': ['A' * 2048]}, {'data': 'data:application/octet-stream;base64,AAA='}]:
            with self.assertRaises(ValueError):
                assert_no_payload(value)
        with self.assertRaises(ValueError):
            scan_sample({'messages': [{'role': 'assistant',
                         'content': json.dumps({'native_snapshot': {'data': 'x'}})}]})

    def test_context_explicit_and_perfect_eval(self):
        raw = raw_example()
        registry = build_registry([encode_v2(raw, 'part')])
        label, context = encode(raw, 'part', registry)
        with self.assertRaises(ValueError):
            restore_label(label)
        with self.assertRaises(ValueError):
            restore_label(label, {**context, 'features': raw['features']}, registry)
        metrics = eval_one(label, json.dumps(label), context, registry)
        self.assertTrue(metrics['schema_ok'])
        self.assertEqual(0, metrics['dim_rel_err_median'])

    def test_hole_internal_extrude_does_not_change_semantic_operation(self):
        raw = raw_example()
        raw['features'][0]['feat_type_name'] = 'PRO_FEAT_HOLE'
        raw['features'][0]['feat_type'] = 911
        registry = build_registry([encode_v2(raw, 'part')])
        label, context = encode(raw, 'part', registry)
        self.assertEqual('hole', label['feats'][0]['operation']['op'])
        self.assertTrue(equivalent(raw, decode(label, context, registry)))


if __name__ == '__main__':
    unittest.main()
