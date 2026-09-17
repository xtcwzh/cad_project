"""Regression tests for observed data corruption and baseline answer leakage."""
import copy
import json
import unittest
from unittest.mock import patch

from prepare_data import simplify_tree, build_sample
from restore import restore_tree
from llm_call import inference_messages, chat
from evaluate import eval_one
from check_lengths import token_budget


def source():
    return {'name': 'part', 'units': 'mm', 'density': 1.0,
            'relations': ['d26=10'], 'native_snapshot': {'payload': 'SECRET'},
            'custom_future_field': {'keep': 0},
            'features': [{'feat_id': 61, 'feat_number': 1,
                          'feat_type': 917, 'feat_type_name': 'PRO_FEAT_PROTRUSION',
                          'parent_ids': [], 'child_ids': [],
                          'params_xml': '<PRO_E_FEATURE_FORM>PRO_EXTRUDE</PRO_E_FEATURE_FORM>',
                          'pattern_xml': '<pattern>original</pattern>',
                          'hole_screw_size': 'M1x.25', 'hole_thread_series': 'ISO',
                          'geom_items': [{'geom_id': 101, 'geom_type': 41}],
                          'dimensions': [{'dim_id': 26, 'symbol': 'd26', 'value': 1.123456789,
                                          'dim_type': 8, 'tolerance_plus': .01}],
                          'elem_refs': [{'elem_id': 6316, 'elem_path': [1, 2], 'ref_id': 101}],
                          'sketch': {'entities': [{'ent_type': 2, 'json_ent_id': 7,
                                                  'is_construction': True}],
                                     'dimensions': [{'entity_ids': [7], 'value': 10}],
                                     'constraints': [{'entity_ids': [7], 'constr_type': 1}]}}]}


class PipelineTests(unittest.TestCase):
    def test_raw_fidelity_and_sparse_entity_reference(self):
        raw = source()
        before = copy.deepcopy(raw)
        label = simplify_tree(raw, 'part')
        restored, _ = restore_tree(label)
        expected = copy.deepcopy(raw)
        expected.pop('native_snapshot')
        expected['import_mode'] = 'feature_rebuild'
        self.assertEqual(expected, restored)
        self.assertEqual(raw, before)
        self.assertNotIn('SECRET', json.dumps(label))

    def test_no_template_substitution_or_source_lookup(self):
        label = simplify_tree(source(), 'part')
        with patch('pathlib.Path.read_text', side_effect=AssertionError('source lookup')):
            restored, _ = restore_tree(label, {'PRO_FEAT_PROTRUSION': '<PRO_SWEEP/>'})
        self.assertEqual(source()['features'][0]['params_xml'], restored['features'][0]['params_xml'])

    def test_empty_xml_is_not_fabricated(self):
        raw = source()
        raw['features'][0]['params_xml'] = ''
        restored, pending = restore_tree(simplify_tree(raw, 'part'), {'PRO_FEAT_PROTRUSION': 'wrong'})
        self.assertEqual('', restored['features'][0]['params_xml'])
        self.assertEqual([61], pending)

    def test_old_and_malformed_labels_rejected(self):
        with self.assertRaises(ValueError):
            restore_tree({'feats': []})
        label = simplify_tree(source(), 'part')
        del label['feats'][0]['sk']['e'][0]['t']
        with self.assertRaises(ValueError):
            restore_tree(label)
        self.assertFalse(eval_one(simplify_tree(source(), 'part'), json.dumps(label))['schema_ok'])

    def test_dangling_reference_rejected(self):
        label = simplify_tree(source(), 'part')
        label['feats'][0]['sk']['e'][0]['id'] = 0
        with self.assertRaises(ValueError):
            restore_tree(label)

    def test_alias_collision_is_not_silent(self):
        raw = source()
        raw['features'][0]['id'] = 999
        with self.assertRaises(ValueError):
            simplify_tree(raw, 'part')

    def test_complete_evidence_no_target_leakage(self):
        evidence = {'faces': [{'face_id': 0, 'geometry': {'radius': 2}}],
                    'feature_candidates': [{'kind': 'slot', 'bbox_min': [81, -22, -9]}],
                    'llm_brief': {'candidate_summary': [{'kind': 'slot'}]}}
        sample = build_sample(simplify_tree(source(), 'part'), evidence)
        payload = json.loads(sample['messages'][1]['content'].split('\n', 1)[1])
        self.assertEqual(evidence['faces'], payload['faces'])
        self.assertEqual(evidence['feature_candidates'], payload['feature_candidates'])
        self.assertNotIn('llm_brief', payload)
        sent = inference_messages(sample['messages'])
        self.assertEqual(['system', 'user'], [m['role'] for m in sent])
        self.assertNotIn('PRO_EXTRUDE', json.dumps(sent))
        captured = []
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                return b'{"choices":[{"message":{"content":"{}"},"finish_reason":"stop"}]}'
        def fake_send(req, **kwargs):
            captured.append(json.loads(req.data))
            return Response()
        with patch('urllib.request.urlopen', side_effect=fake_send):
            chat('https://invalid.test/v1', 'dummy', 'model', sent)
        self.assertEqual(sent, captured[0]['messages'])
        with self.assertRaises(ValueError):
            chat('https://invalid.test/v1', 'dummy', 'model', sample['messages'])

    def test_perfect_prediction_metrics(self):
        label = simplify_tree(source(), 'part')
        metrics = eval_one(label, json.dumps(label))
        self.assertTrue(metrics['schema_ok'])
        self.assertEqual(0, metrics['dim_rel_err_median'])
        self.assertEqual(1, metrics['dim_exact_rate'])
        self.assertEqual(1, metrics['parent_edge_f1'])

    def test_token_budget_rejects_short_output_and_context(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                assert [m['role'] for m in messages] == ['system', 'user']
                return [0] * 20
            def encode(self, text, **kwargs): return [0] * 30
        messages = build_sample(simplify_tree(source(), 'part'), {})['messages']
        self.assertFalse(token_budget(messages, Tokenizer(), 100, 10)['fits'])
        self.assertFalse(token_budget(messages, Tokenizer(), 50, 40)['fits'])
        self.assertTrue(token_budget(messages, Tokenizer(), 100, 40)['fits'])

    def test_snapshot_and_missing_dimension_identity_rejected(self):
        label = simplify_tree(source(), 'part')
        label['native_snapshot'] = {'payload': 'forbidden'}
        with self.assertRaises(ValueError):
            restore_tree(label)
        label.pop('native_snapshot')
        del label['feats'][0]['dims'][0]['symbol']
        with self.assertRaises(ValueError):
            restore_tree(label)


if __name__ == '__main__':
    unittest.main()
