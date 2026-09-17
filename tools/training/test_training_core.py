import json
from pathlib import Path
import tempfile
import unittest
from training_core import encode_record, prepare, offline, write_json, gpu_indices


class Tokenizer:
    chat_template = 'test-template'
    def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
        value = ''.join('<' + m['role'] + '>' + m['content'] + '</end>' for m in messages)
        return value + ('<assistant>' if add_generation_prompt else '')
    def encode(self, text, **kwargs): return [ord(c) for c in text]


def record(part='one'):
    return {'part': part, 'messages': [{'role': 'system', 'content': 'system'},
        {'role': 'user', 'content': 'evidence'}, {'role': 'assistant', 'content': '{"label":1}'}]}


class TrainingTests(unittest.TestCase):
    def test_complete_template_and_mask(self):
        rec = record()
        tok = Tokenizer()
        item = encode_record(rec, tok)
        full = tok.encode(tok.apply_chat_template(rec['messages'], False))
        prefix = tok.encode(tok.apply_chat_template(rec['messages'][:-1], True))
        self.assertEqual(item['input_ids'], full)
        self.assertEqual(item['labels'][:len(prefix)], [-100] * len(prefix))
        self.assertEqual(item['labels'][len(prefix):], full[len(prefix):])
        self.assertEqual(len(item['attention_mask']), len(full))

    def test_prefix_mismatch_refused(self):
        class Bad(Tokenizer):
            def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
                return 'different' if add_generation_prompt else 'full'
        with self.assertRaises(ValueError): encode_record(record(), Bad())

    def test_token_boundary_mismatch_refused(self):
        class Bad(Tokenizer):
            def encode(self, text, **kwargs): return [len(text)]
        with self.assertRaises(ValueError): encode_record(record(), Bad())

    def test_oversize_retained_and_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'samples'
            p.mkdir()
            for split in ('train', 'val', 'test'):
                (p / (split + '.jsonl')).write_text(json.dumps(record(split)) + '\n')
            c = {'sft_root': temp, 'max_seq_length': 2, 'enable_thinking': False}
            datasets, report = prepare(c, Tokenizer())
            self.assertTrue(all(not r['fits'] for r in report['rows']))
            self.assertEqual(len(datasets['train']), 1)
            self.assertGreater(len(datasets['train'][0][1]['input_ids']), 2)
            (p / 'test.jsonl').write_text(json.dumps(record('train')) + '\n')
            with self.assertRaises(ValueError): prepare(c, Tokenizer())

    def test_gpu_selection(self):
        self.assertEqual(gpu_indices('0,1,2'), [0, 1, 2])
        self.assertEqual(gpu_indices('2,3,4'), [2, 3, 4])
        for value in ('0,0', '0,', '-1', 'all'):
            with self.assertRaises(ValueError): gpu_indices(value)

    def test_distributed_launch_rejected(self):
        from unittest.mock import patch
        with patch.dict('os.environ', {'WORLD_SIZE': '3'}):
            with self.assertRaises(ValueError): offline('0,1,2')

    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'report.json'
            write_json(p, {'ok': True})
            self.assertEqual(json.loads(p.read_text()), {'ok': True})
            self.assertFalse(p.with_suffix('.json.tmp').exists())


if __name__ == '__main__': unittest.main()
