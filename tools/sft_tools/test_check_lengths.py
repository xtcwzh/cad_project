import unittest
from collections import UserDict
from check_lengths import single_sequence_length, token_budget


class LengthTests(unittest.TestCase):
    def test_supported_results(self):
        for value in ([1, 2, 3], [[1, 2, 3]],
                      UserDict(input_ids=[1, 2, 3], attention_mask=[1, 1, 1]),
                      {'input_ids': [[1, 2, 3]], 'attention_mask': [[1, 1, 1]]}):
            self.assertEqual(single_sequence_length(value), 3)

    def test_tensor_like(self):
        class Tensor:
            def tolist(self): return [[1, 2, 3, 4]]
        self.assertEqual(single_sequence_length({'input_ids': Tensor()}), 4)

    def test_invalid_results(self):
        for value in ({'attention_mask': [1]}, 'rendered', [[1], [2]], [True], [1.5]):
            with self.assertRaises(ValueError):
                single_sequence_length(value)

    def test_mapping_cannot_bypass_budget(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return UserDict(input_ids=[0] * 9000, attention_mask=[1] * 9000)
            def encode(self, text, **kwargs): return [0] * 100
        messages = [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'u'},
                    {'role': 'assistant', 'content': 'a'}]
        result = token_budget(messages, Tokenizer(), 8192, 4096)
        self.assertEqual(result['prompt_tokens'], 9000)
        self.assertEqual(result['minimum_sft_tokens'], 9108)
        self.assertFalse(result['fits'])


if __name__ == '__main__':
    unittest.main()
