import unittest
from semantic_v5 import pack, unpack
from semantic_v4 import dumps


class PoolTests(unittest.TestCase):
    def test_exact_numbers_and_structure(self):
        vector = [1.234567890123456, -0.0, 1.0, 1, None, '']
        value = {'items': [vector for _ in range(30)], 'other': [0.0, -0.0, 1.0, 1]}
        packed = pack(value)
        self.assertTrue(packed['pool'])
        self.assertLess(len(dumps(packed)), len(dumps(value)))
        self.assertEqual(dumps(unpack(packed)), dumps(value))
        self.assertEqual(dumps(pack(value)), dumps(packed))

    def test_reject_bad_reference(self):
        for index in (-1, 1, True, '0'):
            with self.assertRaises(ValueError):
                unpack({'pool': ['value'], 'body': {'$ref': index}})

    def test_reject_cycles_and_reserved_collision(self):
        with self.assertRaises(ValueError):
            unpack({'pool': [{'$ref': 0}], 'body': {'$ref': 0}})
        with self.assertRaises(ValueError):
            pack({'$ref': 'ordinary source field'})

    def test_mutable_values_are_independent(self):
        out = unpack({'pool': [[1, 2]], 'body': [{'$ref': 0}, {'$ref': 0}]})
        out[0][0] = 99
        self.assertEqual(out[1], [1, 2])

    def test_expansion_limit(self):
        with self.assertRaises(ValueError):
            unpack({'pool': ['a' * 100], 'body': [{'$ref': 0}] * 10}, max_expanded_chars=200)


if __name__ == '__main__':
    unittest.main()
