import unittest
from semantic_v6 import compact, expand, pack, unpack
from semantic_v4 import dumps


class ColumnTests(unittest.TestCase):
    def test_integer_progression(self):
        value = list(range(1000, 1500, 3))
        compressed = compact(value)
        self.assertIn('$range', compressed)
        self.assertEqual(expand(compressed), value)

    def test_float_precision_and_signed_zero(self):
        value = [1.234567890123456] * 100
        value[3:7] = [-0.0, 0.0, 1, 1.0]
        self.assertEqual(dumps(expand(compact(value))), dumps(value))
        self.assertEqual(dumps(unpack(pack(value))), dumps(value))

    def test_column_table_and_independent_mutables(self):
        value = {'$table': ['id', 'point'], 'rows': [[i, [1.0, -0.0, 3.0]] for i in range(100)]}
        compressed = compact(value)
        self.assertIn('$columns', compressed)
        restored = expand(compressed)
        self.assertEqual(dumps(restored), dumps(value))
        restored['rows'][0][1][0] = 99
        self.assertEqual(restored['rows'][1][1][0], 1.0)

    def test_invalid_forms(self):
        bad = [{'$range': [2, 0.0, 1]}, {'$range': [-1, 0, 1]},
               {'$range': [1000000000, 0, 1]},
               {'$sparse': [2, 0, [[1, 2], [1, 3]]]},
               {'$columns': ['a'], 'count': 2, 'values': [[1]]}]
        for value in bad:
            with self.assertRaises(ValueError):
                expand(value)

    def test_reserved_marker(self):
        with self.assertRaises(ValueError):
            compact({'$range': 'original field'})

    def test_heterogeneous_order(self):
        value = {'$records': [['id', 'point'], ['id', 'radius']], 'rows': []}
        for i in range(100):
            value['rows'].append([0, i, [1.234567890123, 0.0, 0.0]])
            value['rows'].append([1, i + 1000, 3.141592653589])
        self.assertEqual(dumps(expand(compact(value))), dumps(value))
        self.assertEqual(dumps(unpack(pack(value))), dumps(value))
        with self.assertRaises(ValueError):
            expand({'$groups': [{'$table': ['x'], 'rows': [[1]]}], 'order': []})


if __name__ == '__main__':
    unittest.main()
