"""Exact sparse sequences, integer progressions and columnar tables over v4.

All optional forms expand before v4 restoration. No float arithmetic/rounding.
"""
from collections import Counter
from copy import deepcopy
from semantic_v4 import dumps, decode as decode_v4
from semantic_v5 import pack as pool_pack, unpack as pool_unpack

MARKERS = {'$sparse', '$range', '$columns', '$groups'}
MAX_CELLS = 2_000_000


def sequence(values):
    candidates = [values]
    if len(values) > 3:
        signatures = [dumps(v) for v in values]
        common, _ = Counter(signatures).most_common(1)[0]
        default = values[signatures.index(common)]
        candidates.append({'$sparse': [len(values), default,
            [[i, v] for i, v in enumerate(values) if signatures[i] != common]]})
        if all(type(v) is int for v in values):
            step = values[1] - values[0]
            if all(v == values[0] + i * step for i, v in enumerate(values)):
                candidates.append({'$range': [len(values), values[0], step]})
    return min(candidates, key=lambda x: len(dumps(x)))


def compact(value):
    if isinstance(value, list):
        return sequence([compact(v) for v in value])
    if not isinstance(value, dict):
        return value
    if set(value) & MARKERS:
        raise ValueError('Reserved v6 marker in source')
    plain = {k: compact(v) for k, v in value.items()}
    if list(value) == ['$records', 'rows'] and value['rows']:
        layouts, rows = value['$records'], value['rows']
        groups = [[] for _ in layouts]
        order = []
        for row in rows:
            index = row[0]
            if type(index) is not int or not 0 <= index < len(layouts) or len(row) != len(layouts[index]) + 1:
                raise ValueError('Invalid heterogeneous row')
            groups[index].append(row[1:])
            order.append(index)
        candidate = {'$groups': [compact({'$table': columns, 'rows': group})
                                  for columns, group in zip(layouts, groups)],
                     'order': sequence(order)}
        if len(dumps(candidate)) < len(dumps(plain)):
            return candidate
    if list(value) == ['$table', 'rows'] and value['rows']:
        columns, rows = value['$table'], value['rows']
        if not all(len(row) == len(columns) for row in rows):
            raise ValueError('Invalid source table')
        candidate = {'$columns': columns, 'count': len(rows),
                     'values': [sequence([compact(row[i]) for row in rows]) for i in range(len(columns))]}
        if len(dumps(candidate)) < len(dumps(plain)):
            return candidate
    return plain


def expand(value):
    remaining = MAX_CELLS
    def spend(n):
        nonlocal remaining
        if type(n) is not int or n < 0 or n > remaining:
            raise ValueError('Invalid/oversized expanded sequence')
        remaining -= n
    def visit(item):
        if isinstance(item, list):
            spend(len(item))
            return [visit(v) for v in item]
        if not isinstance(item, dict):
            return item
        if '$groups' in item:
            if set(item) != {'$groups', 'order'} or not isinstance(item['$groups'], list):
                raise ValueError('Invalid grouped records')
            groups = [visit(v) for v in item['$groups']]
            if not all(isinstance(g, dict) and set(g) == {'$table', 'rows'} and
                       isinstance(g['$table'], list) and isinstance(g['rows'], list) and
                       all(isinstance(r, list) and len(r) == len(g['$table']) for r in g['rows']) for g in groups):
                raise ValueError('Invalid record group')
            order = visit(item['order'])
            if not isinstance(order, list):
                raise ValueError('Invalid group order')
            positions, rows = [0] * len(groups), []
            for i in order:
                if type(i) is not int or not 0 <= i < len(groups) or positions[i] >= len(groups[i]['rows']):
                    raise ValueError('Invalid group occurrence')
                rows.append([i] + groups[i]['rows'][positions[i]])
                positions[i] += 1
            if any(p != len(g['rows']) for p, g in zip(positions, groups)):
                raise ValueError('Unused grouped rows')
            return {'$records': [g['$table'] for g in groups], 'rows': rows}
        if '$range' in item:
            spec = item['$range']
            if set(item) != {'$range'} or not isinstance(spec, list) or len(spec) != 3 or not all(type(v) is int for v in spec):
                raise ValueError('Invalid integer range')
            n, start, step = spec
            spend(n)
            return [start + i * step for i in range(n)]
        if '$sparse' in item:
            spec = item['$sparse']
            if set(item) != {'$sparse'} or not isinstance(spec, list) or len(spec) != 3:
                raise ValueError('Invalid sparse sequence')
            n, default, exceptions = spec
            spend(n)
            if not isinstance(exceptions, list):
                raise ValueError('Invalid sparse exceptions')
            # Charge nested defaults on each occurrence, preventing expansion bombs.
            result = [visit(default) for _ in range(n)]
            seen = set()
            for pair in exceptions:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError('Invalid sparse exception')
                i, v = pair
                if type(i) is not int or not 0 <= i < n or i in seen:
                    raise ValueError('Invalid/duplicate sparse index')
                seen.add(i)
                result[i] = visit(v)
            return result
        if '$columns' in item:
            if set(item) != {'$columns', 'count', 'values'}:
                raise ValueError('Invalid column table')
            columns, n, values = item['$columns'], item['count'], item['values']
            if not isinstance(columns, list) or not all(isinstance(k, str) for k in columns) or len(columns) != len(set(columns)):
                raise ValueError('Invalid columns')
            spend(n)
            if not isinstance(values, list) or len(values) != len(columns):
                raise ValueError('Column count mismatch')
            expanded = [visit(v) for v in values]
            if not all(isinstance(v, list) and len(v) == n for v in expanded):
                raise ValueError('Column length mismatch')
            return {'$table': deepcopy(columns), 'rows': [[v[i] for v in expanded] for i in range(n)]}
        return {k: visit(v) for k, v in item.items()}
    return visit(value)


def pack(value):
    # Whole-document selection protects against losing useful row-level pooling.
    original = pool_pack(value)
    columnar = pool_pack(compact(value))
    return min((original, columnar), key=lambda x: len(dumps(x)))


def unpack(value):
    return expand(pool_unpack(value))


def encode(label):
    if label.get('label_version') != 4:
        raise ValueError('Expected v4 source')
    return {'label_version': 6, **pack(label)}


def to_v4(label):
    if not isinstance(label, dict) or set(label) != {'label_version', 'pool', 'body'} or label['label_version'] != 6:
        raise ValueError('Expected v6 envelope')
    result = unpack({'pool': label['pool'], 'body': label['body']})
    if not isinstance(result, dict) or result.get('label_version') != 4:
        raise ValueError('Expected expanded v4 label')
    return result


def decode(label, context, registry):
    return decode_v4(to_v4(label), context, registry)
