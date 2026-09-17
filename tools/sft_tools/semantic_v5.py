"""Self-contained exact-value pooling. No rounding, external lookup or binary data.

Pool entries are ordinary JSON, never references. Only identical serialized
values are shared, preserving integer/float distinctions and signed zero.
"""
from collections import Counter
from copy import deepcopy
from semantic_v4 import dumps, decode as decode_v4
from payload_guard import assert_no_payload


def walk(value):
    yield value
    if isinstance(value, dict):
        if '$ref' in value:
            raise ValueError('Reserved $ref key in source/pool')
        for item in value.values():
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item)


def pack(value):
    assert_no_payload(value)
    counts, originals = Counter(), {}
    for item in walk(value):
        text = dumps(item)
        # Large subtrees stay readable; small repeated vectors/records/scalars
        # are useful local dictionary candidates.
        if 16 <= len(text) <= 4096:
            counts[text] += 1
            originals.setdefault(text, item)
    candidates = {text for text, n in counts.items()
                  if n > 1 and (n - 1) * len(text) > n * 16 + 8}
    # Recalculate actual occurrences after parent pooling to avoid dead entries.
    while True:
        used = Counter()
        def visit(item):
            text = dumps(item)
            if text in candidates:
                used[text] += 1
                return
            if isinstance(item, dict):
                for child in item.values():
                    visit(child)
            elif isinstance(item, list):
                for child in item:
                    visit(child)
        visit(value)
        active = {text for text, n in used.items()
                  if n > 1 and (n - 1) * len(text) > n * 16 + 8}
        if active == candidates:
            break
        candidates = active
    ids, pool = {}, []
    def replace(item):
        text = dumps(item)
        if text in candidates:
            if text not in ids:
                ids[text] = len(pool)
                pool.append(deepcopy(originals[text]))
            return {'$ref': ids[text]}
        if isinstance(item, dict):
            return {k: replace(v) for k, v in item.items()}
        if isinstance(item, list):
            return [replace(v) for v in item]
        return item
    body = replace(value)
    result = {'pool': pool, 'body': body}
    if len(dumps(result)) >= len(dumps({'pool': [], 'body': value})):
        result = {'pool': [], 'body': deepcopy(value)}
    return result


def unpack(record, max_expanded_chars=100_000_000):
    if not isinstance(record, dict) or set(record) != {'pool', 'body'} or not isinstance(record['pool'], list):
        raise ValueError('Expected pool/body envelope')
    assert_no_payload(record)
    pool = record['pool']
    sizes = []
    for item in pool:
        for _ in walk(item):
            pass
        sizes.append(len(dumps(item)))
    budget = len(dumps(record['body']))
    def expand(item):
        nonlocal budget
        if isinstance(item, dict):
            if '$ref' in item:
                index = item['$ref']
                if set(item) != {'$ref'} or type(index) is not int or not 0 <= index < len(pool):
                    raise ValueError('Invalid pool reference')
                budget += sizes[index]
                if budget > max_expanded_chars:
                    raise ValueError('Expanded document exceeds limit')
                return deepcopy(pool[index])
            return {k: expand(v) for k, v in item.items()}
        if isinstance(item, list):
            return [expand(v) for v in item]
        return item
    if budget > max_expanded_chars:
        raise ValueError('Document exceeds limit')
    return expand(record['body'])


def encode(label):
    if label.get('label_version') != 4:
        raise ValueError('Expected v4 source')
    return {'label_version': 5, **pack(label)}


def to_v4(label):
    if not isinstance(label, dict) or set(label) != {'label_version', 'pool', 'body'} or label['label_version'] != 5:
        raise ValueError('Expected v5 envelope')
    result = unpack({'pool': label['pool'], 'body': label['body']})
    if not isinstance(result, dict) or result.get('label_version') != 4:
        raise ValueError('v5 body must restore v4')
    return result


def decode(label, context, registry):
    return decode_v4(to_v4(label), context, registry)
