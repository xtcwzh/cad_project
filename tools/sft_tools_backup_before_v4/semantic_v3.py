"""Inspectable XML operation parameters and lossless structural compression.

Schemas contain only XML tags and allowed type/header attributes, never leaf
values. Freeze schemas from train only. Unseen/unsafe XML stays as explicit XML.
Geometry tables preserve all mapping data until the Creo interface can change.
The codec is deterministic and batch-safe; it never looks up a source part.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import re
import xml.etree.ElementTree as ET

from label_codec import encode as encode_v2, decode as decode_v2
from payload_guard import assert_no_payload

ET.register_namespace('xsi', 'http://www.w3.org/2001/XMLSchema-instance')
HEADER_ATTRS = {'type', 'AppName', 'AppVersion',
                '{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation'}
CONTEXT_FIELDS = ('name', 'units', 'density', 'material')
SUPPORTED_OPS = {'extrude', 'revolve', 'hole', 'round', 'chamfer', 'pattern', 'datum'}
DEFAULTS = {'st': 0, 'feat_visible': 1, 'group_id': 0, 'is_pattern_leader': False,
            'layer_ids': [], 'par': [], 'chld': [], 'refs': [], 'dims': [],
            'feat_params': [], 'sk': None, 'geom_items': [], 'sub': '',
            'pattern_axis_angle': 0, 'pattern_axis_geom_id': -1,
            'pattern_axis_geom_type': 0, 'pattern_axis_owner_feat_id': -1,
            'pattern_class': 0, 'pattern_elem_refs': [],
            'pattern_first_dir_increment': 0, 'pattern_first_dir_instances': 0,
            'pattern_header_id': -1, 'pattern_leader_id': -1,
            'pattern_member_ids': [], 'pattern_second_dir_instances': 0,
            'pattern_status': 0}


def compact(value):
    """Tables save repeated field names; no missing-vs-null ambiguity."""
    if isinstance(value, list):
        if len(value) > 1 and all(isinstance(x, dict) for x in value):
            columns = list(value[0])
            if all(set(x) == set(columns) for x in value):
                return {'$table': columns,
                        'rows': [[compact(x[k]) for k in columns] for x in value]}
            layouts, rows = [], []
            for record in value:
                columns = list(record)
                if columns not in layouts:
                    layouts.append(columns)
                rows.append([layouts.index(columns)] + [compact(record[k]) for k in columns])
            table = {'$records': layouts, 'rows': rows}
            plain = [{k: compact(v) for k, v in x.items()} for x in value]
            if len(json.dumps(table)) < len(json.dumps(plain)):
                return table
            return plain
        return [compact(x) for x in value]
    if isinstance(value, dict):
        if '$table' in value or '$records' in value:
            raise ValueError('Reserved table marker in source')
        return {k: compact(v) for k, v in value.items()}
    return value


def expand(value):
    if isinstance(value, dict):
        if '$records' in value:
            if set(value) != {'$records', 'rows'}:
                raise ValueError('Invalid record table')
            layouts = value['$records']
            if not isinstance(layouts, list) or not layouts:
                raise ValueError('Missing table layouts')
            result = []
            for row in value['rows']:
                if not isinstance(row, list) or not row or type(row[0]) is not int or not 0 <= row[0] < len(layouts):
                    raise ValueError('Invalid record layout index')
                columns = layouts[row[0]]
                result.extend(expand({'$table': columns, 'rows': [row[1:]]}))
            return result
        if '$table' in value:
            if set(value) != {'$table', 'rows'}:
                raise ValueError('Invalid table fields')
            columns, rows = value['$table'], value['rows']
            if not isinstance(columns, list) or not all(isinstance(k, str) for k in columns):
                raise ValueError('Invalid columns')
            if len(columns) != len(set(columns)) or not isinstance(rows, list):
                raise ValueError('Duplicate columns or invalid rows')
            if any(not isinstance(row, list) or len(row) != len(columns) for row in rows):
                raise ValueError('Table row length mismatch')
            return [{k: expand(v) for k, v in zip(columns, row)} for row in rows]
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(x) for x in value]
    return value


def parse_xml(text):
    if '<!DOCTYPE' in text.upper() or '<!ENTITY' in text.upper():
        raise ValueError('DTD/entity not supported')
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True, insert_pis=True))
    return ET.fromstring(text, parser=parser)


def shape(node):
    if not isinstance(node.tag, str) or any(
            key not in HEADER_ATTRS and not (key == 'point_3d' and val == 'array')
            for key, val in node.attrib.items()):
        raise ValueError('Unsupported XML structure: preserve raw')
    if len(node) and node.text and node.text.strip():
        raise ValueError('Mixed XML content: preserve raw')
    if any(child.tail and child.tail.strip() for child in node):
        raise ValueError('Mixed XML tail: preserve raw')
    return [node.tag, dict(node.attrib), [shape(c) for c in node]]


def schema_id(structure):
    return hashlib.sha256(json.dumps(structure, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()[:16]


def leaves(node, prefix=''):
    counts = Counter(c.tag for c in node)
    seen = Counter()
    for child in node:
        seen[child.tag] += 1
        suffix = f'[{seen[child.tag]}]' if counts[child.tag] > 1 else ''
        name = re.sub(r'^PRO_(?:E|XML)_', '', child.tag).lower()
        path = prefix + name + suffix
        if len(child):
            yield from leaves(child, path + '/')
        else:
            yield path, child


def default_text(node):
    # These are format rules, NOT values copied from another part.
    return {'int': '0', 'double': '0.00'}.get(node.get('type'))


def slots(node):
    """Readable short leaf names when unique; full indexed paths otherwise."""
    rows = list(leaves(node))
    names = [path.rsplit('/', 1)[-1] for path, _ in rows]
    counts = Counter(names)
    return [(name if counts[name] == 1 else path, child)
            for (path, child), name in zip(rows, names)]


def classify(text, feature_type):
    if not text:
        return 'source_xml_missing'
    try:
        node = parse_xml(text)
        if node.tag == 'PRO_E_PATTERN_ROOT':
            return 'pattern'
        # Creo holes can internally carry EXTRUDE/REVOLVE forms; keep the
        # semantic hole operation and preserve that internal form as a slot.
        if feature_type in ('PRO_FEAT_HOLE', 'PRO_FEAT_ROUND', 'PRO_FEAT_CHAMFER'):
            return {'PRO_FEAT_HOLE': 'hole', 'PRO_FEAT_ROUND': 'round',
                    'PRO_FEAT_CHAMFER': 'chamfer'}[feature_type]
        form = node.find('.//PRO_E_FEATURE_FORM')
        if form is not None:
            return {'PRO_EXTRUDE': 'extrude', 'PRO_REVOLVE': 'revolve',
                    'PRO_SWEEP': 'sweep'}.get(form.text, 'other_form')
    except (ET.ParseError, ValueError):
        return 'opaque_xml'
    return {'PRO_FEAT_HOLE': 'hole', 'PRO_FEAT_ROUND': 'round',
            'PRO_FEAT_CHAMFER': 'chamfer', 'PRO_FEAT_CUT': 'cut',
            'PRO_FEAT_DATUM': 'datum'}.get(feature_type, 'other')


def build_registry(training_labels):
    schemas = {}
    for label in training_labels:
        for f in label['feats']:
            for key in ('params_xml', 'pattern_xml'):
                text = f.get(key, '')
                if not text:
                    continue
                if classify(text, f['type']) not in SUPPORTED_OPS:
                    continue
                try:
                    structure = shape(parse_xml(text))
                except (ET.ParseError, ValueError):
                    continue
                schemas[schema_id(structure)] = structure
    return {'version': 1, 'source_split': 'train', 'schemas': schemas}


def pack_xml(text, feature_type, registry):
    op = classify(text, feature_type)
    if not text:
        return {'op': op, 'empty': True}
    if op not in SUPPORTED_OPS:
        return {'op': op, 'raw_xml': text}
    try:
        node = parse_xml(text)
        structure = shape(node)
        sid = schema_id(structure)
        if registry['schemas'].get(sid) != structure or not len(node):
            raise ValueError('Schema not in frozen training registry')
        params = {path: child.text for path, child in slots(node)
                  if child.text != default_text(child)}
        return {'op': op, 'schema': sid, 'params': params}
    except (ET.ParseError, ValueError):
        return {'op': op, 'raw_xml': text}


def materialize(structure):
    tag, attrs, children = structure
    node = ET.Element(tag, attrs)
    for child in children:
        node.append(materialize(child))
    if not children:
        node.text = default_text(node)
    return node


def unpack_xml(record, feature_type, registry):
    variants = [key for key in ('empty', 'raw_xml', 'schema') if key in record]
    if len(variants) != 1:
        raise ValueError('XML record must have exactly one representation')
    if variants[0] == 'empty':
        if record['empty'] is not True or set(record) != {'op', 'empty'}:
            raise ValueError('Invalid empty XML')
        text = ''
    elif variants[0] == 'raw_xml':
        if set(record) != {'op', 'raw_xml'}:
            raise ValueError('Invalid raw XML')
        text = record['raw_xml']
    else:
        if set(record) != {'op', 'schema', 'params'}:
            raise ValueError('Invalid parameter XML')
        sid = record['schema']
        structure = registry['schemas'].get(sid)
        if structure is None or schema_id(structure) != sid:
            raise ValueError('Missing or changed frozen XML schema')
        node = materialize(structure)
        slot_map = dict(slots(node))
        if not isinstance(record['params'], dict) or not set(record['params']) <= set(slot_map):
            raise ValueError('Unknown XML parameter path')
        for path, text_value in record['params'].items():
            if text_value is not None and not isinstance(text_value, str):
                raise ValueError('XML text must preserve lexical string values')
            slot_map[path].text = text_value
        # Empty tags retain explicit open/close form for legacy importer regexes.
        text = '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(
            node, encoding='unicode', short_empty_elements=False)
    if classify(text, feature_type) != record.get('op'):
        raise ValueError('Operation and XML parameters disagree')
    return text


def encode(raw, part, registry):
    v2 = encode_v2(raw, part)
    context = {k: v2.pop(k) for k in CONTEXT_FIELDS if k in v2}
    v2.pop('part')
    v2.pop('label_version')
    features = v2.pop('feats')
    result = {'label_version': 3, 'model': compact(v2), 'feats': []}
    for f in features:
        f = deepcopy(f)
        operation = pack_xml(f.pop('params_xml'), f['type'], registry)
        pattern = pack_xml(f.pop('pattern_xml'), f['type'], registry)
        absent = [k for k in DEFAULTS if k not in f]
        for k, default in DEFAULTS.items():
            if k in f and type(f[k]) is type(default) and f[k] == default:
                del f[k]
        item = {'data': compact(f), 'operation': operation, 'pattern': pattern}
        if absent:
            item['absent'] = absent
        result['feats'].append(item)
    assert_no_payload(result)
    assert_no_payload(context)
    return result, context


def decode(label, context, registry):
    assert_no_payload(label)
    assert_no_payload(context)
    assert_no_payload(registry)
    if label.get('label_version') != 3 or set(label) != {'label_version', 'model', 'feats'}:
        raise ValueError('Expected semantic v3 envelope')
    if not set(context) <= set(CONTEXT_FIELDS):
        raise ValueError('Context may only contain declared caller metadata')
    v2 = expand(label['model'])
    if set(v2) & (set(CONTEXT_FIELDS) | {'part', 'label_version', 'feats'}):
        raise ValueError('Unexpected model envelope field')
    v2.update(deepcopy(context))
    v2.update(label_version=2, part=context.get('name', ''), feats=[])
    for item in label['feats']:
        if not set(item) <= {'data', 'operation', 'pattern', 'absent'}:
            raise ValueError('Unexpected feature envelope field')
        absent = item.get('absent', [])
        if not isinstance(absent, list) or not set(absent) <= set(DEFAULTS):
            raise ValueError('Invalid absent-default list')
        f = {k: deepcopy(v) for k, v in DEFAULTS.items() if k not in absent}
        data = expand(item['data'])
        if set(data) & set(absent) or set(data) & {'params_xml', 'pattern_xml'}:
            raise ValueError('Conflicting feature representation')
        f.update(data)
        f['params_xml'] = unpack_xml(item['operation'], f['type'], registry)
        f['pattern_xml'] = unpack_xml(item['pattern'], f['type'], registry)
        v2['feats'].append(f)
    return decode_v2(v2)


def canonical_xml(text):
    if not text:
        return text
    try:
        root = parse_xml(text)
        def canonical(node):
            return [str(node.tag), sorted(node.attrib.items()),
                    node.text if not len(node) else (node.text or '').strip(),
                    [canonical(c) for c in node],
                    (node.tail or '').strip()]
        return canonical(root)
    except (ET.ParseError, ValueError):
        return text


def equivalent(raw, restored):
    expected = deepcopy(raw)
    expected.pop('native_snapshot', None)
    expected['import_mode'] = 'feature_rebuild'
    actual = deepcopy(restored)
    for tree in (expected, actual):
        for f in tree['features']:
            for key in ('params_xml', 'pattern_xml'):
                f[key] = canonical_xml(f[key])
    return expected == actual
