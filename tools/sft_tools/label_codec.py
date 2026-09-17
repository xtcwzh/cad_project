"""Reversible v2 aliases; preserve fields/precision/XML without source lookup.

Only native_snapshot is excluded. import_mode is forced to feature_rebuild.
Unknown fields survive; collisions fail rather than silently overwrite data.
"""
from copy import deepcopy
from payload_guard import assert_no_payload

VERSION = 2
ROOT_KEYS = {'features': 'feats', 'material_name': 'material', 'relations': 'rel'}
FEAT_KEYS = {'feat_id': 'id', 'feat_number': 'n', 'feat_type_name': 'type',
             'feat_type': 'type_id', 'feat_name': 'name', 'feat_status': 'st',
             'feat_subtype': 'sub', 'parent_ids': 'par', 'child_ids': 'chld',
             'dimensions': 'dims', 'sketch': 'sk', 'elem_refs': 'refs'}
SKETCH_KEYS = {'location_matrix': 'm', 'intent_manager': 'im', 'entities': 'e',
               'dimensions': 'd', 'projection_refs': 'pr', 'constraints': 'c'}
ENTITY_KEYS = {'ent_type': 't', 'json_ent_id': 'id'}


def rename(obj, keys, reverse=False):
    if not isinstance(obj, dict):
        raise ValueError('Expected a JSON object')
    mapping = {v: k for k, v in keys.items()} if reverse else keys
    reserved = set(mapping.values())
    result = {}
    for key, value in obj.items():
        if key not in mapping and key in reserved:
            raise ValueError(f'Reserved alias collision: {key}')
        result[mapping.get(key, key)] = deepcopy(value)
    return result


def sketch_codec(sketch, reverse=False):
    if sketch is None:
        return None
    result = rename(sketch, SKETCH_KEYS, reverse)
    key = 'entities' if reverse else 'e'
    if key in result:
        result[key] = [rename(e, ENTITY_KEYS, reverse) for e in result[key]]
    return result


def feature_codec(feature, reverse=False):
    result = rename(feature, FEAT_KEYS, reverse)
    key = 'sketch' if reverse else 'sk'
    if key in result:
        result[key] = sketch_codec(result[key], reverse)
    return result


def encode(tree, part):
    if any(k in tree for k in ('label_version', 'part')):
        raise ValueError('Source uses a reserved label envelope field')
    source = deepcopy(tree)
    source.pop('native_snapshot', None)
    source['import_mode'] = 'feature_rebuild'
    result = rename(source, ROOT_KEYS)
    result['feats'] = [feature_codec(f) for f in source['features']]
    result.update(label_version=VERSION, part=part)
    assert_no_payload(result)
    return result


def decode(label):
    assert_no_payload(label)
    if not isinstance(label, dict) or label.get('label_version') != VERSION:
        raise ValueError('Unsafe legacy label: regenerate v2 from raw exports; '
                         'lost fields cannot be recovered from v1 or type templates')
    if 'native_snapshot' in label:
        raise ValueError('native_snapshot is forbidden in a prediction')
    if not isinstance(label.get('part'), str):
        raise ValueError('Missing part identifier')
    if not isinstance(label.get('feats'), list) or not label['feats']:
        raise ValueError('Expected a nonempty feats array')
    source = {k: v for k, v in label.items() if k not in ('part', 'label_version')}
    result = rename(source, ROOT_KEYS, True)
    result['features'] = [feature_codec(f, True) for f in label['feats']]
    result['import_mode'] = 'feature_rebuild'
    validate_tree(result)
    return result


def validate_tree(tree):
    """Structural gate, not a guarantee of Creo or geometric correctness."""
    ids = set()
    for f in tree['features']:
        for key in ('feat_id', 'feat_number', 'feat_type', 'feat_type_name',
                    'params_xml', 'pattern_xml', 'geom_items', 'dimensions',
                    'parent_ids', 'child_ids'):
            if key not in f:
                raise ValueError(f'Feature missing required field {key}')
        for key in ('feat_id', 'feat_number', 'feat_type'):
            if type(f[key]) is not int:
                raise ValueError(f'{key} must be an integer')
        if f['feat_id'] in ids:
            raise ValueError(f'Duplicate feature ID {f["feat_id"]}')
        ids.add(f['feat_id'])
        if not isinstance(f['feat_type_name'], str) or not f['feat_type_name']:
            raise ValueError('Invalid feature type name')
        for key in ('params_xml', 'pattern_xml'):
            if not isinstance(f[key], str):
                raise ValueError(f'{key} must be a string; empty source XML is allowed')
        for key in ('geom_items', 'dimensions', 'parent_ids', 'child_ids'):
            if not isinstance(f[key], list):
                raise ValueError(f'{key} must be an array')
        for key in ('parent_ids', 'child_ids'):
            if not all(type(x) is int for x in f[key]):
                raise ValueError(f'{key} must contain integer IDs')
        for dim in f['dimensions']:
            if not isinstance(dim, dict) or not all(k in dim for k in ('dim_id', 'symbol', 'value')):
                raise ValueError('Dimension must retain ID, symbol and value')
            if type(dim['value']) not in (int, float):
                raise ValueError('Dimension value must be numeric')
        sk = f.get('sketch')
        if sk:
            if not isinstance(sk, dict) or not all(k in sk for k in ('entities', 'dimensions', 'constraints')):
                raise ValueError('Incomplete sketch')
            entity_ids = set()
            for e in sk['entities']:
                if type(e.get('ent_type')) is not int or type(e.get('json_ent_id')) is not int:
                    raise ValueError('Sketch entity needs ent_type and json_ent_id')
                if e['json_ent_id'] in entity_ids:
                    raise ValueError('Duplicate sketch entity ID')
                entity_ids.add(e['json_ent_id'])
            for dim in sk['dimensions']:
                if any(i not in entity_ids for i in dim.get('entity_ids', [])):
                    raise ValueError('Sketch dimension references missing entity')
