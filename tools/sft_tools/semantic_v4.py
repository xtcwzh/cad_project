"""Reversible v3 transport compression; no source-part lookup or numeric rounding."""
from copy import deepcopy
import json
import xml.etree.ElementTree as ET
from semantic_v3 import parse_xml, shape, slots, materialize, decode as decode_v3


def dumps(x):
    return json.dumps(x, ensure_ascii=False, separators=(',', ':'))


def xml_tree(node):
    # shape rejects comments, processing instructions and mixed content.
    shape(node)
    tag = node.tag
    if tag.startswith('PRO_E_'):
        tag = 'e:' + tag[6:]
    elif tag.startswith('PRO_XML_'):
        tag = 'x:' + tag[8:]
    return [tag, dict(node.attrib), node.text if not len(node) else None,
            [xml_tree(c) for c in node]]


def tree_xml(tree):
    if not isinstance(tree, list) or len(tree) != 4:
        raise ValueError('XML tree requires tag, attributes, text, children')
    tag, attrs, text, children = tree
    if not isinstance(tag, str) or not isinstance(attrs, dict) or not isinstance(children, list):
        raise ValueError('Invalid XML tree')
    if text is not None and not isinstance(text, str):
        raise ValueError('XML text must be lexical string or null')
    if tag.startswith('e:'):
        tag = 'PRO_E_' + tag[2:]
    elif tag.startswith('x:'):
        tag = 'PRO_XML_' + tag[2:]
    node = ET.Element(tag, attrs)
    node.text = text
    for child in children:
        node.append(tree_xml(child))
    shape(node)
    return node


def encode(label, registry):
    result = deepcopy(label)
    result['label_version'] = 4
    for f in result['feats']:
        for key in ('operation', 'pattern'):
            rec = f[key]
            if 'schema' in rec:
                names = [name for name, _ in slots(materialize(registry['schemas'][rec['schema']]))]
                rec['values'] = [[names.index(k), v] for k, v in rec.pop('params').items()]
            elif 'raw_xml' in rec:
                try:
                    tree = xml_tree(parse_xml(rec['raw_xml']))
                    candidate = {'op': rec['op'], 'tree': tree}
                    if len(dumps(candidate)) < len(dumps(rec)):
                        f[key] = candidate
                except (ValueError, ET.ParseError):
                    pass
    return result


def to_v3(label, registry):
    if label.get('label_version') != 4:
        raise ValueError('Expected label_version=4')
    result = deepcopy(label)
    result['label_version'] = 3
    for f in result['feats']:
        for key in ('operation', 'pattern'):
            rec = f[key]
            if 'values' in rec:
                if set(rec) != {'op', 'schema', 'values'}:
                    raise ValueError('Invalid indexed XML record')
                names = [name for name, _ in slots(materialize(registry['schemas'][rec['schema']]))]
                params = {}
                for pair in rec.pop('values'):
                    if not isinstance(pair, list) or len(pair) != 2:
                        raise ValueError('Invalid XML slot pair')
                    i, value = pair
                    if type(i) is not int or not 0 <= i < len(names) or names[i] in params:
                        raise ValueError('Invalid or duplicate XML slot')
                    params[names[i]] = value
                rec['params'] = params
            elif 'tree' in rec:
                if set(rec) != {'op', 'tree'}:
                    raise ValueError('Invalid XML tree record')
                node = tree_xml(rec.pop('tree'))
                rec['raw_xml'] = '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(
                    node, encoding='unicode', short_empty_elements=False)
    return result


def decode(label, context, registry):
    return decode_v3(to_v3(label, registry), context, registry)
