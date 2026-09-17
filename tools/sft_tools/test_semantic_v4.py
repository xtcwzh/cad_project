import unittest
from copy import deepcopy
from semantic_v4 import encode, to_v3, xml_tree, tree_xml
from semantic_v3 import canonical_xml, parse_xml, schema_id


class CodecTests(unittest.TestCase):
    def test_xml_lexical_values_and_references(self):
        xml = '<PRO_E_ROOT type="compound"><PRO_XML_REFERENCE type="selection"><PRO_XML_ID type="int">0012</PRO_XML_ID><PRO_XML_V type="double">-0.000</PRO_XML_V></PRO_XML_REFERENCE><EMPTY></EMPTY></PRO_E_ROOT>'
        import xml.etree.ElementTree as ET
        restored = ET.tostring(tree_xml(xml_tree(parse_xml(xml))), encoding='unicode')
        self.assertEqual(canonical_xml(xml), canonical_xml(restored))

    def test_indexed_slots_preserve_null_and_empty(self):
        structure = ['ROOT', {}, [['A', {}, []], ['B', {}, []]]]
        sid = schema_id(structure)
        registry = {'schemas': {sid: structure}}
        label = {'label_version': 3, 'model': {}, 'feats': [{'operation': {
            'op': 'other', 'schema': sid, 'params': {'a': None, 'b': ''}},
            'pattern': {'op': 'source_xml_missing', 'empty': True}}]}
        packed = encode(label, registry)
        self.assertEqual(to_v3(packed, registry), label)
        bad = deepcopy(packed)
        bad['feats'][0]['operation']['values'].append([0, 'bad'])
        with self.assertRaises(ValueError):
            to_v3(bad, registry)

    def test_mixed_content_refused(self):
        with self.assertRaises(ValueError):
            xml_tree(parse_xml('<ROOT>important<A/>tail</ROOT>'))


if __name__ == '__main__':
    unittest.main()
