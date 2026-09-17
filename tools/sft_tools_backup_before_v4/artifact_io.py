"""Version dispatch with explicit caller context and frozen v3 schemas."""
from label_codec import decode as decode_v2
from semantic_v3 import decode as decode_v3


def restore_label(label, context=None, registry=None):
    if isinstance(label, dict) and label.get('label_version') == 3:
        if context is None or registry is None:
            raise ValueError('v3 requires explicit caller context and frozen XML schema registry')
        return decode_v3(label, context, registry)
    return decode_v2(label)
