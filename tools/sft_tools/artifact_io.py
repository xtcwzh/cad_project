"""Version dispatch with explicit caller context and frozen v3 schemas."""
from label_codec import decode as decode_v2
from semantic_v3 import decode as decode_v3


def restore_label(label, context=None, registry=None):
    if isinstance(label, dict) and label.get('label_version') == 6:
        if context is None or registry is None:
            raise ValueError('v6 requires explicit context and frozen schemas')
        from semantic_v6 import decode as decode_v6
        return decode_v6(label, context, registry)
    if isinstance(label, dict) and label.get('label_version') == 5:
        if context is None or registry is None:
            raise ValueError('v5 requires explicit context and frozen schemas')
        from semantic_v5 import decode as decode_v5
        return decode_v5(label, context, registry)
    if isinstance(label, dict) and label.get('label_version') == 4:
        if context is None or registry is None:
            raise ValueError('v4 requires explicit caller context and frozen XML schema registry')
        from semantic_v4 import decode as decode_v4
        return decode_v4(label, context, registry)
    if isinstance(label, dict) and label.get('label_version') == 3:
        if context is None or registry is None:
            raise ValueError('v3 requires explicit caller context and frozen XML schema registry')
        return decode_v3(label, context, registry)
    return decode_v2(label)
