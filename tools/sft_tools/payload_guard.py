"""Reject native model snapshots/binary base64 in model-facing artifacts."""
import re

LONG_BASE64 = re.compile(r'[A-Za-z0-9+/_-]{1024,}={0,2}\Z')


def assert_no_payload(value, path='$'):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() == 'native_snapshot' or 'base64' in key.lower():
                raise ValueError(f'Binary payload key at {path}.{key}')
            if key.lower() == 'encoding' and str(item).lower() == 'base64':
                raise ValueError(f'Base64 encoding at {path}')
            assert_no_payload(item, f'{path}.{key}')
    elif isinstance(value, list):
        for i, item in enumerate(value):
            assert_no_payload(item, f'{path}[{i}]')
    elif isinstance(value, str):
        if ';base64,' in value.lower() or LONG_BASE64.fullmatch(''.join(value.split())):
            raise ValueError(f'Possible binary base64 at {path}')


def scan_sample(sample):
    """Assistant content is serialized JSON, so inspect it as JSON too."""
    import json
    assert_no_payload(sample)
    for message in sample.get('messages', []):
        if message.get('role') == 'assistant':
            assert_no_payload(json.loads(message['content']))
