from pyvoteview.core import get_records_by_congress

CONGRESS_NUMBER = 118

raw = get_records_by_congress(CONGRESS_NUMBER, chamber='House')
print('TYPE:', type(raw))
try:
    import pandas as pd
    print('pandas installed')
    print('has rename:', hasattr(raw, 'rename'))
    if hasattr(raw, 'columns'):
        print('columns sample:', list(getattr(raw, 'columns'))[:10])
except Exception as e:
    print('Error introspecting raw:', e)

# if raw is iterable, show first element keys
try:
    first = next(iter(raw))
    print('first element type:', type(first))
    if isinstance(first, dict):
        print('first keys:', list(first.keys())[:20])
except Exception as e:
    print('Could not iterate raw:', e)
