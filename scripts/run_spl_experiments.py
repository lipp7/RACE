"""Immutable JSON protocol helper; no historical run registry."""
import json
from pathlib import Path
def freeze(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Existing protocol differs: ' + str(path))
    else:
        path.write_text(json.dumps(value, indent=2) + '\n')
