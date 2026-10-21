"""Shared baseline grids and pinned upstream identities."""
import json
from pathlib import Path
BETAS = (.1, .5, 1., 2., 3., 5., 10.)
KERNEL_LAMBDAS = (.001, .01, .1, 1., 10.)
def reference_commit(name):
    return json.loads(Path('third_party/versions.json').read_text())[name]['commit']
