"""Load the original implementations without changing their import namespaces."""
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SEQUENTIAL = ROOT / 'problem-2-min-error-seq/src'
JOINT = ROOT / 'problem-3-global-pgm/src'
FIELDS = ROOT / 'problem-1-zero-error/src/field_models.py'


def _load(name, path, *, package=False):
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def _legacy_imports():
    names = ('input_data', 'actual_error', 'strategy')
    previous = {name: sys.modules.get(name) for name in names}
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@lru_cache(maxsize=1)
def implementations():
    """Resolve legacy absolute imports temporarily; keep original function objects."""
    fields = _load('_comparison_fields', FIELDS)
    with _legacy_imports():
        sequential = {}
        for name in ('input_data', 'actual_error', 'strategy', 'heuristic', 'greedy'):
            module = _load(f'_comparison_seq_{name}', SEQUENTIAL / f'{name}.py')
            sequential[name] = module
            if name in ('input_data', 'actual_error', 'strategy'):
                sys.modules[name] = module
    joint = _load('_comparison_joint', JOINT / '__init__.py', package=True)
    from importlib import import_module
    return fields, sequential, joint, import_module('_comparison_joint.input_data'), import_module('_comparison_joint.normalization')


def source_hashes():
    paths = [FIELDS, *SEQUENTIAL.glob('*.py'), *JOINT.glob('*.py'),
             *Path(__file__).parent.glob('*.py')]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}
