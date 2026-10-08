"""Helpers shared by the exercise tests. (There are no exercise solutions in here.)"""
import contextlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / 'fixtures'

# --------------------------------------------------------------------------- optional packages
# Packages that are only needed to run the full pipeline (or parts of it that the tests do not
# touch). If one of them is not installed, a stub module is registered so that the repository
# modules can still be imported. Using anything from a stub skips the test that touched it.
OPTIONAL_PACKAGES = [
    'mmpose', 'mmcv', 'mmdet', 'mmengine', 'ultralytics', 'supervision', 'sports', 'moviepy',
    'torchinfo', 'torchvision', 'tensorflow', 'seaborn', 'parse', 'skimage', 'sklearn', 'scipy',
    'matplotlib', 'PIL', 'yaml', 'tqdm', 'einops', 'cv2', 'pandas',
]
STUBBED = set()


def _installed(name):
    if name in STUBBED:
        return False
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


class _Missing:
    """Placeholder for anything taken from a package that is not installed."""

    def __init__(self, name):
        self._missing_name = name

    def _skip(self, *args, **kwargs):
        pytest.skip(f"needs the package '{self._missing_name.split('.')[0]}' "
                    f"(used: {self._missing_name}), which is not installed")

    __call__ = __getitem__ = __setitem__ = __iter__ = __enter__ = __len__ = _skip

    def __mro_entries__(self, bases):
        # Allow imports that define subclasses of an optional dependency, but
        # still skip if a test tries to instantiate one of those subclasses.
        missing = self

        class MissingBase:
            def __new__(cls, *args, **kwargs):
                missing._skip()

        return (MissingBase,)

    def __getattr__(self, attr):
        if attr.startswith('__'):
            raise AttributeError(attr)
        return _Missing(f'{self._missing_name}.{attr}')

    def __repr__(self):
        return f'<not installed: {self._missing_name}>'


class _StubModule(types.ModuleType):
    def __getattr__(self, attr):
        if attr.startswith('__'):
            raise AttributeError(attr)
        return _Missing(f'{self.__name__}.{attr}')


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition('.')[0] in STUBBED:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        module = _StubModule(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module):
        pass


def install_stubs():
    for name in OPTIONAL_PACKAGES:
        if not _installed(name):
            STUBBED.add(name)
    if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
        sys.meta_path.append(_StubFinder())


def require(*packages):
    """Skip the current test unless all `packages` are really installed."""
    missing = [p for p in packages if not _installed(p)]
    if missing:
        pytest.skip(f"needs {', '.join(missing)} (not installed)")


def import_repo(module, *packages):
    """Import a repository module (skipping the test if a third-party package is missing)."""
    require(*packages)
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as e:
        top = (e.name or '').partition('.')[0]
        if top and not (ROOT / top).exists() and not (ROOT / f'{top}.py').exists():
            pytest.skip(f"cannot import {module}: package '{e.name}' is not installed")
        raise


def add_to_path(folder):
    """Some scripts import their siblings directly (e.g. `from utility_tempose import ...`)."""
    path = str(ROOT / folder)
    if path not in sys.path:
        sys.path.insert(1, path)   # after the repository root, so ROOT/utils.py keeps priority


# --------------------------------------------------------------------------- deterministic data
_MASK64 = (1 << 64) - 1


def det_uniform(shape, seed=0):
    """Deterministic pseudo-random numbers in [-1, 1).

    Pure integer hashing (splitmix64), so the values are identical on every platform and
    every numpy / torch version - unlike np.random or torch.manual_seed.
    """
    shape = tuple(int(s) for s in shape)
    n = int(np.prod(shape, dtype=np.int64))
    offset = np.uint64(((seed + 1) * 0x9E3779B97F4A7C15) & _MASK64)
    with np.errstate(over='ignore'):
        z = np.arange(1, n + 1, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15) + offset
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return ((z >> np.uint64(11)).astype(np.float64) * (2.0 / 2 ** 53) - 1.0).reshape(shape)


def det_tensor(shape, seed=0, scale=1.0):
    import torch
    return torch.as_tensor(det_uniform(shape, seed) * scale, dtype=torch.float32)


def fill_params(module, seed=0):
    """Overwrite every parameter / buffer of `module` with deterministic values.

    The values only depend on the parameter NAME and shape, so the model built from your code
    and the reference model get exactly the same weights (do not rename layers!).
    """
    import torch
    with torch.no_grad():
        for name, t in module.state_dict().items():
            if not t.is_floating_point():
                continue
            v = det_uniform(t.shape, seed * 1_000_003 + zlib.crc32(name.encode()))
            if name.endswith('running_var'):
                v = 1.0 + 0.5 * np.abs(v)
            elif name.endswith('running_mean'):
                v = 0.1 * v
            elif t.dim() <= 1:
                v = 1.0 + 0.2 * v if name.endswith('weight') else 0.1 * v
            else:
                v = v * np.sqrt(3.0 / max(1, t[0].numel()))
            t.copy_(torch.as_tensor(v, dtype=t.dtype))
    return module


# --------------------------------------------------------------------------- comparisons
def to_numpy(x):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def sample_idx(n, k=256):
    return np.unique(np.linspace(0, n - 1, min(k, n)).round().astype(np.int64))


def fingerprint(x, k=256):
    """Small summary of a (possibly big) array: shape, mean, std and k sampled values."""
    a = to_numpy(x).astype(np.float64)
    flat = a.reshape(-1)
    return {'shape': np.array(a.shape), 'mean': flat.mean(), 'std': flat.std(),
            'sample': flat[sample_idx(flat.size, k)]}


def describe_mismatch(actual, expected):
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    i = np.unravel_index(np.nanargmax(np.where(np.isnan(diff), np.inf, diff)), diff.shape)
    return f'max |diff| = {np.nanmax(diff):.4g} at index {tuple(int(v) for v in i)}: got {actual[i]!r}, expected {expected[i]!r}'


def assert_close(actual, expected, what, rtol=1e-4, atol=1e-5):
    actual, expected = to_numpy(actual), np.asarray(expected)
    assert actual.shape == tuple(expected.shape), f'{what}: shape {actual.shape}, expected {tuple(expected.shape)}'
    if actual.dtype == bool or expected.dtype == bool or actual.dtype.kind in 'iu':
        actual, expected = actual.astype(np.float64), expected.astype(np.float64)
    assert np.isfinite(actual).all(), f'{what}: contains NaN/inf'
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise AssertionError(f'{what}: values differ from the reference ({describe_mismatch(actual, expected)})')


def matches_fingerprint(x, fp, rtol=1e-4, atol=1e-5):
    a = to_numpy(x).astype(np.float64)
    if tuple(a.shape) != tuple(fp['shape']):
        return False
    flat = a.reshape(-1)
    return (np.allclose(flat[sample_idx(flat.size, len(fp['sample']))], fp['sample'], rtol=rtol, atol=atol)
            and np.isclose(flat.mean(), fp['mean'], rtol=rtol, atol=atol)
            and np.isclose(flat.std(), fp['std'], rtol=rtol, atol=atol))


def load_fixture(name):
    path = FIXTURES / f'{name}.npz'
    assert path.exists(), f'missing test fixture {path}'
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def get_fingerprint(fx, key):
    return {part: fx[f'{key}:{part}'] for part in ('shape', 'mean', 'std', 'sample')}


# --------------------------------------------------------------------------- tracing
@contextlib.contextmanager
def trace(model, names):
    """Record the first positional input (and `mask`) and the output of the named submodules."""
    record = {}
    handles = []

    def make_hook(name):
        def hook(module, args, kwargs, output):
            if name in record:           # keep the first call only
                return
            inputs = list(args) + [v for k, v in kwargs.items() if k != 'mask']
            mask = kwargs.get('mask', args[1] if len(args) > 1 else None)
            out = output[0] if isinstance(output, tuple) else output
            record[name] = {'in': inputs[0] if inputs else None, 'mask': mask, 'out': out}
        return hook

    for name in names:
        handles.append(model.get_submodule(name).register_forward_hook(make_hook(name), with_kwargs=True))
    try:
        yield record
    finally:
        for h in handles:
            h.remove()


def check_stages(record, fx, prefix, stages, hints):
    """Point at the first traced stage whose input differs from the reference."""
    for name, part in stages:
        key = f'{prefix}/{name}/{part}'
        if f'{key}:shape' not in fx:
            continue
        value = record.get(name, {}).get(part)
        what = f"the {'input' if part == 'in' else part} of `{name}`"
        hint = hints.get((name, part), hints.get(name, ''))
        if value is None:
            return f'{what} was never computed (is `{name}` called?). {hint}'
        if not matches_fingerprint(value, get_fingerprint(fx, key)):
            got, expected = tuple(to_numpy(value).shape), tuple(int(s) for s in fx[key + ':shape'])
            problem = f'has shape {got}, expected {expected}' if got != expected else 'has different values'
            return f'{what} {problem}. {hint}'
    return None


# --------------------------------------------------------------------------- small utilities
def write_video(path, frames, fps=30):
    """Write BGR uint8 frames to an .mp4 file (skip the test if OpenCV cannot do it here)."""
    require('cv2')
    import cv2
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n != len(frames):
        pytest.skip('this OpenCV build cannot write/read mp4 videos')
    return path
