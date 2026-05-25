"""
conftest.py for kv_migration unit tests.

This file is loaded by pytest before any test module is imported.
It stubs out GPU/heavy-ML dependencies (triton, torchvision, etc.) so
that the pure-Python / PyTorch-only modules under sglang.srt.kv_migration
and sglang.srt.mem_cache can be imported on machines without a full GPU stack
(e.g. macOS dev laptops, CI runners that lack CUDA).

Nothing here affects the semantics of the code under test — the stubs only
satisfy import-time name resolution; they are never called by the unit tests.
"""

import importlib.util
import sys
import types
from unittest.mock import MagicMock

SGLANG_PYTHON_ROOT = str(
    (
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "python"
    )
)

# ---------------------------------------------------------------------------
# 1.  Stub out modules that are unavailable in this dev environment.
# ---------------------------------------------------------------------------
_MOCK_PACKAGES = [
    "torchvision",
    "torchvision.io",
    "torchvision.transforms",
    "torchvision.transforms.functional",
    "torchvision.transforms.v2",
    "torchvision.transforms.v2.functional",
    "IPython",
    "IPython.display",
    "sentencepiece",
    "deep_gemm",
]

for _pkg in _MOCK_PACKAGES:
    if _pkg not in sys.modules:
        _m = MagicMock()
        _m.__path__ = []
        _m.__package__ = _pkg
        _m.__name__ = _pkg
        _m.__spec__ = types.SimpleNamespace(
            name=_pkg, submodule_search_locations=[]
        )
        sys.modules[_pkg] = _m

# ---------------------------------------------------------------------------
# 2.  Install the sglang triton stub (replaces the real triton package with a
#     pass-through decorator shim so @triton.jit definitions don't raise).
# ---------------------------------------------------------------------------
if "triton" not in sys.modules:
    _triton_spec = importlib.util.spec_from_file_location(
        "sglang._triton_stub",
        f"{SGLANG_PYTHON_ROOT}/sglang/_triton_stub.py",
    )
    _triton_stub_mod = importlib.util.module_from_spec(_triton_spec)
    sys.modules["sglang._triton_stub"] = _triton_stub_mod
    _triton_spec.loader.exec_module(_triton_stub_mod)
    _triton_stub_mod.install()

# ---------------------------------------------------------------------------
# 3.  Create a minimal sglang namespace package that satisfies Python's
#     package-path resolution without executing sglang/__init__.py (which
#     would pull in the full ML stack).
# ---------------------------------------------------------------------------
if "sglang" not in sys.modules:
    _sglang = types.ModuleType("sglang")
    _sglang.__path__ = [f"{SGLANG_PYTHON_ROOT}/sglang"]
    _sglang.__package__ = "sglang"
    _sglang.__file__ = f"{SGLANG_PYTHON_ROOT}/sglang/__init__.py"
    sys.modules["sglang"] = _sglang

# ---------------------------------------------------------------------------
# 4.  Install the sglang MPS stub (patches torch.mps with Stream / Event etc.
#     that are absent on macOS but referenced at module-load time by
#     distributed/parallel_state.py).
# ---------------------------------------------------------------------------
_mps_spec = importlib.util.spec_from_file_location(
    "sglang._mps_stub",
    f"{SGLANG_PYTHON_ROOT}/sglang/_mps_stub.py",
)
_mps_stub_mod = importlib.util.module_from_spec(_mps_spec)
sys.modules["sglang._mps_stub"] = _mps_stub_mod
_mps_spec.loader.exec_module(_mps_stub_mod)
_mps_stub_mod.install()

# ---------------------------------------------------------------------------
# 5.  Add python/ to sys.path so `import sglang.srt.*` resolves correctly.
# ---------------------------------------------------------------------------
if SGLANG_PYTHON_ROOT not in sys.path:
    sys.path.insert(0, SGLANG_PYTHON_ROOT)
