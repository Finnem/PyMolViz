"""Optional CuPy array module. NumPy is the default and the fallback.

CuPy on ``sys.path`` is not enough: a driver/runtime mismatch raises on the
first CUDA call. Probe a device before handing back the module.
"""

from __future__ import annotations

import numpy as np

_CACHED = "unset"
_ANNOUNCED = False


def reset_array_module():
    """Drop the cached backend (tests)."""
    global _CACHED, _ANNOUNCED
    _CACHED = "unset"
    _ANNOUNCED = False


def array_module(prefer=None):
    """Return CuPy when it can run a kernel, otherwise NumPy.

    ``prefer="numpy"`` skips the probe. ``prefer="cupy"`` still falls back if
    the device is unusable.
    """
    if prefer == "numpy":
        return np
    global _CACHED
    if prefer is None and _CACHED != "unset":
        return _CACHED
    xp = _probe_cupy() if prefer in (None, "cupy") else np
    if prefer is None:
        _CACHED = xp
    return xp


def array_backend_name(prefer=None) -> str:
    xp = array_module(prefer=prefer)
    name = getattr(xp, "__name__", "")
    return "cupy" if name == "cupy" else "numpy"


def as_numpy(values) -> np.ndarray:
    """Copy a CuPy array back to NumPy; NumPy input is returned as float."""
    getter = getattr(values, "get", None)
    if callable(getter) and not isinstance(values, np.ndarray):
        values = getter()
    return np.asarray(values, dtype=float)


def cuda_startup_message() -> str | None:
    """One-line console notice when CuPy can run, else ``None``."""
    if array_backend_name() != "cupy":
        return None
    gpu = _cupy_device_label()
    if gpu:
        return "PyMOLViz: CUDA detected (%s). Gaussian surfaces will use the GPU." % gpu
    return "PyMOLViz: CUDA detected. Gaussian surfaces will use the GPU."


def announce_cuda(cmd=None) -> bool:
    """Print ``cuda_startup_message`` once (PyMOL console / stdout)."""
    global _ANNOUNCED
    if _ANNOUNCED:
        return False
    msg = cuda_startup_message()
    if not msg:
        return False
    _ANNOUNCED = True
    echo = getattr(cmd, "echo", None) if cmd is not None else None
    if callable(echo):
        try:
            echo(msg)
            return True
        except Exception:
            pass
    print(msg)
    return True


def _cupy_device_label() -> str:
    try:
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"] if props is not None else ""
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        return str(name or "").strip()
    except Exception:
        return ""


def _probe_cupy():
    try:
        import cupy as cp
    except Exception:
        return np
    try:
        if not bool(cp.cuda.is_available()):
            return np
        cp.zeros(1, dtype=cp.float64)
    except Exception:
        return np
    return cp
