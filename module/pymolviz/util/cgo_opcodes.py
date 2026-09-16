"""PyMOL CGO opcode values for headless script generation (no ``pymol`` import)."""

from types import SimpleNamespace

# Matches PyMOL 2.x / 3.x ``pymol.cgo`` (float opcodes).
_FALLBACK = {
    "POINTS": 0.0,
    "LINES": 1.0,
    "BEGIN": 2.0,
    "END": 3.0,
    "VERTEX": 4.0,
    "NORMAL": 5.0,
    "COLOR": 6.0,
    "SPHERE": 7.0,
    "CYLINDER": 9.0,
    "LINEWIDTH": 10.0,
    "ENABLE": 12.0,
    "DISABLE": 13.0,
    "ALPHA": 25.0,
    "CONE": 27.0,
    "TRIANGLES": 4.0,
    "LIGHTING": 2896.0,
}

_CGO_TOKEN_NAMES = tuple(_FALLBACK.keys())


def get_pymol_cgo():
    """Return ``pymol.cgo`` when PyMOL is installed, else a namespace with the same constants."""
    try:
        from pymol import cgo as pymol_cgo
    except ImportError:
        return SimpleNamespace(**_FALLBACK)
    return pymol_cgo
