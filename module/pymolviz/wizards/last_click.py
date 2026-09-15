"""Last atom clicked in the 3D viewer (Selecting mode does not set pk1)."""

from __future__ import annotations

from typing import Optional, Tuple

_LAST: Optional[Tuple[str, int]] = None


def last_clicked_atom():
    """``(model, atom_id)`` from the most recent viewer click, or None."""
    return _LAST


def set_last_clicked_atom(model=None, atom_id=None) -> None:
    global _LAST
    if model is None or atom_id is None:
        _LAST = None
        return
    try:
        _LAST = (str(model), int(atom_id))
    except Exception:
        _LAST = None


def parse_atom_sele(sele) -> Optional[Tuple[str, int]]:
    """Parse ``(object)`id`` from :func:`pick.atom_sele`."""
    text = str(sele or "").strip()
    if not text:
        return None
    tick = text.rfind("`")
    if tick <= 0:
        return None
    model = text[:tick].strip()
    if model.startswith("(") and model.endswith(")"):
        model = model[1:-1].strip()
    if not model:
        return None
    try:
        atom_id = int(text[tick + 1 :])
    except Exception:
        return None
    return (model, atom_id)
