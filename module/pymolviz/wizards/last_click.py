"""Last atom PyMOL reported from a viewer click.

Selecting mode expands ``sele`` to residue/chain/molecule, but the console
line ``You clicked /obj//A/GLY`77/C`` is always the exact atom. Pick/edit
modes may append `` -> (obj`id)`` or `` -> (pk1)``.

PyMOL's Qt GUI drains those lines via ``cmd._get_feedback()``. Observe
that list; do not scrape Qt widgets or wrap the C drain primitive.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

_LAST: Optional[Tuple[str, int]] = None
_LAST_PATH: Optional[str] = None
_FEEDBACK_HOOKS = []

_YOU_CLICKED = re.compile(r"You clicked\s+(\S+)", re.I)
_CLICK_IDENT = re.compile(r"->\s*\(\s*([^)]+?)\s*\)")


def last_clicked_atom():
    """``(model, atom_id)`` from the most recent viewer click, or None."""
    return _LAST


def last_clicked_path():
    """PyMOL atom path ``/obj//A/GLY`77/C``, or None."""
    return _LAST_PATH


def clicked_atom_expr():
    """Selection for the last clicked atom (``(object)`id`` or the path)."""
    if _LAST is not None:
        return "(%s)`%d" % (_LAST[0], _LAST[1])
    if _LAST_PATH:
        return _LAST_PATH
    return None


def set_last_clicked_atom(model=None, atom_id=None) -> None:
    global _LAST, _LAST_PATH
    if model is None or atom_id is None:
        _LAST = None
        _LAST_PATH = None
        return
    try:
        _LAST = (str(model), int(atom_id))
    except Exception:
        _LAST = None


def set_last_clicked_path(path=None) -> None:
    global _LAST_PATH
    text = str(path or "").strip()
    _LAST_PATH = text or None


def parse_atom_sele(sele) -> Optional[Tuple[str, int]]:
    """Parse ``(object)`id`` from :func:`pick.atom_sele`."""
    text = str(sele or "").strip()
    if not text:
        return None
    if "/" in text and text.startswith("/"):
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


def last_you_clicked_line(text, tail_lines: int = 16):
    """Most recent ``You clicked …`` line in ``text``, or None."""
    lines = str(text or "").splitlines()
    if tail_lines and len(lines) > int(tail_lines):
        lines = lines[-int(tail_lines) :]
    last = None
    for line in lines:
        if "You clicked" in line or "you clicked" in line.lower():
            last = line
    return last


def note_click_feedback(text) -> bool:
    """Record the atom from PyMOL's ``You clicked`` feedback. Returns True if parsed."""
    global _LAST
    raw = str(text or "")
    if "You clicked" not in raw and "you clicked" not in raw.lower():
        return False
    ident = None
    arrow = _CLICK_IDENT.search(raw)
    if arrow:
        ident = parse_atom_sele(arrow.group(1))
        if ident is None:
            ident = parse_atom_sele("(" + arrow.group(1).strip() + ")")
    match = _YOU_CLICKED.search(raw)
    path = match.group(1).strip().rstrip(",.") if match else None
    if path and not path.startswith("/"):
        if ident is None:
            ident = parse_atom_sele(path)
        path = None
    if path and path.startswith("/"):
        set_last_clicked_path(path)
    if ident is not None:
        set_last_clicked_atom(ident[0], ident[1])
        if path and path.startswith("/"):
            set_last_clicked_path(path)
        return True
    if path and path.startswith("/"):
        _LAST = None
        return True
    return False


def resolve_clicked_atom(cmd_):
    """Fill ``(model, id)`` from a stored click path when ``cmd`` is available."""
    if _LAST is not None:
        return _LAST
    path = _LAST_PATH
    if not path or cmd_ is None:
        return None
    for python_expr in (
        "ids.append((model, ID))",
        "ids.append((model, index))",
    ):
        ids = []
        try:
            cmd_.iterate("first (%s)" % path, python_expr, space={"ids": ids})
        except Exception:
            ids = []
        if ids:
            set_last_clicked_atom(ids[0][0], ids[0][1])
            set_last_clicked_path(path)
            return last_clicked_atom()
    return None


def record_named_pick(cmd_, name: str = "pk1") -> bool:
    """Record ``(model, id)`` from a one-atom pick name. Does not unpick."""
    if cmd_ is None:
        return False
    name = str(name or "").strip() or "pk1"
    try:
        n = int(cmd_.count_atoms("(%s)" % name))
    except Exception:
        return False
    if n != 1:
        return False
    expr = "first (%s)" % name
    for python_expr in (
        "ids.append((model, ID))",
        "ids.append((model, index))",
    ):
        ids = []
        try:
            cmd_.iterate(expr, python_expr, space={"ids": ids})
        except Exception:
            ids = []
        if ids:
            try:
                set_last_clicked_atom(ids[0][0], ids[0][1])
            except Exception:
                return False
            return last_clicked_atom() is not None
    return False


def record_pymol_click(cmd_=None) -> bool:
    """Install the feedback observer; do not call ``cmd`` (wizard lock)."""
    install_click_feedback_hook()
    return last_clicked_atom() is not None or last_clicked_path() is not None


MOUSE_SELECTION_ATOMS = 0
_SAVED_SELECTION_MODE = None
_ATOM_MODE_DEPTH = 0


def _read_selection_mode(cmd_) -> int:
    getter = getattr(cmd_, "get_setting_int", None)
    if callable(getter):
        try:
            return int(getter("mouse_selection_mode"))
        except Exception:
            pass
    try:
        return int(float(cmd_.get("mouse_selection_mode")))
    except Exception:
        return MOUSE_SELECTION_ATOMS


def _set_selection_mode(cmd_, mode: int) -> None:
    try:
        cmd_.set("mouse_selection_mode", int(mode), quiet=1)
    except TypeError:
        try:
            cmd_.set("mouse_selection_mode", int(mode))
        except Exception:
            pass
    except Exception:
        pass


def use_atom_selection_mode(cmd_, on: bool) -> None:
    """While Add Clicked Atoms is on, PyMOL Selecting uses atoms (mode 0)."""
    global _SAVED_SELECTION_MODE, _ATOM_MODE_DEPTH
    if cmd_ is None:
        return
    if on:
        if _ATOM_MODE_DEPTH <= 0:
            _SAVED_SELECTION_MODE = _read_selection_mode(cmd_)
            _set_selection_mode(cmd_, MOUSE_SELECTION_ATOMS)
            _ATOM_MODE_DEPTH = 1
            return
        _ATOM_MODE_DEPTH += 1
        return
    if _ATOM_MODE_DEPTH <= 0:
        return
    _ATOM_MODE_DEPTH -= 1
    if _ATOM_MODE_DEPTH <= 0:
        _ATOM_MODE_DEPTH = 0
        if _SAVED_SELECTION_MODE is not None:
            _set_selection_mode(cmd_, _SAVED_SELECTION_MODE)
            _SAVED_SELECTION_MODE = None


def restore_atom_selection_mode(cmd_) -> None:
    """Undo atom-mode hold on wizard exit, even if a toggle is still on."""
    global _SAVED_SELECTION_MODE, _ATOM_MODE_DEPTH
    if _ATOM_MODE_DEPTH <= 0:
        _SAVED_SELECTION_MODE = None
        return
    _ATOM_MODE_DEPTH = 0
    if cmd_ is not None and _SAVED_SELECTION_MODE is not None:
        _set_selection_mode(cmd_, _SAVED_SELECTION_MODE)
    _SAVED_SELECTION_MODE = None


def _note_feedback_value(value) -> None:
    if value is None:
        return
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8", "replace")
        except Exception:
            return
    if isinstance(value, (list, tuple)):
        for item in value:
            _note_feedback_value(item)
        return
    note_click_feedback(value)


def _wrap_get_feedback(fn):
    def wrapped(*args, **kwargs):
        lines = fn(*args, **kwargs)
        try:
            _note_feedback_value(lines)
        except Exception:
            pass
        return lines

    wrapped._pmv_click_hook = True
    return wrapped


def _feedback_owners():
    owners = []
    try:
        from pymol import cmd as pymol_cmd
    except Exception:
        pymol_cmd = None
    if pymol_cmd is not None:
        owners.append(pymol_cmd)
    try:
        from pymol import internal
    except Exception:
        internal = None
    if internal is not None and internal not in owners:
        owners.append(internal)
    return owners


def install_click_feedback_hook() -> None:
    """Observe ``You clicked`` lines when the GUI drains ``cmd._get_feedback``."""
    if _FEEDBACK_HOOKS:
        return
    orig = None
    owners = _feedback_owners()
    for owner in owners:
        current = getattr(owner, "_get_feedback", None)
        if callable(current) and not getattr(current, "_pmv_click_hook", False):
            orig = current
            break
    if orig is None:
        return
    wrapped = _wrap_get_feedback(orig)
    for owner in owners:
        current = getattr(owner, "_get_feedback", None)
        if current is orig:
            try:
                setattr(owner, "_get_feedback", wrapped)
            except Exception:
                continue
            _FEEDBACK_HOOKS.append((owner, "_get_feedback", orig))


def uninstall_click_feedback_hook() -> None:
    hooks = list(_FEEDBACK_HOOKS)
    _FEEDBACK_HOOKS.clear()
    for owner, attr, orig in reversed(hooks):
        current = getattr(owner, attr, None)
        if callable(current) and getattr(current, "_pmv_click_hook", False):
            try:
                setattr(owner, attr, orig)
            except Exception:
                pass
