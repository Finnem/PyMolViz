"""Unique default names for wizard-created PyMOL objects, fields, and visuals."""

from __future__ import annotations


def _cmd_object_names(cmd_):
    names = set()
    if cmd_ is None:
        return names
    for kind in ("objects", "public"):
        getter = getattr(cmd_, "get_names", None)
        if not callable(getter):
            break
        try:
            names.update(str(name) for name in (getter(kind) or ()) if name)
        except TypeError:
            try:
                names.update(str(name) for name in (getter() or ()) if name)
            except Exception:
                break
        except Exception:
            continue
    return names


def _label_on(obj):
    stored = getattr(obj, "_name", None)
    if stored:
        return str(stored)
    return ""


def _session_object_names():
    """User-facing names of Fields and visuals in the current session."""
    names = set()
    try:
        from ...runtime.session import all_objects

        objects = all_objects()
    except Exception:
        return names
    for obj in objects:
        label = _label_on(obj)
        if label:
            names.add(label)
        grid = getattr(obj, "grid_data", None)
        if grid is not None:
            glabel = _label_on(grid)
            if glabel:
                names.add(glabel)
        oid = str(getattr(obj, "id", "") or "")
        if oid.startswith("pymol_map:"):
            names.add(oid.split(":", 1)[-1])
    return names


def unused_object_name(base, cmd_=None, taken=(), keep=None):
    """Return ``base`` or ``base_1``, ``base_2``, ... not already used.

    ``keep`` is allowed even if it appears in the taken set (edit-in-place).
    Counts PyMOL objects and session Fields / field visuals.
    """
    base = str(base).strip() or "object"
    names = {str(name) for name in taken if name}
    names.update(_cmd_object_names(cmd_))
    names.update(_session_object_names())
    if keep:
        names.discard(str(keep))
    if base not in names:
        return base
    index = 1
    while True:
        candidate = "%s_%d" % (base, index)
        if candidate not in names:
            return candidate
        index += 1
