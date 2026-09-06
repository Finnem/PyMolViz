"""Unique default names for wizard-created PyMOL objects."""

from __future__ import annotations


def unused_object_name(base, cmd_=None, taken=(), keep=None):
    """Return ``base`` or ``base_1``, ``base_2``, ... not already used.

    ``keep`` is allowed even if it appears in the taken set (edit-in-place).
    """
    base = str(base).strip() or "object"
    names = {str(name) for name in taken if name}
    if cmd_ is not None:
        try:
            names.update(str(name) for name in cmd_.get_names("objects"))
        except Exception:
            pass
        try:
            from ...runtime.session import all_objects
            from ..catalog import display_name

            for obj in all_objects():
                label = display_name(obj)
                if label:
                    names.add(str(label))
        except Exception:
            pass
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
