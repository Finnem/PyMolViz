"""Read/write ``pymol.session.pymolviz`` as plain dicts."""

from __future__ import annotations

import types

from ..serialization import SCHEMA_VERSION, displayable_from_dict, displayable_to_dict

PREVIEW_ID_PREFIX = "preview_"
PREVIEW_NAME_PREFIX = "_pmv_prev_"

_live = {}


def session_ns():
    import pymol
    sess = getattr(pymol, "session", None)
    if sess is None:
        pymol.session = types.SimpleNamespace()
        sess = pymol.session
    return sess


def is_ephemeral(obj) -> bool:
    if str(getattr(obj, "id", "")).startswith(PREVIEW_ID_PREFIX):
        return True
    name = getattr(obj, "_name", None) or ""
    return str(name).startswith(PREVIEW_NAME_PREFIX)


def all_objects():
    return list(_live.values())


def get(model_id):
    return _live.get(str(model_id))


def persist():
    sess = session_ns()
    sess.pymolviz = {
        "schema": SCHEMA_VERSION,
        "objects": [displayable_to_dict(obj) for obj in _live.values()],
    }
    return sess.pymolviz


def add(obj):
    if is_ephemeral(obj):
        return
    _live[str(obj.id)] = obj
    persist()


def add_object(cmd, obj):
    """Wizard-facing alias: persist a collection and return it as a list."""
    add(obj)
    return [obj]


def read_session(cmd=None):
    return restore_from_session()


def remove(obj):
    _live.pop(str(getattr(obj, "id", obj)), None)
    persist()


def clear():
    _live.clear()


def read_blob():
    sess = session_ns()
    data = getattr(sess, "pymolviz", None)
    if not isinstance(data, dict):
        return {"schema": SCHEMA_VERSION, "objects": []}
    return data


def restore_from_session():
    data = read_blob()
    _live.clear()
    for item in data.get("objects", []):
        if not isinstance(item, dict):
            continue
        try:
            obj = displayable_from_dict(item)
        except Exception:
            continue
        if is_ephemeral(obj):
            continue
        _live[str(obj.id)] = obj
    return all_objects()
