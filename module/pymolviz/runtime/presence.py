"""Keep the live session in sync with PyMOL's object list."""

from __future__ import annotations

from contextlib import contextmanager

from .renderer import FIELD_VISUAL_TYPES, renders_cgo

_PAUSE = 0


@contextmanager
def pause_presence_sync():
    """Ignore PyMOL deletes while we replace or reload our own objects."""
    global _PAUSE
    _PAUSE += 1
    try:
        yield
    finally:
        _PAUSE -= 1


def presence_sync_paused() -> bool:
    return _PAUSE > 0


def tracks_pymol_object(obj) -> bool:
    """True when this session object must exist as a named PyMOL object."""
    from .session import is_ephemeral

    if obj is None or is_ephemeral(obj):
        return False
    kind = type(obj).__name__
    if kind in FIELD_VISUAL_TYPES:
        return True
    if kind == "Field":
        from ..util.field_sample import PYMOL_MAP_ID_PREFIX

        oid = str(getattr(obj, "id", "") or "")
        return oid.startswith(PYMOL_MAP_ID_PREFIX)
    return renders_cgo(obj)


def object_pymol_name(obj, runtime=None) -> str:
    """PyMOL object name for a session displayable, or empty if untracked."""
    if obj is None:
        return ""
    kind = type(obj).__name__
    if kind == "Field":
        from ..util.field_sample import PYMOL_MAP_ID_PREFIX

        oid = str(getattr(obj, "id", "") or "")
        if oid.startswith(PYMOL_MAP_ID_PREFIX):
            return oid[len(PYMOL_MAP_ID_PREFIX):]
        return ""
    if runtime is not None:
        binding = runtime.bindings.get(getattr(obj, "id", None))
        if binding is not None and binding.pymol_name:
            return str(binding.pymol_name)
    stored = getattr(obj, "_name", None)
    if stored:
        return str(stored)
    from .runtime import binding_name

    return binding_name(obj)


def sync_session_with_pymol(cmd) -> list:
    """Drop session objects whose PyMOL counterpart was deleted. Returns dropped."""
    from .runtime import get_runtime
    from .session import all_objects, remove as session_remove

    if cmd is None:
        return []
    try:
        existing = {str(n) for n in cmd.get_names("objects")}
    except Exception:
        return []
    runtime = get_runtime(cmd)
    dropped = []
    for obj in list(all_objects()):
        if not tracks_pymol_object(obj):
            continue
        name = object_pymol_name(obj, runtime)
        if not name or name in existing:
            continue
        dropped.append(obj)
        try:
            runtime.bindings.pop(getattr(obj, "id", None))
        except Exception:
            pass
        session_remove(obj)
    return dropped


def after_pymol_delete(cmd, name="") -> list:
    """Prune the session after a user ``cmd.delete``. No-op while we are replacing."""
    if presence_sync_paused() or cmd is None:
        return []
    dropped = sync_session_with_pymol(cmd)
    if dropped:
        _refresh_open_windows(cmd)
    return dropped


def _refresh_open_windows(cmd) -> None:
    getter = getattr(cmd, "get_wizard", None)
    if not callable(getter):
        return
    try:
        wizard = getter()
    except Exception:
        return
    refresh = getattr(wizard, "_refresh_open_windows", None)
    if callable(refresh):
        try:
            refresh()
        except Exception:
            pass


def install_delete_hook(cmd) -> None:
    if cmd is None or getattr(cmd, "_pmv_delete_hook", False):
        return
    original = cmd.delete

    def delete(name="", *args, **kwargs):
        result = original(name, *args, **kwargs)
        try:
            after_pymol_delete(cmd, name)
        except Exception:
            pass
        return result

    cmd.delete = delete
    cmd._pmv_delete_orig = original
    cmd._pmv_delete_hook = True


def uninstall_delete_hook(cmd) -> None:
    if cmd is None or not getattr(cmd, "_pmv_delete_hook", False):
        return
    original = getattr(cmd, "_pmv_delete_orig", None)
    if original is not None:
        cmd.delete = original
    try:
        delattr(cmd, "_pmv_delete_orig")
        delattr(cmd, "_pmv_delete_hook")
    except Exception:
        cmd._pmv_delete_hook = False
        cmd._pmv_delete_orig = None
