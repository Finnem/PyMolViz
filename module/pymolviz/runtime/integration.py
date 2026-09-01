"""Install / uninstall session hooks, follow callback, and commands."""

from __future__ import annotations

FOLLOW_NAME = "_pmv_follow"
_WIZARD_COMMANDS = ("pymolviz_wizard", "pmvw", "pymolviz_reload_wizard", "pymolviz_exit_wizard")

_INSTALLED = False
_SAVE_TASK = None
_SAVE_PURGE_TASK = None
_RESTORE_TASK = None
_FOLLOW_TIMER = None


def _session_save_purge_ephemeral(*_args, **_kwargs):
    """Drop wizard-only CGOs before PyMOL serializes the object list."""
    from pymol import cmd

    from ..util.pymol_helpers import purge_preview_objects
    from ..wizard import is_pymolviz_wizard

    purge_preview_objects(cmd)
    wizard = cmd.get_wizard()
    is_pmv = is_pymolviz_wizard(wizard)
    try:
        cmd.delete("pmv_camera_center")
    except Exception:
        pass
    if is_pmv and hasattr(wizard, "_ensure_runtime") and not getattr(wizard, "_closed", False):
        wizard._ensure_runtime()
    return 1


def _session_save(*_args, **_kwargs):
    from .session import persist
    persist()
    return 1


def _session_restore(*_args, **_kwargs):
    from .runtime import get_runtime
    from .session import restore_from_session
    from ..wizard import reconcile_wizard_after_session_load

    objects = restore_from_session()
    get_runtime().reconcile(objects)
    reconcile_wizard_after_session_load()
    return 1


def _task_lists():
    import pymol
    save = getattr(pymol, "_session_save_tasks", None)
    restore = getattr(pymol, "_session_restore_tasks", None)
    if save is None:
        pymol._session_save_tasks = []
        save = pymol._session_save_tasks
    if restore is None:
        pymol._session_restore_tasks = []
        restore = pymol._session_restore_tasks
    return save, restore


def _stop_follow_timer():
    global _FOLLOW_TIMER
    timer = _FOLLOW_TIMER
    _FOLLOW_TIMER = None
    if timer is None:
        return
    try:
        timer.stop()
    except Exception:
        pass
    try:
        timer.deleteLater()
    except Exception:
        pass


def _install_follow(cmd):
    """Follow after mouse/wheel on the viewer — never a periodic cmd poll."""
    from .follow import ensure_follow_input_hook

    _uninstall_follow(cmd)
    ensure_follow_input_hook()


def _uninstall_follow(cmd):
    _stop_follow_timer()
    try:
        from .follow import uninstall_follow_hooks
        uninstall_follow_hooks()
    except Exception:
        pass
    try:
        cmd.delete(FOLLOW_NAME)
    except Exception:
        pass


def _unextend_wizard_commands(cmd):
    for name in _WIZARD_COMMANDS:
        try:
            del cmd.__dict__[name]
        except (KeyError, AttributeError, TypeError):
            pass


def _pymolviz_sync(*_args, **_kwargs):
    from ..points import has_dynamic_sources
    from .runtime import get_runtime
    from .session import all_objects
    runtime = get_runtime()
    for obj in all_objects():
        if has_dynamic_sources(obj):
            runtime.sync(obj)
        elif runtime.bindings.get(obj.id) is None:
            runtime.reconcile([obj])


def _pymolviz_reload(*_args, **_kwargs):
    reload_pymolviz(restart_wizard=True)


def _pymolviz_wizard(*_args, **_kwargs):
    from ..wizard import start_wizard
    start_wizard()


def _pymolviz_reload_wizard(*_args, **_kwargs):
    reload_pymolviz(restart_wizard=True)


def _pymolviz_exit_wizard(*_args, **_kwargs):
    from ..wizard import exit_wizard
    exit_wizard()


def get_runtime(cmd=None):
    from .runtime import get_runtime as _get
    return _get(cmd)


def install(cmd=None):
    """Register session hooks, follow callback, and commands. Idempotent."""
    global _INSTALLED, _SAVE_TASK, _SAVE_PURGE_TASK, _RESTORE_TASK
    if cmd is None:
        from pymol import cmd

    if _INSTALLED:
        return

    _SAVE_PURGE_TASK = _session_save_purge_ephemeral
    _SAVE_TASK = _session_save
    _RESTORE_TASK = _session_restore
    save, restore = _task_lists()
    if _SAVE_PURGE_TASK not in save:
        save.insert(0, _SAVE_PURGE_TASK)
    if _SAVE_TASK not in save:
        save.append(_SAVE_TASK)
    if _RESTORE_TASK not in restore:
        restore.append(_RESTORE_TASK)

    _install_follow(cmd)
    _unextend_wizard_commands(cmd)
    from ..util.pymol_helpers import extend_cmd

    extend_cmd(cmd, "pymolviz_sync", _pymolviz_sync)
    extend_cmd(cmd, "pymolviz_reload", _pymolviz_reload)
    extend_cmd(cmd, "pymolviz_wizard", _pymolviz_wizard)
    extend_cmd(cmd, "pmvw", _pymolviz_wizard)
    extend_cmd(cmd, "pymolviz_reload_wizard", _pymolviz_reload_wizard)
    extend_cmd(cmd, "pymolviz_exit_wizard", _pymolviz_exit_wizard)
    _INSTALLED = True

    from .runtime import get_runtime
    from .session import restore_from_session
    objects = restore_from_session()
    if objects:
        get_runtime(cmd).reconcile(objects)


def uninstall(cmd=None):
    """Remove hooks and follow callback so hot-reload does not stack them."""
    global _INSTALLED, _SAVE_TASK, _SAVE_PURGE_TASK, _RESTORE_TASK
    from .follow import reset_follow_state
    from .runtime import reset_runtime
    from .session import clear

    try:
        if cmd is None:
            from pymol import cmd
        _uninstall_follow(cmd)
        _unextend_wizard_commands(cmd)
    except Exception:
        pass

    save, restore = _task_lists()
    if _SAVE_PURGE_TASK is not None:
        while _SAVE_PURGE_TASK in save:
            save.remove(_SAVE_PURGE_TASK)
    if _SAVE_TASK is not None:
        while _SAVE_TASK in save:
            save.remove(_SAVE_TASK)
    if _RESTORE_TASK is not None:
        while _RESTORE_TASK in restore:
            restore.remove(_RESTORE_TASK)

    reset_follow_state()
    reset_runtime()
    clear()
    _SAVE_PURGE_TASK = None
    _SAVE_TASK = None
    _RESTORE_TASK = None
    _INSTALLED = False


def reload_pymolviz(restart_wizard=True):
    """Uninstall hooks, purge pymolviz.*, reinstall, reconcile."""
    import importlib
    import sys

    from pymol import cmd

    from ..util.pymol_helpers import restore_view
    from ..wizard import exit_wizard, is_pymolviz_wizard

    saved_view = cmd.get_view()

    wizard_was_active = False
    try:
        wizard = cmd.get_wizard()
        wizard_was_active = is_pymolviz_wizard(wizard)
    except Exception:
        wizard_was_active = False

    try:
        exit_wizard(cmd)
    except Exception:
        try:
            cmd.set_wizard()
        except Exception:
            pass

    try:
        uninstall()
    except Exception:
        pass

    for name in list(sys.modules):
        if name == "pymolviz" or name.startswith("pymolviz."):
            del sys.modules[name]

    integration = importlib.import_module("pymolviz.runtime.integration")
    integration.install()

    if restart_wizard or wizard_was_active:
        wizard_mod = importlib.import_module("pymolviz.wizard")
        wizard_mod.start_wizard()

    restore_view(cmd, saved_view)
