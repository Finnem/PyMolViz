"""Install / uninstall session hooks, follow callback, and commands."""

from __future__ import annotations

FOLLOW_NAME = "_pmv_follow"
_WIZARD_COMMANDS = ("pymolviz_wizard", "pymolviz_reload_wizard")

_INSTALLED = False
_SAVE_TASK = None
_RESTORE_TASK = None


def _session_save(*_args, **_kwargs):
    from .session import persist
    persist()
    return 1


def _session_restore(*_args, **_kwargs):
    from .runtime import get_runtime
    from .session import restore_from_session
    objects = restore_from_session()
    get_runtime().reconcile(objects)
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


def _install_follow(cmd):
    from .follow import pymolviz_follow_callback
    _uninstall_follow(cmd)
    try:
        cmd.load_callback(pymolviz_follow_callback, FOLLOW_NAME, 0)
    except TypeError:
        cmd.load_callback(pymolviz_follow_callback, FOLLOW_NAME)


def _uninstall_follow(cmd):
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


def get_runtime(cmd=None):
    from .runtime import get_runtime as _get
    return _get(cmd)


def install(cmd=None):
    """Register session hooks, follow callback, and commands. Idempotent."""
    global _INSTALLED, _SAVE_TASK, _RESTORE_TASK
    if cmd is None:
        from pymol import cmd

    if _INSTALLED:
        return

    _SAVE_TASK = _session_save
    _RESTORE_TASK = _session_restore
    save, restore = _task_lists()
    if _SAVE_TASK not in save:
        save.append(_SAVE_TASK)
    if _RESTORE_TASK not in restore:
        restore.append(_RESTORE_TASK)

    _install_follow(cmd)
    _unextend_wizard_commands(cmd)
    cmd.extend("pymolviz_sync", _pymolviz_sync)
    cmd.extend("pymolviz_reload", _pymolviz_reload)
    cmd.extend("pymolviz_wizard", _pymolviz_wizard)
    cmd.extend("pymolviz_reload_wizard", _pymolviz_reload_wizard)
    _INSTALLED = True

    from .runtime import get_runtime
    from .session import restore_from_session
    objects = restore_from_session()
    if objects:
        get_runtime(cmd).reconcile(objects)


def uninstall(cmd=None):
    """Remove hooks and follow callback so hot-reload does not stack them."""
    global _INSTALLED, _SAVE_TASK, _RESTORE_TASK
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
    if _SAVE_TASK is not None:
        while _SAVE_TASK in save:
            save.remove(_SAVE_TASK)
    if _RESTORE_TASK is not None:
        while _RESTORE_TASK in restore:
            restore.remove(_RESTORE_TASK)

    reset_follow_state()
    reset_runtime()
    clear()
    _SAVE_TASK = None
    _RESTORE_TASK = None
    _INSTALLED = False


def reload_pymolviz(restart_wizard=True):
    """Uninstall hooks, purge pymolviz.*, reinstall, reconcile."""
    import importlib
    import sys

    from pymol import cmd

    wizard_was_active = False
    try:
        wizard = cmd.get_wizard()
        wizard_was_active = wizard is not None and type(wizard).__name__ == "PyMolVizWizard"
    except Exception:
        wizard_was_active = False

    try:
        uninstall()
    except Exception:
        pass

    try:
        cmd.set_wizard()
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
