"""Import/export the live session from the wizard panel (dialogs optional)."""

from __future__ import annotations

IMPORT_LABEL = "Import PyMolViz File"
EXPORT_LABEL = "Export Session"
IMPORT_TITLE = "Import PyMolViz"
EXPORT_TITLE = "Export session"
EMPTY_EXPORT_MSG = "Nothing to export."
DEFAULT_EXPORT_NAME = "pymolviz"


def session_export_items() -> list:
    """Non-preview objects currently in the live session."""
    from ..runtime.session import all_objects, is_ephemeral

    return [obj for obj in all_objects() if not is_ephemeral(obj)]


def import_session_path(cmd, path) -> list:
    """Load a ``.pmv`` pack and intern it into the live session."""
    from ..io import intern_loaded, load

    return intern_loaded(cmd, load(path))


def prompt_import_session(parent, cmd):
    """Ask for a pack and intern it. ``None`` if the dialog is cancelled."""
    from ..io import OPEN_NATIVE_FILTER
    from .pick import overlay_get_open_file_name, overlay_warning

    path, _filt = overlay_get_open_file_name(
        parent,
        IMPORT_TITLE,
        "",
        OPEN_NATIVE_FILTER,
    )
    if not path:
        return None
    try:
        return import_session_path(cmd, path)
    except Exception as exc:
        overlay_warning(parent, IMPORT_TITLE, str(exc))
        raise


def prompt_export_session(parent):
    """Write the live session.

    Returns the path, ``False`` if there is nothing to export, or ``None``
    if the save dialog is cancelled.
    """
    from .builders.export import export_objects
    from .pick import overlay_information

    items = session_export_items()
    if not items:
        overlay_information(parent, EXPORT_TITLE, EMPTY_EXPORT_MSG)
        return False
    return export_objects(
        parent,
        items,
        DEFAULT_EXPORT_NAME,
        title=EXPORT_TITLE,
    )
