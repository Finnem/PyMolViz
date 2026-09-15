"""Shared export dialog for builder pages (script and native pack)."""

from __future__ import annotations

from ..pick import overlay_get_save_file_name
from ...io import SAVE_FILTER, complete_save_path, save


def export_objects(parent, obj, default_name, title="Export"):
    """Ask for a path and write ``obj`` as ``.pmv`` or ``.py``."""
    stem = str(default_name or "pmv_object").strip() or "pmv_object"
    path, filt = overlay_get_save_file_name(
        parent,
        title,
        "%s.pmv" % stem,
        SAVE_FILTER,
    )
    if not path:
        return None
    path, fmt = complete_save_path(path, filt)
    save(obj, path, format=fmt)
    return path
