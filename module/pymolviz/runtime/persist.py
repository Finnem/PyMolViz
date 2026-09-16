"""Persist session displayables into PyMOL (CGO collections and field visuals)."""

from __future__ import annotations

from typing import Optional

from ..meshes.CGOCollection import CGOCollection


def persist_collection(cmd_, collection: CGOCollection, obj_id=None):
    """Write to session and materialize. ``obj_id`` replaces an existing visual."""
    from .integration import install
    from .runtime import get_runtime
    from .session import add as session_add
    from .session import get as session_get

    try:
        install(cmd_)
    except Exception:
        pass
    if obj_id:
        collection.id = str(obj_id)
    runtime = get_runtime(cmd_)
    existing = session_get(collection.id)
    if existing is not None:
        try:
            runtime.remove(existing)
        except Exception:
            pass
    session_add(collection)
    runtime.materialize(collection, rebuild=False)


def persist_field_visual(cmd, visual) -> None:
    from ..wizards.catalog import display_name
    from ..wizards.builders.object_names import unused_object_name
    from ..wizards.builders.preview import set_visual_enabled
    from ..volumetric.map_load import ensure_map_loaded, sync_visual_grid_from_field
    from .presence import pause_presence_sync
    from .runtime import get_runtime
    from .session import add as session_add
    from .session import get as session_get

    grid = sync_visual_grid_from_field(visual, cmd)
    if cmd is not None:
        ensure_map_loaded(cmd, grid, reload=True)
    raw = display_name(visual) or getattr(visual, "_name", None)
    vid = str(getattr(visual, "id", "") or "")
    keep = None
    try:
        existing = session_get(vid) if vid else None
        if existing is visual and raw:
            keep = raw
    except Exception:
        keep = raw
    name = unused_object_name(raw or "visual", cmd, keep=keep)
    if name and name != raw:
        try:
            visual.name = name
        except Exception:
            visual._name = name
    with pause_presence_sync():
        if cmd is not None and name:
            try:
                existing = [str(n) for n in cmd.get_names("objects")]
            except Exception:
                existing = []
            if str(name) in existing:
                try:
                    cmd.delete(str(name))
                except Exception:
                    pass
                visual.is_loaded = False
        session_add(visual)
        if cmd is not None:
            get_runtime(cmd).materialize(visual)
            set_visual_enabled(cmd, visual, True)
