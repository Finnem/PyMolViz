"""Construct and persist volumetric field visuals (no Qt)."""

from __future__ import annotations

from typing import Optional

from ..catalog import display_name, is_field_visual

from ...volumetric.kinds import FIELD_VISUAL_KINDS, ISO_KINDS, VOLUME_KINDS
from ...volumetric.map_load import (
    bind_iso_color_ramp,
    clip_map_name,
    color_ramp_for_field,
    colormap_for_color_field,
    ensure_grid_ready,
    ensure_map_loaded,
    grid_map_name,
    is_native_grid,
    load_geometry_map,
    load_grid_as_map,
    named_grid_copy,
    resolve_color_source,
    sync_visual_grid_from_field,
)

KIND_DEFAULT_NAMES = {
    "Volume": "pmv_volume",
    "IsoVolume": "pmv_isovolume",
    "IsoSurface": "pmv_isosurface",
    "IsoMesh": "pmv_isomesh",
}


def field_visual_options(obj) -> dict:
    """Persisted visual knobs for the Field Visual editor (no Qt)."""
    from ...fields.isovalues import primary_isovalue, primary_side

    entries = getattr(obj, "isovalues", None) or []
    side = primary_side(entries, default_side=int(getattr(obj, "side", 1) or 1))
    if len(entries) >= 2:
        side_index = 2
    else:
        side_index = 0 if side >= 0 else 1
    color = getattr(obj, "color", None)
    rgb = None
    if color is not None and not hasattr(color, "name"):
        try:
            rgb = (float(color[0]), float(color[1]), float(color[2]))
        except (TypeError, IndexError, ValueError):
            rgb = None
    cmap_obj = getattr(obj, "colormap", None)
    cmap = getattr(cmap_obj, "preset", None)
    if not cmap:
        cmap = cmap_obj if isinstance(cmap_obj, str) else None
    spec = getattr(obj, "colormap_spec", None)
    range_mode = getattr(cmap_obj, "range_mode", None) if cmap_obj is not None else None
    clims = getattr(obj, "clims", None)
    pair = None
    if clims is not None and len(clims) >= 2:
        pair = (float(clims[0]), float(clims[-1]))
    from ...util.colormap_spec import normalization_from_stored_spec

    stored_norm = normalization_from_stored_spec(spec)
    if stored_norm is not None:
        range_mode = stored_norm.mode
        if stored_norm.vmin is not None and stored_norm.vmax is not None:
            pair = (float(stored_norm.vmin), float(stored_norm.vmax))
    geom = getattr(obj, "geometry_field_id", None)
    color_id = getattr(obj, "color_field_id", None)
    sel = getattr(obj, "selection", None)
    carve = getattr(obj, "carve", None)
    carve_sel = str(sel).strip() if sel else ""
    carve_radius = None
    if carve is not None:
        try:
            carve_radius = float(carve)
        except (TypeError, ValueError):
            carve_radius = None
    from .preview_mode import read_preview_mode
    from ...util.clip_gizmo import read_clip_gizmo_state

    return {
        "kind": type(obj).__name__,
        "geometry_field_id": str(geom) if geom else None,
        "color_field_id": str(color_id) if color_id else None,
        "level": float(primary_isovalue(entries, default_level=float(getattr(obj, "level", 0) or 0))),
        "side": int(side),
        "side_index": int(side_index),
        "transparency": float(getattr(obj, "transparency", 0) or 0),
        "color": rgb,
        "clip_aabb": getattr(obj, "clip_aabb", None),
        "selection": carve_sel or None,
        "carve": carve_radius,
        "colormap": cmap if isinstance(cmap, str) else ((spec or {}).get("preset") if spec else None),
        "colormap_spec": spec,
        "range_mode": range_mode,
        "clims": pair,
        "preview_mode": read_preview_mode(obj),
        "clip_gizmos": read_clip_gizmo_state(obj),
    }


def default_visual_name(kind, field=None) -> str:
    base = KIND_DEFAULT_NAMES.get(str(kind), "pmv_field_visual")
    if field is None:
        return base
    from ...util.field_sample import field_label

    label = field_label(field)
    if not label:
        return base
    suffix = {
        "Volume": "volume",
        "IsoVolume": "isovolume",
        "IsoSurface": "isosurface",
        "IsoMesh": "isomesh",
    }.get(str(kind), "visual")
    return "%s_%s" % (label, suffix)


def default_iso_level(grid) -> float:
    import numpy as np

    values = getattr(grid, "values", None)
    if values is None:
        return 0.0
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def resolve_field_grid(field):
    from ...util.field_sample import resolve_grid

    return resolve_grid(field)


def translate_visual_clip_aabb(visual, delta) -> None:
    """Move a world-space crop with an origin-only lattice translation."""
    if visual is None:
        return
    from ...fields.clip import translate_clip_aabb

    visual.clip_aabb = translate_clip_aabb(getattr(visual, "clip_aabb", None), delta)


def retarget_visual_clip_after_wrap(visual, old_aabb, new_aabb, origin_delta, copied) -> None:
    if visual is None:
        return
    from ...fields.clip import retarget_clip_aabb

    visual.clip_aabb = retarget_clip_aabb(
        getattr(visual, "clip_aabb", None),
        old_aabb,
        new_aabb,
        origin_delta=origin_delta,
        copied=copied,
    )


def make_field_visual(
    kind,
    grid,
    name,
    *,
    colormap="RdYlBu_r",
    level=None,
    color=None,
    transparency=0.0,
    obj_id=None,
    geometry_field_id=None,
    color_field_id=None,
    isovalues=None,
    clip_aabb=None,
    side=1,
    transfer_stops=None,
    color_src=None,
    clims=None,
    colormap_spec=None,
    selection=None,
    carve=None,
):
    """Build a Volume / IsoVolume / IsoSurface / IsoMesh wrapping ``grid``."""
    from ...util.colormap_spec import (
        ColormapDefinition,
        apply_colormap_alpha_to_volume_ramp,
        persist_colormap_attrs,
        volume_colormap_arg,
    )
    from .carve_around import normalize_carve_args

    kind = str(kind)
    carve_sel, carve_radius = normalize_carve_args(selection, carve)
    cmap_arg = volume_colormap_arg(colormap, colormap_spec)
    field = None
    try:
        from ...fields.field import as_field, ensure_brick

        field = as_field(grid) if grid is not None else None
        if field is not None:
            brick = ensure_brick(field)
            if brick is not None:
                grid = brick
            if not geometry_field_id:
                geometry_field_id = str(field.id)
    except Exception:
        field = None
    if grid is None:
        raise ValueError("Field visual requires a grid")
    geometry_field_id = str(geometry_field_id) if geometry_field_id else str(getattr(grid, "id", "") or "") or None
    color_field_id = str(color_field_id) if color_field_id else None
    if color_src is not None and not color_field_id:
        color_field_id = str(getattr(color_src, "id", "") or "") or None
    color_ramp = (
        color_ramp_for_field(color_field_id, colormap=cmap_arg, color_src=color_src, clims=clims)
        if (color_field_id or color_src is not None)
        else None
    )
    if kind in VOLUME_KINDS:
        kwargs = dict(
            name=name,
            colormap=cmap_arg,
            geometry_field_id=geometry_field_id,
            color_field_id=color_field_id,
            clip_aabb=clip_aabb,
            transfer_stops=transfer_stops,
            selection=carve_sel,
            carve=carve_radius,
        )
        if clims is not None:
            kwargs["clims"] = clims
        if kind == "IsoVolume":
            from ...volumetric.IsoVolume import IsoVolume

            visual = IsoVolume(grid, **kwargs)
        else:
            from ...volumetric.Volume import Volume

            visual = Volume(grid, **kwargs)
    elif kind == "IsoMesh":
        from ...volumetric.IsoMesh import IsoMesh
        from ...fields.isovalues import normalize_isovalues, primary_isovalue

        if isovalues is None and level is None:
            level = default_iso_level(grid)
        entries = normalize_isovalues(isovalues, default_level=level if level is not None else default_iso_level(grid), default_color=color, default_side=side)
        visual = IsoMesh(
            grid,
            primary_isovalue(entries),
            name=name,
            color=color_ramp if color_ramp is not None else color,
            transparency=float(transparency or 0),
            geometry_field_id=geometry_field_id,
            color_field_id=color_field_id,
            isovalues=entries,
            clip_aabb=clip_aabb,
            side=side,
            selection=carve_sel or "",
            carve=carve_radius,
        )
    elif kind == "IsoSurface":
        from ...volumetric.IsoSurface import IsoSurface
        from ...fields.isovalues import normalize_isovalues, primary_isovalue

        if isovalues is None and level is None:
            level = default_iso_level(grid)
        entries = normalize_isovalues(isovalues, default_level=level if level is not None else default_iso_level(grid), default_color=color, default_side=side)
        visual = IsoSurface(
            grid,
            primary_isovalue(entries),
            name=name,
            color=color_ramp if color_ramp is not None else color,
            transparency=float(transparency or 0),
            geometry_field_id=geometry_field_id,
            color_field_id=color_field_id,
            isovalues=entries,
            clip_aabb=clip_aabb,
            side=side,
            selection=carve_sel or "",
            carve=carve_radius,
        )
    else:
        raise ValueError("Unknown field visual type %r" % kind)
    if obj_id:
        visual.id = str(obj_id)
    _preset, stored = persist_colormap_attrs(colormap, colormap_spec)
    if stored:
        visual.colormap_spec = stored
    elif _preset:
        visual.colormap_spec = None
    defn = None
    if colormap_spec:
        defn = ColormapDefinition.from_dict(colormap_spec)
    if defn is not None and getattr(visual, "alphas", None) is not None and getattr(visual, "clims", None) is not None:
        visual.alphas = apply_colormap_alpha_to_volume_ramp(visual.clims, visual.alphas, defn)
        from ...fields.isovalues import stops_from_volume_ramp

        visual.transfer_stops = stops_from_volume_ramp(visual.clims, visual.alphas)
    return visual


def symmetrize_field_to_selection(cmd, field, selection=None, padding=0.0, cell=None):
    """Lattice-wrap the field so the selection lies inside, then reload maps/visuals."""
    import numpy as np

    from ...fields.crystal import apply_symmetrize_field
    from ...fields.dependents import dependents_of_field
    from ...fields.field import ensure_brick
    from ..catalog import is_field_visual

    before = ensure_brick(field, cmd=cmd)
    from ...fields.domain import brick_aabb

    old_aabb = brick_aabb(before)
    old_origin = None
    if before is not None:
        from ...fields.domain import grid_local_to_world

        old_origin = grid_local_to_world(before, getattr(before, "origin", (0.0, 0.0, 0.0))).reshape(3)
    wrapped, grid, info = apply_symmetrize_field(
        cmd, field, selection=selection, padding=padding, cell=cell
    )
    load_grid_as_map(grid, name=info.get("map_name"), cmd=cmd)
    new_aabb = brick_aabb(grid)
    delta = None
    if old_origin is not None and grid is not None:
        from ...fields.domain import grid_local_to_world

        new_origin = grid_local_to_world(grid, getattr(grid, "origin", (0.0, 0.0, 0.0))).reshape(3)
        delta = new_origin - old_origin
    copied = bool(info.get("copied"))
    for obj in dependents_of_field(wrapped):
        if is_field_visual(obj):
            retarget_visual_clip_after_wrap(obj, old_aabb, new_aabb, delta, copied)
    for obj in dependents_of_field(wrapped):
        if is_field_visual(obj):
            persist_field_visual(cmd, obj)
    info = dict(info)
    info["old_aabb"] = old_aabb
    info["origin_delta"] = None if delta is None else delta.tolist()
    return wrapped, info


def persist_field_visual(cmd, visual) -> None:
    from ...runtime.persist import persist_field_visual as _persist_field_visual

    return _persist_field_visual(cmd, visual)


def delete_field_visual(cmd, visual) -> None:
    from ...runtime.session import remove as session_remove
    from .preview import delete_visual

    try:
        delete_visual(cmd, visual)
    except Exception:
        session_remove(visual)
        name = display_name(visual)
        if cmd is not None and name:
            try:
                cmd.delete(str(name))
            except Exception:
                pass


def delete_field_and_visuals(cmd, field, visuals=None) -> None:
    from ...fields.dependents import dependents_of_field
    from ...fields.field import forget_field_wrap
    from ...runtime.session import get as session_get
    from ...runtime.session import remove as session_remove
    from ...util.field_sample import forget_native_grid

    if visuals is None:
        visuals = dependents_of_field(field)
    for obj in list(visuals or ()):
        if is_field_visual(obj):
            delete_field_visual(cmd, obj)
            continue
        try:
            from .preview import delete_visual

            delete_visual(cmd, obj)
        except Exception:
            session_remove(obj)
            name = display_name(obj)
            if cmd is not None and name:
                try:
                    cmd.delete(str(name))
                except Exception:
                    pass
    name = display_name(field) or grid_map_name(field)
    if cmd is not None and name:
        try:
            cmd.delete(str(name))
        except Exception:
            pass
    forget_native_grid(getattr(field, "id", None))
    forget_field_wrap(getattr(field, "id", None))
    if session_get(str(getattr(field, "id", "") or "")) is not None:
        session_remove(field)


def convert_isosurface_visual(cmd, visual, name=None):
    """Convert an isosurface Field Visual to an explicit Surface collection."""
    from ...fields.convert import convert_isosurface_to_surface
    from ...meshes.CGOCollection import CGOCollection
    from .object_names import unused_object_name
    from .preview import persist_collection

    label = unused_object_name(name or ((getattr(visual, "_name", None) or "iso") + "_surface"), cmd)
    surface = convert_isosurface_to_surface(visual, name=label)
    collection = CGOCollection([surface], name=label)
    persist_collection(cmd, collection)
    return collection


def pymol_name_for(obj, cmd=None) -> Optional[str]:
    from .preview import _pymol_object_name

    if cmd is not None:
        name = _pymol_object_name(cmd, obj)
        if name:
            return name
    stored = getattr(obj, "_name", None)
    if stored:
        return str(stored)
    return None


def iter_visuals_for_field(field, objects=None):
    from ..catalog import field_identity_keys, visual_field_keys

    if objects is None:
        from ...runtime.session import all_objects

        objects = all_objects()
    keys = set(field_identity_keys(field))
    out = []
    for obj in objects:
        if not is_field_visual(obj):
            continue
        if keys & set(visual_field_keys(obj)):
            out.append(obj)
    return out


def existing_visual_of_kind(field, kind, objects=None):
    """First session visual of ``kind`` bound to ``field``, or None."""
    want = str(kind)
    for vis in iter_visuals_for_field(field, objects):
        if type(vis).__name__ == want:
            return vis
    return None


def visual_is_in_pymol(cmd, visual) -> bool:
    name = display_name(visual) or getattr(visual, "name", None)
    if cmd is None or not name:
        return False
    try:
        return str(name) in [str(n) for n in cmd.get_names("objects")]
    except Exception:
        return False


def ensure_field_visual(
    cmd,
    field,
    kind="IsoSurface",
    *,
    name=None,
    level=None,
    color_field_id=None,
    colormap="RdYlBu_r",
    color=None,
    persist=True,
):
    """Reuse a visual of ``kind`` for ``field``, or create and load one.

    Identical From Selection recipes intern the same Field; this does not
    add a second IsoSurface when one already exists. If the session visual is
    missing from PyMOL, it is loaded again.
    """
    kind = str(kind or "IsoSurface")
    existing = existing_visual_of_kind(field, kind)
    if existing is not None:
        if persist and not visual_is_in_pymol(cmd, existing):
            persist_field_visual(cmd, existing)
        return existing
    from .object_names import unused_object_name

    label = name or default_visual_name(kind, field)
    vis_name = unused_object_name(label, cmd)
    visual = make_field_visual(
        kind,
        field,
        vis_name,
        colormap=colormap,
        level=level,
        color=color,
        geometry_field_id=getattr(field, "id", None),
        color_field_id=color_field_id,
    )
    if persist:
        persist_field_visual(cmd, visual)
    else:
        try:
            from ...runtime.session import add as session_add

            session_add(visual)
        except Exception:
            pass
    return visual
