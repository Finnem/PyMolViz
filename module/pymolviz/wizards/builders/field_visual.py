"""Construct and persist volumetric field visuals (no Qt)."""

from __future__ import annotations

from typing import Optional

from ..catalog import display_name, is_field_visual, is_native_field

VOLUME_KINDS = frozenset({"Volume", "IsoVolume"})
ISO_KINDS = frozenset({"IsoSurface", "IsoMesh"})
FIELD_VISUAL_KINDS = tuple(sorted(VOLUME_KINDS | ISO_KINDS))

KIND_DEFAULT_NAMES = {
    "Volume": "pmv_volume",
    "IsoVolume": "pmv_isovolume",
    "IsoSurface": "pmv_isosurface",
    "IsoMesh": "pmv_isomesh",
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


def sync_visual_grid_from_field(visual, cmd=None):
    """Use the Field's current brick so Volume/iso follow lattice origin shifts.

    Preview copies (``named_grid_copy``) keep the pre-wrap origin; ``cmd.volume``
    would otherwise keep drawing that stale map.
    """
    if visual is None:
        return None
    fid = str(getattr(visual, "geometry_field_id", "") or "")
    if not fid:
        return getattr(visual, "grid_data", None)
    try:
        from ...fields.field import ensure_brick
        from ...runtime.session import get as session_get

        field = session_get(fid)
        brick = ensure_brick(field, cmd=cmd) if field is not None else None
        if brick is not None:
            visual.grid_data = brick
            try:
                import numpy as np

                from ...fields.domain import grid_affine

                visual.A_to = np.array(grid_affine(brick), copy=True)
            except Exception:
                pass
            return brick
    except Exception:
        pass
    return getattr(visual, "grid_data", None)


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


def is_native_grid(grid) -> bool:
    return is_native_field(grid)


def grid_map_name(grid) -> str:
    stored = getattr(grid, "_name", None)
    if stored:
        return str(stored)
    return str(getattr(grid, "name", "") or "")


def _discrete_rgb_colormap(colors):
    from ...ColorMap import ColorMap

    rows = []
    for item in colors or ():
        try:
            rows.append([float(item[0]), float(item[1]), float(item[2])])
        except (TypeError, ValueError, IndexError):
            continue
    if not rows:
        return None
    if len(rows) == 1:
        rows = [rows[0], rows[0]]
    return ColorMap(rows, values_are_single_color=False)


def _blended_rgb_colormap(color_src):
    """Interpolating (t, RGB) stops along the spatial color axis, not index chips."""
    from ...ColorMap import ColorMap

    stops = getattr(color_src, "color_stops", None)
    if not stops:
        grid = getattr(color_src, "grid_data", None)
        stops = getattr(grid, "color_stops", None) if grid is not None else None
    if not stops:
        from ...fields.color_blend import color_blend_stops
        from ...fields.generators import _atom_rgb, _atom_xyz

        gen = getattr(color_src, "generator", None) or {}
        atoms = gen.get("atoms") or ()
        centers = _atom_xyz(atoms)
        if int(centers.shape[0]) == 0:
            return None
        colors = [_atom_rgb(item) for item in list(atoms)[: int(centers.shape[0])]]
        stops = color_blend_stops(centers, colors)
    try:
        return ColorMap(list(stops))
    except Exception:
        return None


def colormap_for_color_field(color_src, colormap="RdYlBu_r"):
    """Use snapshotted RGB for nearest-atom color fields, not RdYlBu_r."""
    from ...util.field_sample import is_rgb_color_field

    if is_rgb_color_field(color_src):
        blended = _blended_rgb_colormap(color_src)
        if blended is not None:
            return blended
        discrete = _discrete_rgb_colormap(getattr(color_src, "categories", None))
        if discrete is not None:
            return discrete
    return colormap


def resolve_color_source(color_field_id):
    if not color_field_id:
        return None
    try:
        from ...runtime.session import get as session_get

        return session_get(str(color_field_id))
    except Exception:
        return None


def color_ramp_for_field(color_field_id, colormap="RdYlBu_r", color_src=None, clims=None):
    """Native ColorRamp on a second map. None if the color Field cannot be baked.

    ``color_src`` is an already-resolved Field (preview bricks are not interned).
    """
    if color_src is None and not color_field_id:
        return None
    try:
        from ...fields.field import ensure_brick as _ensure
        from ...fields.identity import KIND_CATEGORICAL, normalize_kind
        from ...util.field_sample import is_rgb_color_field
        from ...volumetric.ColorRamp import ColorRamp

        if color_src is None:
            color_src = resolve_color_source(color_field_id)
        color_grid = _ensure(color_src) if color_src is not None else None
        if color_grid is None:
            return None
        cmap = colormap_for_color_field(color_src, colormap)
        categorical = (
            normalize_kind(getattr(color_src, "kind", None)) == KIND_CATEGORICAL
            and not is_rgb_color_field(color_src)
        )
        return ColorRamp(color_grid, colormap=cmap, clims=clims, interpolate=not categorical)
    except Exception:
        return None


def bind_iso_color_ramp(visual, colormap="RdYlBu_r"):
    """Attach a ColorRamp to ``visual.color`` from ``color_field_id`` when needed."""
    if visual is None:
        return visual
    color = getattr(visual, "color", None)
    from ...volumetric.ColorRamp import ColorRamp

    if issubclass(type(color), ColorRamp):
        return visual
    fid = getattr(visual, "color_field_id", None)
    ramp = color_ramp_for_field(fid, colormap=colormap)
    if ramp is not None:
        visual.color = ramp
        deps = list(getattr(visual, "dependencies", None) or [])
        if ramp not in deps:
            deps.append(ramp)
            visual.dependencies = deps
    return visual


def named_grid_copy(grid, name):
    """Copy ``grid`` under a distinct PyMOL map name without mutating the source."""
    if grid is None:
        return None
    import numpy as np

    from ...volumetric.GridData import GridData

    copied = GridData(
        np.asarray(grid.values, dtype=float).reshape(-1),
        step_sizes=tuple(float(v) for v in np.asarray(grid.step_sizes).reshape(3)),
        step_counts=tuple(int(v) for v in np.asarray(grid.step_counts).reshape(3)),
        origin=tuple(float(v) for v in np.asarray(grid.origin).reshape(3)),
        name=name,
    )
    src_ttt = getattr(grid, "A_to", None)
    if src_ttt is not None:
        copied.A_to = np.array(src_ttt, copy=True)
    return copied


def apply_grid_object_ttt(cmd, name, grid) -> None:
    """Place a map/volume/iso with the brick's crystal ``A_to`` (not identity)."""
    if cmd is None or not name or grid is None:
        return
    import numpy as np

    from ...fields.domain import grid_affine

    setter = getattr(cmd, "set_object_ttt", None)
    if not callable(setter):
        return
    try:
        setter(str(name), [float(v) for v in np.asarray(grid_affine(grid)).reshape(-1)])
    except Exception:
        pass


def _display_grid_for_pymol(grid, name):
    """Copy with Å origin/spacing and identity TTT so native volumes draw."""
    import numpy as np

    from ...fields.domain import grid_pymol_brick_params

    display = named_grid_copy(grid, name)
    origin, step = grid_pymol_brick_params(grid)
    display.origin = tuple(float(v) for v in np.asarray(origin).reshape(3))
    display.step_sizes = tuple(float(v) for v in np.asarray(step).reshape(3))
    display.A_to = np.eye(4)
    return display


def _stamp_cartesian_map_cell(cmd, name, grid) -> None:
    """P1 cell matching the brick AABB — never the molecule CRYST1 (that shears volumes)."""
    import numpy as np

    from ...fields.domain import grid_pymol_brick_params

    setter = getattr(cmd, "set_symmetry", None)
    if not callable(setter):
        return
    origin, step = grid_pymol_brick_params(grid)
    counts = np.maximum(np.asarray(getattr(grid, "step_counts", (1, 1, 1)), dtype=float).reshape(3), 1.0)
    extent = np.maximum(np.abs(step) * counts, 1e-3)
    setter(name, float(extent[0]), float(extent[1]), float(extent[2]), 90.0, 90.0, 90.0, "P1")
    try:
        cmd.set("map_auto_expand_sym", 0)
    except Exception:
        pass
    try:
        cmd.set("map_auto_expand_sym", 0, name)
    except Exception:
        pass


def load_grid_as_map(grid, name=None, cmd=None):
    """Load a brick into PyMOL as a named map (interned or preview)."""
    if grid is None:
        return None
    label = str(name or grid_map_name(grid) or "map")
    try:
        grid._name = label
    except Exception:
        pass
    if cmd is None:
        from ...Displayable import call_load

        call_load(grid)
        return grid
    try:
        existing = [str(n) for n in cmd.get_names("objects")]
    except Exception:
        existing = []
    if label in existing:
        from ...fields.crystal import grid_is_crystal_axis

        if grid_is_crystal_axis(grid) and not getattr(grid, "_crystal_lookup", False):
            grid.is_loaded = True
            return grid
        try:
            cmd.delete(label)
        except Exception:
            pass
    load_brick = getattr(cmd, "load_brick", None)
    if callable(load_brick):
        import numpy as np

        try:
            from chempy.brick import Brick

            values = np.asarray(grid.values).reshape(
                tuple(int(c) + 1 for c in np.asarray(grid.step_counts).reshape(3))
            )
            display = _display_grid_for_pymol(grid, label)
            payload = Brick.from_numpy(values, display.step_sizes, origin=display.origin)
            load_brick(payload, label)
        except Exception:
            try:
                load_brick(_display_grid_for_pymol(grid, label), label)
            except Exception:
                from ...Displayable import call_load

                call_load(grid, cmd)
                return grid
        try:
            cmd.set("volume_mode", 0)
            display = _display_grid_for_pymol(grid, label)
            apply_grid_object_ttt(cmd, label, display)
            _stamp_cartesian_map_cell(cmd, label, display)
        except Exception:
            pass
        grid.is_loaded = True
        return grid
    from ...Displayable import call_load

    call_load(grid, cmd)
    return grid


def _object_names(cmd) -> set:
    if cmd is None:
        return set()
    try:
        return {str(n) for n in cmd.get_names("objects")}
    except Exception:
        return set()


def ensure_grid_ready(cmd, grid) -> None:
    """Load ``grid`` into PyMOL unless the map object already exists."""
    if grid is None:
        return
    name = grid_map_name(grid)
    existing = _object_names(cmd)
    if name and name in existing:
        grid.is_loaded = True
        return
    if is_native_grid(grid):
        # Native wrappers cache voxel data locally. Do not mark them loaded when
        # the PyMOL map object is gone — Volume/iso would skip load_brick and
        # point at a missing map.
        try:
            grid.is_loaded = False
        except Exception:
            pass
    load_grid_as_map(grid, name=name, cmd=cmd)


def ensure_map_loaded(cmd, grid, *, reload=False) -> Optional[str]:
    """Return the PyMOL map name for ``grid``, loading it when absent.

    ``reload=True`` replaces an existing map object (cropped bricks, preview).
    """
    if grid is None:
        return None
    name = grid_map_name(grid)
    if not name:
        return None
    if reload:
        try:
            grid.is_loaded = False
        except Exception:
            pass
        load_grid_as_map(grid, name=name, cmd=cmd)
        return name
    ensure_grid_ready(cmd, grid)
    if cmd is not None and name not in _object_names(cmd):
        try:
            grid.is_loaded = False
        except Exception:
            pass
        load_grid_as_map(grid, name=name, cmd=cmd)
    return name


def clip_map_name(grid) -> str:
    """Sibling map name for a cropped copy; preview maps are replaced in place."""
    base = str(grid_map_name(grid) or "map")
    if base.endswith("_clip") or base.startswith("_pmv_prev"):
        return base
    return "%s_clip" % base


def load_geometry_map(cmd, grid, clip_aabb=None):
    """Load the map a Volume/iso draws from, cropping to ``clip_aabb`` when set.

    The Field brick is not mutated. Cropped voxels go to ``{map}_clip`` (or
    replace a ``_pmv_prev_*`` preview map). Returns ``(map_name, reloaded)``.
    """
    from ...fields.clip import crop_grid_to_aabb, normalize_clip_aabb

    if grid is None:
        return None, False
    box = normalize_clip_aabb(clip_aabb)
    if box is None:
        return ensure_map_loaded(cmd, grid), False
    cropped = crop_grid_to_aabb(grid, box)
    if cropped is None:
        return ensure_map_loaded(cmd, grid), False
    name = clip_map_name(grid)
    try:
        cropped._name = name
    except Exception:
        pass
    load_grid_as_map(cropped, name=name, cmd=cmd)
    return name, True


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
):
    """Build a Volume / IsoVolume / IsoSurface / IsoMesh wrapping ``grid``."""
    from ...util.colormap_spec import persist_colormap_attrs, volume_colormap_arg

    kind = str(kind)
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
    from ...runtime.session import add as session_add

    grid = sync_visual_grid_from_field(visual, cmd)
    if cmd is not None:
        ensure_map_loaded(cmd, grid, reload=True)
    name = display_name(visual) or getattr(visual, "name", None)
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
    from ...Displayable import call_load
    from .preview import set_visual_enabled

    call_load(visual, cmd)
    set_visual_enabled(cmd, visual, True)
    session_add(visual)


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
