"""Load Field bricks into PyMOL maps for Volume / Iso* visuals (no Qt)."""

from __future__ import annotations

from typing import Optional

def sync_visual_grid_from_field(visual, cmd=None):
    """Use the Field's current brick so Volume/iso follow lattice origin shifts.

    Preview copies (``named_grid_copy``) keep the pre-wrap origin; ``cmd.volume``
    would otherwise keep drawing that stale map. Preview visuals (``preview_``
    ids / ``_pmv_prev_*`` names) keep their own brick so a crop is not undone.
    """
    if visual is None:
        return None
    ident = str(getattr(visual, "id", "") or "")
    name = str(getattr(visual, "name", "") or "")
    if ident.startswith("preview_") or name.startswith("_pmv_prev"):
        return getattr(visual, "grid_data", None)
    fid = str(getattr(visual, "geometry_field_id", "") or "")
    if not fid:
        return getattr(visual, "grid_data", None)
    try:
        from ..fields.field import ensure_brick
        from ..runtime.session import get as session_get

        field = session_get(fid)
        brick = ensure_brick(field, cmd=cmd) if field is not None else None
        if brick is not None:
            visual.grid_data = brick
            try:
                import numpy as np

                from ..fields.domain import grid_affine

                visual.A_to = np.array(grid_affine(brick), copy=True)
            except Exception:
                pass
            return brick
    except Exception:
        pass
    return getattr(visual, "grid_data", None)

def is_native_grid(grid) -> bool:
    from ..util.field_sample import PYMOL_MAP_ID_PREFIX

    oid = str(getattr(grid, "id", "") or "")
    return oid.startswith(PYMOL_MAP_ID_PREFIX)


def grid_map_name(grid) -> str:
    stored = getattr(grid, "_name", None)
    if stored:
        return str(stored)
    return str(getattr(grid, "name", "") or "")


def _discrete_rgb_colormap(colors):
    from ..ColorMap import ColorMap

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
    from ..ColorMap import ColorMap

    stops = getattr(color_src, "color_stops", None)
    if not stops:
        grid = getattr(color_src, "grid_data", None)
        stops = getattr(grid, "color_stops", None) if grid is not None else None
    if not stops:
        from ..fields.color_blend import color_blend_stops
        from ..fields.generators import _atom_rgb, _atom_xyz

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
    from ..util.field_sample import is_rgb_color_field

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
        from ..runtime.session import get as session_get

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
        from ..fields.field import ensure_brick as _ensure
        from ..fields.identity import KIND_CATEGORICAL, normalize_kind
        from ..util.field_sample import is_rgb_color_field
        from .ColorRamp import ColorRamp

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
    from .ColorRamp import ColorRamp

    if issubclass(type(color), ColorRamp):
        return visual
    fid = getattr(visual, "color_field_id", None)
    spec = getattr(visual, "colormap_spec", None)
    if spec:
        from ..util.colormap_spec import volume_colormap_arg

        colormap = volume_colormap_arg(colormap, spec)
    ramp = color_ramp_for_field(
        fid,
        colormap=colormap,
        clims=getattr(visual, "clims", None),
    )
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

    from ..fields.field import Field

    copied = Field(
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

    from ..fields.domain import grid_affine

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

    from ..fields.domain import grid_pymol_brick_params

    display = named_grid_copy(grid, name)
    origin, step = grid_pymol_brick_params(grid)
    display.origin = tuple(float(v) for v in np.asarray(origin).reshape(3))
    display.step_sizes = tuple(float(v) for v in np.asarray(step).reshape(3))
    display.A_to = np.eye(4)
    return display


def _stamp_cartesian_map_cell(cmd, name, grid) -> None:
    """P1 cell matching the brick AABB — never the molecule CRYST1 (that shears volumes)."""
    import numpy as np

    from ..fields.domain import grid_pymol_brick_params

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
        from ..Displayable import call_load

        call_load(grid)
        return grid
    try:
        existing = [str(n) for n in cmd.get_names("objects")]
    except Exception:
        existing = []
    if label in existing:
        from ..fields.crystal import grid_is_crystal_axis

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
                from ..Displayable import call_load

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
    from ..Displayable import call_load

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
    from ..fields.clip import crop_grid_to_aabb, normalize_clip_aabb

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
