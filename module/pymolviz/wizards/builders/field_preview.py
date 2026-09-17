"""Ephemeral native IsoSurface / IsoMesh / Volume preview for field builders (no Qt).

Live preview loads the same PyMOL objects Done creates (``cmd.isosurface``,
``cmd.isomesh``, ``cmd.volume``). Geometry and color bricks are copied into
temporary maps (``_pmv_prev_geom_map``, ``_pmv_prev_color_map``) so native
commands have a map even when the Field is ``register=False``.

Volume is a real volume, not an isosurface stand-in.
``VOLUME_PREVIEW_IS_ISO_PROXY`` is False; there is no silent iso-proxy fallback.
"""

from __future__ import annotations

import uuid
from typing import Optional, Sequence

from ...fields.field import ensure_brick
from ...fields.identity import (
    GEN_DISTANCE,
    GEN_GAUSSIAN,
    GEN_NEAREST_COLOR,
    GEN_NEAREST_PROP,
    GEN_SIGNED_VDW,
)
from ...meshes.CGOCollection import CGOCollection
from .field_params import DEFAULT_DISTANCE_ISOLEVEL, default_preset_iso_level, normalize_field_model
from ...volumetric.kinds import VOLUME_KINDS
from .field_visual import (
    load_grid_as_map,
    make_field_visual,
    named_grid_copy,
)
from .load_field import field_from_selection
from .points import VisualPoint, enabled_points
from .surface_params import COLOR_MODE_FIELD
from ..widgets.theme import PRIMARY

PREVIEW_FIELD_ISO_NAME = "_pmv_prev_field_iso"
PREVIEW_FIELD_VISUAL_NAME = "_pmv_prev_field_visual"
PREVIEW_FIELD_VOLUME_NAME = "_pmv_prev_field_volume"
PREVIEW_DOMAIN_NAME = "_pmv_prev_domain"
PREVIEW_GEOM_MAP_NAME = "_pmv_prev_geom_map"
PREVIEW_COLOR_MAP_NAME = "_pmv_prev_color_map"
PREVIEW_COLOR_RAMP_NAME = "_pmv_prev_color_ramp"
PREVIEW_FIELD_CLIP_NAME = "_pmv_prev_field_clip"

PREVIEW_NATIVE_PREFIXES = (
    PREVIEW_FIELD_ISO_NAME,
    PREVIEW_FIELD_VISUAL_NAME,
    PREVIEW_FIELD_VOLUME_NAME,
    PREVIEW_GEOM_MAP_NAME,
    PREVIEW_COLOR_MAP_NAME,
    PREVIEW_COLOR_RAMP_NAME,
    PREVIEW_DOMAIN_NAME,
    PREVIEW_FIELD_CLIP_NAME,
)

_DOMAIN_BOX_COLOR = tuple(c / 255.0 for c in PRIMARY)

VOLUME_PREVIEW_IS_ISO_PROXY = False
_VOLUME_PREVIEW_KINDS = VOLUME_KINDS
_ISO_PREVIEW_ALGORITHMS = frozenset({
    GEN_GAUSSIAN,
    GEN_DISTANCE,
    GEN_SIGNED_VDW,
    GEN_NEAREST_PROP,
    GEN_NEAREST_COLOR,
})


def field_supports_iso_preview(algorithm) -> bool:
    text = str(algorithm or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in _ISO_PREVIEW_ALGORITHMS:
        return True
    if str(algorithm) in _VOLUME_PREVIEW_KINDS or str(algorithm) in ("IsoSurface", "IsoMesh"):
        return True
    return normalize_field_model(algorithm) in _ISO_PREVIEW_ALGORITHMS


def default_field_iso_level(algorithm, grid=None) -> float:
    if normalize_field_model(algorithm) == GEN_DISTANCE:
        return float(DEFAULT_DISTANCE_ISOLEVEL)
    return default_preset_iso_level(algorithm, grid)


def volume_preview_is_iso_proxy(kind) -> bool:
    """Volume preview used to be a translucent iso; it is native ``cmd.volume`` now."""
    return False


def preview_uses_wireframe(kind) -> bool:
    """IsoMesh is native ``cmd.isomesh``, not a CGO wireframe."""
    return False


def _preview_color(points: Sequence[VisualPoint]):
    active = enabled_points(points)
    if not active:
        return (0.20, 0.60, 0.90)
    return tuple(float(c) for c in active[0].color[:3])


def _clims_key(clims) -> Optional[tuple]:
    if clims is None:
        return None
    try:
        return (round(float(clims[0]), 5), round(float(clims[1]), 5))
    except (TypeError, ValueError, IndexError):
        return None


def _colormap_spec_key(spec) -> Optional[str]:
    if not spec:
        return None
    if isinstance(spec, dict):
        import json

        return json.dumps(spec, sort_keys=True, default=str)
    return str(spec)


def _aabb_key(aabb) -> Optional[tuple]:
    if aabb is None:
        return None
    try:
        return tuple(round(float(v), 4) for row in aabb for v in row)
    except (TypeError, ValueError, IndexError):
        return None


def domain_box_key(aabb, grid=None) -> Optional[tuple]:
    from ...fields.domain import aabb_has_extent, grid_affine, normalize_aabb

    box = normalize_aabb(aabb)
    if box is None or not aabb_has_extent(box):
        return None
    key = _aabb_key(box)
    if grid is None:
        return key
    import numpy as np

    A = grid_affine(grid)
    if np.allclose(A, np.eye(4), atol=1e-12):
        return key
    return key + tuple(round(float(v), 4) for v in A.reshape(-1))


def build_domain_box_collection(aabb, name: str = PREVIEW_DOMAIN_NAME, grid=None) -> CGOCollection:
    """Ephemeral wire Domain: parallelepiped when the brick is sheared, else AABB."""
    from ...fields.domain import aabb_center_extent, aabb_has_extent, grid_has_crystal_shear, grid_world_edges
    from ...meshes.Arrows import Arrows
    from ...meshes.CenteredBox import CenteredBox
    from ...util.line_style import LineStyle

    if grid is not None and grid_has_crystal_shear(grid):
        starts, ends = grid_world_edges(grid)
        if starts is not None and len(starts) == 12:
            mesh = Arrows(
                starts=starts,
                ends=ends,
                color=_DOMAIN_BOX_COLOR,
                bypass_colormap=True,
                line_style=LineStyle(ends="None"),
                quality=0,
                shaft_radius=0.05,
                name=name,
            )
            return CGOCollection([mesh], name=name)
    if not aabb_has_extent(aabb):
        return CGOCollection([], name=name)
    pose = aabb_center_extent(aabb)
    if pose is None:
        return CGOCollection([], name=name)
    center, extent = pose
    mesh = CenteredBox(
        center,
        extent,
        color=_DOMAIN_BOX_COLOR,
        wireframe=True,
        bypass_colormap=True,
        name=name,
    )
    return CGOCollection([mesh], name=name)


def estimate_grid_iso_job(grid) -> dict:
    """Cheap voxel-count estimate for a native field preview (map + iso/volume)."""
    import numpy as np

    from ...util.solvent_params import (
        HEAVY_SURFACE_SECONDS,
        HEAVY_SURFACE_VOXELS,
        _ISO_SEC_PER_VOXEL,
        _MC_SEC_PER_CUBE,
    )

    n_vox = 0
    values = getattr(grid, "values", None)
    if values is not None:
        n_vox = int(np.asarray(values).size)
    cubes = min(float(n_vox), 0.05 * float(n_vox))
    seconds = float(n_vox) * _ISO_SEC_PER_VOXEL + cubes * _MC_SEC_PER_CUBE
    return {
        "algorithm": "ISO",
        "quality": 0,
        "n_atoms": 0,
        "voxels": n_vox,
        "seconds": float(seconds),
        "heavy": bool(seconds >= HEAVY_SURFACE_SECONDS or n_vox >= HEAVY_SURFACE_VOXELS),
    }


def _retarget_preview_maps(visual, *, color_src=None, clip_aabb=None):
    """Point geometry/color bricks at the temporary preview map names."""
    from ...fields.clip import normalize_clip_aabb
    from ...volumetric.ColorRamp import ColorRamp

    if visual is None:
        return None
    brick = getattr(visual, "grid_data", None)
    box = normalize_clip_aabb(clip_aabb if clip_aabb is not None else getattr(visual, "clip_aabb", None))
    # Crop once in ``load_geometry_map`` from the full preview brick; pre-cropping
    # here would apply world-space ``clip_aabb`` twice and breaks flip/slide.
    if brick is not None:
        visual.grid_data = named_grid_copy(brick, PREVIEW_GEOM_MAP_NAME)
    visual.clip_aabb = box
    color = getattr(visual, "color", None)
    if issubclass(type(color), ColorRamp):
        data = getattr(color, "data", None)
        if data is not None:
            color.data = named_grid_copy(data, PREVIEW_COLOR_MAP_NAME)
            color.dependencies = [color.data]
        try:
            color._name = PREVIEW_COLOR_RAMP_NAME
        except Exception:
            pass
    deps = [visual.grid_data] if getattr(visual, "grid_data", None) is not None else []
    if issubclass(type(color), ColorRamp):
        deps.append(color)
    visual.dependencies = deps
    visual.id = "preview_" + uuid.uuid4().hex
    return visual


def make_unregistered_preview_visual(
    field,
    *,
    kind="IsoSurface",
    name=PREVIEW_FIELD_VISUAL_NAME,
    iso_level=None,
    side="positive",
    clip_aabb=None,
    selection=None,
    carve=None,
    color=None,
    transparency=0.0,
    color_field_id=None,
    color_src=None,
    colormap="RdYlBu_r",
    colormap_spec=None,
    cmd=None,
    clims=None,
):
    """Build an IsoSurface / IsoMesh / Volume that is not interned in the catalog.

    Geometry (and ColorRamp data) are copied onto ``_pmv_prev_*`` map names.
    """
    from ...fields.isovalues import isovalues_for_side
    from .field_visual import default_iso_level, resolve_field_grid

    grid = resolve_field_grid(field)
    if grid is None:
        grid = ensure_brick(field, cmd=cmd)
    if grid is None:
        return None
    kind = str(kind or "IsoSurface")
    level = iso_level
    isovalues = None
    if kind not in VOLUME_KINDS:
        if level is None:
            level = default_iso_level(grid)
        isovalues = isovalues_for_side(level, side)
    visual = make_field_visual(
        kind,
        field if field is not None else grid,
        name,
        colormap=colormap,
        colormap_spec=colormap_spec,
        level=level,
        color=color,
        transparency=float(transparency or 0.0),
        geometry_field_id=getattr(field, "id", None),
        color_field_id=color_field_id,
        color_src=color_src,
        isovalues=isovalues,
        clip_aabb=clip_aabb,
        selection=selection,
        carve=carve,
        side=1,
        clims=clims,
    )
    return _retarget_preview_maps(visual, color_src=color_src, clip_aabb=clip_aabb)


def preview_pymol_names(visual) -> tuple:
    """PyMOL object names created for a preview visual (maps, ramp, iso/volume)."""
    if visual is None:
        return ()
    names = [str(visual.name)]
    grid = getattr(visual, "grid_data", None)
    stored = getattr(grid, "_name", None) if grid is not None else None
    if stored:
        names.append(str(stored))
    elif grid is not None:
        names.append(str(grid.name))
    from ...volumetric.ColorRamp import ColorRamp

    color = getattr(visual, "color", None)
    if issubclass(type(color), ColorRamp):
        names.append(str(color.name))
        data = getattr(color, "data", None)
        if data is not None:
            names.append(str(getattr(data, "_name", None) or data.name))
    names.append("%s_volume_color_ramp" % visual.name)
    return tuple(n for n in names if n)


def load_preview_field_visual(cmd, visual) -> None:
    """Load maps then the native iso/mesh/volume via ``visual.load(cmd)``."""
    if visual is None:
        return
    try:
        if cmd is not None:
            cmd.delete(visual.name)
    except Exception:
        pass
    try:
        visual.is_loaded = False
    except Exception:
        pass
    geom = getattr(visual, "grid_data", None)
    if geom is not None and getattr(visual, "clip_aabb", None) is None:
        load_grid_as_map(geom, getattr(geom, "_name", None) or geom.name, cmd=cmd)
    from ...volumetric.ColorRamp import ColorRamp

    color = getattr(visual, "color", None)
    if issubclass(type(color), ColorRamp):
        data = getattr(color, "data", None)
        if data is not None:
            load_grid_as_map(data, getattr(data, "_name", None) or data.name, cmd=cmd)
    from ...Displayable import call_load

    call_load(visual, cmd)


def build_field_iso_preview_visual(
    cmd,
    points: Sequence[VisualPoint],
    *,
    algorithm,
    domain,
    quality,
    resolution,
    property_key="b_factor",
    iso_level=None,
    name: str = PREVIEW_FIELD_ISO_NAME,
    color_mode=None,
    color_field_id=None,
    colormap=None,
    colormap_spec=None,
    clims=None,
    kind="IsoSurface",
    volume_kind=None,
):
    """Native IsoSurface (and optional Volume) of a draft field (not interned)."""
    active = enabled_points(points)
    if not active or not field_supports_iso_preview(algorithm):
        return None
    field = field_from_selection(
        cmd,
        points,
        "_pmv_field_preview",
        algorithm=algorithm,
        domain=domain,
        quality=quality,
        resolution=resolution,
        property_key=property_key,
        register=False,
    )
    if field is None:
        return None
    mode = str(color_mode or "")
    color_src = None
    uniform = None
    cmap = colormap or "RdYlBu_r"
    transparency = 1.0 - min(float(pt.alpha) for pt in active)
    if mode != COLOR_MODE_FIELD:
        color_field_id = None
        uniform = _preview_color(points)
    level = float(
        iso_level if iso_level is not None else default_field_iso_level(algorithm, ensure_brick(field, cmd=cmd))
    )
    visual = make_unregistered_preview_visual(
        field,
        kind=kind,
        name=name,
        iso_level=level,
        color=uniform,
        transparency=transparency,
        color_field_id=color_field_id,
        color_src=color_src,
        colormap=cmap,
        colormap_spec=colormap_spec,
        cmd=cmd,
        clims=clims,
    )
    extra = None
    if volume_kind and str(volume_kind) in VOLUME_KINDS:
        extra = make_unregistered_preview_visual(
            field,
            kind=volume_kind,
            name=PREVIEW_FIELD_VOLUME_NAME,
            iso_level=level,
            color=uniform,
            transparency=transparency,
            color_field_id=color_field_id,
            color_src=color_src,
            colormap=cmap,
            colormap_spec=colormap_spec,
            cmd=cmd,
            clims=clims,
        )
        if extra is not None and extra.grid_data is not None and visual is not None:
            extra.grid_data = visual.grid_data
    if extra is not None:
        return visual, extra
    return visual


def build_grid_preview_visual(
    field,
    *,
    kind="IsoSurface",
    iso_level=None,
    side="positive",
    clip_aabb=None,
    selection=None,
    carve=None,
    color=(1.0, 1.0, 1.0),
    transparency=0.0,
    color_field_id=None,
    color_src=None,
    colormap=None,
    colormap_spec=None,
    clims=None,
    name: str = PREVIEW_FIELD_VISUAL_NAME,
    cmd=None,
):
    """Native IsoSurface / IsoMesh / Volume of an existing Field (not interned)."""
    return make_unregistered_preview_visual(
        field,
        kind=kind,
        name=name,
        iso_level=iso_level,
        side=side,
        clip_aabb=clip_aabb,
        selection=selection,
        carve=carve,
        color=color if color is not None else (1.0, 1.0, 1.0),
        transparency=transparency,
        color_field_id=color_field_id,
        color_src=color_src,
        colormap=colormap or "RdYlBu_r",
        colormap_spec=colormap_spec,
        cmd=cmd,
        clims=clims,
    )


# Older names kept as aliases so imports do not break mid-refactor.
build_field_iso_preview_collection = build_field_iso_preview_visual
build_grid_iso_preview_collection = build_grid_preview_visual


def iso_preview_key(
    points: Sequence[VisualPoint],
    *,
    algorithm,
    domain,
    quality,
    resolution,
    property_key,
    iso_level,
    color_mode=None,
    color_field_id=None,
    colormap=None,
    colormap_spec=None,
    clims=None,
) -> Optional[tuple]:
    """Hashable preview fingerprint, including a cheap color fingerprint."""
    active = enabled_points(points)
    if not active:
        return None
    centers = []
    colors = []
    for pt in active:
        try:
            xyz = pt.resolve()
        except Exception:
            xyz = pt.xyz()
        centers.append(tuple(round(float(v), 4) for v in xyz))
        rgba = pt.rgba() if hasattr(pt, "rgba") else pt.color
        colors.append(tuple(round(float(c), 4) for c in list(rgba)[:3]))
    domain_key = None
    if domain is not None:
        getter = getattr(domain, "to_dict", None)
        if callable(getter):
            domain_key = tuple(sorted(getter().items()))
        else:
            domain_key = str(domain)
    return (
        normalize_field_model(algorithm),
        int(quality),
        round(float(resolution), 4),
        str(property_key or ""),
        round(float(iso_level), 5),
        domain_key,
        tuple(centers),
        tuple(bool(getattr(pt, "enabled", True)) for pt in points),
        str(color_mode or ""),
        str(color_field_id or ""),
        str(colormap or ""),
        _colormap_spec_key(colormap_spec),
        _clims_key(clims),
        tuple(colors),
    )


def field_visual_preview_key(
    *,
    kind,
    field_id,
    iso_level,
    side="positive",
    clip_aabb=None,
    selection=None,
    carve=None,
    color_field_id=None,
    colormap=None,
    colormap_spec=None,
    clims=None,
    color=None,
    transparency=0.0,
    origin=None,
) -> Optional[tuple]:
    rgb = None
    if color is not None:
        try:
            rgb = tuple(round(float(c), 4) for c in list(color)[:3])
        except (TypeError, ValueError, IndexError):
            rgb = None
    origin_key = None
    if origin is not None:
        try:
            vals = getattr(origin, "tolist", lambda: origin)()
            origin_key = tuple(round(float(v), 4) for v in list(vals)[:3])
        except (TypeError, ValueError):
            origin_key = None
    return (
        str(kind),
        str(field_id or ""),
        round(float(iso_level), 5),
        str(side or ""),
        _aabb_key(clip_aabb),
        str(selection or ""),
        round(float(carve), 4) if carve is not None else None,
        str(color_field_id or ""),
        str(colormap or ""),
        _colormap_spec_key(colormap_spec),
        _clims_key(clims),
        rgb,
        round(float(transparency or 0.0), 4),
        origin_key,
    )
