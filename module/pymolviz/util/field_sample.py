"""Sample a regular 3D field at points and map scalars through a colormap."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

PYMOL_MAP_ID_PREFIX = "pymol_map:"
_NATIVE_GRIDS = {}

DEFAULT_SURFACE_COLORMAP = "RdYlBu_r"
# Voxel-space Gaussian when painting a fine mesh from a coarse map.
SURFACE_FIELD_SMOOTH = 0.75
# 4-split only near-equilateral faces. SAS saddles are strips; 4-splitting
# those needles makes starbursts. Longest-edge splits stay conforming.
SKINNY_FACE_ASPECT = 2.8
SURFACE_COLORMAPS = (
    "RdYlBu_r",
    "coolwarm",
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "turbo",
    "seismic",
    "PiYG",
    "onwhite",
    "onwhite_r",
)
FIELD_COLORMAPS = SURFACE_COLORMAPS


def resolve_grid(obj):
    """Return a ``GridData`` from a field object, or None."""
    if obj is None:
        return None
    if type(obj).__name__ == "GridData":
        return obj
    nested = getattr(obj, "grid_data", None)
    if nested is not None and type(nested).__name__ == "GridData":
        return nested
    if type(obj).__name__ == "Field":
        try:
            from ..fields.field import ensure_brick

            return ensure_brick(obj)
        except Exception:
            return None
    return None


def forget_native_grid(field_id) -> None:
    """Drop a cached native PyMOL map wrapper."""
    if not field_id:
        return
    _NATIVE_GRIDS.pop(str(field_id), None)


def remember_field(obj) -> None:
    """Keep a sampleable field in the live session catalog for the Surface wizard."""
    if resolve_grid(obj) is None and type(obj).__name__ != "Field":
        return
    name = str(getattr(obj, "_name", None) or "")
    if name.startswith("cbar_dummy"):
        return
    try:
        from ..fields.field import as_field, intern_field

        field = as_field(obj)
        if field is None:
            return
        intern_field(field)
    except Exception:
        try:
            from ..runtime.session import add
            add(obj)
        except Exception:
            pass


def resolve_grid_from_session(field_id):
    if not field_id:
        return None
    key = str(field_id)
    cached = _NATIVE_GRIDS.get(key)
    if cached is not None:
        return resolve_grid(cached)
    try:
        from ..runtime.session import get
        grid = resolve_grid(get(key))
        if grid is not None:
            return grid
    except Exception:
        pass
    name = key[len(PYMOL_MAP_ID_PREFIX):] if key.startswith(PYMOL_MAP_ID_PREFIX) else key
    return resolve_grid(grid_from_pymol_map(name))


def resolve_field_from_session(field_id):
    """Return the session Field for ``field_id``, or None."""
    if not field_id:
        return None
    try:
        from ..runtime.session import get

        obj = get(str(field_id))
    except Exception:
        obj = None
    if obj is not None and type(obj).__name__ == "Field":
        return obj
    try:
        from ..fields.field import as_field

        return as_field(obj) if obj is not None else None
    except Exception:
        return None


def is_rgb_color_field(field) -> bool:
    if field is None:
        return False
    from ..fields.identity import GEN_NEAREST_COLOR, KIND_VECTOR, normalize_kind

    gen = getattr(field, "generator", None) or {}
    if str(gen.get("type") or "") == GEN_NEAREST_COLOR:
        return True
    return normalize_kind(getattr(field, "kind", None)) == KIND_VECTOR and bool(
        getattr(field, "categories", None)
    )


def rgb_from_color_categories(values, categories):
    """Map color-index voxels onto snapshotted RGB (no named colormap)."""
    colors = []
    for item in categories or ():
        try:
            colors.append([float(item[0]), float(item[1]), float(item[2])])
        except (TypeError, ValueError, IndexError):
            continue
    values = np.asarray(values, dtype=float).reshape(-1)
    if not colors:
        return np.zeros((int(values.shape[0]), 3), dtype=float)
    palette = np.asarray(colors, dtype=float).reshape(-1, 3)
    n = int(palette.shape[0])
    if n == 1:
        return np.broadcast_to(palette[0], (int(values.shape[0]), 3)).copy()
    v = np.clip(values, 0.0, float(n - 1))
    lo = np.floor(v).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    t = (v - lo)[:, None]
    return palette[lo] * (1.0 - t) + palette[hi] * t


def rgb_from_color_field(values, field):
    """Map sampled color-field scalars through spatial RGB stops when present."""
    stops = getattr(field, "color_stops", None)
    if not stops:
        grid = getattr(field, "grid_data", None)
        stops = getattr(grid, "color_stops", None) if grid is not None else None
    if stops:
        from ..ColorMap import ColorMap

        try:
            cmap = ColorMap(list(stops))
            return np.asarray(cmap.get_color(values)[:, :3], dtype=float)
        except Exception:
            pass
    return rgb_from_color_categories(values, getattr(field, "categories", None))


def _pymol_cmd(cmd=None):
    if cmd is not None:
        return cmd
    try:
        from pymol import cmd as pymol_cmd
        return pymol_cmd
    except Exception:
        return None


def _skip_pymol_field_name(name) -> bool:
    text = str(name or "")
    return (
        not text
        or text.startswith("_pmv_prev_")
        or text.startswith("cbar_dummy")
        or text.startswith("preview_")
    )


def _pymol_object_is_field(cmd, name) -> bool:
    getter = getattr(cmd, "get_type", None)
    if callable(getter):
        try:
            typ = str(getter(name) or "").lower()
        except Exception:
            typ = ""
        if typ:
            return ("map" in typ) or ("volume" in typ)
    reader = getattr(cmd, "get_volume_field", None)
    if not callable(reader):
        return False
    try:
        data = np.asarray(reader(name), dtype=float)
    except Exception:
        return False
    return data.ndim == 3 and data.size > 0


def iter_pymol_field_names(cmd=None) -> List[str]:
    """PyMOL object names that expose a regular 3D field (map or volume)."""
    cmd = _pymol_cmd(cmd)
    if cmd is None:
        return []
    names = []
    typed = getattr(cmd, "get_names_of_type", None)
    if callable(typed):
        for kind in ("object:map", "object:volume"):
            try:
                names.extend(str(n) for n in typed(kind) or ())
            except Exception:
                pass
    if not names:
        try:
            names = [str(n) for n in cmd.get_names("objects")]
        except Exception:
            return []
        names = [n for n in names if _pymol_object_is_field(cmd, n)]
    out = []
    seen = set()
    for name in names:
        if _skip_pymol_field_name(name) or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def grid_from_pymol_map(name, cmd=None):
    """Wrap a native PyMOL map/volume as ``GridData``, or None."""
    name = str(name or "")
    if _skip_pymol_field_name(name):
        return None
    key = PYMOL_MAP_ID_PREFIX + name
    cached = _NATIVE_GRIDS.get(key)
    if cached is not None:
        return cached
    cmd = _pymol_cmd(cmd)
    if cmd is None or not hasattr(cmd, "get_volume_field"):
        return None
    try:
        values = np.asarray(cmd.get_volume_field(name), dtype=float)
    except Exception:
        return None
    if values.ndim != 3 or values.size == 0:
        return None
    try:
        cmd.set("map_auto_expand_sym", 0)
        cmd.set("map_auto_expand_sym", 0, name)
    except Exception:
        pass
    origin = np.zeros(3, dtype=float)
    step = np.ones(3, dtype=float)
    try:
        extent = cmd.get_extent(name)
        lo = np.asarray(extent[0], dtype=float).reshape(3)
        hi = np.asarray(extent[1], dtype=float).reshape(3)
        origin = lo
        denom = np.maximum(np.array(values.shape, dtype=float) - 1.0, 1.0)
        step = (hi - lo) / denom
        if (not np.all(np.isfinite(step))) or np.any(np.abs(step) < 1e-18):
            step = np.ones(3, dtype=float)
            origin = np.zeros(3, dtype=float)
    except Exception:
        pass
    from ..volumetric.GridData import GridData

    counts = np.asarray(values.shape, dtype=int) - 1
    grid = GridData(
        values.reshape(-1),
        step_sizes=step,
        step_counts=counts,
        origin=origin,
        name=name,
    )
    grid._name = name
    grid.id = key
    try:
        parsed = None
        getter = getattr(cmd, "get_symmetry", None)
        if callable(getter):
            from ..fields.crystal import parse_cell_params, _is_dummy_cell, _cell_is_orthogonal, attach_crystal_axis_frame

            parsed = parse_cell_params(getter(name))
            if parsed is not None and not _is_dummy_cell(parsed[0]) and not _cell_is_orthogonal(parsed[0]):
                attach_crystal_axis_frame(grid, parsed[0])
    except Exception:
        pass
    _NATIVE_GRIDS[key] = grid
    return grid


def _field_names(obj) -> List[str]:
    names = []
    for attr in ("_name", "name"):
        val = getattr(obj, attr, None)
        if val:
            names.append(str(val))
    nested = getattr(obj, "grid_data", None)
    if nested is not None:
        for attr in ("_name", "name"):
            val = getattr(nested, attr, None)
            if val:
                names.append(str(val))
    return names


def _session_visual_pymol_names(objects, cmd=None) -> set:
    """PyMOL object names owned by interned field visuals (not sampleable fields)."""
    from ..wizards.catalog import is_field_visual
    from ..wizards.builders.field_visual import pymol_name_for
    from ..volumetric.map_load import grid_map_name

    names = set()
    for obj in objects or ():
        if not is_field_visual(obj):
            continue
        label = pymol_name_for(obj, cmd)
        if not label:
            label = getattr(obj, "_name", None) or getattr(obj, "name", None)
        if label:
            names.add(str(label))
        grid = getattr(obj, "grid_data", None)
        if grid is not None:
            map_name = grid_map_name(grid)
            if map_name:
                names.add(str(map_name))
    return names


def discover_fields(objects: Optional[Iterable] = None, cmd=None) -> list:
    """Session fields plus native PyMOL maps/volumes that can be sampled."""
    if objects is None:
        try:
            from ..runtime.session import all_objects
            objects = all_objects()
        except Exception:
            objects = []
    from ..fields.field import as_field

    out = []
    seen_ids = set()
    seen_names = set()
    visual_pymol_names = _session_visual_pymol_names(objects, cmd=cmd)
    for obj in objects:
        if type(obj).__name__ in ("Volume", "IsoVolume", "IsoSurface", "IsoMesh"):
            continue
        field = as_field(obj) if type(obj).__name__ == "Field" or resolve_grid(obj) is not None else None
        if field is None:
            continue
        oid = str(getattr(field, "id", "") or "")
        if oid:
            if oid in seen_ids:
                continue
            seen_ids.add(oid)
        for name in _field_names(field):
            seen_names.add(name)
        out.append(field)
    for name in iter_pymol_field_names(cmd):
        if name in seen_names or name in visual_pymol_names:
            continue
        grid = grid_from_pymol_map(name, cmd=cmd)
        if grid is None:
            continue
        field = as_field(grid)
        if field is None:
            continue
        oid = str(getattr(field, "id", "") or "")
        if oid in seen_ids:
            continue
        seen_ids.add(oid)
        seen_names.add(name)
        out.append(field)
    return out


def field_label(obj) -> str:
    """User-facing name for a sampleable field."""
    for attr in ("_name", "name"):
        val = getattr(obj, attr, None)
        if val:
            text = str(val)
            if text:
                return text
    oid = str(getattr(obj, "id", "") or "")
    if oid.startswith(PYMOL_MAP_ID_PREFIX):
        return oid[len(PYMOL_MAP_ID_PREFIX):]
    return oid or "field"


def field_picker_label(obj) -> str:
    """Combo label: name, kind, grid size, units — no raw ids."""
    name = field_label(obj) or "field"
    parts = [name]
    kind = getattr(obj, "kind", None)
    if kind:
        parts.append(str(kind))
    grid = resolve_grid(obj)
    if grid is not None:
        counts = np.asarray(getattr(grid, "step_counts", None), dtype=int).reshape(-1)
        if counts.size >= 3:
            shape = (int(counts[0]) + 1, int(counts[1]) + 1, int(counts[2]) + 1)
            parts.append("%dx%dx%d" % shape)
    units = getattr(obj, "units", None)
    if units:
        parts.append(str(units))
    return " · ".join(parts)


def field_choices(objects: Optional[Iterable] = None, cmd=None) -> list:
    """``[(field_id, label), ...]`` for currently sampleable fields."""
    out = []
    seen = set()
    for obj in discover_fields(objects, cmd=cmd):
        oid = str(getattr(obj, "id", "") or "")
        if not oid or oid in seen:
            continue
        seen.add(oid)
        out.append((oid, field_label(obj)))
    return out


def sample_rgb_at(xyz, field_id, colormap=DEFAULT_SURFACE_COLORMAP, clims=None, smooth=None):
    """RGB at one point, or None if the field cannot be sampled."""
    grid = resolve_grid_from_session(field_id)
    if grid is None:
        return None
    if smooth is None:
        smooth = SURFACE_FIELD_SMOOTH
    values = sample_grid(grid, [xyz], smooth=smooth)
    field = resolve_field_from_session(field_id)
    if is_rgb_color_field(field):
        rgb = rgb_from_color_field(values, field)
    else:
        rgb, _ = rgb_from_scalars(values, colormap, clims)
    row = np.asarray(rgb, dtype=float).reshape(-1, 3)[0]
    return (float(row[0]), float(row[1]), float(row[2]))


# Voxel-space Gaussian used when painting a fine mesh from a coarse map.
# Small faces otherwise trace trilinear creases (axis-aligned "stretchmarks").
_SMOOTH_ATTR = "_pymolviz_gauss"


def grid_values_3d(grid) -> np.ndarray:
    counts = np.asarray(getattr(grid, "step_counts"), dtype=int).reshape(3)
    shape = (int(counts[0]) + 1, int(counts[1]) + 1, int(counts[2]) + 1)
    values = np.asarray(grid.values, dtype=float).reshape(-1)
    if values.size != int(np.prod(shape)):
        raise ValueError("Grid values do not match step_counts.")
    return values.reshape(shape)


def grid_min_edge(grid) -> float:
    """Do not field-split edges shorter than this (voxel size)."""
    if grid is None:
        return 0.0
    step = np.abs(np.asarray(getattr(grid, "step_sizes"), dtype=float).reshape(3))
    finite = step[np.isfinite(step) & (step > 1e-18)]
    if finite.size == 0:
        return 0.0
    return 0.75 * float(np.min(finite))


def _gaussian_field(values, sigma) -> np.ndarray:
    sigma = float(sigma)
    values = np.asarray(values, dtype=float)
    if sigma <= 1e-6 or min(values.shape) < 2:
        return values
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(values, sigma=sigma, mode="nearest")


def grid_field_for_sample(grid, smooth=0.0) -> np.ndarray:
    values = grid_values_3d(grid)
    sigma = float(smooth or 0.0)
    if sigma <= 1e-6:
        return values
    cached = getattr(grid, _SMOOTH_ATTR, None)
    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == sigma:
        return cached[1]
    smoothed = _gaussian_field(values, sigma)
    try:
        setattr(grid, _SMOOTH_ATTR, (sigma, smoothed))
    except Exception:
        pass
    return smoothed


def sample_grid(grid, xyz, smooth=0.0) -> np.ndarray:
    """Sample ``grid`` at ``xyz``. Out-of-bounds positions clamp to the edge.

    ``smooth`` is a Gaussian sigma in voxel units applied before trilinear
    interpolation. Use :data:`SURFACE_FIELD_SMOOTH` when the mesh is finer
    than the map so voxel faces do not print through as streaks.
    """
    field = grid_field_for_sample(grid, smooth)
    origin = np.asarray(grid.origin, dtype=float).reshape(3)
    step = np.asarray(grid.step_sizes, dtype=float).reshape(3)
    from ..fields.domain import grid_world_to_local

    local = grid_world_to_local(grid, xyz)
    return sample_regular_grid(field, origin, step, local)


def sample_regular_grid(values, origin, step_sizes, xyz) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 3:
        raise ValueError("Expected a 3D value array.")
    origin = np.asarray(origin, dtype=float).reshape(3)
    step = np.asarray(step_sizes, dtype=float).reshape(3)
    step = np.maximum(step, 1e-18)
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    n = int(xyz.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=float)
    nx, ny, nz = values.shape
    last = np.array([max(nx - 1, 0), max(ny - 1, 0), max(nz - 1, 0)], dtype=float)
    p = np.clip((xyz - origin) / step, 0.0, last)
    i0 = np.floor(p).astype(np.int32)
    i0[:, 0] = np.clip(i0[:, 0], 0, nx - 1)
    i0[:, 1] = np.clip(i0[:, 1], 0, ny - 1)
    i0[:, 2] = np.clip(i0[:, 2], 0, nz - 1)
    i1 = np.stack(
        (
            np.minimum(i0[:, 0] + 1, nx - 1),
            np.minimum(i0[:, 1] + 1, ny - 1),
            np.minimum(i0[:, 2] + 1, nz - 1),
        ),
        axis=1,
    )
    f = p - i0.astype(float)
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]

    def corner(ix, iy, iz):
        return values[ix, iy, iz]

    c000 = corner(i0[:, 0], i0[:, 1], i0[:, 2])
    c100 = corner(i1[:, 0], i0[:, 1], i0[:, 2])
    c010 = corner(i0[:, 0], i1[:, 1], i0[:, 2])
    c110 = corner(i1[:, 0], i1[:, 1], i0[:, 2])
    c001 = corner(i0[:, 0], i0[:, 1], i1[:, 2])
    c101 = corner(i1[:, 0], i0[:, 1], i1[:, 2])
    c011 = corner(i0[:, 0], i1[:, 1], i1[:, 2])
    c111 = corner(i1[:, 0], i1[:, 1], i1[:, 2])
    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


def project_edge_midpoint(a, b, na, nb):
    """Lift a chord midpoint onto the sphere implied by endpoints and normals.

    CGO interpolates RGB across triangle corners. Midpoints that stay on the
    chord sit inside a curved solvent surface and sample the field at the
    wrong place. A two-point sphere fit keeps the new vertex on the patch.
    """
    a = np.asarray(a, dtype=float).reshape(3)
    b = np.asarray(b, dtype=float).reshape(3)
    na = np.asarray(na, dtype=float).reshape(3)
    nb = np.asarray(nb, dtype=float).reshape(3)
    mid = 0.5 * (a + b)
    la = float(np.linalg.norm(na))
    lb = float(np.linalg.norm(nb))
    if la > 1e-18:
        na = na / la
    if lb > 1e-18:
        nb = nb / lb
    n = na + nb
    ln = float(np.linalg.norm(n))
    if ln > 1e-18:
        n = n / ln
    elif la > 1e-18:
        n = na
    else:
        n = np.array([0.0, 0.0, 1.0], dtype=float)
    dn = na - nb
    denom = float(np.dot(dn, dn))
    if denom > 1e-8:
        r = float(np.dot(a - b, dn) / denom)
        center = 0.5 * ((a - r * na) + (b - r * nb))
        vec = mid - center
        lv = float(np.linalg.norm(vec))
        if lv > 1e-12 and abs(r) > 1e-4:
            radial = vec / lv
            n_surf = radial if r > 0.0 else -radial
            if float(np.dot(n_surf, n)) < 0.0:
                n_surf = -n_surf
            return center + abs(r) * radial, n_surf
    return mid, n


def union_sphere_midpoint_projector(centers, radii):
    """Snap a sphere-fit midpoint onto the union of ``centers``/``radii``."""
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    radii = np.asarray(radii, dtype=float).reshape(-1)

    def _nearest(point):
        dist = np.linalg.norm(np.asarray(point, dtype=float).reshape(1, 3) - centers, axis=1)
        return int(np.argmin(np.abs(dist - radii)))

    def project(a, b, na, nb):
        p, n = project_edge_midpoint(a, b, na, nb)
        ia, ib = _nearest(a), _nearest(b)
        if ia == ib:
            idx = ia
        else:
            dist = np.linalg.norm(p.reshape(1, 3) - centers, axis=1)
            idx = int(np.argmin(dist - radii))
        center = centers[idx]
        radius = float(radii[idx])
        vec = p - center
        ln = float(np.linalg.norm(vec))
        if ln < 1e-18:
            return p, n
        radial = vec / ln
        n_surf = radial
        if float(np.dot(n_surf, n)) < 0.0:
            n_surf = -n_surf
        return center + radius * radial, n_surf

    return project


def _split_triangle(a, b, c, ab, bc, ca):
    """Return triangles covering ABC. ``ab``/``bc``/``ca`` are mid indices or None."""
    if ab is None and bc is None and ca is None:
        return [(a, b, c)]
    if ab is not None and bc is not None and ca is not None:
        return [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    if ab is not None and bc is None and ca is None:
        return [(a, ab, c), (ab, b, c)]
    if ab is None and bc is not None and ca is None:
        return [(a, b, bc), (a, bc, c)]
    if ab is None and bc is None and ca is not None:
        return [(a, b, ca), (b, c, ca)]
    if ab is None:
        return [(a, b, bc), (a, bc, ca), (ca, bc, c)]
    if bc is None:
        return [(a, ab, ca), (ab, b, c), (ca, ab, c)]
    return [(a, ab, c), (ab, b, bc), (ab, bc, c)]


def refine_mesh_for_field(
    vertices,
    normals,
    faces,
    values,
    grid=None,
    max_level=3,
    max_frac=0.08,
    max_vertices=80000,
    span=None,
    project=None,
    smooth=0.0,
    min_edge=None,
    skip_edge=None,
):
    """Split triangles whose corners span too much of the sampled field.

    CGO interpolates RGB, not the scalar. Near-equilateral steep faces mark
    every long edge (4-split). High-aspect SAS saddles only mark their long
    edges, in a shared set so both incident faces split the same edges.
    ``skip_edge(i, j)`` keeps a frozen edge out of that set (both faces agree).
    """
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    faces = np.asarray(faces, dtype=int).reshape(-1, 3)
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    values = np.asarray(values, dtype=float).reshape(-1)
    if faces.shape[0] == 0 or vertices.shape[0] < 3:
        return vertices, normals, faces, values
    if normals.shape[0] != vertices.shape[0]:
        normals = np.zeros_like(vertices)
        normals[:, 2] = 1.0
    if values.shape[0] != vertices.shape[0]:
        values = np.zeros((int(vertices.shape[0]),), dtype=float)
    verts = vertices.tolist()
    norms = normals.tolist()
    vals = values.tolist()
    faces_list = [tuple(int(i) for i in tri) for tri in faces]
    max_level = max(0, int(max_level))
    max_frac = float(max_frac)
    max_vertices = max(int(max_vertices), int(vertices.shape[0]))
    project = project_edge_midpoint if project is None else project
    if min_edge is None:
        min_edge = grid_min_edge(grid)
    min_edge = float(min_edge or 0.0)
    smooth = float(smooth or 0.0)

    def _span():
        if span is not None:
            width = abs(float(span))
            if width > 1e-12:
                return width
        arr = np.asarray(vals, dtype=float)
        finite = np.isfinite(arr)
        if not np.any(finite):
            return 1.0
        return float(max(np.max(arr[finite]) - np.min(arr[finite]), 1e-12))

    for _ in range(max_level):
        if len(verts) >= max_vertices:
            break
        tol = max_frac * _span()
        mids = {}
        new_faces = []
        split = False

        def _key(i, j):
            return (i, j) if i < j else (j, i)

        def _mid(i, j):
            key = _key(i, j)
            found = mids.get(key)
            if found is not None:
                return found
            pa = np.asarray(verts[i], dtype=float)
            pb = np.asarray(verts[j], dtype=float)
            na = np.asarray(norms[i], dtype=float)
            nb = np.asarray(norms[j], dtype=float)
            p, n = project(pa, pb, na, nb)
            p = np.asarray(p, dtype=float).reshape(3)
            n = np.asarray(n, dtype=float).reshape(3)
            ln = float(np.linalg.norm(n))
            n = (n / ln) if ln > 1e-18 else na
            if grid is not None:
                scalar = float(sample_grid(grid, p.reshape(1, 3), smooth=smooth)[0])
            else:
                scalar = 0.5 * (float(vals[i]) + float(vals[j]))
            idx = len(verts)
            verts.append(p.tolist())
            norms.append(n.tolist())
            vals.append(scalar)
            mids[key] = idx
            return idx

        def _long(i, j):
            if min_edge <= 0.0:
                return True
            pa = np.asarray(verts[i], dtype=float)
            pb = np.asarray(verts[j], dtype=float)
            return float(np.linalg.norm(pa - pb)) >= min_edge

        def _should_split(i, j):
            if skip_edge is not None and skip_edge(i, j):
                return False
            return abs(float(vals[i]) - float(vals[j])) > tol and _long(i, j)

        def _edge_len(i, j):
            pa = np.asarray(verts[i], dtype=float)
            pb = np.asarray(verts[j], dtype=float)
            return float(np.linalg.norm(pa - pb))

        marked = set()
        for a, b, c in faces_list:
            corners = ((a, b), (b, c), (c, a))
            if not any(_should_split(i, j) for i, j in corners):
                continue
            lengths = [_edge_len(i, j) for i, j in corners]
            longest = max(lengths)
            shortest = max(min(lengths), 1e-18)
            skinny = (longest / shortest) > SKINNY_FACE_ASPECT
            if skinny:
                thresh = 0.70 * longest
                for (i, j), length in zip(corners, lengths):
                    if skip_edge is not None and skip_edge(i, j):
                        continue
                    if length >= thresh and _long(i, j):
                        marked.add(_key(i, j))
            else:
                for i, j in corners:
                    if skip_edge is not None and skip_edge(i, j):
                        continue
                    if _long(i, j):
                        marked.add(_key(i, j))

        for i, j in marked:
            if len(verts) >= max_vertices and _key(i, j) not in mids:
                continue
            _mid(i, j)
            split = True

        for a, b, c in faces_list:
            ab = mids.get(_key(a, b))
            bc = mids.get(_key(b, c))
            ca = mids.get(_key(c, a))
            new_faces.extend(_split_triangle(a, b, c, ab, bc, ca))
        faces_list = new_faces
        if not split:
            break
    return (
        np.asarray(verts, dtype=float).reshape(-1, 3),
        np.asarray(norms, dtype=float).reshape(-1, 3),
        np.asarray(faces_list, dtype=int).reshape(-1, 3),
        np.asarray(vals, dtype=float).reshape(-1),
    )


def rgb_from_scalars(values, colormap=DEFAULT_SURFACE_COLORMAP, clims=None):
    """Map scalars to RGB via :class:`pymolviz.ColorMap`. Returns ``(rgb, (vmin, vmax))``."""
    values = np.asarray(values, dtype=float).reshape(-1)
    from .colormap_spec import coerce_definition, map_scalars, uses_stop_sampling

    defn = coerce_definition(colormap)
    if defn is not None and uses_stop_sampling(defn):
        if clims is None:
            finite = np.isfinite(values)
            if np.any(finite):
                vmin = float(np.min(values[finite]))
                vmax = float(np.max(values[finite]))
            else:
                vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = float(clims[0]), float(clims[1])
        if abs(vmax - vmin) < 1e-15:
            vmax = vmin + 1.0
        rgba = map_scalars(values, defn, vmin, vmax)
        return rgba[:, :3], (vmin, vmax)
    if defn is not None:
        colormap = defn.preset or DEFAULT_SURFACE_COLORMAP
    finite = np.isfinite(values)
    if clims is None:
        if np.any(finite):
            vmin = float(np.min(values[finite]))
            vmax = float(np.max(values[finite]))
        else:
            vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(clims[0]), float(clims[1])
    if abs(vmax - vmin) < 1e-15:
        vmax = vmin + 1.0
    from ..ColorMap import ColorMap
    cmap = ColorMap([vmin, vmax], colormap, values_are_single_color=False)
    rgb = np.zeros((int(values.shape[0]), 3), dtype=float)
    if np.any(finite):
        mapped = np.asarray(cmap.get_color(values[finite]), dtype=float)
        rgb[finite] = mapped[:, :3]
    return rgb, (vmin, vmax)


def normalize_clims(clims) -> Optional[List[float]]:
    if clims is None:
        return None
    try:
        lo, hi = float(clims[0]), float(clims[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    if hi < lo:
        lo, hi = hi, lo
    return [lo, hi]


def _linear_midpoint(a, b, na, nb):
    a = np.asarray(a, dtype=float).reshape(3)
    b = np.asarray(b, dtype=float).reshape(3)
    na = np.asarray(na, dtype=float).reshape(3)
    nb = np.asarray(nb, dtype=float).reshape(3)
    return 0.5 * (a + b), 0.5 * (na + nb)


def _field_projector(mesh):
    kind = type(mesh).__name__
    if kind == "Sphere":
        try:
            from ..points import resolve_xyz
            center = resolve_xyz(getattr(mesh, "position", None))
            radius = float(getattr(mesh, "geom_radius", getattr(mesh, "radius", 0.0)))
            return union_sphere_midpoint_projector([center], [radius])
        except Exception:
            return None
    if kind == "CenteredBox":
        return _linear_midpoint
    return None


def _point_color_paint_key(vertices, centers, colors, influence_radii):
    verts = np.asarray(vertices, dtype=float).reshape(-1, 3)
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    colors = np.asarray(colors, dtype=float).reshape(-1, 3)
    radii = np.asarray(influence_radii, dtype=float).reshape(-1)
    vmean = tuple(np.round(verts.mean(axis=0), 5).tolist()) if verts.size else ()
    return (
        int(verts.shape[0]),
        vmean,
        tuple(np.round(centers, 5).ravel().tolist()),
        tuple(np.round(colors, 5).ravel().tolist()),
        tuple(np.round(radii, 5).tolist()),
    )


def paint_mesh_by_point_colors(mesh, centers, colors, influence_radii) -> bool:
    """Blend anchor RGB onto mesh vertices using squared falloff inside each sphere.

    Each anchor *i* has center ``centers[i]``, color ``colors[i]``, and influence
    radius ``influence_radii[i]`` (typically atom radius + probe used for the
    solvent surface).  For vertex *v* at distance ``d_i`` from anchor *i*,

        w_i = max(0, 1 - d_i / r_i)^2

    Weights are normalized per vertex; if all weights vanish, the nearest anchor
    wins (hard region assignment).

    Returns False when vertices/centers/colors are unchanged (no CGO invalidate).
    """
    vertices = np.asarray(getattr(mesh, "vertices", []), dtype=float).reshape(-1, 3)
    if vertices.shape[0] == 0:
        return False
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    colors = np.asarray(colors, dtype=float).reshape(-1, 3)
    radii = np.maximum(np.asarray(influence_radii, dtype=float).reshape(-1), 1e-6)
    if centers.shape[0] == 0:
        return False
    key = _point_color_paint_key(vertices, centers, colors, radii)
    if getattr(mesh, "_point_color_paint_key", None) == key:
        return False
    if centers.shape[0] == 1:
        mesh.color = np.broadcast_to(colors[0], (vertices.shape[0], 3)).copy()
        mesh.bypass_colormap = True
        mesh._point_color_paint_key = key
        if hasattr(mesh, "invalidate_cgo_cache"):
            mesh.invalidate_cgo_cache()
        return True
    diff = vertices[:, None, :] - centers[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    t = np.clip(1.0 - dist / radii[None, :], 0.0, 1.0)
    weights = t * t
    sums = weights.sum(axis=1, keepdims=True)
    zero = sums.squeeze(-1) < 1e-12
    if np.any(zero):
        nearest = np.argmin(dist, axis=1)
        weights[zero] = 0.0
        weights[zero, nearest[zero]] = 1.0
        sums = weights.sum(axis=1, keepdims=True)
    rgb = (weights @ colors) / np.maximum(sums, 1e-12)
    mesh.color = rgb
    mesh.bypass_colormap = True
    mesh._point_color_paint_key = key
    if hasattr(mesh, "invalidate_cgo_cache"):
        mesh.invalidate_cgo_cache()
    return True


def paint_surface_mesh_from_anchors(mesh, context=None) -> bool:
    """Re-paint a Surface from ``point_sources`` and stored ``point_colors``."""
    if type(mesh).__name__ != "Surface":
        return False
    point_colors = getattr(mesh, "point_colors", None)
    if not point_colors:
        return False
    from ..points import resolve_xyz
    from .solvent_surface import (
        normalize_point_enabled,
        normalize_point_radii,
        resolve_atom_radii,
    )

    sources = list(getattr(mesh, "point_sources", None) or ())
    flags = normalize_point_enabled(getattr(mesh, "point_enabled", None), len(sources))
    if flags is None:
        flags = [True] * len(sources)
    radii_all = normalize_point_radii(getattr(mesh, "point_radii", None), len(sources))
    centers = []
    colors = []
    active_sources = []
    active_radii = []
    for i, (src, on) in enumerate(zip(sources, flags)):
        if not on or i >= len(point_colors):
            continue
        centers.append(resolve_xyz(src, context))
        colors.append(point_colors[i][:3])
        active_sources.append(src)
        active_radii.append(radii_all[i] if radii_all is not None else None)
    if not centers:
        return False
    atom_radii = resolve_atom_radii(
        active_sources,
        len(active_sources),
        float(getattr(mesh, "atom_radius", 1.5)),
        getattr(mesh, "radius_mode", "uniform"),
        float(getattr(mesh, "vdw_scale", 1.0) or 1.0),
        active_radii,
        context,
    )
    probe = float(getattr(mesh, "probe_radius", 1.4))
    influence = [float(r) + probe for r in atom_radii]
    return paint_mesh_by_point_colors(mesh, centers, colors, influence)


def clear_mesh_field(mesh) -> None:
    mesh.field_id = None
    mesh.field_colormap = None
    mesh.field_colormap_spec = None
    mesh.field_clims = None
    mesh.field_clim_mode = None


def _store_mesh_colormap(mesh, colormap, spec=None) -> None:
    from .colormap_spec import persist_colormap_attrs

    name, stored = persist_colormap_attrs(colormap, spec)
    mesh.field_colormap = name
    mesh.field_colormap_spec = stored


def _paint_arrows_by_field(mesh, grid, field_id, colormap, clims, spec=None) -> bool:
    from .colormap_spec import sampling_colormap

    verts = np.asarray(getattr(mesh, "vertices", []), dtype=float).reshape(-1, 3)
    if verts.shape[0] < 2:
        return False
    pairs = verts.reshape(-1, 2, 3)
    mids = 0.5 * (pairs[:, 0] + pairs[:, 1])
    values = sample_grid(grid, mids, smooth=0.0)
    rgb, used = rgb_from_scalars(values, sampling_colormap(colormap, spec), clims)
    mesh.color = rgb
    mesh.bypass_colormap = True
    mesh.field_id = str(field_id)
    _store_mesh_colormap(mesh, colormap, spec)
    mesh.field_clims = list(used)
    if hasattr(mesh, "invalidate_cgo_cache"):
        mesh.invalidate_cgo_cache()
    return True


def paint_mesh_by_field(
    mesh,
    field_id=None,
    colormap=None,
    clims=None,
    refine=True,
    smooth=None,
    colormap_spec=None,
    clim_mode=None,
) -> bool:
    """Sample ``field_id`` at mesh vertices and store per-vertex RGB.

    Returns False if the field is missing or the mesh has no vertices.
    """
    from .colormap_spec import sampling_colormap

    field_id = field_id if field_id is not None else getattr(mesh, "field_id", None)
    if not field_id:
        return False
    if colormap_spec is None:
        colormap_spec = getattr(mesh, "field_colormap_spec", None)
    grid = resolve_grid_from_session(field_id)
    if grid is None:
        mesh.field_id = str(field_id)
        if colormap is not None or colormap_spec is not None:
            _store_mesh_colormap(mesh, colormap, colormap_spec)
        if clims is not None:
            mesh.field_clims = normalize_clims(clims)
        if clim_mode:
            mesh.field_clim_mode = str(clim_mode)
        return False
    colormap = (
        colormap
        if colormap is not None
        else (getattr(mesh, "field_colormap", None) or DEFAULT_SURFACE_COLORMAP)
    )
    if clims is None:
        clims = getattr(mesh, "field_clims", None)
    clims = normalize_clims(clims)
    if clim_mode:
        mesh.field_clim_mode = str(clim_mode)
    kind = type(mesh).__name__
    if kind == "Arrows":
        return _paint_arrows_by_field(mesh, grid, field_id, colormap, clims, spec=colormap_spec)
    vertices = np.asarray(getattr(mesh, "vertices", []), dtype=float).reshape(-1, 3)
    if vertices.shape[0] == 0:
        return False
    if smooth is None:
        smooth = SURFACE_FIELD_SMOOTH if kind in ("Surface", "Sphere") else 0.0
    values = sample_grid(grid, vertices, smooth=smooth)
    faces = getattr(mesh, "faces", None)
    normals = getattr(mesh, "normals", None)
    if refine and faces is not None and np.asarray(faces).size:
        vertices, normals, faces, values = refine_mesh_for_field(
            vertices,
            normals,
            faces,
            values,
            grid=grid,
            project=_field_projector(mesh),
            smooth=smooth,
        )
        mesh.vertices = vertices
        if normals is not None:
            mesh.normals = normals
        mesh.faces = faces
    field = resolve_field_from_session(field_id)
    if is_rgb_color_field(field):
        rgb = rgb_from_color_field(values, field)
        stops = getattr(field, "color_stops", None) or getattr(grid, "color_stops", None)
        if stops:
            used = [float(stops[0][0]), float(stops[-1][0])]
        else:
            ncat = max(len(getattr(field, "categories", None) or []) - 1, 1)
            used = [0.0, float(ncat)]
        cmap_name = None
        colormap_spec = None
    else:
        rgb, used = rgb_from_scalars(values, sampling_colormap(colormap, colormap_spec), clims)
        cmap_name = True
    mesh.color = rgb
    mesh.bypass_colormap = True
    mesh.field_id = str(field_id)
    if cmap_name:
        _store_mesh_colormap(mesh, colormap, colormap_spec)
    mesh.field_clims = list(used)
    if hasattr(mesh, "invalidate_cgo_cache"):
        mesh.invalidate_cgo_cache()
    return True


def apply_stored_field_color(mesh, data=None, refine=True) -> bool:
    """Copy field attrs from a persistence dict (optional) and paint."""
    if data:
        field_id = data.get("field_id")
        if field_id:
            mesh.field_id = str(field_id)
            cmap = data.get("field_colormap")
            spec = data.get("field_colormap_spec")
            if spec or cmap:
                _store_mesh_colormap(mesh, cmap, spec)
            mesh.field_clims = normalize_clims(data.get("field_clims"))
            mode = data.get("field_clim_mode")
            if mode:
                mesh.field_clim_mode = str(mode)
    return paint_mesh_by_field(mesh, refine=refine)
