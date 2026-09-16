"""Place a map on the crystal lattice so a selection lies inside the brick.

Repeating physics is integer unit-cell translations (and space-group
operators when available), not a free Cartesian drag. If one translated
copy of the stored brick cannot cover the selection, neighboring cells are
tiled by wrapping sample coordinates through those operators.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .identity import GEN_IMPORTED, GEN_PYMOL_MAP

_DEG = np.pi / 180.0
_EPS = 1e-6
_MAX_VOXELS = 192 ** 3
_SHIFT_SEARCH = 2
# PyMOL reports this for objects with no CRYST1; it is not a real unit cell.
_DUMMY_CELL_LENGTH = 1.0


from .crystal_cell import (
    CrystalError,
    _EPS,
    _is_dummy_cell,
    aabb_contains_points,
    attach_crystal_cell,
    best_lattice_shift,
    cell_block_aabb,
    cell_from_aabb,
    covering_map_copy_range,
    field_supports_symmetrize,
    grid_contains_points,
    grid_is_crystal_axis,
    map_corners,
    map_world_aabb,
    ops_from_spacegroup,
    parse_cell_params,
    resample_crystal_around_points,
    wrap_coords_into_brick,
)
from .crystal_coverage import selection_points


def _symmetry_from_cmd(cmd, name):
    if cmd is None or not name:
        return None
    getter = getattr(cmd, "get_symmetry", None)
    if not callable(getter):
        return None
    try:
        return parse_cell_params(getter(name))
    except Exception:
        return None


def cell_for_grid(grid, cmd=None, map_name=None, cell=None, spacegroup=None):
    """Prefer a real unit cell (map or molecule CRYST1); never PyMOL's 1 Å dummy."""
    lo, hi = map_corners(grid)
    parsed = parse_cell_params(cell)
    if parsed is not None and _is_dummy_cell(parsed[0]):
        parsed = None
    if parsed is None and cmd is not None:
        name = str(map_name or getattr(grid, "_name", None) or getattr(grid, "name", "") or "")
        parsed = _symmetry_from_cmd(cmd, name)
        if parsed is not None and _is_dummy_cell(parsed[0]):
            parsed = None
        if parsed is None:
            try:
                others = [str(n) for n in cmd.get_names("objects", 1)]
            except Exception:
                others = []
            for mol in others:
                if mol == name:
                    continue
                cand = _symmetry_from_cmd(cmd, mol)
                if cand is not None and not _is_dummy_cell(cand[0]):
                    parsed = cand
                    break
    if parsed is not None and not _is_dummy_cell(parsed[0]):
        O, sg = parsed
        if spacegroup:
            sg = str(spacegroup)
        return O, np.zeros(3, dtype=float), sg, False
    O, cell_origin = cell_from_aabb(lo, hi)
    return O, cell_origin, str(spacegroup or "P1"), True


def _copy_grid_meta(src, dest):
    for attr in ("id", "_name", "_crystal_cell"):
        val = getattr(src, attr, None)
        if val:
            try:
                setattr(dest, attr, val)
            except Exception:
                pass
    ttt = getattr(src, "A_to", None)
    if ttt is not None:
        dest.A_to = np.array(ttt, copy=True)
    return dest


def _brick_dwarfs_selection(map_lo, map_hi, points, cell_matrix, factor=0.15) -> bool:
    """True when the brick is the unit-cell AABB around a much smaller selection."""
    lo = np.asarray(map_lo, dtype=float).reshape(3)
    hi = np.asarray(map_hi, dtype=float).reshape(3)
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    O = np.asarray(cell_matrix, dtype=float).reshape(3, 3)
    map_span = np.maximum(hi - lo, 1e-8)
    sel = np.maximum(pts.max(axis=0) - pts.min(axis=0), 1e-8)
    corners = []
    for ix in (0, 1):
        for iy in (0, 1):
            for iz in (0, 1):
                corners.append(O @ np.array([ix, iy, iz], dtype=float))
    cell_span = np.ptp(np.asarray(corners, dtype=float), axis=0)
    cell_span = np.maximum(cell_span, 1e-8)
    fills_cell = bool(np.allclose(map_span, cell_span, rtol=float(factor), atol=1.0))
    return fills_cell and bool(np.max(sel) > 5.0) and bool(np.min(map_span) > 2.5 * np.max(sel))


def _clear_crystal_tile_meta(grid):
    for attr in ("_crystal_O", "_crystal_origin", "_crystal_nmin", "_crystal_nmax"):
        try:
            if hasattr(grid, attr):
                delattr(grid, attr)
        except Exception:
            try:
                setattr(grid, attr, None)
            except Exception:
                pass
    return grid


def _shift_map_onto_points(grid, points, cell_matrix):
    """Lattice image if it covers the selection, else Cartesian sit-on-protein."""
    lo, hi = map_world_aabb(grid)
    O = np.asarray(cell_matrix, dtype=float).reshape(3, 3)
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    shift, all_inside = best_lattice_shift(lo, hi, pts, O)
    delta = O @ shift.astype(float)
    dwarfs = _brick_dwarfs_selection(lo, hi, pts, O)
    if (not dwarfs) and (all_inside or aabb_contains_points(lo + delta, hi + delta, pts)):
        moved = _shift_origin(grid, delta)
        return _clear_crystal_tile_meta(moved), shift, True
    map_center = 0.5 * (lo + hi)
    prot_center = pts.mean(axis=0)
    cart = prot_center - map_center
    if aabb_contains_points(lo + cart, hi + cart, pts):
        moved = _shift_origin(grid, cart)
        moved = _clear_crystal_tile_meta(moved)
        return moved, shift, True
    if dwarfs:
        moved = _shift_origin(grid, cart)
        moved = _clear_crystal_tile_meta(moved)
        from .clip import crop_grid_to_aabb

        pad = 2.0
        plo = pts.min(axis=0) - pad
        phi = pts.max(axis=0) + pad
        cropped = crop_grid_to_aabb(moved, [plo.tolist(), phi.tolist()])
        if cropped is not None:
            try:
                cropped._name = getattr(moved, "_name", None) or getattr(moved, "name", None)
            except Exception:
                pass
            return cropped, shift, True
        return moved, shift, True
    return grid, shift, False


def _shift_origin(grid, delta):
    delta = np.asarray(delta, dtype=float).reshape(3)
    A = np.asarray(getattr(grid, "A_to", np.eye(4)), dtype=float).reshape(4, 4)
    if np.allclose(A, np.eye(4), atol=1e-12):
        grid.origin = np.asarray(grid.origin, dtype=float).reshape(3) + delta
    else:
        A = np.array(A, copy=True)
        A[:3, 3] = A[:3, 3] + delta
        grid.A_to = A
    grid.is_loaded = False
    return grid


def _original_cell_span(grid, cell_matrix, cell_origin):
    """How many unit cells the stored brick already spans (per axis)."""
    from .domain import grid_world_corners

    O = np.asarray(cell_matrix, dtype=float).reshape(3, 3)
    c0 = np.asarray(cell_origin, dtype=float).reshape(3)
    corners = grid_world_corners(grid)
    if corners is None:
        return np.ones(3, dtype=float)
    frac = (corners - c0) @ np.linalg.inv(O).T
    return np.maximum(frac.max(axis=0) - frac.min(axis=0), 1e-6)


def _tile_point_counts(grid, n_cells, cell_matrix, cell_origin, max_voxels=_MAX_VOXELS):
    """Keep original Å resolution; shrink uniformly if the tile exceeds the voxel cap."""
    n_cells = np.maximum(np.asarray(n_cells, dtype=int).reshape(3), 1)
    orig_counts = np.maximum(np.asarray(grid.step_counts, dtype=int).reshape(3), 1)
    span = _original_cell_span(grid, cell_matrix, cell_origin)
    per_cell = np.maximum(np.rint(orig_counts.astype(float) / span), 1.0)
    n_pts = np.maximum(np.rint(per_cell * n_cells.astype(float)).astype(int), n_cells) + 1
    n_pts = np.maximum(n_pts, 2)
    prod = int(np.prod(n_pts))
    if prod <= int(max_voxels):
        return n_pts
    scale = (float(max_voxels) / float(prod)) ** (1.0 / 3.0)
    fitted = np.maximum(np.floor(n_pts.astype(float) * scale).astype(int), 2)
    while int(np.prod(fitted)) > int(max_voxels) and int(np.max(fitted)) > 2:
        fitted[int(np.argmax(fitted))] -= 1
        fitted = np.maximum(fitted, 2)
    return fitted


def _cartesian_point_counts(grid, lo, hi, max_voxels=_MAX_VOXELS):
    """Point counts for a world-space XYZ brick covering ``[lo, hi]``."""
    from .domain import grid_pymol_brick_params

    lo = np.asarray(lo, dtype=float).reshape(3)
    hi = np.asarray(hi, dtype=float).reshape(3)
    _origin, step = grid_pymol_brick_params(grid)
    step = np.maximum(np.abs(step), 1e-18)
    span = np.maximum(hi - lo, step)
    n_pts = np.maximum(np.rint(span / step).astype(int), 1) + 1
    n_pts = np.maximum(n_pts, 2)
    prod = int(np.prod(n_pts))
    if prod <= int(max_voxels):
        return n_pts
    scale = (float(max_voxels) / float(prod)) ** (1.0 / 3.0)
    fitted = np.maximum(np.floor(n_pts.astype(float) * scale).astype(int), 2)
    while int(np.prod(fitted)) > int(max_voxels) and int(np.max(fitted)) > 2:
        fitted[int(np.argmax(fitted))] -= 1
        fitted = np.maximum(fitted, 2)
    return fitted


def _expand_grid(grid, nmin, nmax, cell_matrix, ops, cell_origin, cover_lo=None, cover_hi=None):
    """Tile copies of the stored map brick (Cartesian XYZ), not wallpaper the unit cell."""
    from ..util.field_sample import grid_values_3d
    from ..volumetric.GridData import GridData

    O = np.asarray(cell_matrix, dtype=float).reshape(3, 3)
    c0 = np.asarray(cell_origin, dtype=float).reshape(3)
    nmin = np.asarray(nmin, dtype=int).reshape(3)
    nmax = np.asarray(nmax, dtype=int).reshape(3)
    wrap_lo, wrap_hi = map_world_aabb(grid)
    block_lo, block_hi = cell_block_aabb(wrap_lo, wrap_hi, O, nmin, nmax)
    n_pts = _cartesian_point_counts(grid, block_lo, block_hi)
    origin_src = np.asarray(grid.origin, dtype=float).reshape(3)
    step_src = np.maximum(np.abs(np.asarray(grid.step_sizes, dtype=float).reshape(3)), 1e-18)
    values = grid_values_3d(grid)
    shape = np.asarray(values.shape, dtype=int)
    gx = np.linspace(block_lo[0], block_hi[0], int(n_pts[0]))
    gy = np.linspace(block_lo[1], block_hi[1], int(n_pts[1]))
    gz = np.linspace(block_lo[2], block_hi[2], int(n_pts[2]))
    xx, yy, zz = np.meshgrid(gx, gy, gz, indexing="ij")
    sample_xyz = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
    wrapped = wrap_coords_into_brick(sample_xyz, wrap_lo, wrap_hi, O, ops, c0)
    from .domain import grid_world_to_local

    local = grid_world_to_local(grid, wrapped)
    u = (local - origin_src) / step_src
    idx = np.rint(u).astype(int)
    last = shape - 1
    inside = np.all(
        (local >= origin_src - _EPS) & (local <= origin_src + step_src * last + _EPS),
        axis=1,
    )
    valid = inside & np.all((idx >= 0) & (idx <= last), axis=1)
    sampled = np.zeros(idx.shape[0], dtype=float)
    if np.any(valid):
        take = idx[valid]
        sampled[valid] = values[take[:, 0], take[:, 1], take[:, 2]]
    step_world = (block_hi - block_lo) / np.maximum(n_pts - 1, 1)
    name = getattr(grid, "_name", None) or getattr(grid, "name", None) or "field"
    dest = GridData(
        sampled.reshape(-1),
        step_sizes=step_world,
        step_counts=n_pts.astype(int) - 1,
        origin=tuple(float(v) for v in block_lo),
        name=name,
    )
    dest = _copy_grid_meta(grid, dest)
    dest.A_to = np.eye(4)
    return dest


def symmetrize_grid_to_points(
    grid,
    points,
    *,
    cell=None,
    spacegroup=None,
    padding=0.0,
    cmd=None,
    map_name=None,
):
    """Move and/or tile ``grid`` so ``points`` lie inside the brick.

    Returns ``(grid, info)`` where ``info`` has ``moved``, ``copied``, ``shift``,
    and ``brick_cell`` (True when the lattice was inferred from the map box).
    """
    if grid is None:
        raise CrystalError("No map to symmetrize.")
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.size == 0 or not np.all(np.isfinite(pts)):
        raise CrystalError("Selection has no finite coordinates.")
    lo, hi = map_world_aabb(grid)
    O, cell_origin, sg, brick_cell = cell_for_grid(
        grid, cmd=cmd, map_name=map_name, cell=cell, spacegroup=spacegroup
    )
    attach_crystal_cell(grid, O, sg)
    if grid_is_crystal_axis(grid) and not brick_cell:
        dest = resample_crystal_around_points(
            grid, pts, O, padding=padding, cell_origin=cell_origin
        )
        attach_crystal_cell(dest, O, sg)
        return dest, {
            "moved": True,
            "copied": True,
            "resampled": True,
            "shift": [0, 0, 0],
            "brick_cell": brick_cell,
            "spacegroup": sg,
        }
    ops = ops_from_spacegroup(sg)
    already = grid_contains_points(grid, pts)
    dwarfs = _brick_dwarfs_selection(lo, hi, pts, O)
    if already and not dwarfs:
        return _clear_crystal_tile_meta(grid), {
            "moved": False,
            "copied": False,
            "shift": [0, 0, 0],
            "brick_cell": brick_cell,
            "spacegroup": sg,
        }
    moved, shift, placed = _shift_map_onto_points(grid, pts, O)
    if placed:
        attach_crystal_cell(moved, O, sg)
        return moved, {
            "moved": True,
            "copied": False,
            "shift": np.asarray(shift, dtype=int).reshape(3).tolist(),
            "brick_cell": brick_cell,
            "spacegroup": sg,
        }
    pad = float(padding or 0.0)
    nmin, nmax = covering_map_copy_range(lo, hi, pts, O, padding=pad)
    n_cells = np.maximum(np.asarray(nmax, dtype=int) - np.asarray(nmin, dtype=int), 1)
    if int(np.prod(n_cells)) > 8:
        moved, shift, placed = _shift_map_onto_points(grid, pts, O)
        attach_crystal_cell(moved, O, sg)
        return moved, {
            "moved": True,
            "copied": False,
            "shift": np.asarray(shift, dtype=int).reshape(3).tolist(),
            "brick_cell": brick_cell,
            "spacegroup": sg,
        }
    expanded = _expand_grid(grid, nmin, nmax, O, ops, cell_origin)
    expanded = _clear_crystal_tile_meta(expanded)
    attach_crystal_cell(expanded, O, sg)
    return expanded, {
        "moved": True,
        "copied": True,
        "shift": np.asarray(shift, dtype=int).reshape(3).tolist(),
        "brick_cell": brick_cell,
        "spacegroup": sg,
        "nmin": nmin.tolist(),
        "nmax": nmax.tolist(),
    }


def bind_grid_to_field(field, grid) -> None:
    if field is None or grid is None:
        return
    from .domain import Domain

    if type(field).__name__ == "GridData":
        return
    old = getattr(field, "grid_data", None)
    field.grid_data = grid
    try:
        field.domain = Domain.from_grid(grid)
    except Exception:
        pass
    field.dependencies = [grid]
    gen = dict(getattr(field, "generator", None) or {})
    if str(gen.get("type") or GEN_IMPORTED) in (GEN_IMPORTED, GEN_PYMOL_MAP, ""):
        gen["origin"] = [float(v) for v in np.asarray(grid.origin, dtype=float).reshape(3)]
        gen["step_sizes"] = [float(v) for v in np.asarray(grid.step_sizes, dtype=float).reshape(3)]
        gen["step_counts"] = [int(v) for v in np.asarray(grid.step_counts, dtype=int).reshape(3)]
        ttt = getattr(grid, "A_to", None)
        if ttt is not None and not np.allclose(np.asarray(ttt, dtype=float).reshape(4, 4), np.eye(4)):
            gen["A_to"] = [float(v) for v in np.asarray(ttt, dtype=float).reshape(-1)]
        field.generator = gen
    fid = str(getattr(field, "id", "") or "")
    try:
        from ..runtime.session import all_objects, get as session_get

        objects = list(all_objects())
        interned = session_get(fid) if fid else None
        if interned is not None and interned is not field and type(interned).__name__ == "Field":
            interned.grid_data = grid
            try:
                interned.domain = Domain.from_grid(grid)
            except Exception:
                pass
            interned.dependencies = [grid]
            if interned not in objects:
                objects.append(interned)
    except Exception:
        objects = ()
    for obj in objects:
        if obj is field:
            continue
        if type(obj).__name__ == "Field" and fid and str(getattr(obj, "id", "") or "") == fid:
            obj.grid_data = grid
            try:
                obj.domain = Domain.from_grid(grid)
            except Exception:
                pass
            continue
        held = getattr(obj, "grid_data", None)
        if held is old or (fid and str(getattr(obj, "geometry_field_id", "") or "") == fid):
            obj.grid_data = grid


def apply_symmetrize_field(
    cmd,
    field,
    selection=None,
    padding=0.0,
    cell=None,
):
    """Move/tile the field brick. Does not reload PyMOL objects."""
    from .field import as_field, ensure_brick

    wrapped = as_field(field) if type(field).__name__ != "Field" else field
    if wrapped is None:
        wrapped = field
    if not field_supports_symmetrize(wrapped):
        raise CrystalError("Only stored maps can be symmetrized to a selection.")
    grid = ensure_brick(wrapped, cmd=cmd)
    if grid is None:
        raise CrystalError("This field has no sampleable map yet.")
    points = selection_points(cmd, selection)
    map_name = getattr(grid, "_name", None) or getattr(grid, "name", None)
    new_grid, info = symmetrize_grid_to_points(
        grid,
        points,
        cell=cell,
        padding=padding,
        cmd=cmd,
        map_name=map_name,
    )
    bind_grid_to_field(wrapped, new_grid)
    try:
        from ..util.field_sample import PYMOL_MAP_ID_PREFIX, forget_native_grid, remember_field

        fid = str(getattr(wrapped, "id", "") or "")
        forget_native_grid(fid)
        remember_field(wrapped)
        if fid.startswith(PYMOL_MAP_ID_PREFIX):
            from ..util import field_sample as sample_mod

            sample_mod._NATIVE_GRIDS[fid] = new_grid
    except Exception:
        pass
    info = dict(info)
    info["map_name"] = map_name
    return wrapped, new_grid, info
