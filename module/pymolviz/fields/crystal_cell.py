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


class CrystalError(ValueError):
    """User-facing failure: missing selection, cell, or an oversize expand."""


def orthogonalization_matrix(a, b, c, alpha=90.0, beta=90.0, gamma=90.0) -> np.ndarray:
    """PDB/IUC orthogonalization: columns are Cartesian lattice vectors a, b, c."""
    a = float(a)
    b = float(b)
    c = float(c)
    al = float(alpha) * _DEG
    be = float(beta) * _DEG
    ga = float(gamma) * _DEG
    va = np.array([a, 0.0, 0.0], dtype=float)
    vb = np.array([b * np.cos(ga), b * np.sin(ga), 0.0], dtype=float)
    cx = c * np.cos(be)
    sin_ga = np.sin(ga)
    if abs(sin_ga) < 1e-12:
        raise CrystalError("Unit-cell gamma is degenerate.")
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / sin_ga
    cz2 = c * c - cx * cx - cy * cy
    if cz2 < -1e-8:
        raise CrystalError("Unit-cell angles do not form a valid parallelepiped.")
    vc = np.array([cx, cy, np.sqrt(max(cz2, 0.0))], dtype=float)
    return np.column_stack((va, vb, vc))


def map_corners(grid) -> Tuple[np.ndarray, np.ndarray]:
    """Local-axis corners of the stored array (not sheared by ``A_to``)."""
    origin = np.asarray(getattr(grid, "origin", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
    step = np.asarray(getattr(grid, "step_sizes", (1.0, 1.0, 1.0)), dtype=float).reshape(3)
    counts = np.asarray(getattr(grid, "step_counts", (1, 1, 1)), dtype=float).reshape(3)
    return origin, origin + step * counts


def map_world_aabb(grid) -> Tuple[np.ndarray, np.ndarray]:
    from .domain import grid_world_corners

    corners = grid_world_corners(grid)
    if corners is None:
        return map_corners(grid)
    return np.min(corners, axis=0), np.max(corners, axis=0)


def _cell_is_orthogonal(cell_matrix, atol=1e-3) -> bool:
    O = np.asarray(cell_matrix, dtype=float).reshape(3, 3)
    gram = O.T @ O
    off = gram - np.diag(np.diag(gram))
    scale = max(float(np.max(np.abs(gram))), 1e-12)
    return bool(np.allclose(off, 0.0, atol=atol * scale))


def grid_contains_points(grid, points, eps=_EPS) -> bool:
    return point_coverage_status(grid, points, eps=eps) == "inside"


def point_coverage_status(grid, points, eps=_EPS) -> str:
    """``inside``, ``partial``, ``outside``, or ``empty`` vs the stored map brick."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.size == 0 or not np.all(np.isfinite(pts)):
        return "empty"
    from .domain import grid_world_to_local

    local = grid_world_to_local(grid, pts)
    lo, hi = map_corners(grid)
    inside = np.all((local >= lo - eps) & (local <= hi + eps), axis=1)
    n_in = int(np.count_nonzero(inside))
    if n_in <= 0:
        return "outside"
    if n_in >= int(inside.shape[0]):
        return "inside"
    return "partial"


def coverage_sketch(map_lo, map_hi, target_lo=None, target_hi=None, axes=None, pad=0.08):
    """Normalized rectangles in 0–1, +second-axis up.

    When ``axes`` is omitted, the plane includes the largest map–target
    offset so a Z miss is not hidden in an XY overlap.
    """
    if axes is None:
        axes = choose_coverage_axes(map_lo, map_hi, target_lo, target_hi)
    ax, ay = (int(axes[0]), int(axes[1]))
    names = (AXIS_NAMES[ax], AXIS_NAMES[ay])

    def xy(lo, hi):
        lo = np.asarray(lo, dtype=float).reshape(3)
        hi = np.asarray(hi, dtype=float).reshape(3)
        return np.array([lo[ax], lo[ay]], dtype=float), np.array([hi[ax], hi[ay]], dtype=float)

    m0, m1 = xy(map_lo, map_hi)
    corners = [m0, m1]
    t0 = t1 = None
    if target_lo is not None and target_hi is not None:
        t0, t1 = xy(target_lo, target_hi)
        corners.extend([t0, t1])
    stack = np.vstack(corners)
    gmin = stack.min(axis=0)
    gmax = stack.max(axis=0)
    span = np.maximum(gmax - gmin, 1e-6)
    gmin = gmin - float(pad) * span
    gmax = gmax + float(pad) * span
    span = np.maximum(gmax - gmin, 1e-6)

    def norm_rect(lo, hi):
        p0 = (lo - gmin) / span
        p1 = (hi - gmin) / span
        x0 = float(min(p0[0], p1[0]))
        x1 = float(max(p0[0], p1[0]))
        y_lo = float(min(p0[1], p1[1]))
        y_hi = float(max(p0[1], p1[1]))
        return {
            "x": x0,
            "y": 1.0 - y_hi,
            "w": max(x1 - x0, 0.02),
            "h": max(y_hi - y_lo, 0.02),
        }

    out = {
        "map": norm_rect(m0, m1),
        "target": None,
        "axes": (ax, ay),
        "axis_names": names,
    }
    if t0 is not None:
        out["target"] = norm_rect(t0, t1)
    return out


AXIS_NAMES = ("X", "Y", "Z")


def choose_coverage_axes(map_lo, map_hi, target_lo=None, target_hi=None):
    """Return ``(horizontal, vertical)`` world axes; vertical is the largest offset."""
    if target_lo is None or target_hi is None:
        return (0, 1)
    map_lo = np.asarray(map_lo, dtype=float).reshape(3)
    map_hi = np.asarray(map_hi, dtype=float).reshape(3)
    tgt_lo = np.asarray(target_lo, dtype=float).reshape(3)
    tgt_hi = np.asarray(target_hi, dtype=float).reshape(3)
    sep = np.maximum(map_lo - tgt_hi, tgt_lo - map_hi)
    sep = np.maximum(sep, 0.0)
    offset = np.abs(0.5 * (tgt_lo + tgt_hi) - 0.5 * (map_lo + map_hi))
    score = np.where(sep > 1e-8, sep, offset)
    if float(np.max(score)) < 1e-8:
        return (0, 1)
    ax_disp = int(np.argmax(score))
    rest = [i for i in range(3) if i != ax_disp]
    span = np.maximum(map_hi - map_lo, 1e-12) + np.maximum(tgt_hi - tgt_lo, 0.0)
    ax_other = max(rest, key=lambda i: (float(score[i]), float(span[i]), -i))
    return (ax_other, ax_disp)


def cell_from_aabb(lo, hi) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(lo, dtype=float).reshape(3)
    hi = np.asarray(hi, dtype=float).reshape(3)
    extent = np.maximum(hi - lo, 1e-8)
    return np.diag(extent), lo


def parse_cell_params(cell) -> Optional[Tuple[np.ndarray, str]]:
    """Return ``(orthogonalization_matrix, spacegroup)`` or None."""
    if cell is None:
        return None
    if isinstance(cell, dict):
        lengths = cell.get("abc") or cell.get("lengths")
        if lengths is not None:
            a, b, c = (float(v) for v in np.asarray(lengths, dtype=float).reshape(3))
        else:
            a = float(cell.get("a", 0.0) or 0.0)
            b = float(cell.get("b", 0.0) or 0.0)
            c = float(cell.get("c", 0.0) or 0.0)
        angles = cell.get("angles") or (
            cell.get("alpha", 90.0),
            cell.get("beta", 90.0),
            cell.get("gamma", 90.0),
        )
        alpha, beta, gamma = (float(v) for v in np.asarray(angles, dtype=float).reshape(3))
        sg = str(cell.get("spacegroup") or cell.get("sg") or "P1")
        if min(a, b, c) < 1e-6:
            return None
        return orthogonalization_matrix(a, b, c, alpha, beta, gamma), sg
    if isinstance(cell, (list, tuple)) and len(cell) >= 6 and not isinstance(cell[0], (list, tuple, np.ndarray)):
        a, b, c, alpha, beta, gamma = (float(v) for v in cell[:6])
        if min(a, b, c) < 1e-6:
            return None
        sg = str(cell[6] or "P1") if len(cell) >= 7 else "P1"
        return orthogonalization_matrix(a, b, c, alpha, beta, gamma), sg
    arr = np.asarray(cell, dtype=float)
    if arr.ndim == 2 and arr.shape == (3, 3):
        return arr.astype(float), "P1"
    flat = arr.reshape(-1)
    if flat.size >= 6:
        a, b, c, alpha, beta, gamma = (float(v) for v in flat[:6])
        if min(a, b, c) < 1e-6:
            return None
        return orthogonalization_matrix(a, b, c, alpha, beta, gamma), "P1"
    return None


def ops_from_spacegroup(spacegroup) -> list:
    """Fractional operators ``(R, t)``. Identity if the space group is unknown."""
    identity = [(np.eye(3), np.zeros(3))]
    name = str(spacegroup or "P1").strip() or "P1"
    compact = name.replace(" ", "").upper()
    if compact in ("P1", "1", "C1"):
        return identity
    try:
        import gemmi

        sg = gemmi.SpaceGroup(name)
        ops = []
        for op in sg.operations():
            t = np.asarray(op.apply_to_xyz([0.0, 0.0, 0.0]), dtype=float).reshape(3)
            cols = []
            for i in range(3):
                e = [0.0, 0.0, 0.0]
                e[i] = 1.0
                cols.append(np.asarray(op.apply_to_xyz(e), dtype=float).reshape(3) - t)
            ops.append((np.column_stack(cols), t))
        return ops or identity
    except Exception:
        return identity


def _is_dummy_cell(cell_matrix) -> bool:
    """True for PyMOL's placeholder 1×1×1 Å P1 cell."""
    O = np.asarray(cell_matrix, dtype=float).reshape(3, 3)
    lengths = np.linalg.norm(O, axis=0)
    return bool(np.allclose(lengths, _DUMMY_CELL_LENGTH, atol=0.2))


def lattice_shift_indices(map_lo, map_hi, points, cell_matrix) -> np.ndarray:
    """Integer cell indices that move the map so it covers as much of the selection as possible."""
    n, _all_inside = best_lattice_shift(map_lo, map_hi, points, cell_matrix)
    return n


def best_lattice_shift(map_lo, map_hi, points, cell_matrix, search=_SHIFT_SEARCH):
    """Return ``(n, all_inside)`` for the lattice translation covering the most atoms."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    lo = np.asarray(map_lo, dtype=float).reshape(3)
    hi = np.asarray(map_hi, dtype=float).reshape(3)
    O = np.asarray(cell_matrix, dtype=float).reshape(3, 3)
    inv = np.linalg.inv(O)
    center = 0.5 * (lo + hi)
    n0 = np.rint(inv @ (pts.mean(axis=0) - center)).astype(int)
    rel = (pts - center) @ inv.T
    seeds = {tuple(int(v) for v in n0.tolist())}
    for row in np.rint(rel).astype(int):
        seeds.add(tuple(int(v) for v in row.tolist()))
    best_n = n0
    best_count = -1
    best_norm = 10 ** 9
    n_pts = int(pts.shape[0])
    radius = max(int(search), 0)
    seen = set()
    for seed in seeds:
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                for dk in range(-radius, radius + 1):
                    tup = (seed[0] + di, seed[1] + dj, seed[2] + dk)
                    if tup in seen:
                        continue
                    seen.add(tup)
                    n = np.array(tup, dtype=int)
                    delta = O @ n.astype(float)
                    inside = np.all(
                        (pts >= lo + delta - _EPS) & (pts <= hi + delta + _EPS),
                        axis=1,
                    )
                    count = int(np.sum(inside))
                    norm = int(np.sum(np.abs(n)))
                    if count > best_count or (count == best_count and norm < best_norm):
                        best_n = n
                        best_count = count
                        best_norm = norm
    return best_n, best_count == n_pts


def aabb_contains_points(lo, hi, points, eps=_EPS) -> bool:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    lo = np.asarray(lo, dtype=float).reshape(3)
    hi = np.asarray(hi, dtype=float).reshape(3)
    return bool(np.all((pts >= lo - eps) & (pts <= hi + eps)))


def _fold_into_aabb(xyz, lo, hi):
    """P1 wrap of an axis-aligned brick (map box used as a P1 cell)."""
    span = np.maximum(np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float), 1e-12)
    lo = np.asarray(lo, dtype=float).reshape(3)
    frac = (np.asarray(xyz, dtype=float).reshape(-1, 3) - lo) / span
    return lo + (frac - np.floor(frac)) * span


def fold_into_unit_cell(xyz, cell_matrix, cell_origin):
    """Wrap Cartesian points into the home unit cell (parallelepiped faces)."""
    O = np.asarray(cell_matrix, dtype=float).reshape(3, 3)
    origin = np.asarray(cell_origin, dtype=float).reshape(3)
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    if _is_dummy_cell(O):
        return xyz
    inv = np.linalg.inv(O)
    frac = (xyz - origin) @ inv.T
    frac = frac - np.floor(frac)
    return frac @ O.T + origin


def grid_is_crystal_axis(grid) -> bool:
    """True when array indices run along a, b, c — not orthogonal XYZ Å."""
    return bool(getattr(grid, "_crystal_axis_grid", False))


def attach_crystal_axis_frame(grid, cell_matrix):
    """Index (i,j,k) → Cartesian ``O @ (i/nu, j/nv, k/nw)`` (full-cell CCP4)."""
    from ..util.field_sample import grid_values_3d

    O = np.asarray(cell_matrix, dtype=float).reshape(3, 3)
    values = grid_values_3d(grid)
    nu, nv, nw = (int(v) for v in values.shape)
    scale = np.diag([1.0 / max(nu, 1), 1.0 / max(nv, 1), 1.0 / max(nw, 1)])
    A = np.eye(4)
    A[:3, :3] = O @ scale
    grid.origin = (0.0, 0.0, 0.0)
    grid.step_sizes = (1.0, 1.0, 1.0)
    grid.step_counts = (nu - 1, nv - 1, nw - 1)
    grid.A_to = A
    grid._crystal_axis_grid = True
    grid._crystal_O = O.tolist()
    attach_crystal_cell(grid, O)
    return grid


def _trilinear_periodic(values, frac):
    """``frac`` in [0, 1)³ indexes a full unit-cell grid along a, b, c."""
    values = np.asarray(values, dtype=float)
    nu, nv, nw = values.shape
    n = np.array([nu, nv, nw], dtype=float)
    u = np.asarray(frac, dtype=float).reshape(-1, 3) * n
    i0 = np.floor(u).astype(int)
    t = u - i0
    i0 = np.mod(i0, np.array([nu, nv, nw], dtype=int))
    i1 = (i0 + 1) % np.array([nu, nv, nw], dtype=int)
    c000 = values[i0[:, 0], i0[:, 1], i0[:, 2]]
    c100 = values[i1[:, 0], i0[:, 1], i0[:, 2]]
    c010 = values[i0[:, 0], i1[:, 1], i0[:, 2]]
    c110 = values[i1[:, 0], i1[:, 1], i0[:, 2]]
    c001 = values[i0[:, 0], i0[:, 1], i1[:, 2]]
    c101 = values[i1[:, 0], i0[:, 1], i1[:, 2]]
    c011 = values[i0[:, 0], i1[:, 1], i1[:, 2]]
    c111 = values[i1[:, 0], i1[:, 1], i1[:, 2]]
    t0, t1, t2 = t[:, 0], t[:, 1], t[:, 2]
    c00 = c000 * (1.0 - t0) + c100 * t0
    c10 = c010 * (1.0 - t0) + c110 * t0
    c01 = c001 * (1.0 - t0) + c101 * t0
    c11 = c011 * (1.0 - t0) + c111 * t0
    c0 = c00 * (1.0 - t1) + c10 * t1
    c1 = c01 * (1.0 - t1) + c11 * t1
    return c0 * (1.0 - t2) + c1 * t2


def interpolate_unit_cell(values, cell_matrix, xyz, cell_origin=None):
    """Look up Cartesian Å points in a full-cell crystal-axis grid (periodic)."""
    O = np.asarray(cell_matrix, dtype=float).reshape(3, 3)
    origin = np.zeros(3, dtype=float) if cell_origin is None else np.asarray(cell_origin, dtype=float).reshape(3)
    frac = (np.asarray(xyz, dtype=float).reshape(-1, 3) - origin) @ np.linalg.inv(O).T
    frac = frac - np.floor(frac + 1e-15)
    frac = np.clip(frac, 0.0, 1.0 - 1e-15)
    return _trilinear_periodic(values, frac)


def resample_crystal_around_points(grid, points, cell_matrix, padding=4.0, cell_origin=None):
    """Cartesian XYZ brick around ``points``, sampled via unit-cell periodicity."""
    from ..util.field_sample import grid_values_3d
    from ..volumetric.GridData import GridData

    O = np.asarray(cell_matrix, dtype=float).reshape(3, 3)
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    pad = max(float(padding or 0.0), 2.0)
    lo = pts.min(axis=0) - pad
    hi = pts.max(axis=0) + pad
    values = grid_values_3d(grid)
    from .crystal_symmetrize import _cartesian_point_counts, _copy_grid_meta

    n_pts = _cartesian_point_counts(grid, lo, hi)
    gx = np.linspace(lo[0], hi[0], int(n_pts[0]))
    gy = np.linspace(lo[1], hi[1], int(n_pts[1]))
    gz = np.linspace(lo[2], hi[2], int(n_pts[2]))
    xx, yy, zz = np.meshgrid(gx, gy, gz, indexing="ij")
    sample_xyz = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
    sampled = interpolate_unit_cell(values, O, sample_xyz, cell_origin=cell_origin)
    step_world = (hi - lo) / np.maximum(n_pts - 1, 1)
    name = getattr(grid, "_name", None) or getattr(grid, "name", None) or "field"
    dest = GridData(
        sampled.reshape(-1),
        step_sizes=step_world,
        step_counts=n_pts.astype(int) - 1,
        origin=tuple(float(v) for v in lo),
        name=name,
    )
    dest = _copy_grid_meta(grid, dest)
    dest.A_to = np.eye(4)
    dest._crystal_axis_grid = False
    dest._crystal_lookup = True
    return dest


def wrap_coords_into_brick(xyz, lo, hi, cell_matrix, ops, cell_origin):
    """Pull lattice images onto the stored map; leave gaps outside the brick.

    Only integer translations of the Cartesian map brick (``O @ n``).
    Space-group operators are not applied: the stored array is a local box,
    not an asymmetric unit to wallpaper through the cell.
    """
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    lo = np.asarray(lo, dtype=float).reshape(3)
    hi = np.asarray(hi, dtype=float).reshape(3)
    O = np.asarray(cell_matrix, dtype=float).reshape(3, 3)
    if _is_dummy_cell(O):
        return _fold_into_aabb(xyz, lo, hi)
    inv = np.linalg.inv(O)
    chosen = np.array(xyz, copy=True)
    filled = np.all((xyz >= lo - _EPS) & (xyz < hi), axis=1)
    n = np.floor((xyz - lo) @ inv.T + 1e-12)
    cand = xyz - n @ O.T
    take = np.all((cand >= lo - _EPS) & (cand <= hi + _EPS), axis=1) & ~filled
    if np.any(take):
        chosen[take] = cand[take]
    return chosen


def covering_cell_range(points, cell_matrix, cell_origin, padding=0.0):
    """Integer cell index range ``[nmin, nmax)`` covering ``points``."""
    O = np.asarray(cell_matrix, dtype=float).reshape(3, 3)
    origin = np.asarray(cell_origin, dtype=float).reshape(3)
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    inv = np.linalg.inv(O)
    frac = (pts - origin) @ inv.T
    lengths = np.maximum(np.linalg.norm(O, axis=0), 1e-8)
    pad_frac = float(padding or 0.0) / lengths
    nmin = np.floor(frac.min(axis=0) - pad_frac - 1e-9).astype(int)
    nmax = np.ceil(frac.max(axis=0) + pad_frac + 1e-9).astype(int)
    nmax = np.maximum(nmax, nmin + 1)
    return nmin, nmax


def covering_map_copy_range(map_lo, map_hi, points, cell_matrix, padding=0.0):
    """Integer ``n`` such that map AABB + ``O @ n`` covers ``points``."""
    O = np.asarray(cell_matrix, dtype=float).reshape(3, 3)
    lo = np.asarray(map_lo, dtype=float).reshape(3)
    hi = np.asarray(map_hi, dtype=float).reshape(3)
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    inv = np.linalg.inv(O)
    center = 0.5 * (lo + hi)
    n = np.rint((pts - center) @ inv.T).astype(int)
    lengths = np.maximum(np.linalg.norm(O, axis=0), 1e-8)
    extra = np.ceil(float(padding or 0.0) / lengths).astype(int)
    nmin = n.min(axis=0) - extra
    nmax = n.max(axis=0) + extra + 1
    nmax = np.maximum(nmax, nmin + 1)
    return nmin, nmax


def cell_block_aabb(lo, hi, cell_matrix, nmin, nmax):
    """Cartesian AABB of whole-cell copies of the brick for ``n`` in ``[nmin, nmax)``."""
    O = np.asarray(cell_matrix, dtype=float).reshape(3, 3)
    lo = np.asarray(lo, dtype=float).reshape(3)
    hi = np.asarray(hi, dtype=float).reshape(3)
    nmin = np.asarray(nmin, dtype=int).reshape(3)
    nmax = np.asarray(nmax, dtype=int).reshape(3)
    los = []
    his = []
    for ix in (int(nmin[0]), int(nmax[0]) - 1):
        for iy in (int(nmin[1]), int(nmax[1]) - 1):
            for iz in (int(nmin[2]), int(nmax[2]) - 1):
                delta = O @ np.array([ix, iy, iz], dtype=float)
                los.append(lo + delta)
                his.append(hi + delta)
    return np.min(np.stack(los), axis=0), np.max(np.stack(his), axis=0)


def cell_lengths_angles(cell_matrix):
    O = np.asarray(cell_matrix, dtype=float).reshape(3, 3)
    va, vb, vc = O[:, 0], O[:, 1], O[:, 2]
    a = float(np.linalg.norm(va))
    b = float(np.linalg.norm(vb))
    c = float(np.linalg.norm(vc))

    def _angle(u, v):
        den = max(float(np.linalg.norm(u) * np.linalg.norm(v)), 1e-18)
        return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / den, -1.0, 1.0))))

    return a, b, c, _angle(vb, vc), _angle(va, vc), _angle(va, vb)


def attach_crystal_cell(grid, cell_matrix, spacegroup="P1"):
    if grid is None:
        return
    try:
        a, b, c, alpha, beta, gamma = cell_lengths_angles(cell_matrix)
        grid._crystal_cell = [a, b, c, alpha, beta, gamma, str(spacegroup or "P1")]
    except Exception:
        pass


def field_supports_symmetrize(field) -> bool:
    """True for stored maps (imported / native), not atom-generated recipes."""
    if field is None:
        return False
    if type(field).__name__ == "GridData":
        return True
    gen = getattr(field, "generator", None) or {}
    kind = str(gen.get("type") or GEN_IMPORTED)
    if kind in (GEN_IMPORTED, GEN_PYMOL_MAP):
        return True
    if getattr(field, "grid_data", None) is not None and kind in ("", "imported"):
        return True
    return False

