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


def selection_points(cmd, selection=None) -> np.ndarray:
    if cmd is None:
        raise CrystalError("No PyMOL session to read a selection from.")
    candidates = []
    if selection:
        candidates.append(str(selection))
    candidates.extend(("sele", "(sele)"))
    try:
        names = [str(n) for n in cmd.get_names("objects", 1)]
    except Exception:
        names = []
    for name in names:
        try:
            typ = str(cmd.get_type(name) or "")
        except Exception:
            typ = ""
        if "molecule" in typ.lower():
            candidates.append(name)
    candidates.append("all")
    seen = set()
    for expr in candidates:
        if expr in seen:
            continue
        seen.add(expr)
        try:
            n = int(cmd.count_atoms(expr) or 0)
        except Exception:
            continue
        if n <= 0:
            continue
        try:
            coords = cmd.get_coords(expr)
        except Exception:
            coords = None
        pts = np.asarray(coords if coords is not None else [], dtype=float)
        if pts.size == 0:
            continue
        return pts.reshape(-1, 3)
    raise CrystalError("Select some atoms first (the named selection \"sele\").")


_SKIP_EXTEND_PREFIXES = ("_pmv_",)
_SKIP_EXTEND_NAMES = frozenset({"pmv_camera_center"})
_SKIP_EXTEND_TYPE_TOKENS = (
    "map", "volume", "mesh", "cgo", "ramp", "surface", "callback", "gadget", "group",
)
_COVERAGE_SORT = {"outside": 0, "partial": 1, "inside": 2, "empty": 3}


def _is_extend_skip_name(name: str) -> bool:
    text = str(name or "")
    if text in _SKIP_EXTEND_NAMES:
        return True
    return any(text.startswith(prefix) for prefix in _SKIP_EXTEND_PREFIXES)


def _object_is_coordinate_target(cmd, name: str) -> bool:
    try:
        typ = str(cmd.get_type(name) or "").lower()
    except Exception:
        typ = ""
    if any(token in typ for token in _SKIP_EXTEND_TYPE_TOKENS):
        return False
    return True


def _coords_for_name(cmd, name):
    try:
        n = int(cmd.count_atoms(name) or 0)
    except Exception:
        n = 0
    if n <= 0:
        return None
    try:
        coords = cmd.get_coords(name)
    except Exception:
        coords = None
    pts = np.asarray(coords if coords is not None else [], dtype=float)
    if pts.size == 0 or not np.all(np.isfinite(pts)):
        return None
    return pts.reshape(-1, 3)


def iter_extend_target_names(cmd) -> List[Tuple[str, str]]:
    """``(kind, name)`` for molecule objects and named selections with atoms."""
    found = []
    seen = set()
    try:
        objects = [str(n) for n in cmd.get_names("objects") or []]
    except Exception:
        objects = []
    for name in objects:
        if not name or name in seen or _is_extend_skip_name(name):
            continue
        if not _object_is_coordinate_target(cmd, name):
            continue
        seen.add(name)
        found.append(("object", name))
    try:
        selections = [str(n) for n in cmd.get_names("selections") or []]
    except Exception:
        selections = []
    for name in selections:
        if not name or name in seen or _is_extend_skip_name(name):
            continue
        seen.add(name)
        found.append(("selection", name))
    return found


def extend_target_rows(cmd, grid, skip_names=()) -> List[dict]:
    """Coverage of each object/selection against the current map brick."""
    skip = {str(n) for n in (skip_names or ()) if n}
    rows = []
    if cmd is None or grid is None:
        return rows
    for kind, name in iter_extend_target_names(cmd):
        if name in skip:
            continue
        pts = _coords_for_name(cmd, name)
        if pts is None:
            continue
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        rows.append({
            "kind": kind,
            "name": name,
            "status": point_coverage_status(grid, pts),
            "n_atoms": int(pts.shape[0]),
            "lo": lo.tolist(),
            "hi": hi.tolist(),
        })
    rows.sort(
        key=lambda row: (
            0 if row["kind"] == "object" else 1,
            _COVERAGE_SORT.get(row["status"], 9),
            str(row["name"]).lower(),
        )
    )
    return rows


def default_extend_target_name(rows) -> Optional[str]:
    rows = list(rows or [])
    if not rows:
        return None
    for row in rows:
        if row.get("name") in ("sele", "(sele)") and row.get("status") != "empty":
            return str(row["name"])
    for row in rows:
        if row.get("status") in ("outside", "partial"):
            return str(row["name"])
    return str(rows[0]["name"])


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
