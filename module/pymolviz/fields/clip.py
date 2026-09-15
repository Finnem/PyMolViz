"""Axis-aligned clip/crop of a regular grid (Field Visual crop)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .domain import aabb_has_extent, normalize_aabb

# Keep-side normals point into the crop box (same convention as mesh clip gizmos).
AABB_FACE_SPECS = (
    (0, False, (1.0, 0.0, 0.0)),
    (0, True, (-1.0, 0.0, 0.0)),
    (1, False, (0.0, 1.0, 0.0)),
    (1, True, (0.0, -1.0, 0.0)),
    (2, False, (0.0, 0.0, 1.0)),
    (2, True, (0.0, 0.0, -1.0)),
)
_FACE_EPS = 1e-4


def normalize_clip_aabb(aabb) -> Optional[list]:
    return normalize_aabb(aabb)


def aabb_corners(aabb):
    """Eight corners of an axis-aligned box, or None."""
    box = normalize_clip_aabb(aabb)
    if box is None:
        return None
    lo = np.asarray(box[0], dtype=float).reshape(3)
    hi = np.asarray(box[1], dtype=float).reshape(3)
    xs, ys, zs = (lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])
    return np.array(
        [[x, y, z] for x in xs for y in ys for z in zs],
        dtype=float,
    )


def aabb_to_axis_planes(aabb, scale=5.0):
    """Six cardinal crop faces as clip-gizmo planes (inward keep normals)."""
    box = normalize_clip_aabb(aabb)
    if box is None or not aabb_has_extent(box):
        return []
    lo = np.asarray(box[0], dtype=float).reshape(3)
    hi = np.asarray(box[1], dtype=float).reshape(3)
    mid = 0.5 * (lo + hi)
    planes = []
    for axis, is_hi, normal in AABB_FACE_SPECS:
        origin = mid.copy()
        origin[axis] = hi[axis] if is_hi else lo[axis]
        planes.append({
            "origin": [float(origin[0]), float(origin[1]), float(origin[2])],
            "normal": [float(normal[0]), float(normal[1]), float(normal[2])],
            "scale": float(scale),
            "axis": int(axis),
            "hi": bool(is_hi),
        })
    return planes


def apply_axis_origin_to_aabb(aabb, axis, is_hi, origin, eps=_FACE_EPS):
    """Move one crop face; the opposite face is a hard stop."""
    box = normalize_clip_aabb(aabb)
    if box is None:
        return None
    lo = np.asarray(box[0], dtype=float).reshape(3)
    hi = np.asarray(box[1], dtype=float).reshape(3)
    axis = int(axis)
    if axis < 0 or axis > 2:
        return box
    value = float(np.asarray(origin, dtype=float).reshape(3)[axis])
    gap = float(eps)
    if is_hi:
        hi[axis] = max(value, float(lo[axis]) + gap)
    else:
        lo[axis] = min(value, float(hi[axis]) - gap)
    return normalize_clip_aabb([lo, hi])


def translate_clip_aabb(aabb, delta):
    box = normalize_clip_aabb(aabb)
    if box is None:
        return None
    shift = np.asarray(delta, dtype=float).reshape(3)
    if not np.any(np.abs(shift) > 1e-12):
        return box
    return normalize_clip_aabb([
        (np.asarray(box[0], dtype=float).reshape(3) + shift).tolist(),
        (np.asarray(box[1], dtype=float).reshape(3) + shift).tolist(),
    ])


def retarget_clip_aabb(clip, old_aabb, new_aabb, *, origin_delta=None, copied=False):
    """Keep a crop on the map after a lattice wrap.

    A crop that was the full old brick (the default seeded cube) is replaced
    by the new brick. A crop that no longer overlaps the map is shifted or
    reseeded so gizmos do not stay on the original cell.
    """
    from .domain import aabbs_close, aabbs_overlap

    clip = normalize_clip_aabb(clip)
    new_box = normalize_clip_aabb(new_aabb)
    if clip is None:
        return None
    if new_box is None:
        return clip
    old_box = normalize_clip_aabb(old_aabb)
    if old_box is not None and aabbs_close(clip, old_box):
        return new_box
    if aabbs_overlap(clip, new_box):
        return clip
    shifted = translate_clip_aabb(clip, origin_delta) if (origin_delta is not None and not copied) else None
    if shifted is not None and aabbs_overlap(shifted, new_box):
        return shifted
    return new_box


def crop_grid_to_aabb(grid, aabb):
    """Return a new GridData covering the overlap of ``grid`` and ``aabb``.

    Step sizes are unchanged. Empty overlap returns None.
    """
    box = normalize_clip_aabb(aabb)
    if grid is None or box is None:
        return grid
    from ..util.field_sample import grid_values_3d
    from ..volumetric.GridData import GridData

    origin = np.asarray(grid.origin, dtype=float).reshape(3)
    step = np.asarray(grid.step_sizes, dtype=float).reshape(3)
    step = np.maximum(step, 1e-18)
    values = grid_values_3d(grid)
    nx, ny, nz = values.shape
    from .domain import grid_world_to_local

    local = grid_world_to_local(grid, aabb_corners(box))
    lo = np.min(local, axis=0)
    hi = np.max(local, axis=0)
    i0 = np.floor((lo - origin) / step).astype(int)
    i1 = np.ceil((hi - origin) / step).astype(int)
    i0 = np.clip(i0, 0, [nx - 1, ny - 1, nz - 1])
    i1 = np.clip(i1, 0, [nx - 1, ny - 1, nz - 1])
    if np.any(i1 <= i0):
        return None
    sl = values[i0[0]:i1[0] + 1, i0[1]:i1[1] + 1, i0[2]:i1[2] + 1]
    new_origin = origin + i0.astype(float) * step
    counts = np.array(sl.shape, dtype=int) - 1
    cropped = GridData(
        sl.reshape(-1),
        step_sizes=step,
        step_counts=counts,
        origin=new_origin,
        name=(getattr(grid, "_name", None) or getattr(grid, "name", None) or "field") + "_crop",
    )
    ttt = getattr(grid, "A_to", None)
    if ttt is not None:
        cropped.A_to = np.array(ttt, copy=True)
    return cropped
