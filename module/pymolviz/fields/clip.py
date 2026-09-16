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


def _as_axis(axis) -> Optional[int]:
    if axis is None:
        return None
    if isinstance(axis, str):
        key = str(axis).strip().upper()
        mapping = {"X": 0, "Y": 1, "Z": 2, "0": 0, "1": 1, "2": 2}
        return mapping.get(key)
    try:
        value = int(axis)
    except (TypeError, ValueError):
        return None
    if value < 0 or value > 2:
        return None
    return value


def normalize_cardinal_plane(plane) -> Optional[dict]:
    """One axis-aligned half-space: ``{axis, position, hi}``.

    ``hi=False`` keeps the +axis side (plane is the new lo face).
    ``hi=True`` keeps the -axis side (plane is the new hi face).
    """
    if not isinstance(plane, dict):
        return None
    axis = _as_axis(plane.get("axis"))
    if axis is None:
        return None
    if "position" in plane:
        try:
            position = float(plane["position"])
        except (TypeError, ValueError):
            return None
    else:
        try:
            position = float(np.asarray(plane.get("origin"), dtype=float).reshape(3)[axis])
        except (TypeError, ValueError, IndexError):
            return None
    if not np.isfinite(position):
        return None
    return {"axis": int(axis), "position": float(position), "hi": bool(plane.get("hi"))}


def normalize_cardinal_planes(planes) -> list:
    """At most two planes per axis (lo and hi faces), ordered X/Y/Z then lo, hi."""
    if not planes:
        return []
    by_axis = {0: {}, 1: {}, 2: {}}
    for item in planes:
        plane = normalize_cardinal_plane(item)
        if plane is None:
            continue
        by_axis[plane["axis"]][bool(plane["hi"])] = plane
    out = []
    for axis in range(3):
        faces = by_axis[axis]
        if False in faces:
            out.append(faces[False])
        if True in faces:
            out.append(faces[True])
    return out


def default_cardinal_plane(axis, domain_aabb) -> dict:
    """Mid-domain plane that keeps the +axis half (immediately visible clip)."""
    axis = int(_as_axis(axis) if _as_axis(axis) is not None else 0)
    box = normalize_clip_aabb(domain_aabb)
    if box is None:
        position = 0.0
    else:
        position = 0.5 * (float(box[0][axis]) + float(box[1][axis]))
    return {"axis": axis, "position": float(position), "hi": False}


def clamp_cardinal_plane(plane, domain_aabb, eps=_FACE_EPS):
    plane = normalize_cardinal_plane(plane)
    if plane is None:
        return None
    box = normalize_clip_aabb(domain_aabb)
    if box is None:
        return plane
    axis = plane["axis"]
    lo = float(box[0][axis])
    hi = float(box[1][axis])
    gap = float(eps)
    pos = float(plane["position"])
    if plane["hi"]:
        pos = min(max(pos, lo + gap), hi)
    else:
        pos = max(min(pos, hi - gap), lo)
    return {"axis": axis, "position": float(pos), "hi": bool(plane["hi"])}


def clamp_cardinal_planes(planes, domain_aabb, eps=_FACE_EPS, lock=None) -> list:
    """Clamp each face to the domain, then keep lo behind hi on the same axis.

    ``lock`` is ``(axis, hi)`` of a face that should keep its position when the
    pair would otherwise collide; the other face is the stop.
    """
    gap = float(eps)
    lock_axis = None
    lock_hi = None
    if lock is not None:
        try:
            lock_axis = int(lock[0])
            lock_hi = bool(lock[1])
        except (TypeError, ValueError, IndexError):
            lock_axis = None
    faces = {}
    for plane in normalize_cardinal_planes(planes):
        clamped = clamp_cardinal_plane(plane, domain_aabb, eps=eps)
        if clamped is None:
            continue
        faces.setdefault(clamped["axis"], {})[bool(clamped["hi"])] = clamped
    out = []
    for axis in range(3):
        pair = faces.get(axis) or {}
        lo = pair.get(False)
        hi = pair.get(True)
        if lo is not None and hi is not None and lo["position"] > hi["position"] - gap:
            if lock_axis == axis and lock_hi:
                hi = dict(hi)
                hi["position"] = float(lo["position"]) + gap
                clamped = clamp_cardinal_plane(hi, domain_aabb, eps=eps)
                if clamped is not None:
                    hi = clamped
            else:
                lo = dict(lo)
                lo["position"] = float(hi["position"]) - gap
                clamped = clamp_cardinal_plane(lo, domain_aabb, eps=eps)
                if clamped is not None:
                    lo = clamped
            if lo["position"] > hi["position"] - gap:
                hi = dict(hi)
                hi["position"] = float(lo["position"]) + gap
                clamped = clamp_cardinal_plane(hi, domain_aabb, eps=eps)
                if clamped is not None:
                    hi = clamped
        if lo is not None:
            out.append(lo)
        if hi is not None:
            out.append(hi)
    return out


def opposite_cardinal_plane(existing, domain_aabb, eps=_FACE_EPS) -> dict:
    """The other face on the same axis, placed between the current plane and the far bound."""
    existing = normalize_cardinal_plane(existing)
    if existing is None:
        return default_cardinal_plane(0, domain_aabb)
    axis = existing["axis"]
    other_hi = not bool(existing["hi"])
    gap = float(eps)
    pos = float(existing["position"])
    box = normalize_clip_aabb(domain_aabb)
    if box is not None:
        lo = float(box[0][axis])
        hi = float(box[1][axis])
        if other_hi:
            pos = 0.5 * (float(existing["position"]) + hi)
            pos = max(pos, float(existing["position"]) + gap)
        else:
            pos = 0.5 * (lo + float(existing["position"]))
            pos = min(pos, float(existing["position"]) - gap)
    plane = {"axis": axis, "position": float(pos), "hi": other_hi}
    clamped = clamp_cardinal_planes([existing, plane], domain_aabb, eps=eps)
    for item in clamped:
        if item["axis"] == axis and bool(item["hi"]) == other_hi:
            return item
    return plane


def cardinal_planes_to_aabb(planes, domain_aabb, eps=_FACE_EPS) -> Optional[list]:
    """Intersect enabled half-spaces with the field domain. ``None`` if none on."""
    planes = clamp_cardinal_planes(planes, domain_aabb, eps=eps)
    if not planes:
        return None
    box = normalize_clip_aabb(domain_aabb)
    if box is None or not aabb_has_extent(box):
        return None
    lo = np.asarray(box[0], dtype=float).reshape(3)
    hi = np.asarray(box[1], dtype=float).reshape(3)
    gap = float(eps)
    for plane in planes:
        clamped = clamp_cardinal_plane(plane, box, eps=eps)
        if clamped is None:
            continue
        axis = clamped["axis"]
        pos = float(clamped["position"])
        if clamped["hi"]:
            hi[axis] = max(pos, float(lo[axis]) + gap)
        else:
            lo[axis] = min(pos, float(hi[axis]) - gap)
    return normalize_clip_aabb([lo, hi])


def aabb_to_cardinal_planes(aabb, domain_aabb, *, atol=1e-3) -> list:
    """Infer enabled lo/hi faces for each inset axis (up to two per axis)."""
    clip = normalize_clip_aabb(aabb)
    domain = normalize_clip_aabb(domain_aabb)
    if clip is None or domain is None:
        return []
    planes = []
    tol = float(atol)
    for axis in range(3):
        lo_inset = float(clip[0][axis]) - float(domain[0][axis])
        hi_inset = float(domain[1][axis]) - float(clip[1][axis])
        if lo_inset > tol:
            planes.append({
                "axis": axis,
                "position": float(clip[0][axis]),
                "hi": False,
            })
        if hi_inset > tol:
            planes.append({
                "axis": axis,
                "position": float(clip[1][axis]),
                "hi": True,
            })
    return normalize_cardinal_planes(planes)


def cardinal_gizmo_plane(plane, domain_aabb, scale=5.0) -> Optional[dict]:
    plane = normalize_cardinal_plane(plane)
    if plane is None:
        return None
    axis = plane["axis"]
    origin = [0.0, 0.0, 0.0]
    box = normalize_clip_aabb(domain_aabb)
    if box is not None:
        mid = 0.5 * (
            np.asarray(box[0], dtype=float).reshape(3)
            + np.asarray(box[1], dtype=float).reshape(3)
        )
        origin = [float(mid[0]), float(mid[1]), float(mid[2])]
    origin[axis] = float(plane["position"])
    normal = [0.0, 0.0, 0.0]
    normal[axis] = -1.0 if plane["hi"] else 1.0
    return {
        "origin": origin,
        "normal": normal,
        "scale": float(scale),
        "axis": int(axis),
        "hi": bool(plane["hi"]),
        "position": float(plane["position"]),
    }


def cardinal_planes_to_gizmos(planes, domain_aabb, scale=5.0) -> list:
    gizmos = []
    for plane in normalize_cardinal_planes(planes):
        gizmo = cardinal_gizmo_plane(plane, domain_aabb, scale=scale)
        if gizmo is not None:
            gizmos.append(gizmo)
    return gizmos


def apply_origin_to_cardinal_planes(planes, axis, origin, domain_aabb=None, hi=None):
    """Move the enabled plane on ``axis`` (and optional ``hi`` face) to ``origin``."""
    axis = _as_axis(axis)
    current = normalize_cardinal_planes(planes)
    if axis is None:
        return current
    try:
        position = float(np.asarray(origin, dtype=float).reshape(3)[axis])
    except (TypeError, ValueError, IndexError):
        return current
    faces = [plane for plane in current if plane["axis"] == axis]
    if hi is None:
        if len(faces) != 1:
            return current
        hi = bool(faces[0]["hi"])
    hi = bool(hi)
    updated = []
    found = False
    for plane in current:
        if plane["axis"] != axis or bool(plane["hi"]) != hi:
            updated.append(plane)
            continue
        found = True
        moved = dict(plane)
        moved["position"] = position
        updated.append(moved)
    if not found:
        return current
    return clamp_cardinal_planes(updated, domain_aabb, lock=(axis, hi))


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
