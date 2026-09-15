"""Spatial domain of a Field: bounds, padding, and voxel spacing."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

BOUNDS_AROUND_SELECTION = "around_selection"
BOUNDS_OBJECT = "object"
BOUNDS_CUSTOM_BOX = "custom_box"
BOUNDS_MODES = (
    BOUNDS_AROUND_SELECTION,
    BOUNDS_OBJECT,
    BOUNDS_CUSTOM_BOX,
)

DEFAULT_PADDING = 0.0
# Match explicit GAUSS quality 3 (ISO_SPACING[3] = 0.16 Å).
DEFAULT_SPACING = 0.16
# ColorRamp maps need finer voxels than the geometry iso brick so trilinear
# sampling does not smear nearest-atom / categorical colors across cells.
COLOR_FIELD_SPACING_FACTOR = 1.0 / 3.0
_MIN_COLOR_SPACING = 1e-4


def normalize_bounds_mode(mode) -> str:
    text = str(mode or BOUNDS_AROUND_SELECTION).strip().lower().replace("-", "_")
    aliases = {
        "selection": BOUNDS_AROUND_SELECTION,
        "around": BOUNDS_AROUND_SELECTION,
        "atoms": BOUNDS_AROUND_SELECTION,
        "box": BOUNDS_CUSTOM_BOX,
        "custom": BOUNDS_CUSTOM_BOX,
        "aabb": BOUNDS_CUSTOM_BOX,
        "obj": BOUNDS_OBJECT,
    }
    text = aliases.get(text, text)
    if text in BOUNDS_MODES:
        return text
    return BOUNDS_AROUND_SELECTION


def normalize_aabb(aabb) -> Optional[list]:
    if aabb is None:
        return None
    try:
        arr = np.asarray(aabb, dtype=float).reshape(2, 3)
    except (TypeError, ValueError):
        return None
    lo = np.minimum(arr[0], arr[1])
    hi = np.maximum(arr[0], arr[1])
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        return None
    return [
        [float(lo[0]), float(lo[1]), float(lo[2])],
        [float(hi[0]), float(hi[1]), float(hi[2])],
    ]


def _points_array(points) -> Optional[np.ndarray]:
    if points is None:
        return None
    try:
        pts = np.asarray(points, dtype=float)
    except (TypeError, ValueError):
        return None
    if pts.size == 0:
        return None
    try:
        pts = pts.reshape(-1, 3)
    except ValueError:
        return None
    finite = np.all(np.isfinite(pts), axis=1)
    if not np.all(finite):
        pts = pts[finite]
    if pts.size == 0:
        return None
    return pts


def aabb_from_points(points, padding=0.0) -> Optional[list]:
    """Axis-aligned box around ``points``, expanded by ``padding`` on each side."""
    pts = _points_array(points)
    if pts is None:
        return None
    pad = float(padding or 0.0)
    lo = np.min(pts, axis=0) - pad
    hi = np.max(pts, axis=0) + pad
    return [
        [float(lo[0]), float(lo[1]), float(lo[2])],
        [float(hi[0]), float(hi[1]), float(hi[2])],
    ]


def aabb_center_extent(aabb, min_extent=0.0):
    """``((cx, cy, cz), (dx, dy, dz))`` for a ``CenteredBox``, or None."""
    box = normalize_aabb(aabb)
    if box is None:
        return None
    lo = np.asarray(box[0], dtype=float)
    hi = np.asarray(box[1], dtype=float)
    center = 0.5 * (lo + hi)
    extent = np.abs(hi - lo)
    if min_extent:
        extent = np.maximum(extent, float(min_extent))
    return (
        (float(center[0]), float(center[1]), float(center[2])),
        (float(extent[0]), float(extent[1]), float(extent[2])),
    )


def aabb_has_extent(aabb, eps=1e-8) -> bool:
    box = normalize_aabb(aabb)
    if box is None:
        return False
    extent = np.abs(np.asarray(box[1], dtype=float) - np.asarray(box[0], dtype=float))
    return float(np.max(extent)) > float(eps)


def aabbs_close(a, b, atol=1e-3) -> bool:
    left = normalize_aabb(a)
    right = normalize_aabb(b)
    if left is None or right is None:
        return False
    return bool(
        np.allclose(np.asarray(left, dtype=float), np.asarray(right, dtype=float), atol=float(atol), rtol=0.0)
    )


def aabbs_overlap(a, b, eps=1e-8) -> bool:
    left = normalize_aabb(a)
    right = normalize_aabb(b)
    if left is None or right is None:
        return False
    lo = np.maximum(np.asarray(left[0], dtype=float), np.asarray(right[0], dtype=float))
    hi = np.minimum(np.asarray(left[1], dtype=float), np.asarray(right[1], dtype=float))
    return bool(np.all(hi - lo > float(eps)))


def brick_aabb(grid) -> Optional[list]:
    """World-space AABB of a regular brick, or None."""
    if grid is None:
        return None
    try:
        corners = grid_world_corners(grid)
        if corners is None:
            return Domain.from_grid(grid).aabb
        lo = np.min(corners, axis=0)
        hi = np.max(corners, axis=0)
        return normalize_aabb([lo.tolist(), hi.tolist()])
    except Exception:
        return None


def grid_affine(grid):
    """4×4 map from local brick coordinates to world, or identity."""
    raw = getattr(grid, "A_to", None) if grid is not None else None
    if raw is None:
        return np.eye(4)
    try:
        return np.asarray(raw, dtype=float).reshape(4, 4)
    except (TypeError, ValueError):
        return np.eye(4)


def grid_pymol_brick_params(grid):
    """Å origin and spacing for ``load_brick`` (PyMOL volumes ignore sheared TTT)."""
    origin = np.asarray(getattr(grid, "origin", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
    step = np.asarray(getattr(grid, "step_sizes", (1.0, 1.0, 1.0)), dtype=float).reshape(3)
    A = grid_affine(grid)
    if np.allclose(A, np.eye(4), atol=1e-12):
        return origin, np.maximum(np.abs(step), 1e-18)
    O = A[:3, :3]
    t = A[:3, 3]
    lengths = np.maximum(np.linalg.norm(O, axis=0), 1e-18)
    return O @ origin + t, np.maximum(np.abs(step) * lengths, 1e-18)


def grid_has_shear(grid, atol=1e-8) -> bool:
    A = grid_affine(grid)
    return not bool(np.allclose(A, np.eye(4), atol=atol))


def grid_local_corners(grid):
    """Eight corners in local index space (origin + step × counts)."""
    if grid is None:
        return None
    origin = np.asarray(getattr(grid, "origin", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
    step = np.asarray(getattr(grid, "step_sizes", (1.0, 1.0, 1.0)), dtype=float).reshape(3)
    counts = np.asarray(getattr(grid, "step_counts", (1, 1, 1)), dtype=float).reshape(3)
    hi = origin + step * counts
    pts = []
    for x in (origin[0], hi[0]):
        for y in (origin[1], hi[1]):
            for z in (origin[2], hi[2]):
                pts.append([x, y, z])
    return np.asarray(pts, dtype=float)


def grid_world_to_local(grid, xyz):
    """World Cartesian points → local brick coordinates (inverts ``A_to``)."""
    pts = np.asarray(xyz, dtype=float).reshape(-1, 3)
    A = grid_affine(grid)
    if np.allclose(A, np.eye(4), atol=1e-12):
        return pts
    hom = np.hstack([pts, np.ones((pts.shape[0], 1))])
    return (np.linalg.inv(A) @ hom.T).T[:, :3]


def grid_local_to_world(grid, xyz):
    pts = np.asarray(xyz, dtype=float).reshape(-1, 3)
    A = grid_affine(grid)
    if np.allclose(A, np.eye(4), atol=1e-12):
        return pts
    hom = np.hstack([pts, np.ones((pts.shape[0], 1))])
    return (A @ hom.T).T[:, :3]


def grid_world_corners(grid):
    """Eight corners of the brick in world space (applies crystal ``A_to``)."""
    corners = grid_local_corners(grid)
    if corners is None:
        return None
    return grid_local_to_world(grid, corners)


def crystal_world_corners(grid):
    """Eight corners of the lattice parallelepiped, or None if not tiled."""
    raw_O = getattr(grid, "_crystal_O", None) if grid is not None else None
    nmin = getattr(grid, "_crystal_nmin", None)
    nmax = getattr(grid, "_crystal_nmax", None)
    if raw_O is None or nmin is None or nmax is None:
        return None
    try:
        O = np.asarray(raw_O, dtype=float).reshape(3, 3)
        nmin = np.asarray(nmin, dtype=float).reshape(3)
        nmax = np.asarray(nmax, dtype=float).reshape(3)
        c0 = np.asarray(getattr(grid, "_crystal_origin", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
    except (TypeError, ValueError):
        return None
    pts = []
    for x in (nmin[0], nmax[0]):
        for y in (nmin[1], nmax[1]):
            for z in (nmin[2], nmax[2]):
                pts.append(O @ np.array([x, y, z], dtype=float) + c0)
    return np.asarray(pts, dtype=float)


def crystal_world_edges(grid):
    """Twelve edges of the lattice parallelepiped as ``(starts, ends)``."""
    corners = crystal_world_corners(grid)
    if corners is None or len(corners) != 8:
        return None, None
    local = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=float)
    starts, ends = [], []
    for i in range(8):
        for j in range(i + 1, 8):
            if int(np.sum(np.abs(local[i] - local[j]) > 1e-8)) == 1:
                starts.append(corners[i])
                ends.append(corners[j])
    if len(starts) != 12:
        return None, None
    return np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)


def grid_has_crystal_shear(grid, atol=1e-3) -> bool:
    """True when the lattice wire is a non-orthogonal parallelepiped."""
    raw_O = getattr(grid, "_crystal_O", None) if grid is not None else None
    if raw_O is None:
        return grid_has_shear(grid)
    O = np.asarray(raw_O, dtype=float).reshape(3, 3)
    gram = O.T @ O
    off = gram - np.diag(np.diag(gram))
    scale = max(float(np.max(np.abs(gram))), 1e-12)
    return not bool(np.allclose(off, 0.0, atol=atol * scale))


def grid_world_edges(grid):
    """Twelve edges of the (possibly sheared) brick as ``(starts, ends)``."""
    crystal = crystal_world_edges(grid)
    if crystal[0] is not None:
        return crystal
    local = grid_local_corners(grid)
    if local is None:
        return None, None
    world = grid_local_to_world(grid, local)
    starts, ends = [], []
    for i in range(8):
        for j in range(i + 1, 8):
            if int(np.sum(np.abs(local[i] - local[j]) > 1e-8)) == 1:
                starts.append(world[i])
                ends.append(world[j])
    if len(starts) != 12:
        return None, None
    return np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)


def field_display_aabb(field, grid=None) -> Optional[list]:
    """Box for domain preview and crop gizmos. The live brick wins over Domain."""
    if grid is None:
        grid = getattr(field, "grid_data", None) if field is not None else None
    box = brick_aabb(grid)
    if box is not None:
        return box
    domain = getattr(field, "domain", None) if field is not None else None
    if domain is not None:
        try:
            resolved = domain.resolve_aabb()
        except Exception:
            resolved = None
        if resolved is not None:
            return resolved
    return None


def domain_preview_aabbs(domain, centers=None):
    """Outer Domain AABB and unpadded selection hull, sharing ``resolve_aabb``.

    Returns ``(domain_aabb, selection_aabb)``. Either side may be None.
    """
    if domain is None:
        domain = Domain()
    elif not isinstance(domain, Domain):
        domain = Domain.from_dict(domain)
    selection = aabb_from_points(centers, 0.0)
    outer = domain.resolve_aabb(centers)
    return outer, selection


class Domain:
    """Bounds + grid spacing that participate in Field identity."""

    def __init__(
        self,
        bounds_mode=BOUNDS_AROUND_SELECTION,
        padding=DEFAULT_PADDING,
        spacing=DEFAULT_SPACING,
        aabb=None,
        object_name=None,
    ):
        self.bounds_mode = normalize_bounds_mode(bounds_mode)
        self.padding = float(padding if padding is not None else DEFAULT_PADDING)
        self.spacing = float(spacing if spacing is not None else DEFAULT_SPACING)
        if self.spacing <= 0.0:
            self.spacing = DEFAULT_SPACING
        self.aabb = normalize_aabb(aabb)
        self.object_name = str(object_name).strip() if object_name else None

    def with_spacing(self, spacing) -> "Domain":
        """Copy with a different voxel step. Bounds and padding are unchanged."""
        data = self.to_dict()
        data["spacing"] = float(spacing)
        return Domain.from_dict(data)

    def to_dict(self) -> dict:
        data = {
            "bounds_mode": self.bounds_mode,
            "padding": float(self.padding),
            "spacing": float(self.spacing),
        }
        if self.aabb is not None:
            data["aabb"] = [list(self.aabb[0]), list(self.aabb[1])]
        if self.object_name:
            data["object_name"] = self.object_name
        return data

    @classmethod
    def from_dict(cls, data) -> "Domain":
        if isinstance(data, Domain):
            return data
        data = data or {}
        if not isinstance(data, dict):
            return cls()
        return cls(
            bounds_mode=data.get("bounds_mode"),
            padding=data.get("padding", DEFAULT_PADDING),
            spacing=data.get("spacing", DEFAULT_SPACING),
            aabb=data.get("aabb"),
            object_name=data.get("object_name"),
        )

    @classmethod
    def from_grid(cls, grid) -> "Domain":
        corners = grid_world_corners(grid)
        if corners is None:
            origin = np.asarray(getattr(grid, "origin", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
            step = np.asarray(getattr(grid, "step_sizes", (1.0, 1.0, 1.0)), dtype=float).reshape(3)
            counts = np.asarray(getattr(grid, "step_counts", (1, 1, 1)), dtype=float).reshape(3)
            hi = origin + step * counts
            corners = np.array([origin, hi])
        lo = np.min(corners, axis=0)
        hi = np.max(corners, axis=0)
        step = np.asarray(getattr(grid, "step_sizes", (1.0, 1.0, 1.0)), dtype=float).reshape(3)
        spacing = float(np.mean(np.abs(step)))
        if spacing <= 1e-18:
            spacing = DEFAULT_SPACING
        A = getattr(grid, "A_to", None)
        if A is not None and not np.allclose(np.asarray(A, dtype=float).reshape(4, 4), np.eye(4)):
            lengths = np.linalg.norm(np.asarray(A, dtype=float).reshape(4, 4)[:3, :3], axis=0)
            spacing = float(np.mean(np.abs(lengths)) * np.mean(np.abs(step)))
        return cls(
            bounds_mode=BOUNDS_CUSTOM_BOX,
            padding=0.0,
            spacing=spacing,
            aabb=[lo.tolist(), hi.tolist()],
        )

    def resolve_aabb(self, centers=None) -> Optional[list]:
        if self.bounds_mode == BOUNDS_CUSTOM_BOX and self.aabb is not None:
            return normalize_aabb(self.aabb)
        if self.aabb is not None and self.bounds_mode != BOUNDS_AROUND_SELECTION:
            return normalize_aabb(self.aabb)
        return aabb_from_points(centers, self.padding)

    def grid_shape(self, aabb=None, centers=None, extra_pad=0.0):
        """Return ``(origin, step, shape)`` for a cubic lattice, or None."""
        box = normalize_aabb(aabb) if aabb is not None else self.resolve_aabb(centers)
        if box is None:
            return None
        h = float(self.spacing)
        if h <= 1e-12:
            return None
        pad = float(extra_pad or 0.0)
        lo = np.asarray(box[0], dtype=float) - pad
        hi = np.asarray(box[1], dtype=float) + pad
        span = np.maximum(hi - lo, h)
        shape = np.floor(span / h).astype(np.int32) + 1
        shape = np.maximum(shape, 2)
        return lo, h, (int(shape[0]), int(shape[1]), int(shape[2]))


def color_field_domain(domain=None, factor=COLOR_FIELD_SPACING_FACTOR) -> Domain:
    """Copy of ``domain`` with finer spacing for ColorRamp / nearest-color bricks.

    Identity includes this spacing via ``canonical_field_spec`` (``domain.spacing``).
    """
    base = domain if isinstance(domain, Domain) else Domain.from_dict(domain)
    try:
        scale = float(factor if factor is not None else COLOR_FIELD_SPACING_FACTOR)
    except (TypeError, ValueError):
        scale = COLOR_FIELD_SPACING_FACTOR
    if scale <= 0.0:
        scale = COLOR_FIELD_SPACING_FACTOR
    h = max(float(base.spacing) * scale, _MIN_COLOR_SPACING)
    return base.with_spacing(h)
