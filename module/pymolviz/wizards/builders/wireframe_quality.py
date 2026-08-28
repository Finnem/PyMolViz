"""Mesh detail presets with guards for large point counts."""

from __future__ import annotations

from typing import NamedTuple

DEFAULT_WIREFRAME_QUALITY = 3
MAX_WIREFRAME_QUALITY = 5
MIN_WIREFRAME_QUALITY = 1

# Total CGO primitive budget across all preview/commit spheres.
MAX_WIREFRAME_CYLINDERS = 40000
MAX_MESH_TRIANGLES = 80000


class WireframeQuality(NamedTuple):
    level: int
    n_lon: int
    n_lat: int
    n_seg: int
    mesh_resolution: int
    subdivisions: int
    frequency: int


WIREFRAME_QUALITY_PRESETS = {
    1: WireframeQuality(1, 4, 2, 6, 8, 1, 2),
    2: WireframeQuality(2, 5, 3, 8, 10, 1, 3),
    3: WireframeQuality(3, 6, 4, 10, 12, 2, 4),
    4: WireframeQuality(4, 8, 5, 12, 14, 2, 6),
    5: WireframeQuality(5, 10, 6, 14, 16, 3, 8),
}


def _clamp_level(level: int) -> int:
    return max(MIN_WIREFRAME_QUALITY, min(MAX_WIREFRAME_QUALITY, int(level)))


def cylinders_per_sphere(quality: WireframeQuality) -> int:
    return 30 * (quality.frequency ** 2)


def faces_per_sphere(quality: WireframeQuality) -> int:
    return 20 * (quality.frequency ** 2)


def max_allowed_wireframe_quality(point_count: int, wireframe: bool = False) -> int:
    """Highest user quality level allowed for the current point count."""
    count = max(int(point_count), 1)
    budget = MAX_WIREFRAME_CYLINDERS if wireframe else MAX_MESH_TRIANGLES
    cost_fn = cylinders_per_sphere if wireframe else faces_per_sphere
    for level in range(MAX_WIREFRAME_QUALITY, MIN_WIREFRAME_QUALITY - 1, -1):
        preset = WIREFRAME_QUALITY_PRESETS[level]
        if cost_fn(preset) * count <= budget:
            return level
    return MIN_WIREFRAME_QUALITY


def effective_wireframe_quality(
    requested: int,
    point_count: int,
    wireframe: bool = False,
) -> WireframeQuality:
    """Map requested quality to a preset, capped by the point-count budget."""
    requested = _clamp_level(requested)
    allowed = max_allowed_wireframe_quality(point_count, wireframe=wireframe)
    level = min(requested, allowed)
    return WIREFRAME_QUALITY_PRESETS[level]
