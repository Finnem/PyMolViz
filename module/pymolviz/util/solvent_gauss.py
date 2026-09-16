"""PyMOL-style Gaussian solvent surface."""

from __future__ import annotations

import math

import numpy as np

from .gaussian_map import (
    DEFAULT_GAUSSIAN_B_FLOOR,
    DEFAULT_GAUSSIAN_ISOLEVEL,
    DEFAULT_GAUSSIAN_RESOLUTION,
    gaussian_pad_radius,
    normalize_gaussian_map,
    paint_gaussian_density,
)
from .solvent_mesh import (
    _cubes_with_sign_change,
    _face_normals,
    _isosurface_from_signed_field,
    _orient_faces_to_vertex_normals,
)
from .solvent_params import DEFAULT_QUALITY, MAX_SAS_CUBES, MAX_SAS_VOXELS, gauss_spacing

def _gauss_pad_radius(elements, resolution, b_floor) -> float:
    return gaussian_pad_radius(elements, resolution, b_floor)
def _gauss_grid(centers, elements, spacing, resolution, b_floor, isolevel, _depth=0):
    """Normalized PyMOL Gaussian brick; signed field is isolevel - density."""
    h = float(spacing)
    if h < 1e-8 or centers.shape[0] == 0:
        return None
    pad = _gauss_pad_radius(elements, resolution, b_floor) + 2.0 * h
    origin = np.min(centers, axis=0) - pad
    hi = np.max(centers, axis=0) + pad
    shape = np.floor((hi - origin) / h).astype(np.int32) + 3
    shape = np.maximum(shape, 2)
    n_vox = int(shape[0]) * int(shape[1]) * int(shape[2])
    if n_vox > MAX_SAS_VOXELS and _depth < 8:
        scale = (float(n_vox) / float(MAX_SAS_VOXELS)) ** (1.0 / 3.0)
        return _gauss_grid(
            centers, elements, h * max(scale, 1.12), resolution, b_floor, isolevel,
            _depth=_depth + 1,
        )
    try:
        density = paint_gaussian_density(
            shape, origin, h, centers, elements,
            resolution=resolution, b_floor=b_floor,
        )
        if not np.any(np.abs(density) > 1e-12):
            return None
        signed = float(isolevel) - normalize_gaussian_map(density)
        gx, gy, gz = np.gradient(signed, h, h, h)
        grad = np.stack((gx, gy, gz), axis=-1)
    except MemoryError:
        if _depth < 8:
            return _gauss_grid(
                centers, elements, h * 1.25, resolution, b_floor, isolevel,
                _depth=_depth + 1,
            )
        return None
    return origin, h, signed, grad
def _gauss_component(centers, elements, spacing, resolution, b_floor, isolevel, _depth=0):
    empty = (
        np.zeros((0, 3), dtype=float),
        np.zeros((0, 3), dtype=float),
        np.zeros((0, 3), dtype=int),
    )
    built = _gauss_grid(centers, elements, spacing, resolution, b_floor, isolevel, _depth=_depth)
    if built is None:
        return empty
    origin, h, field, grad = built
    cubes = _cubes_with_sign_change(field)
    if cubes.shape[0] == 0:
        return empty
    if cubes.shape[0] > MAX_SAS_CUBES and _depth < 6:
        return _gauss_component(
            centers, elements,
            h * max(math.sqrt(float(cubes.shape[0]) / float(MAX_SAS_CUBES)), 1.15),
            resolution, b_floor, isolevel, _depth=_depth + 1,
        )
    return _isosurface_from_signed_field(origin, h, field, grad)
def _gauss_mesh(centers, elements, quality: int):
    empty = (
        np.zeros((0, 3), dtype=float),
        np.zeros((0, 3), dtype=float),
        np.zeros((0, 3), dtype=int),
    )
    n = int(np.asarray(centers).reshape(-1, 3).shape[0])
    if n == 0:
        return empty
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    elems = [str(e or "C") for e in list(elements or [])]
    if len(elems) < n:
        elems.extend(["C"] * (n - len(elems)))
    elif len(elems) > n:
        elems = elems[:n]
    spacing = gauss_spacing(quality)
    return _gauss_component(
        centers, elems, spacing,
        DEFAULT_GAUSSIAN_RESOLUTION, DEFAULT_GAUSSIAN_B_FLOOR, DEFAULT_GAUSSIAN_ISOLEVEL,
    )
