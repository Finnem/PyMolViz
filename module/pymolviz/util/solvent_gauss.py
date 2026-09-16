"""PyMOL-style Gaussian solvent surface."""

from __future__ import annotations

import math
from functools import lru_cache

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
    _isosurface_from_signed_field,
    _trilinear_sample,
)
from .solvent_params import (
    DEFAULT_ATOM_RADIUS,
    MAX_SAS_CUBES,
    MAX_SAS_VOXELS,
    gauss_spacing,
    vdw_for_element,
)


def _gauss_pad_radius(elements, resolution, b_floor, pad_scales=None) -> float:
    pad = gaussian_pad_radius(elements, resolution, b_floor)
    if pad_scales is None:
        return pad
    scales = np.asarray(pad_scales, dtype=float).reshape(-1)
    if scales.size == 0:
        return pad
    return pad * float(np.max(scales))


def _uses_radius_scaling(radius_scales) -> bool:
    if radius_scales is None:
        return False
    scales = np.asarray(radius_scales, dtype=float).reshape(-1)
    if scales.size == 0:
        return False
    return float(np.max(np.abs(scales - 1.0))) > 1e-6


@lru_cache(maxsize=128)
def gaussian_isosurface_baseline_radius(elem: str) -> float:
    """Default isosurface median radius for one atom (quality 1, default PyMOL map)."""
    key = str(elem or "C").strip().upper()
    spacing = gauss_spacing(1)
    verts, _, _ = _gauss_component(
        np.array([[0.0, 0.0, 0.0]], dtype=float),
        [key],
        spacing,
        DEFAULT_GAUSSIAN_RESOLUTION,
        DEFAULT_GAUSSIAN_B_FLOOR,
        DEFAULT_GAUSSIAN_ISOLEVEL,
    )
    if verts.shape[0] == 0:
        return float(vdw_for_element(key))
    return float(np.median(np.linalg.norm(verts, axis=1)))


def _gauss_radius_scales(elements, atom_radii):
    radii = np.asarray(atom_radii, dtype=float).reshape(-1)
    n = len(elements)
    if n == 0:
        return None
    if radii.size == 1:
        radii = np.repeat(radii, n)
    elif radii.size != n:
        radii = np.resize(radii, n)
    baselines = np.array(
        [gaussian_isosurface_baseline_radius(str(e or "C")) for e in elements],
        dtype=float,
    )
    baselines = np.maximum(baselines, 1e-6)
    return radii / baselines


def _gauss_median_radius_for_peak_fraction(
    elem,
    target_radius,
    spacing,
    resolution,
    b_floor,
    peak_fraction,
):
    """Single-atom reference mesh for calibrating unnormalized isolevel (no vertex warp)."""
    center = np.array([[0.0, 0.0, 0.0]], dtype=float)
    elems = [str(elem or "C")]
    radii = np.array([float(target_radius)], dtype=float)
    scales = _gauss_radius_scales(elems, radii)
    h = float(spacing)
    pad = _gauss_pad_radius(elems, resolution, b_floor, scales) + 2.0 * h
    origin = np.full(3, -float(pad), dtype=float)
    n = int(np.floor(2.0 * float(pad) / h)) + 3
    shape = (max(n, 2), max(n, 2), max(n, 2))
    density = paint_gaussian_density(
        shape, origin, h, center, elems,
        resolution=resolution, b_floor=b_floor, radius_scales=scales,
    )
    peak = float(_trilinear_sample(density, origin, h, center)[0])
    level = peak * float(peak_fraction)
    signed = level - np.asarray(density, dtype=float)
    gx, gy, gz = np.gradient(signed, h, h, h)
    grad = np.stack((gx, gy, gz), axis=-1)
    verts, _, _ = _isosurface_from_signed_field(origin, h, signed, grad)
    if verts.shape[0] == 0:
        return 0.0
    return float(np.median(np.linalg.norm(verts, axis=1)))


@lru_cache(maxsize=256)
def _gauss_peak_fraction(elem: str, target_radius: float, spacing: float) -> float:
    """Fraction of center peak density that yields isosurface median ``target_radius``."""
    key = str(elem or "C").strip().upper()
    target = max(float(target_radius), 1e-6)
    h = float(spacing)
    resolution = DEFAULT_GAUSSIAN_RESOLUTION
    b_floor = DEFAULT_GAUSSIAN_B_FLOOR
    lo, hi = 0.05, 0.995
    med_lo = _gauss_median_radius_for_peak_fraction(key, target, h, resolution, b_floor, lo)
    med_hi = _gauss_median_radius_for_peak_fraction(key, target, h, resolution, b_floor, hi)
    if med_lo <= 0.0 and med_hi <= 0.0:
        return 0.85
    if target <= med_hi:
        return hi
    if target >= med_lo:
        return lo
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        med = _gauss_median_radius_for_peak_fraction(key, target, h, resolution, b_floor, mid)
        if med < target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _gauss_signed_field(
    density,
    isolevel,
    radius_scales,
    origin,
    h,
    centers,
    atom_radii,
    elements,
):
    """PyMOL normalize for default blobs; scaled splats use a calibrated raw density level."""
    if not _uses_radius_scaling(radius_scales):
        return float(isolevel) - normalize_gaussian_map(density)
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    radii = np.asarray(atom_radii, dtype=float).reshape(-1)
    elems = [str(e or "C") for e in list(elements or [])]
    if len(elems) < radii.size:
        elems.extend(["C"] * (int(radii.size) - len(elems)))
    elem = elems[0] if elems else "C"
    target = float(np.mean(radii)) if radii.size else float(gaussian_isosurface_baseline_radius(elem))
    frac = _gauss_peak_fraction(elem, target, float(h))
    peaks = np.asarray(
        _trilinear_sample(np.asarray(density, dtype=float), origin, h, centers),
        dtype=float,
    ).reshape(-1)
    peaks = peaks[np.isfinite(peaks)]
    peak = float(np.max(peaks)) if peaks.size else float(np.max(density))
    return peak * frac - np.asarray(density, dtype=float)


def _gauss_grid(
    centers,
    elements,
    spacing,
    resolution,
    b_floor,
    isolevel,
    radius_scales=None,
    atom_radii=None,
    pad_scales=None,
    _depth=0,
):
    """Gaussian density brick; signed field is isolevel minus (normalized or raw) density."""
    h = float(spacing)
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    if h < 1e-8 or centers.shape[0] == 0:
        return None
    pad = _gauss_pad_radius(elements, resolution, b_floor, pad_scales) + 2.0 * h
    origin = np.min(centers, axis=0) - pad
    hi = np.max(centers, axis=0) + pad
    shape = np.floor((hi - origin) / h).astype(np.int32) + 3
    shape = np.maximum(shape, 2)
    n_vox = int(shape[0]) * int(shape[1]) * int(shape[2])
    if n_vox > MAX_SAS_VOXELS and _depth < 8:
        scale = (float(n_vox) / float(MAX_SAS_VOXELS)) ** (1.0 / 3.0)
        return _gauss_grid(
            centers, elements, h * max(scale, 1.12), resolution, b_floor, isolevel,
            radius_scales=radius_scales, atom_radii=atom_radii, pad_scales=pad_scales,
            _depth=_depth + 1,
        )
    try:
        density = paint_gaussian_density(
            shape, origin, h, centers, elements,
            resolution=resolution, b_floor=b_floor, radius_scales=radius_scales,
        )
        if not np.any(np.abs(density) > 1e-12):
            return None
        signed = _gauss_signed_field(
            density, isolevel, radius_scales, origin, h, centers, atom_radii, elements,
        )
        gx, gy, gz = np.gradient(signed, h, h, h)
        grad = np.stack((gx, gy, gz), axis=-1)
    except MemoryError:
        if _depth < 8:
            return _gauss_grid(
                centers, elements, h * 1.25, resolution, b_floor, isolevel,
                radius_scales=radius_scales, atom_radii=atom_radii, pad_scales=pad_scales,
                _depth=_depth + 1,
            )
        return None
    return origin, h, signed, grad


def _gauss_component(
    centers,
    elements,
    spacing,
    resolution,
    b_floor,
    isolevel,
    radius_scales=None,
    atom_radii=None,
    pad_scales=None,
    _depth=0,
):
    empty = (
        np.zeros((0, 3), dtype=float),
        np.zeros((0, 3), dtype=float),
        np.zeros((0, 3), dtype=int),
    )
    built = _gauss_grid(
        centers, elements, spacing, resolution, b_floor, isolevel,
        radius_scales=radius_scales, atom_radii=atom_radii, pad_scales=pad_scales,
        _depth=_depth,
    )
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
            resolution, b_floor, isolevel,
            radius_scales=radius_scales, atom_radii=atom_radii, pad_scales=pad_scales,
            _depth=_depth + 1,
        )
    return _isosurface_from_signed_field(origin, h, field, grad)


def _gauss_mesh(centers, elements, quality: int, atom_radii=None):
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
    radius_scales = None
    radii_arr = None
    pad_scales = None
    if atom_radii is not None:
        radii_arr = np.asarray(atom_radii, dtype=float).reshape(-1)
        if radii_arr.size == 1:
            radii_arr = np.repeat(radii_arr, n)
        elif radii_arr.size != n:
            radii_arr = np.resize(radii_arr, n)
        if np.allclose(radii_arr, float(DEFAULT_ATOM_RADIUS)):
            radii_arr = None
        else:
            radius_scales = _gauss_radius_scales(elems, radii_arr)
            pad_scales = radius_scales
    return _gauss_component(
        centers, elems, spacing,
        DEFAULT_GAUSSIAN_RESOLUTION, DEFAULT_GAUSSIAN_B_FLOOR, DEFAULT_GAUSSIAN_ISOLEVEL,
        radius_scales=radius_scales, atom_radii=radii_arr, pad_scales=pad_scales,
    )
