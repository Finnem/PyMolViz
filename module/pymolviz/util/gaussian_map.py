"""PyMOL ``map_new gaussian`` atomic density (SelectorMapGaussian).

Each atom is a Cromer–Mann four-Gaussian plus constant, broadened by the
B-factor. Default ``gaussian_resolution`` is 2.0 (``blur = 2 / resolution``).
PyMOL then normalizes the brick to zero mean / unit variance and contours
``isosurface`` at 1.0.
"""

from __future__ import annotations

import math

import numpy as np

from .array_backend import array_module, as_numpy

# Default PyMOL settings (SettingInfo.h / creating.py).
DEFAULT_GAUSSIAN_RESOLUTION = 2.0
DEFAULT_GAUSSIAN_B_FLOOR = 20.0
DEFAULT_GAUSSIAN_ISOLEVEL = 1.0
_ELIM = 7.0
_R_SMALL4 = 1e-4
_R_SMALL8 = 1e-8
_D_SMALL10 = 1e-10

# Cromer–Mann coefficients packed as PyMOL AtomSF: a0,b0, a1,b1, a2,b2, a3,b3, a4,b4.
# Unknown elements use carbon, matching SelectorMapGaussian.
_SCATTERING = {
    "H": (0.493002, 10.510900, 0.322912, 26.125700, 0.140191, 3.142360,
          0.040810, 57.799698, 0.003038, 0.0),
    "C": (2.310000, 20.843899, 1.020000, 10.207500, 1.588600, 0.568700,
          0.865000, 51.651199, 0.215600, 0.0),
    "O": (3.048500, 13.277100, 2.286800, 5.701100, 1.546300, 0.323900,
          0.867000, 32.908897, 0.250800, 0.0),
    "N": (12.212600, 0.005700, 3.132200, 9.893300, 2.012500, 28.997499,
          1.166300, 0.582600, -11.528999, 0.0),
    "S": (6.905300, 1.467900, 5.203400, 22.215099, 1.437900, 0.253600,
          1.586300, 56.172001, 0.866900, 0.0),
    "CL": (11.460400, 0.010400, 7.196400, 1.166200, 6.255600, 18.519400,
           1.645500, 47.778400, 0.866900, 0.0),
    "BR": (17.178900, 2.172300, 5.235800, 16.579599, 5.637700, 0.260900,
           3.985100, 41.432800, 2.955700, 0.0),
    "I": (20.147200, 4.347000, 18.994900, 0.381400, 7.513800, 27.765999,
          2.273500, 66.877602, 4.071200, 0.0),
    "F": (3.539200, 10.282499, 2.641200, 4.294400, 1.517000, 0.261500,
          1.024300, 26.147600, 0.277600, 0.0),
    "K": (8.218599, 12.794900, 7.439800, 0.774800, 1.051900, 213.186996,
          0.865900, 41.684097, 1.422800, 0.0),
    "MG": (5.420400, 2.827500, 2.173500, 79.261101, 1.226900, 0.380800,
           2.307300, 7.193700, 0.858400, 0.0),
    "NA": (4.762600, 3.285000, 3.173600, 8.842199, 1.267400, 0.313600,
           1.112800, 129.423996, 0.676000, 0.0),
    "P": (6.434500, 1.906700, 4.179100, 27.157000, 1.780000, 0.526000,
          1.490800, 68.164497, 1.114900, 0.0),
    "ZN": (14.074300, 3.265500, 7.031800, 0.233300, 5.162500, 10.316299,
           2.410000, 58.709702, 1.304100, 0.0),
    "CA": (8.626600, 10.442100, 7.387300, 0.659900, 1.589900, 85.748398,
           1.021100, 178.436996, 1.375100, 0.0),
    "CU": (13.337999, 3.582800, 7.167600, 0.247000, 5.615800, 11.396600,
           1.673500, 64.812599, 1.191000, 0.0),
    "FE": (11.769500, 4.761100, 7.357300, 0.307200, 3.522200, 15.353500,
           2.304500, 76.880501, 1.036900, 0.0),
    "SE": (17.000599, 2.409800, 5.819600, 0.272600, 3.973100, 15.237200,
           4.354300, 43.816299, 2.840900, 0.0),
}
_SCATTERING["LP"] = _SCATTERING["C"]


def gaussian_blur_factor(resolution: float) -> float:
    resol = float(resolution)
    if resol < 1.0:
        resol = 1.0
    return 2.0 / resol


def scattering_for_element(elem) -> tuple:
    key = str(elem or "C").strip().upper()
    return _SCATTERING.get(key, _SCATTERING["C"])


def atom_gaussian_terms(elem, b_factor, occupancy, blur_factor):
    """Return ``(amps, kappas, rcut)`` for one atom.

    ``rcut`` is the stored PyMOL cutoff compared against ``r * blur_factor``.
    """
    src = scattering_for_element(elem)
    occup = float(occupancy)
    bfact = float(b_factor)
    amps = np.empty(5, dtype=float)
    kappas = np.empty(5, dtype=float)
    rcut2 = 0.0
    for i in range(5):
        sfa = src[2 * i]
        sfb = src[2 * i + 1]
        denom = sfb + bfact
        amp = occup * sfa * (math.sqrt(4.0 * math.pi / denom) ** 3)
        kappa = 4.0 * math.pi * math.pi / denom
        amps[i] = amp
        kappas[i] = kappa
        rcut2 = max(
            rcut2,
            (_ELIM + math.log(max(abs(amp), _D_SMALL10))) / kappa,
        )
    rcut = math.sqrt(rcut2) / float(blur_factor)
    return amps, kappas, rcut


def gaussian_density_at_origin(elem="C", b_factor=DEFAULT_GAUSSIAN_B_FLOOR, occupancy=1.0,
                               resolution=DEFAULT_GAUSSIAN_RESOLUTION) -> float:
    blur = gaussian_blur_factor(resolution)
    amps, kappas, _rcut = atom_gaussian_terms(elem, b_factor, occupancy, blur)
    d2 = _R_SMALL8 * _R_SMALL8
    return float(np.sum(amps * np.exp(-kappas * d2)) * blur)


def paint_gaussian_density(
    shape,
    origin,
    h,
    centers,
    elements,
    b_factors=None,
    occupancies=None,
    resolution=DEFAULT_GAUSSIAN_RESOLUTION,
    b_floor=DEFAULT_GAUSSIAN_B_FLOOR,
    backend=None,
) -> np.ndarray:
    """Splat PyMOL Gaussian density onto a cubic lattice. High inside atoms.

    ``backend`` is ``None`` (auto CuPy/NumPy), ``"numpy"``, or ``"cupy"``.
    The returned field is always a NumPy array.
    """
    nx, ny, nz = (int(shape[0]), int(shape[1]), int(shape[2]))
    empty = np.zeros((nx, ny, nz), dtype=float)
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    n = int(centers.shape[0])
    if n == 0 or h <= 0.0 or nx < 1 or ny < 1 or nz < 1:
        return empty
    xp = array_module(prefer=backend)
    try:
        return as_numpy(
            _paint_gaussian_density_xp(
                xp, (nx, ny, nz), origin, h, centers, elements,
                b_factors, occupancies, resolution, b_floor,
            )
        )
    except Exception:
        if xp is np:
            raise
        return as_numpy(
            _paint_gaussian_density_xp(
                np, (nx, ny, nz), origin, h, centers, elements,
                b_factors, occupancies, resolution, b_floor,
            )
        )


def _paint_gaussian_density_xp(
    xp,
    shape,
    origin,
    h,
    centers,
    elements,
    b_factors,
    occupancies,
    resolution,
    b_floor,
):
    nx, ny, nz = shape
    field = xp.zeros((nx, ny, nz), dtype=float)
    origin = np.asarray(origin, dtype=float).reshape(3)
    h = float(h)
    blur = gaussian_blur_factor(resolution)
    blur2 = blur * blur
    floor = float(b_floor)
    n = int(centers.shape[0])
    if b_factors is None:
        b_factors = np.full(n, floor, dtype=float)
    else:
        b_factors = np.asarray(b_factors, dtype=float).reshape(-1)
        if b_factors.size == 1:
            b_factors = np.repeat(b_factors, n)
        elif b_factors.size != n:
            b_factors = np.resize(b_factors, n)
    if occupancies is None:
        occupancies = np.ones(n, dtype=float)
    else:
        occupancies = np.asarray(occupancies, dtype=float).reshape(-1)
        if occupancies.size == 1:
            occupancies = np.repeat(occupancies, n)
        elif occupancies.size != n:
            occupancies = np.resize(occupancies, n)
    limx, limy, limz = nx - 1, ny - 1, nz - 1
    inv_h = 1.0 / h
    term_cache = {}
    for i in range(n):
        occup = float(occupancies[i])
        bfact = float(b_factors[i])
        if bfact < floor:
            bfact = floor
        if bfact <= _R_SMALL4 or occup <= _R_SMALL4:
            continue
        elem = elements[i] if i < len(elements) else "C"
        cache_key = (str(elem or "C").strip().upper(), bfact, occup)
        packed = term_cache.get(cache_key)
        if packed is None:
            packed = atom_gaussian_terms(elem, bfact, occup, blur)
            term_cache[cache_key] = packed
        amps, kappas, rcut = packed
        center = centers[i]
        extent = rcut / blur if blur > 1e-12 else rcut
        lo0 = int(np.clip(math.floor((center[0] - extent - origin[0]) * inv_h), 0, limx))
        lo1 = int(np.clip(math.floor((center[1] - extent - origin[1]) * inv_h), 0, limy))
        lo2 = int(np.clip(math.floor((center[2] - extent - origin[2]) * inv_h), 0, limz))
        hi0 = int(np.clip(math.ceil((center[0] + extent - origin[0]) * inv_h), 0, limx))
        hi1 = int(np.clip(math.ceil((center[1] + extent - origin[1]) * inv_h), 0, limy))
        hi2 = int(np.clip(math.ceil((center[2] + extent - origin[2]) * inv_h), 0, limz))
        if hi0 < lo0 or hi1 < lo1 or hi2 < lo2:
            continue
        xs = origin[0] + xp.arange(lo0, hi0 + 1, dtype=float) * h - float(center[0])
        ys = origin[1] + xp.arange(lo1, hi1 + 1, dtype=float) * h - float(center[1])
        zs = origin[2] + xp.arange(lo2, hi2 + 1, dtype=float) * h - float(center[2])
        xs2 = xs * xs
        ys2 = ys * ys
        zs2 = zs * zs
        rcut2 = float(rcut) * float(rcut)
        dist2 = blur2 * (
            xs2[:, None, None] + ys2[None, :, None] + zs2[None, None, :]
        )
        keep = dist2 < rcut2
        if not bool(xp.any(keep)):
            continue
        partial = xp.zeros(dist2.shape, dtype=float)
        for amp, kappa in zip(amps, kappas):
            kblur = float(kappa) * blur2
            gx = xp.exp(-kblur * xs2)
            gy = xp.exp(-kblur * ys2)
            gz = xp.exp(-kblur * zs2)
            partial += float(amp) * gx[:, None, None] * gy[None, :, None] * gz[None, None, :]
        field[lo0:hi0 + 1, lo1:hi1 + 1, lo2:hi2 + 1] += xp.where(keep, partial * blur, 0.0)
    return field


def normalize_gaussian_map(density: np.ndarray) -> np.ndarray:
    """PyMOL ``normalize=true``: zero mean, unit sample standard deviation."""
    values = np.asarray(density, dtype=float)
    n = int(values.size)
    if n < 2:
        return np.array(values, copy=True)
    mean = float(values.mean())
    stdev = float(values.std(ddof=1))
    if stdev < _R_SMALL8:
        stdev = _R_SMALL8
    return (values - mean) / stdev


def gaussian_pad_radius(elements, resolution=DEFAULT_GAUSSIAN_RESOLUTION, b_floor=DEFAULT_GAUSSIAN_B_FLOOR) -> float:
    """Half-extent of the Cromer–Mann splat window, plus a small safety margin."""
    blur = gaussian_blur_factor(resolution)
    extent = max(float(resolution), 1.0)
    seen = set()
    for elem in elements or ():
        key = str(elem or "C").strip().upper()
        if key in seen:
            continue
        seen.add(key)
        _amps, _kappas, rcut = atom_gaussian_terms(elem, b_floor, 1.0, blur)
        real = rcut / blur if blur > 1e-12 else rcut
        if real > extent:
            extent = real
    return float(extent)


def gaussian_density_brick(
    centers,
    elements,
    spacing,
    *,
    resolution=DEFAULT_GAUSSIAN_RESOLUTION,
    b_floor=DEFAULT_GAUSSIAN_B_FLOOR,
    max_voxels=9600000,
    _depth=0,
):
    """Normalized PyMOL Gaussian map as ``(origin, step, density)``, or None.

    ``density`` is zero-mean / unit-stdev, matching ``map_new gaussian`` so an
    isolevel of 1.0 is the usual blob surface.
    """
    pts = np.asarray(centers, dtype=float).reshape(-1, 3)
    n = int(pts.shape[0])
    h = float(spacing)
    if n == 0 or h < 1e-8:
        return None
    elems = [str(e or "C") for e in list(elements or [])]
    if len(elems) < n:
        elems.extend(["C"] * (n - len(elems)))
    elif len(elems) > n:
        elems = elems[:n]
    pad = gaussian_pad_radius(elems, resolution, b_floor) + 2.0 * h
    origin = np.min(pts, axis=0) - pad
    hi = np.max(pts, axis=0) + pad
    shape = np.floor((hi - origin) / h).astype(np.int32) + 3
    shape = np.maximum(shape, 2)
    n_vox = int(shape[0]) * int(shape[1]) * int(shape[2])
    if n_vox > int(max_voxels) and _depth < 8:
        scale = (float(n_vox) / float(max_voxels)) ** (1.0 / 3.0)
        return gaussian_density_brick(
            pts, elems, h * max(scale, 1.12),
            resolution=resolution, b_floor=b_floor, max_voxels=max_voxels,
            _depth=_depth + 1,
        )
    try:
        density = paint_gaussian_density(
            shape, origin, h, pts, elems,
            resolution=resolution, b_floor=b_floor,
        )
    except MemoryError:
        if _depth < 8:
            return gaussian_density_brick(
                pts, elems, h * 1.25,
                resolution=resolution, b_floor=b_floor, max_voxels=max_voxels,
                _depth=_depth + 1,
            )
        return None
    if not np.any(np.abs(density) > 1e-12):
        return None
    return origin, h, normalize_gaussian_map(density)
