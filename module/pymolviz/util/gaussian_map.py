"""PyMOL ``map_new gaussian`` atomic density (SelectorMapGaussian).

Each atom is a Cromer–Mann four-Gaussian plus constant, broadened by the
B-factor. Default ``gaussian_resolution`` is 2.0 (``blur = 2 / resolution``).
PyMOL then normalizes the brick to zero mean / unit variance and contours
``isosurface`` at 1.0.
"""

from __future__ import annotations

import math

import numpy as np

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
) -> np.ndarray:
    """Splat PyMOL Gaussian density onto a cubic lattice. High inside atoms."""
    nx, ny, nz = (int(shape[0]), int(shape[1]), int(shape[2]))
    field = np.zeros((nx, ny, nz), dtype=float)
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    n = int(centers.shape[0])
    if n == 0 or h <= 0.0 or nx < 1 or ny < 1 or nz < 1:
        return field
    origin = np.asarray(origin, dtype=float).reshape(3)
    h = float(h)
    blur = gaussian_blur_factor(resolution)
    floor = float(b_floor)
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
    limits = np.array((nx - 1, ny - 1, nz - 1), dtype=np.int32)
    for i in range(n):
        occup = float(occupancies[i])
        bfact = float(b_factors[i]) + 0.0
        if bfact < floor:
            bfact = floor
        if bfact <= _R_SMALL4 or occup <= _R_SMALL4:
            continue
        elem = elements[i] if i < len(elements) else "C"
        amps, kappas, rcut = atom_gaussian_terms(elem, bfact, occup, blur)
        center = centers[i]
        extent = rcut / blur if blur > 1e-12 else rcut
        lo = np.floor((center - extent - origin) / h).astype(np.int32)
        hi = np.ceil((center + extent - origin) / h).astype(np.int32)
        lo = np.clip(lo, 0, limits)
        hi = np.clip(hi, 0, limits)
        if int(hi[0]) < int(lo[0]) or int(hi[1]) < int(lo[1]) or int(hi[2]) < int(lo[2]):
            continue
        xs = np.arange(int(lo[0]), int(hi[0]) + 1)
        ys = np.arange(int(lo[1]), int(hi[1]) + 1)
        zs = np.arange(int(lo[2]), int(hi[2]) + 1)
        if xs.size == 0 or ys.size == 0 or zs.size == 0:
            continue
        ii, jj, kk = np.meshgrid(xs, ys, zs, indexing="ij")
        xyz = origin + np.stack((ii, jj, kk), axis=-1) * h
        delta = xyz - center
        dist = np.linalg.norm(delta, axis=-1) * blur
        keep = dist < rcut
        if not np.any(keep):
            continue
        dist = np.maximum(dist, _R_SMALL8)
        d2 = dist * dist
        partial = np.zeros(dist.shape, dtype=float)
        for amp, kappa in zip(amps, kappas):
            partial += float(amp) * np.exp(-float(kappa) * d2)
        partial *= blur
        field[ii, jj, kk] += np.where(keep, partial, 0.0)
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
