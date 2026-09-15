"""Spatially blended atom RGB for nearest-atom color fields.

PyMOL ``ramp_new`` is 1D scalar → color. Category indices 0..N with a stepped
ramp print Voronoi stairs (the -0.5..N+0.5 chip bar). This module blends
nearby point RGBs with a Gaussian or IDW kernel and assigns each atom a 1D
spatial parameter so the ColorRamp interpolates neighboring colors, not
unrelated palette slots.
"""

from __future__ import annotations

import numpy as np

COLOR_BLEND_KERNEL = "gaussian"
COLOR_BLEND_SIGMA = 0.6
COLOR_BLEND_KERNELS = ("gaussian", "idw")
_MIN_SIGMA = 1e-4
_KNN = 24


def normalize_color_blend(kernel) -> str:
    text = str(kernel or COLOR_BLEND_KERNEL).strip().lower().replace("-", "_")
    aliases = {
        "idw": "idw",
        "inverse_distance": "idw",
        "shepard": "idw",
        "gauss": COLOR_BLEND_KERNEL,
        "gaussian": COLOR_BLEND_KERNEL,
    }
    return aliases.get(text, COLOR_BLEND_KERNEL)


def normalize_color_sigma(sigma, domain=None) -> float:
    if sigma is None:
        try:
            h = float(getattr(domain, "spacing", 0.0) or 0.0)
        except (TypeError, ValueError):
            h = 0.0
        if h > 0.0:
            return max(COLOR_BLEND_SIGMA, 2.0 * h)
        return COLOR_BLEND_SIGMA
    try:
        value = float(sigma)
    except (TypeError, ValueError):
        return COLOR_BLEND_SIGMA
    return max(value, _MIN_SIGMA)


def spatial_color_parameter(centers) -> np.ndarray:
    """1D coordinate along the principal axis of atom positions."""
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    n = int(centers.shape[0])
    if n == 0:
        return np.zeros(0, dtype=float)
    if n == 1:
        return np.zeros(1, dtype=float)
    mean = centers.mean(axis=0)
    offset = centers - mean
    try:
        _u, _s, vt = np.linalg.svd(offset, full_matrices=False)
        param = offset @ vt[0]
        if float(np.ptp(param)) > 1e-8:
            return np.asarray(param, dtype=float)
    except Exception:
        pass
    for dim in range(3):
        param = centers[:, dim]
        if float(np.ptp(param)) > 1e-8:
            return np.asarray(param, dtype=float)
    return np.zeros(n, dtype=float)


def color_blend_stops(centers, colors):
    """``[(t, [r, g, b]), ...]`` sorted along the spatial axis."""
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    colors = np.asarray(colors, dtype=float).reshape(-1, 3)
    n = min(int(centers.shape[0]), int(colors.shape[0]))
    if n == 0:
        return [(0.0, [0.2, 0.6, 0.9]), (1.0, [0.2, 0.6, 0.9])]
    param = spatial_color_parameter(centers[:n])
    rows = sorted(
        (
            (float(param[i]), [float(colors[i, 0]), float(colors[i, 1]), float(colors[i, 2])])
            for i in range(n)
        ),
        key=lambda item: item[0],
    )
    stops = []
    for t_i, rgb in rows:
        if stops and abs(stops[-1][0] - t_i) < 1e-8:
            continue
        stops.append((t_i, rgb))
    if len(stops) == 1:
        t0, rgb0 = stops[0]
        stops = [(t0, rgb0), (t0 + 1.0, list(rgb0))]
    return stops


def blend_values(positions, centers, values, *, sigma, kernel="gaussian") -> np.ndarray:
    """Weighted mix of ``values`` at ``positions``. Nearest color if all weights vanish."""
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    values = np.asarray(values, dtype=float)
    n_pos = int(positions.shape[0])
    n_src = int(centers.shape[0])
    if values.ndim == 1:
        values = values.reshape(-1)
    if n_pos == 0:
        extra = values.shape[1:] if values.ndim > 1 else ()
        return np.zeros((0,) + extra, dtype=float)
    if n_src == 0:
        extra = values.shape[1:] if values.ndim > 1 else ()
        return np.zeros((n_pos,) + extra, dtype=float)
    if n_src == 1:
        return np.broadcast_to(values[0], (n_pos,) + values.shape[1:]).copy()

    k = min(n_src, _KNN)
    dist, idx = _query_knn(centers, positions, k)
    sigma = max(float(sigma), _MIN_SIGMA)
    kind = normalize_color_blend(kernel)
    if kind == "idw":
        weights = 1.0 / (np.power(dist, 2.0) + 1e-8)
    else:
        weights = np.exp(-0.5 * (dist / sigma) ** 2)
    gathered = values[idx]
    if gathered.ndim == 2:
        mixed = (weights * gathered).sum(axis=1)
        denom = weights.sum(axis=1)
        out = mixed / np.maximum(denom, 1e-30)
        empty = denom < 1e-12
        if np.any(empty):
            out[empty] = values[idx[empty, 0]]
        return out
    w = weights[..., None]
    mixed = (w * gathered).sum(axis=1)
    denom = weights.sum(axis=1)[:, None]
    out = mixed / np.maximum(denom, 1e-30)
    empty = denom.reshape(-1) < 1e-12
    if np.any(empty):
        out[empty] = values[idx[empty, 0]]
    return out


def _query_knn(centers, positions, k):
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(centers)
        dist, idx = tree.query(positions, k=k, workers=1)
    except Exception:
        diff = positions[:, None, :] - centers[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        if k >= int(centers.shape[0]):
            idx = np.argsort(dist2, axis=1)
        else:
            part = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
            order = np.argsort(np.take_along_axis(dist2, part, axis=1), axis=1)
            idx = np.take_along_axis(part, order, axis=1)
        dist = np.sqrt(np.take_along_axis(dist2, idx, axis=1))
    dist = np.asarray(dist, dtype=float)
    idx = np.asarray(idx, dtype=int)
    if dist.ndim == 1:
        dist = dist.reshape(-1, 1)
        idx = idx.reshape(-1, 1)
    return dist, idx
