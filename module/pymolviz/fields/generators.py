"""Sample or wrap a GridData brick from a Field generator recipe."""

from __future__ import annotations

import hashlib

import numpy as np

from .color_blend import (
    blend_values,
    color_blend_stops,
    normalize_color_blend,
    normalize_color_sigma,
    spatial_color_parameter,
)
from .domain import Domain
from .identity import (
    GEN_DISTANCE,
    GEN_GAUSSIAN,
    GEN_IMPORTED,
    GEN_NEAREST_COLOR,
    GEN_NEAREST_PROP,
    GEN_PYMOL_MAP,
    GEN_SIGNED_VDW,
    KIND_CATEGORICAL,
    KIND_SCALAR,
    KIND_VECTOR,
)

_MAX_VOXELS = 9600000
_CATEGORICAL_PROPS = frozenset({"elem", "element", "chain", "name", "resn", "resi"})


def imported_values_digest(values) -> str:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def imported_generator_from_grid(grid, path=None, kind=None) -> dict:
    origin = np.asarray(getattr(grid, "origin", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
    step = np.asarray(getattr(grid, "step_sizes", (1.0, 1.0, 1.0)), dtype=float).reshape(3)
    counts = np.asarray(getattr(grid, "step_counts", (1, 1, 1)), dtype=int).reshape(3)
    gen = {
        "type": GEN_IMPORTED,
        "origin": [float(origin[0]), float(origin[1]), float(origin[2])],
        "step_sizes": [float(step[0]), float(step[1]), float(step[2])],
        "step_counts": [int(counts[0]), int(counts[1]), int(counts[2])],
        "values_digest": imported_values_digest(getattr(grid, "values", ())),
    }
    if path:
        gen["path"] = str(path)
    if kind:
        gen["kind"] = str(kind)
    return gen


def _atom_xyz(atoms):
    rows = []
    for item in atoms or ():
        xyz = item.get("xyz") if isinstance(item, dict) else item
        try:
            rows.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
        except (TypeError, ValueError, IndexError):
            continue
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def _atom_elems(atoms, n):
    out = []
    for item in atoms or ():
        if isinstance(item, dict):
            out.append(str(item.get("elem") or "C"))
        else:
            out.append("C")
    if len(out) < n:
        out.extend(["C"] * (n - len(out)))
    return out[:n]


def _atom_vdw(atoms, n):
    from ..util.solvent_surface import BONDI_VDW, DEFAULT_ATOM_RADIUS

    out = np.full(n, float(DEFAULT_ATOM_RADIUS), dtype=float)
    for i, item in enumerate(list(atoms or ())[:n]):
        if not isinstance(item, dict):
            continue
        if item.get("vdw") is not None:
            out[i] = float(item["vdw"])
            continue
        elem = str(item.get("elem") or "C").strip().upper()
        out[i] = float(BONDI_VDW.get(elem, DEFAULT_ATOM_RADIUS))
    return out


def _atom_rgb(item, default=(0.20, 0.60, 0.90)):
    if not isinstance(item, dict):
        return (float(default[0]), float(default[1]), float(default[2]))
    color = item.get("color")
    if color is None:
        color = item.get("rgb")
    try:
        return (float(color[0]), float(color[1]), float(color[2]))
    except (TypeError, ValueError, IndexError):
        return (float(default[0]), float(default[1]), float(default[2]))


def _atom_property(atoms, key):
    key = str(key or "b_factor")
    aliases = {
        "element": "elem",
        "b": "b_factor",
        "bfactor": "b_factor",
        "q": "occupancy",
        "occ": "occupancy",
    }
    key = aliases.get(key.lower(), key)
    values = []
    for item in atoms or ():
        if not isinstance(item, dict):
            values.append(None)
            continue
        if key in item and item[key] is not None:
            values.append(item[key])
            continue
        if key == "property" and item.get("property") is not None:
            values.append(item["property"])
            continue
        values.append(None)
    return values, key


def _coarsen_shape(shape, h, max_voxels=_MAX_VOXELS, depth=0):
    nx, ny, nz = (int(shape[0]), int(shape[1]), int(shape[2]))
    n_vox = nx * ny * nz
    if n_vox <= int(max_voxels) or depth >= 8:
        return (max(nx, 2), max(ny, 2), max(nz, 2)), float(h)
    scale = (float(n_vox) / float(max_voxels)) ** (1.0 / 3.0)
    h = float(h) * max(scale, 1.12)
    shape = tuple(max(int(np.floor(n / max(scale, 1.12))), 2) for n in (nx, ny, nz))
    return _coarsen_shape(shape, h, max_voxels=max_voxels, depth=depth + 1)


def _grid_positions(origin, h, shape):
    nx, ny, nz = shape
    xs = origin[0] + np.arange(nx, dtype=float) * h
    ys = origin[1] + np.arange(ny, dtype=float) * h
    zs = origin[2] + np.arange(nz, dtype=float) * h
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=1)


def _query_nearest(centers, positions):
    if centers.shape[0] == 0 or positions.shape[0] == 0:
        dist = np.full((positions.shape[0],), np.inf, dtype=float)
        idx = np.zeros((positions.shape[0],), dtype=int)
        return dist, idx
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(centers)
        dist, idx = tree.query(positions, k=1, workers=1)
        return np.asarray(dist, dtype=float), np.asarray(idx, dtype=int)
    except Exception:
        diff = positions[:, None, :] - centers[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        idx = np.argmin(dist2, axis=1)
        dist = np.sqrt(dist2[np.arange(dist2.shape[0]), idx])
        return dist, idx


def _grid_from_values(values, origin, h, name):
    from ..volumetric.GridData import GridData

    values = np.asarray(values, dtype=float)
    counts = [int(n) - 1 for n in values.shape]
    grid = GridData(
        values.reshape(-1),
        step_sizes=(float(h), float(h), float(h)),
        step_counts=counts,
        origin=origin,
        name=name,
    )
    return grid


def brick_from_gaussian(generator, domain, name="field"):
    from ..util.gaussian_map import (
        DEFAULT_GAUSSIAN_B_FLOOR,
        DEFAULT_GAUSSIAN_RESOLUTION,
        gaussian_density_brick,
        gaussian_pad_radius,
    )
    from ..util.solvent_surface import DEFAULT_QUALITY, gauss_spacing

    atoms = generator.get("atoms") or ()
    centers = _atom_xyz(atoms)
    n = int(centers.shape[0])
    if n == 0:
        return None
    elems = _atom_elems(atoms, n)
    quality = int(generator.get("quality") or DEFAULT_QUALITY)
    resol = float(generator.get("resolution") or DEFAULT_GAUSSIAN_RESOLUTION)
    floor = float(generator.get("b_floor") or DEFAULT_GAUSSIAN_B_FLOOR)
    spacing = float(getattr(domain, "spacing", None) or gauss_spacing(quality))
    extra = gaussian_pad_radius(elems, resol, floor) + 2.0 * spacing
    geom = domain.grid_shape(centers=centers, extra_pad=extra)
    if geom is None:
        brick = gaussian_density_brick(
            centers, elems, spacing, resolution=resol, b_floor=floor,
        )
        if brick is None:
            return None
        origin, step, density = brick
        return _grid_from_values(density, origin, step, name)
    origin, h, shape = geom
    from ..util.gaussian_map import paint_gaussian_density, normalize_gaussian_map

    try:
        density = paint_gaussian_density(
            shape, origin, h, centers, elems, resolution=resol, b_floor=floor,
        )
    except MemoryError:
        return None
    if not np.any(np.abs(density) > 1e-12):
        return None
    return _grid_from_values(normalize_gaussian_map(density), origin, h, name)


def brick_from_distance(generator, domain, signed_vdw=False, name="field"):
    atoms = generator.get("atoms") or ()
    centers = _atom_xyz(atoms)
    n = int(centers.shape[0])
    if n == 0:
        return None
    geom = domain.grid_shape(centers=centers)
    if geom is None:
        return None
    origin, h, shape = geom
    shape, h = _coarsen_shape(shape, h)
    positions = _grid_positions(origin, h, shape)
    dist, idx = _query_nearest(centers, positions)
    if signed_vdw:
        radii = _atom_vdw(atoms, n)
        values = (dist - radii[idx]).reshape(shape)
    else:
        values = dist.reshape(shape)
    return _grid_from_values(values, origin, h, name)


def brick_from_nearest_property(generator, domain, name="field"):
    atoms = generator.get("atoms") or ()
    centers = _atom_xyz(atoms)
    n = int(centers.shape[0])
    if n == 0:
        return None, KIND_SCALAR, None
    prop_values, key = _atom_property(atoms, generator.get("property"))
    geom = domain.grid_shape(centers=centers)
    if geom is None:
        return None, KIND_SCALAR, None
    origin, h, shape = geom
    shape, h = _coarsen_shape(shape, h)
    positions = _grid_positions(origin, h, shape)
    _dist, idx = _query_nearest(centers, positions)
    categorical = key in _CATEGORICAL_PROPS or any(
        isinstance(v, str) for v in prop_values if v is not None
    )
    if categorical:
        labels = []
        codes = {}
        mapped = np.zeros(n, dtype=float)
        for i, raw in enumerate(prop_values):
            label = str(raw if raw is not None else "")
            if label not in codes:
                codes[label] = len(labels)
                labels.append(label)
            mapped[i] = float(codes[label])
        values = mapped[idx].reshape(shape)
        grid = _grid_from_values(values, origin, h, name)
        return grid, KIND_CATEGORICAL, labels
    mapped = np.zeros(n, dtype=float)
    for i, raw in enumerate(prop_values):
        try:
            mapped[i] = float(raw)
        except (TypeError, ValueError):
            mapped[i] = 0.0
    values = mapped[idx].reshape(shape)
    return _grid_from_values(values, origin, h, name), KIND_SCALAR, None


def brick_from_nearest_color(generator, domain, name="field"):
    """Gaussian/IDW mix of snapshotted RGB. Brick scalars follow a spatial 1D axis.

    Voxel RGB is ``sum(w_i rgb_i)/sum(w_i)``. The scalar uploaded to PyMOL is the
    same blend of each atom's principal-axis coordinate so ``ramp_new`` can
    interpolate neighboring colors. Not integer category slots.
    """
    atoms = generator.get("atoms") or ()
    centers = _atom_xyz(atoms)
    n = int(centers.shape[0])
    if n == 0:
        return None, KIND_VECTOR, None
    colors = np.asarray(
        [_atom_rgb(item) for item in list(atoms)[:n]], dtype=float
    ).reshape(-1, 3)
    unique = []
    seen = set()
    for rgb in colors:
        key = (round(float(rgb[0]), 6), round(float(rgb[1]), 6), round(float(rgb[2]), 6))
        if key in seen:
            continue
        seen.add(key)
        unique.append([float(rgb[0]), float(rgb[1]), float(rgb[2])])
    kernel = normalize_color_blend(generator.get("blend"))
    sigma = normalize_color_sigma(generator.get("sigma"), domain)
    geom = domain.grid_shape(centers=centers)
    if geom is None:
        return None, KIND_VECTOR, unique
    origin, h, shape = geom
    shape, h = _coarsen_shape(shape, h)
    positions = _grid_positions(origin, h, shape)
    param = spatial_color_parameter(centers)
    scalars = blend_values(positions, centers, param, sigma=sigma, kernel=kernel)
    rgb = blend_values(positions, centers, colors, sigma=sigma, kernel=kernel)
    grid = _grid_from_values(scalars.reshape(shape), origin, h, name)
    grid.rgb_values = np.asarray(rgb, dtype=float).reshape(-1, 3)
    grid.color_stops = color_blend_stops(centers, colors)
    return grid, KIND_VECTOR, unique


def brick_from_pymol_map(generator, cmd=None):
    from ..util.field_sample import grid_from_pymol_map

    name = str(generator.get("map_name") or "")
    if not name:
        return None
    return grid_from_pymol_map(name, cmd=cmd)


def generate_brick(generator, domain, name="field", cmd=None):
    """Return ``(grid, kind, categories)``. ``grid`` may be None."""
    gen = generator or {}
    kind = str(gen.get("type") or GEN_IMPORTED)
    domain = domain if isinstance(domain, Domain) else Domain.from_dict(domain)
    if kind == GEN_GAUSSIAN:
        return brick_from_gaussian(gen, domain, name=name), KIND_SCALAR, None
    if kind == GEN_DISTANCE:
        return brick_from_distance(gen, domain, signed_vdw=False, name=name), KIND_SCALAR, None
    if kind == GEN_SIGNED_VDW:
        return brick_from_distance(gen, domain, signed_vdw=True, name=name), KIND_SCALAR, None
    if kind == GEN_NEAREST_PROP:
        return brick_from_nearest_property(gen, domain, name=name)
    if kind == GEN_NEAREST_COLOR:
        return brick_from_nearest_color(gen, domain, name=name)
    if kind == GEN_PYMOL_MAP:
        return brick_from_pymol_map(gen, cmd=cmd), KIND_SCALAR, None
    return None, KIND_SCALAR, None
