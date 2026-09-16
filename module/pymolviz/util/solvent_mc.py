"""Marching-cubes solvent-excluded surface via EDT."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import distance_transform_edt

from .marching_cubes import march_cubes_mesh
from .solvent_connolly import _overlap_components
from .solvent_mesh import (
    _cubes_with_sign_change,
    _face_normals,
    _isosurface_from_signed_field,
    _orient_faces_to_vertex_normals,
    _trilinear_sample,
    _unique_faces,
)
from .solvent_params import MAX_SAS_CUBES, MAX_SAS_VOXELS, edt_spacing, expanded_radii

def _paint_ball_union(shape, origin, h, centers, radii):
    occ = np.zeros(tuple(int(s) for s in shape), dtype=bool)
    origin = np.asarray(origin, dtype=float)
    h = float(h)
    nx, ny, nz = occ.shape
    limits = np.array((nx - 1, ny - 1, nz - 1), dtype=np.int32)
    for center, radius in zip(centers, radii):
        center = np.asarray(center, dtype=float)
        r = float(radius)
        if r <= 0.0:
            continue
        lo = np.floor((center - r - origin) / h).astype(np.int32)
        hi = np.ceil((center + r - origin) / h).astype(np.int32)
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
        inside = np.linalg.norm(xyz - center, axis=-1) <= r + 1e-8
        occ[ii, jj, kk] |= inside
    return occ
def _edt_ses_grid(centers, expanded, probe, spacing, _depth=0):
    """Accessible-union EDT: zero of (probe - dist_to_solvent) is the rolling-ball SES."""
    h = float(spacing)
    probe = float(probe)
    if h < 1e-8 or centers.shape[0] == 0:
        return None
    pad = float(np.max(expanded)) + 2.0 * h
    origin = np.min(centers, axis=0) - pad
    hi = np.max(centers, axis=0) + pad
    shape = np.floor((hi - origin) / h).astype(np.int32) + 3
    shape = np.maximum(shape, 2)
    n_vox = int(shape[0]) * int(shape[1]) * int(shape[2])
    if n_vox > MAX_SAS_VOXELS and _depth < 8:
        scale = (float(n_vox) / float(MAX_SAS_VOXELS)) ** (1.0 / 3.0)
        return _edt_ses_grid(centers, expanded, probe, h * max(scale, 1.12), _depth=_depth + 1)
    try:
        occ = _paint_ball_union(shape, origin, h, centers, expanded)
        if not np.any(occ):
            return None
        edt = distance_transform_edt(occ, sampling=(h, h, h))
        field = float(probe) - edt
        gx, gy, gz = np.gradient(edt, h, h, h)
        grad = np.stack((-gx, -gy, -gz), axis=-1)
    except MemoryError:
        if _depth < 8:
            return _edt_ses_grid(centers, expanded, probe, h * 1.25, _depth=_depth + 1)
        return None
    return origin, h, field, grad
def _edt_ses_component(centers, expanded, probe, spacing, _depth=0):
    empty = (
        np.zeros((0, 3), dtype=float),
        np.zeros((0, 3), dtype=float),
        np.zeros((0, 3), dtype=int),
    )
    built = _edt_ses_grid(centers, expanded, probe, spacing)
    if built is None:
        return empty
    origin, h, field, grad = built
    cubes = _cubes_with_sign_change(field)
    if cubes.shape[0] == 0:
        return empty
    if cubes.shape[0] > MAX_SAS_CUBES and _depth < 6:
        return _edt_ses_component(
            centers, expanded, probe, h * max(math.sqrt(float(cubes.shape[0]) / float(MAX_SAS_CUBES)), 1.15),
            _depth=_depth + 1,
        )
    return _isosurface_from_signed_field(origin, h, field, grad)
def _mc_mesh(centers, radii, quality: int, probe_radius: float = 0.0):
    empty = (
        np.zeros((0, 3), dtype=float),
        np.zeros((0, 3), dtype=float),
        np.zeros((0, 3), dtype=int),
    )
    probe = float(probe_radius)
    expanded = np.asarray(radii, dtype=float).reshape(-1)
    n = int(np.asarray(centers).reshape(-1, 3).shape[0])
    if n == 0:
        return empty
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    spacing = edt_spacing(quality, probe)
    parts_v = []
    parts_n = []
    parts_f = []
    offset = 0
    for group in _overlap_components(centers, expanded):
        idx = np.asarray(group, dtype=int)
        vertices, normals, faces = _edt_ses_component(
            centers[idx], expanded[idx], probe, spacing,
        )
        if faces.shape[0] == 0:
            continue
        parts_v.append(vertices)
        parts_n.append(normals)
        parts_f.append(faces + offset)
        offset += int(vertices.shape[0])
    if not parts_v:
        return empty
    vertices = np.vstack(parts_v)
    normals = np.vstack(parts_n)
    faces = np.vstack(parts_f)
    faces = _unique_faces(faces)
    if faces.shape[0] == 0:
        return empty
    used = np.unique(np.asarray(faces, dtype=int).ravel())
    if used.size != vertices.shape[0]:
        remap = np.full(int(vertices.shape[0]), -1, dtype=int)
        remap[used] = np.arange(used.size, dtype=int)
        vertices = vertices[used]
        normals = normals[used]
        faces = remap[faces]
    return vertices, normals, faces
