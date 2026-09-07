"""Solvent surfaces around a set of spheres.

``SAS`` — rolling-ball solvent-excluded surface (Connolly / MSMS): reduced
surface, analytical contact/saddle/reentrant faces, singularity clipping,
then a geodesic template sphere whose contact patches are clipped to the
torus polylines (Sanner, Olson & Spehner, Biopolymers 38:305–320, 1996).
``MC`` — the same rolling-ball SES via a Euclidean distance transform of the
accessible union, extracted with marching cubes (EDTSurf-style). The mesh
is watertight on the voxel grid and has no cap/saddle stitches.
``GAUSS`` — PyMOL ``map_new gaussian`` + ``isosurface``: Cromer–Mann atomic
scattering Gaussians, B-factor floor, normalized brick, contour at 1σ.
``ASA`` — Shrake–Rupley solvent-accessible patches on the expanded spheres
(atom radius + probe), clipped to neighboring intersection circles.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence, Tuple

import math

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import Delaunay, cKDTree

from .geometries import geodesic_icosphere
from .gaussian_map import (
    DEFAULT_GAUSSIAN_B_FLOOR,
    DEFAULT_GAUSSIAN_ISOLEVEL,
    DEFAULT_GAUSSIAN_RESOLUTION,
    atom_gaussian_terms,
    gaussian_blur_factor,
    normalize_gaussian_map,
    paint_gaussian_density,
)
from .marching_cubes import march_cubes

SURFACE_ALGORITHMS = ("SAS", "MC", "GAUSS", "ASA")
_ALGORITHM_ALIASES = {
    "CUBES": "MC",
    "EDT": "MC",
    "MARCHING_CUBES": "MC",
    "MARCHINGCUBES": "MC",
    "GAUSSIAN": "GAUSS",
    "BLOB": "GAUSS",
    "MAP": "GAUSS",
}
RADIUS_MODES = ("uniform", "vdw")
DEFAULT_ATOM_RADIUS = 1.0
DEFAULT_PROBE_RADIUS = 1.4
DEFAULT_QUALITY = 3
DEFAULT_VDW_SCALE = 1.0
DEFAULT_RADIUS_MODE = "vdw"
DEFAULT_ALGORITHM = "GAUSS"

# Bondi (1964) van der Waals radii, Å.
BONDI_VDW = {
    "H": 1.20, "HE": 1.40, "LI": 1.82, "BE": 1.53, "B": 1.92, "C": 1.70,
    "N": 1.55, "O": 1.52, "F": 1.47, "NE": 1.54, "NA": 2.27, "MG": 1.73,
    "AL": 1.84, "SI": 2.10, "P": 1.80, "S": 1.80, "CL": 1.75, "AR": 1.88,
    "K": 2.75, "CA": 2.31, "MN": 1.61, "FE": 1.84, "CO": 1.52, "NI": 1.63,
    "CU": 1.40, "ZN": 1.39, "SE": 1.90, "BR": 1.85, "KR": 2.02, "I": 1.98,
    "XE": 2.16, "MO": 1.90,
}
_TWO_LETTER_ELEM = frozenset(
    key for key in BONDI_VDW if len(key) == 2
)
MAX_SAS_CUBES = 80000
MAX_SAS_VOXELS = 1200000

# Grid step (Å) for the MC EDT isosurface (capped further by probe / 4.5).
SAS_SPACING = {1: 0.90, 2: 0.65, 3: 0.45, 4: 0.32, 5: 0.22}
# Icosphere frequency for ASA patches (same ladder as sphere meshes).
ASA_FREQUENCY = {1: 2, 2: 3, 3: 4, 4: 6, 5: 8}


def _convex_cap_frequency(frequency: int) -> int:
    """Geodesic frequency for VDW contact caps.

    Default quality 3 is ASA frequency 4; lighting needs a denser sphere than
    that or PyMOL's specular exponent turns each cap into a cluster of spots.
    """
    return min(20, max(12, 4 * int(frequency)))


def _torus_n_theta(frequency: int) -> int:
    """Contact-circle samples so the torus rim matches the geodesic cap.

    Quality 3 used 32 theta samples against a 16-frequency VDW ico. Clip hits
    then hung on ~0.18 Å chords and lit as a jagged stitch even after T-split.
    """
    cap_freq = _convex_cap_frequency(frequency)
    return max(24, 8 * int(frequency), 4 * int(cap_freq))


def normalize_algorithm(name) -> str:
    text = str(name or DEFAULT_ALGORITHM).strip().upper().replace("-", "_").replace(" ", "_")
    text = _ALGORITHM_ALIASES.get(text, text)
    if text in SURFACE_ALGORITHMS:
        return text
    return DEFAULT_ALGORITHM


def normalize_radius_mode(name) -> str:
    text = str(name or DEFAULT_RADIUS_MODE).strip().lower().replace("-", "_")
    if text in ("vdw", "vdw_scale", "scale_vdw"):
        return "vdw"
    return "uniform"


def element_from_atom_name(name, elem="") -> str:
    raw = str(elem or "").strip()
    if raw:
        letters = "".join(ch for ch in raw if ch.isalpha()).upper()
        if len(letters) >= 2 and letters[:2] in _TWO_LETTER_ELEM:
            return letters[:2]
        if letters:
            return letters[0]
    letters = "".join(ch for ch in str(name or "") if ch.isalpha()).upper()
    if letters:
        return letters[0]
    return "C"


def vdw_for_atom(name="", elem="", default=DEFAULT_ATOM_RADIUS) -> float:
    """Bondi radius from the ``elem`` field, else the first letter of the atom name.

    Two-letter symbols (``CA`` calcium, ``CL`` chlorine) are only taken from
    ``elem``. A protein atom named ``CA`` with an empty ``elem`` is carbon.
    """
    key = element_from_atom_name(name, elem)
    return float(BONDI_VDW.get(key, default))


def vdw_for_element(elem, default=DEFAULT_ATOM_RADIUS) -> float:
    return vdw_for_atom(elem=elem, default=default)


def lookup_source_vdw(source, context=None):
    """Return an atom van der Waals radius in Å, or None if the source is not an atom."""
    from ..points import AtomPoint

    if not isinstance(source, AtomPoint):
        return None
    if context is not None:
        found = source.lookup_vdw(context)
        if found is not None:
            return float(found)
    cached = getattr(source, "last_vdw", None)
    if cached is not None:
        return float(cached)
    return vdw_for_atom(getattr(source, "name", ""), getattr(source, "elem", ""))


def normalize_point_radii(values, n: int):
    """Length-*n* list of optional floats; ``None`` if every entry is inherited."""
    if n <= 0:
        return None
    if values is None:
        return None
    out = []
    raw = list(values)
    for i in range(int(n)):
        if i >= len(raw) or raw[i] is None:
            out.append(None)
            continue
        try:
            out.append(float(raw[i]))
        except (TypeError, ValueError):
            out.append(None)
    if all(item is None for item in out):
        return None
    return out


def resolve_atom_radii(
    sources,
    n: int,
    atom_radius=DEFAULT_ATOM_RADIUS,
    radius_mode=DEFAULT_RADIUS_MODE,
    vdw_scale=DEFAULT_VDW_SCALE,
    point_radii=None,
    context=None,
) -> np.ndarray:
    """Per-point atom radii before adding the probe."""
    count = max(int(n), 0)
    out = np.full(count, float(atom_radius), dtype=float)
    if count == 0:
        return out
    if normalize_radius_mode(radius_mode) == "vdw" and sources:
        scale = float(vdw_scale) if vdw_scale else DEFAULT_VDW_SCALE
        for i, source in enumerate(sources):
            if i >= count:
                break
            vdw = lookup_source_vdw(source, context)
            if vdw is not None:
                out[i] = float(vdw) * scale
    custom = normalize_point_radii(point_radii, count)
    if custom:
        for i, value in enumerate(custom):
            if value is not None:
                out[i] = float(value)
    return out


def resolve_atom_elements(sources, n: int):
    """Per-point element symbols; carbon when the source has no identity."""
    count = max(int(n), 0)
    out = ["C"] * count
    if not sources:
        return out
    for i, source in enumerate(sources):
        if i >= count:
            break
        out[i] = element_from_atom_name(
            getattr(source, "name", ""),
            getattr(source, "elem", ""),
        )
    return out


def _quality_level(quality: int) -> int:
    return max(1, min(5, int(quality)))


def sas_spacing(quality: int) -> float:
    return float(SAS_SPACING[_quality_level(quality)])


def edt_spacing(quality: int, probe_radius: float) -> float:
    """Voxel size for the rolling-ball EDT. Keep several samples across the probe."""
    h = sas_spacing(quality)
    probe = float(probe_radius)
    if probe < 1e-8:
        return h
    return float(min(h, max(probe / 4.5, 0.16)))


def asa_frequency(quality: int) -> int:
    return int(ASA_FREQUENCY[_quality_level(quality)])


def gauss_spacing(quality: int, resolution: float = DEFAULT_GAUSSIAN_RESOLUTION) -> float:
    """Voxel size for the PyMOL Gaussian map. Default grid is resolution / 3."""
    h = sas_spacing(quality)
    resol = max(float(resolution), 1.0)
    return float(min(h, max(resol / 3.0, 0.16)))


def expanded_radii(atom_radius, n: int, probe_radius: float) -> np.ndarray:
    radii = np.asarray(atom_radius, dtype=float).reshape(-1)
    if radii.size == 1:
        radii = np.repeat(radii, max(int(n), 1))
    elif radii.size != n:
        radii = np.resize(radii, n)
    return radii + float(probe_radius)


def signed_distance(xyz: np.ndarray, centers: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """min_i (|x - c_i| - R_i). Negative inside the union of spheres."""
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    radii = np.asarray(radii, dtype=float).reshape(-1)
    if xyz.shape[0] == 0 or centers.shape[0] == 0:
        return np.zeros((xyz.shape[0],), dtype=float)
    if centers.shape[0] == 1:
        return np.linalg.norm(xyz - centers[0], axis=1) - float(radii[0])
    if float(np.max(np.abs(radii - radii[0]))) < 1e-12:
        tree = cKDTree(centers)
        dist, idx = tree.query(xyz)
        return dist - radii[np.asarray(idx, dtype=int)]
    offset = xyz[:, None, :] - centers[None, :, :]
    return np.min(np.linalg.norm(offset, axis=2) - radii.reshape(1, -1), axis=1)


def _orient(tri, hint):
    a, b, c = tri
    n = np.cross(b - a, c - a)
    if np.dot(n, hint) < 0.0:
        return (a, c, b)
    return tri


def _weld(tris, ndigits=5):
    if not tris:
        return (
            np.zeros((0, 3), dtype=float),
            np.zeros((0, 3), dtype=int),
        )
    index = {}
    verts = []
    faces = []
    for a, b, c in tris:
        ids = []
        for p in (a, b, c):
            key = (round(float(p[0]), ndigits), round(float(p[1]), ndigits), round(float(p[2]), ndigits))
            vid = index.get(key)
            if vid is None:
                vid = len(verts)
                index[key] = vid
                verts.append((float(p[0]), float(p[1]), float(p[2])))
            ids.append(vid)
        if ids[0] != ids[1] and ids[1] != ids[2] and ids[2] != ids[0]:
            faces.append(ids)
    if not faces:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int)
    return np.asarray(verts, dtype=float), _unique_faces(np.asarray(faces, dtype=int))


def _unique_faces(faces):
    """Keep the first winding of each unordered vertex triple."""
    if faces.shape[0] == 0:
        return faces
    faces = np.asarray(faces, dtype=int)
    degenerates = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    )
    if np.any(degenerates):
        faces = faces[~degenerates]
        if faces.shape[0] == 0:
            return faces
    keys = np.sort(faces, axis=1)
    _uniq, index = np.unique(keys, axis=0, return_index=True)
    return faces[np.sort(index)]


def _drop_degenerate_faces(vertices, faces, min_cross=1e-12):
    if faces.shape[0] == 0:
        return faces
    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]
    area2 = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    keep = area2 > float(min_cross)
    if np.all(keep):
        return faces
    return faces[keep]


def _drop_overcovered_edge_faces(vertices, faces):
    """Drop extra slivers on edges that already have two faces.

    A Delaunay cap plus torus share the contact polyline (manifold). Tiny
    leftover clip/fan triangles stacked on those edges z-fight as a dark
    stitch. Keep the two largest faces per over-covered edge.
    """
    faces = np.asarray(faces, dtype=int).reshape(-1, 3)
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    if faces.shape[0] == 0:
        return faces
    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]
    area2 = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    keep = np.ones((int(faces.shape[0]),), dtype=bool)
    changed = True
    while changed:
        changed = False
        edge_faces = {}
        for fi, (a, b, c) in enumerate(faces):
            if not keep[fi]:
                continue
            for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
                key = (u, v) if u < v else (v, u)
                bucket = edge_faces.get(key)
                if bucket is None:
                    edge_faces[key] = [fi]
                else:
                    bucket.append(fi)
        for fis in edge_faces.values():
            if len(fis) <= 2:
                continue
            order = sorted(fis, key=lambda i: (float(area2[i]), int(i)))
            for fi in order[: len(fis) - 2]:
                if keep[fi]:
                    keep[fi] = False
                    changed = True
    if np.all(keep):
        return faces
    return faces[keep]


def _compact_mesh(vertices, faces):
    if faces.shape[0] == 0:
        return vertices, faces
    used = np.unique(np.asarray(faces, dtype=int).ravel())
    if used.size == vertices.shape[0]:
        return vertices, faces
    remap = np.full(int(vertices.shape[0]), -1, dtype=int)
    remap[used] = np.arange(used.size, dtype=int)
    return vertices[used], remap[faces]


def _face_normals(vertices, faces):
    """Area-weighted vertex normals from triangle cross products."""
    if vertices.shape[0] == 0:
        return np.zeros((0, 3), dtype=float)
    normals = np.zeros_like(vertices, dtype=float)
    for face in faces:
        p0, p1, p2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        n = np.cross(p1 - p0, p2 - p0)
        for idx in face:
            normals[idx] += n
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths < 1e-12] = 1.0
    return normals / lengths


def _sas_shell_cubes(centers, radii, origin, spacing: float):
    """Cubes whose interior can cross the union surface (narrow band)."""
    h = float(spacing)
    band = 1.75 * h
    chunks = []
    for center, radius in zip(centers, radii):
        extent = float(radius) + band
        lo = np.floor((center - extent - origin) / h).astype(np.int32)
        hi = np.floor((center + extent - origin) / h).astype(np.int32)
        xs = np.arange(int(lo[0]), int(hi[0]) + 1, dtype=np.int32)
        ys = np.arange(int(lo[1]), int(hi[1]) + 1, dtype=np.int32)
        zs = np.arange(int(lo[2]), int(hi[2]) + 1, dtype=np.int32)
        if xs.size == 0 or ys.size == 0 or zs.size == 0:
            continue
        ii, jj, kk = np.meshgrid(xs, ys, zs, indexing="ij")
        cubes = np.stack((ii.ravel(), jj.ravel(), kk.ravel()), axis=1)
        mid = origin + (cubes.astype(float) + 0.5) * h
        dist = np.linalg.norm(mid - center.reshape(1, 3), axis=1)
        keep = np.abs(dist - float(radius)) <= band
        if np.any(keep):
            chunks.append(cubes[keep])
    if not chunks:
        return np.zeros((0, 3), dtype=np.int32)
    stacked = np.vstack(chunks)
    return np.unique(stacked, axis=0)


def _project_sas(vertices, centers, radii):
    """Snap vertices onto the exact union-of-spheres SAS.

    Ownership is the sphere with the smallest signed distance, not the nearest
    center. Nearest-center projection is wrong once radii differ (VDW).
    """
    if vertices.shape[0] == 0:
        return vertices
    radii = np.asarray(radii, dtype=float).reshape(-1)
    if centers.shape[0] == 1:
        offset = vertices - centers[0]
        length = np.linalg.norm(offset, axis=1, keepdims=True)
        length = np.maximum(length, 1e-12)
        return centers[0] + float(radii[0]) * offset / length
    sdf = _sas_sdf(vertices, centers, radii)
    idx = np.argmin(sdf, axis=1)
    chosen = centers[idx]
    offset = vertices - chosen
    length = np.linalg.norm(offset, axis=1, keepdims=True)
    length = np.maximum(length, 1e-12)
    projected = chosen + radii[idx][:, None] * offset / length
    if centers.shape[0] < 2:
        return projected
    sdf2 = _sas_sdf(projected, centers, radii)
    buried = np.min(sdf2, axis=1) < -1e-7
    if not np.any(buried):
        return projected
    order = np.argsort(sdf2, axis=1)
    for i in np.flatnonzero(buried):
        a = int(order[i, 0])
        b = int(order[i, 1])
        hint = projected[i]
        if centers.shape[0] >= 3 and float(sdf2[i, int(order[i, 2])]) < -1e-7:
            triples = _sphere_triple_points(
                centers[a], float(radii[a]),
                centers[b], float(radii[b]),
                centers[int(order[i, 2])], float(radii[int(order[i, 2])]),
            )
            if triples:
                projected[i] = min(
                    triples,
                    key=lambda point: float(np.linalg.norm(point - hint)),
                )
                continue
        projected[i] = _snap_to_intersection_circle(
            hint, centers[a], float(radii[a]), centers[b], float(radii[b]),
        )
    return projected


def _sas_owners(vertices, centers, radii):
    if vertices.shape[0] == 0:
        return np.zeros((0,), dtype=int)
    if centers.shape[0] == 1:
        return np.zeros((vertices.shape[0]), dtype=int)
    sdf = _sas_sdf(vertices, centers, radii)
    return np.argmin(sdf, axis=1)


def _sas_sdf(vertices, centers, radii):
    offset = vertices[:, None, :] - centers[None, :, :]
    return np.linalg.norm(offset, axis=2) - radii.reshape(1, -1)


def _sas_pin_state(vertices, centers, radii, tol=0.15):
    """Mark vertices that should stay on a circle or triple point."""
    n = int(vertices.shape[0])
    s = int(centers.shape[0])
    pinned = np.zeros(n, dtype=bool)
    pair_a = np.zeros(n, dtype=int)
    pair_b = np.zeros(n, dtype=int)
    pair_c = np.full(n, -1, dtype=int)
    if n == 0 or s < 2:
        return pinned, pair_a, pair_b, pair_c
    sdf = _sas_sdf(vertices, centers, radii)
    order = np.argsort(sdf, axis=1)
    i0 = order[:, 0]
    i1 = order[:, 1]
    d1 = sdf[np.arange(n), i1]
    pinned = d1 <= float(tol)
    pair_a[:] = i0
    pair_b[:] = i1
    if s >= 3:
        i2 = order[:, 2]
        d2 = sdf[np.arange(n), i2]
        triple = pinned & (d2 <= float(tol))
        pair_c[triple] = i2[triple]
    return pinned, pair_a, pair_b, pair_c


def _apply_sas_projection(vertices, centers, radii, pin=None):
    projected = _project_sas(vertices, centers, radii)
    if pin is None or centers.shape[0] < 2:
        return projected
    pinned, pair_a, pair_b, pair_c = pin
    idx = np.flatnonzero(pinned)
    for i in idx:
        a = int(pair_a[i])
        b = int(pair_b[i])
        c = int(pair_c[i])
        hint = vertices[i]
        if c >= 0:
            triples = _sphere_triple_points(
                centers[a], float(radii[a]),
                centers[b], float(radii[b]),
                centers[c], float(radii[c]),
            )
            if triples:
                projected[i] = min(
                    triples,
                    key=lambda point: float(np.linalg.norm(point - hint)),
                )
                continue
        projected[i] = _snap_to_intersection_circle(
            hint, centers[a], float(radii[a]), centers[b], float(radii[b]),
        )
    return projected


def _reweld(vertices, faces, ndigits=5):
    """Merge vertices that round to the same grid point, preserving faces."""
    if faces.shape[0] == 0:
        return vertices, faces
    tris = [
        (vertices[int(a)], vertices[int(b)], vertices[int(c)])
        for a, b, c in faces
    ]
    return _weld(tris, ndigits=ndigits)


def _subdivide_triangle(a, b, c, ab, bc, ca, out):
    """Subdivide one triangle after inserting shared edge midpoints.

    Never emits a split original edge: that would leave a T-junction.
    """
    if ab is None and bc is None and ca is None:
        out.append((a, b, c))
        return
    if ab is not None and bc is None and ca is None:
        out.append((a, ab, c))
        out.append((ab, b, c))
        return
    if bc is not None and ab is None and ca is None:
        out.append((a, b, bc))
        out.append((a, bc, c))
        return
    if ca is not None and ab is None and bc is None:
        out.append((a, b, ca))
        out.append((b, c, ca))
        return
    if ab is not None and bc is not None and ca is None:
        out.append((ab, b, bc))
        out.append((a, ab, c))
        out.append((ab, bc, c))
        return
    if bc is not None and ca is not None and ab is None:
        out.append((bc, c, ca))
        out.append((a, b, bc))
        out.append((a, bc, ca))
        return
    if ca is not None and ab is not None and bc is None:
        out.append((ca, a, ab))
        out.append((b, c, ca))
        out.append((ab, b, ca))
        return
    out.append((ab, b, bc))
    out.append((ab, bc, c))
    out.append((ab, c, ca))
    out.append((ab, ca, a))


def _split_t_junctions(vertices, faces, max_dist=0.04, t_pad=0.02, max_passes=8, cleanup=True):
    """Split a boundary edge when a hanging rim vertex lies on it.

    Cap/torus T-junctions show up as a dense-patch vertex sitting in the
    interior of a coarser boundary edge. Interior chords are left alone:
    on a curved SES those look close in Euclidean space without being
    T-junctions.
    """
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    faces = np.asarray(faces, dtype=int).reshape(-1, 3)
    if faces.shape[0] == 0 or vertices.shape[0] < 4:
        return vertices, faces
    max_dist = float(max_dist)
    t_pad = float(t_pad)

    def edge_key(i, j):
        i, j = int(i), int(j)
        return (i, j) if i < j else (j, i)

    did_split = False
    for _ in range(max(0, int(max_passes))):
        faces = np.asarray(faces, dtype=int).reshape(-1, 3)
        if faces.shape[0] == 0:
            break
        adj = _boundary_adjacency(faces)
        if len(adj) < 3:
            break
        bverts = np.array(list(adj.keys()), dtype=int)
        tree = cKDTree(vertices[bverts])
        boundary_edges = {}
        for u, nbrs in adj.items():
            u = int(u)
            for v in nbrs:
                key = edge_key(u, v)
                if key not in boundary_edges:
                    boundary_edges[key] = True
        incident = {}
        for a, b, c in faces:
            a, b, c = int(a), int(b), int(c)
            for u, v in ((a, b), (b, c), (c, a)):
                key = edge_key(u, v)
                if key not in boundary_edges:
                    continue
                bucket = incident.get(key)
                if bucket is None:
                    incident[key] = {a, b, c}
                else:
                    bucket.update((a, b, c))
        splits = {}
        for (u, v), used in incident.items():
            pu = vertices[u]
            pv = vertices[v]
            span = pv - pu
            length2 = float(np.dot(span, span))
            # Dense clip hits put ~2–3 hanging verts on one torus chord.
            # Skipping anything shorter than 2*max_dist left the last hit
            # on a ~0.05 Å stub and a slit in the contact seam.
            if length2 < 0.02 * 0.02:
                continue
            length = math.sqrt(length2)
            mid = 0.5 * (pu + pv)
            cand = tree.query_ball_point(mid, r=0.5 * length + max_dist)
            best = None
            for ci in cand:
                w = int(bverts[int(ci)])
                if w in used:
                    continue
                t = float(np.dot(vertices[w] - pu, span) / length2)
                if t <= t_pad or t >= 1.0 - t_pad:
                    continue
                dist = float(np.linalg.norm(vertices[w] - (pu + t * span)))
                if dist >= max_dist:
                    continue
                if best is None or dist < best[0]:
                    best = (dist, w)
            if best is not None:
                splits[edge_key(u, v)] = best[1]
        if not splits:
            break
        did_split = True
        new_faces = []
        for a, b, c in faces:
            a, b, c = int(a), int(b), int(c)
            _subdivide_triangle(
                a, b, c,
                splits.get(edge_key(a, b)),
                splits.get(edge_key(b, c)),
                splits.get(edge_key(c, a)),
                new_faces,
            )
        faces = np.asarray(new_faces, dtype=int)
    if did_split and cleanup:
        faces = _unique_faces(faces)
        faces = _drop_degenerate_faces(vertices, faces)
    return vertices, faces


def _weld_contact_seam_stubs(vertices, faces, centers, vdw, tol=0.02):
    """Merge near-duplicate cap/torus rim vertices on the contact circle.

    T-split leaves unmatched 3–4 cycles whose endpoints miss by ~5e-4 Å. Those
    slits line up around the join and render as a dark stitch. Only the
    two-atom contact band is welded so a tight reentrant is not chorded.
    """
    faces = np.asarray(faces, dtype=int).reshape(-1, 3)
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    if faces.shape[0] == 0 or vertices.shape[0] < 2 or centers.shape[0] < 2:
        return vertices, faces
    adj = _boundary_adjacency(faces)
    if len(adj) < 2:
        return vertices, faces
    bverts = np.array(list(adj.keys()), dtype=int)
    sdf = signed_distance(vertices[bverts], centers, vdw)
    keep = np.abs(sdf) < 0.08
    if int(np.count_nonzero(keep)) < 2:
        return vertices, faces
    sel = bverts[keep]
    tree = cKDTree(vertices[sel])
    pairs = tree.query_pairs(r=float(tol))
    if not pairs:
        return vertices, faces
    parent = np.arange(int(vertices.shape[0]), dtype=int)

    def find(i):
        i = int(i)
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in pairs:
        union(int(sel[int(i)]), int(sel[int(j)]))
    clusters = {}
    for i in sel:
        clusters.setdefault(find(int(i)), []).append(int(i))
    out = np.array(vertices, copy=True)
    for root, members in clusters.items():
        if len(members) < 2:
            continue
        out[root] = np.mean(out[np.asarray(members, dtype=int)], axis=0)
    new_faces = []
    for a, b, c in faces:
        a, b, c = find(int(a)), find(int(b)), find(int(c))
        if a == b or b == c or c == a:
            continue
        new_faces.append((a, b, c))
    if not new_faces:
        return vertices, faces
    faces = _unique_faces(np.asarray(new_faces, dtype=int))
    return _compact_mesh(out, faces)


def _split_seam_edges(vertices, faces, centers, radii):
    """Insert shared circle / triple-point vertices on mixed-owner faces."""
    if faces.shape[0] == 0 or centers.shape[0] < 2:
        return vertices, faces
    owner = _sas_owners(vertices, centers, radii)
    verts = [np.asarray(vertices[i], dtype=float) for i in range(vertices.shape[0])]
    mid = {}
    triples = {}

    def edge_mid(a, b):
        if owner[a] == owner[b]:
            return None
        key = (a, b) if a < b else (b, a)
        vid = mid.get(key)
        if vid is None:
            ia, ib = int(owner[key[0]]), int(owner[key[1]])
            point = _snap_to_intersection_circle(
                0.5 * (verts[key[0]] + verts[key[1]]),
                centers[ia], float(radii[ia]),
                centers[ib], float(radii[ib]),
            )
            vid = len(verts)
            verts.append(np.asarray(point, dtype=float))
            mid[key] = vid
        return vid

    def triple_vid(ia, ib, ic, hint):
        pts = _sphere_triple_points(
            centers[ia], float(radii[ia]),
            centers[ib], float(radii[ib]),
            centers[ic], float(radii[ic]),
        )
        if not pts:
            return None
        chosen = min(pts, key=lambda point: float(np.linalg.norm(point - hint)))
        key = (
            tuple(sorted((int(ia), int(ib), int(ic)))),
            tuple(np.round(np.asarray(chosen, dtype=float), 4)),
        )
        vid = triples.get(key)
        if vid is None:
            vid = len(verts)
            verts.append(np.asarray(chosen, dtype=float))
            triples[key] = vid
        return vid

    new_faces = []
    for a, b, c in faces:
        a, b, c = int(a), int(b), int(c)
        ab, bc, ca = edge_mid(a, b), edge_mid(b, c), edge_mid(c, a)
        kinds = {int(owner[a]), int(owner[b]), int(owner[c])}
        if len(kinds) == 3 and ab is not None and bc is not None and ca is not None:
            ia, ib, ic = sorted(kinds)
            centroid = (verts[a] + verts[b] + verts[c]) / 3.0
            tid = triple_vid(ia, ib, ic, centroid)
            if tid is not None:
                span = max(
                    float(np.linalg.norm(verts[a] - verts[b])),
                    float(np.linalg.norm(verts[b] - verts[c])),
                    float(np.linalg.norm(verts[c] - verts[a])),
                    1e-12,
                )
                if float(np.linalg.norm(verts[tid] - centroid)) <= 2.5 * span:
                    ring = (a, ab, b, bc, c, ca)
                    for i, q in enumerate(ring):
                        r = ring[(i + 1) % 6]
                        if q != r and tid != q and tid != r:
                            new_faces.append((tid, q, r))
                    continue
        _subdivide_triangle(a, b, c, ab, bc, ca, new_faces)
    if not new_faces:
        return vertices, faces
    return np.asarray(verts, dtype=float), _unique_faces(np.asarray(new_faces, dtype=int))


def _smooth_on_sas(vertices, faces, centers, radii, pin=None, iterations=8, lam=0.35):
    """Umbrella smooth in the surface, reprojecting onto the SAS each step."""
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return vertices
    n = int(vertices.shape[0])
    e0 = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    e1 = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    src = np.concatenate([e0, e1])
    dst = np.concatenate([e1, e0])
    verts = np.asarray(vertices, dtype=float)
    for _ in range(int(iterations)):
        acc = np.zeros_like(verts)
        np.add.at(acc, src, verts[dst])
        cnt = np.bincount(src, minlength=n).astype(float)
        cnt = np.maximum(cnt, 1.0)
        verts = (1.0 - lam) * verts + lam * (acc / cnt[:, None])
        verts = _apply_sas_projection(verts, centers, radii, pin=pin)
    return verts


def _canonical_cycle(loop):
    n = len(loop)
    seq = list(loop)
    forward = min(tuple(seq[i:] + seq[:i]) for i in range(n))
    rev = list(reversed(seq))
    backward = min(tuple(rev[i:] + rev[:i]) for i in range(n))
    return forward if forward <= backward else backward


def _boundary_adjacency(faces):
    count = {}
    for a, b, c in faces:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (u, v) if u < v else (v, u)
            count[key] = count.get(key, 0) + 1
    adj = {}
    for (u, v), n in count.items():
        if n != 1:
            continue
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    for key, nbrs in adj.items():
        adj[key] = list(dict.fromkeys(nbrs))
    return adj


def _simple_boundary_cycles(adj, max_loop=16):
    """Simple cycles in the boundary graph, including those that share a hub vertex."""
    cycles = []
    seen = set()
    nodes = list(adj.keys())

    def dfs(start, prev, cur, path, in_path):
        if len(path) > max_loop:
            return
        for nxt in adj.get(cur, ()):
            if nxt == prev:
                continue
            if nxt == start:
                if len(path) >= 3:
                    key = _canonical_cycle(path)
                    if key not in seen:
                        seen.add(key)
                        cycles.append(list(path))
                continue
            if nxt in in_path:
                continue
            path.append(nxt)
            in_path.add(nxt)
            dfs(start, cur, nxt, path, in_path)
            path.pop()
            in_path.remove(nxt)

    for start in nodes:
        dfs(start, -1, start, [start], {start})
    return cycles


def _loop_fan_origin(points, loop):
    """Index into *loop* whose fan has the largest minimum triangle area."""
    n = len(loop)
    best_i = 0
    best_score = -1.0
    for i in range(n):
        origin = np.asarray(points[loop[i]], dtype=float)
        min_area = None
        ok = True
        for k in range(1, n - 1):
            a = np.asarray(points[loop[(i + k) % n]], dtype=float)
            b = np.asarray(points[loop[(i + k + 1) % n]], dtype=float)
            area = float(np.linalg.norm(np.cross(a - origin, b - origin)))
            if area <= 1e-16:
                ok = False
                break
            if min_area is None or area < min_area:
                min_area = area
        if ok and min_area is not None and min_area > best_score:
            best_score = min_area
            best_i = i
    return best_i


def _fill_boundary_holes(vertices, faces, max_loop=8, max_edge=0.85):
    """Fan-fill small boundary loops left by cap/torus T-junctions.

    Only short edges are filled so a missing contact cap is not papered over
    with a chord through the groove.
    """
    if faces.shape[0] == 0:
        return vertices, faces
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    extra = []
    used = set()
    max_loop = int(max_loop)
    max_edge = float(max_edge)
    occupancy = {}
    for a, b, c in faces:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (u, v) if u < v else (v, u)
            occupancy[key] = occupancy.get(key, 0) + 1

    def edge_key(u, v):
        u, v = int(u), int(v)
        return (u, v) if u < v else (v, u)

    def try_add_loop(loop):
        n = len(loop)
        if n < 3 or n > max_loop:
            return
        edges = []
        for i, vertex in enumerate(loop):
            nxt = loop[(i + 1) % n]
            key = edge_key(vertex, nxt)
            if key in used:
                return
            dist = float(np.linalg.norm(vertices[int(vertex)] - vertices[int(nxt)]))
            if dist > max_edge:
                return
            edges.append(key)
        origin_i = 0 if n == 3 else _loop_fan_origin(vertices, loop)
        rotated = loop[origin_i:] + loop[:origin_i]
        origin = rotated[0]
        added = []
        for i in range(1, n - 1):
            a, b = rotated[i], rotated[i + 1]
            if origin == a or a == b or b == origin:
                continue
            p0 = vertices[int(origin)]
            p1 = vertices[int(a)]
            p2 = vertices[int(b)]
            area2 = float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)))
            if area2 <= 1e-12:
                continue
            tri_edges = (edge_key(origin, a), edge_key(a, b), edge_key(b, origin))
            if any(occupancy.get(key, 0) >= 2 for key in tri_edges):
                return
            added.append((origin, a, b, tri_edges))
        if not added:
            return
        for origin, a, b, tri_edges in added:
            extra.append((origin, a, b))
            for key in tri_edges:
                occupancy[key] = occupancy.get(key, 0) + 1
        used.update(edges)

    adj = _boundary_adjacency(faces)
    if not adj:
        return vertices, faces
    for v, nbrs in adj.items():
        if len(nbrs) < 2:
            continue
        nbrs_v = list(nbrs)
        for i, a in enumerate(nbrs_v):
            nbrs_a = adj.get(a, ())
            for b in nbrs_v[i + 1:]:
                if b in nbrs_a:
                    try_add_loop([int(v), int(a), int(b)])

    combined = faces
    if extra:
        combined = np.vstack((faces, np.asarray(extra, dtype=int)))
    adj = _boundary_adjacency(combined)
    seen = set()
    for start in adj:
        if start in seen or len(adj.get(start, ())) != 2:
            continue
        cycle = [int(start)]
        prev = None
        cur = int(start)
        ok = True
        for _ in range(max_loop + 1):
            nxt = None
            for cand in adj.get(cur, ()):
                if cand != prev:
                    nxt = int(cand)
                    break
            if nxt is None:
                ok = False
                break
            if nxt == int(start):
                break
            if nxt in seen or nxt in cycle:
                ok = False
                break
            cycle.append(nxt)
            prev, cur = cur, nxt
        else:
            ok = False
        if ok and len(cycle) >= 3:
            try_add_loop(cycle)
        seen.update(cycle)

    if not extra:
        return vertices, faces
    return vertices, np.vstack((faces, np.asarray(extra, dtype=int)))


def _orient_faces_to_vertex_normals(vertices, faces, normals):
    """Flip a triangle when most of its vertex normals oppose the geometric normal."""
    if faces.shape[0] == 0:
        return faces
    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]
    geom = np.cross(p1 - p0, p2 - p0)
    d0 = np.sum(geom * normals[faces[:, 0]], axis=1)
    d1 = np.sum(geom * normals[faces[:, 1]], axis=1)
    d2 = np.sum(geom * normals[faces[:, 2]], axis=1)
    votes = (d0 < 0.0).astype(np.int32) + (d1 < 0.0).astype(np.int32) + (d2 < 0.0).astype(np.int32)
    area2 = np.linalg.norm(geom, axis=1)
    flip = (votes >= 2) & (area2 > 1e-10)
    if not np.any(flip):
        return faces
    out = np.array(faces, copy=True, dtype=int)
    out[flip] = out[flip][:, (0, 2, 1)]
    return out


def _orient_faces_outward(vertices, faces, centers, radii):
    """Flip each triangle so its geometric normal points out of the union.

    PyMOL lights CGO triangles independently. A mesh-wide winding walk inverts
    whole patches that do not share vertex indices; each face is oriented here
    on its own.
    """
    if faces.shape[0] == 0:
        return faces
    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-18)
    unit = normals / lengths
    centroid = (p0 + p1 + p2) / 3.0
    eps = 0.05
    sdf_pos = signed_distance(centroid + eps * unit, centers, radii)
    sdf_neg = signed_distance(centroid - eps * unit, centers, radii)
    owners = _sas_owners(centroid, centers, radii)
    hint = centroid - centers[owners]
    owner_inward = np.sum(normals * hint, axis=1) < 0.0
    both_inside = (sdf_pos < 0.0) & (sdf_neg < 0.0)
    flip = np.where(both_inside, owner_inward, sdf_pos < sdf_neg)
    if not np.any(flip):
        return faces
    out = np.array(faces, copy=True, dtype=int)
    out[flip] = out[flip][:, (0, 2, 1)]
    return out


def _overlap_neighbors(centers, radii):
    n = int(centers.shape[0])
    neigh = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if float(np.linalg.norm(centers[i] - centers[j])) < float(radii[i]) + float(radii[j]) - 1e-10:
                neigh[i].append(j)
                neigh[j].append(i)
    return neigh


def _intersection_circles(centers, radii):
    circles = []
    n = int(centers.shape[0])
    radii = np.asarray(radii, dtype=float).reshape(-1)
    for i in range(n):
        for j in range(i + 1, n):
            circle = _sphere_intersection_circle(
                centers[i], float(radii[i]), centers[j], float(radii[j]),
            )
            if circle is None:
                continue
            origin, normal, rad = circle
            thirds = []
            for k in range(n):
                if k == i or k == j:
                    continue
                if float(np.linalg.norm(centers[k] - origin)) < float(radii[k]) + rad - 1e-8:
                    thirds.append(k)
            circles.append((origin, normal, rad, i, j, tuple(thirds)))
    return circles


def _sas_triple_vertices(centers, radii):
    """Exposed SAS vertices: probe centers sitting against three expanded balls."""
    triples = _iter_probe_triples(centers, radii)
    if not triples:
        return np.zeros((0, 3), dtype=float)
    return np.asarray([item[3] for item in triples], dtype=float)


def _distance_to_vertices(xyz, verts):
    n = int(xyz.shape[0])
    if verts is None:
        verts = np.zeros((0, 3), dtype=float)
    else:
        verts = np.asarray(verts, dtype=float).reshape(-1, 3)
    if n == 0 or verts.shape[0] == 0:
        return np.full(n, np.inf), np.zeros((n, 3), dtype=float)
    tree = cKDTree(verts)
    dist, idx = tree.query(xyz)
    return np.asarray(dist, dtype=float).reshape(n), verts[np.asarray(idx, dtype=int).reshape(n)]


def _ball_band_cubes(center, radius, origin, spacing, band):
    h = float(spacing)
    center = np.asarray(center, dtype=float)
    extent = float(radius) + float(band)
    lo = np.floor((center - extent - origin) / h).astype(np.int32)
    hi = np.floor((center + extent - origin) / h).astype(np.int32)
    xs = np.arange(int(lo[0]), int(hi[0]) + 1, dtype=np.int32)
    ys = np.arange(int(lo[1]), int(hi[1]) + 1, dtype=np.int32)
    zs = np.arange(int(lo[2]), int(hi[2]) + 1, dtype=np.int32)
    if xs.size == 0 or ys.size == 0 or zs.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    ii, jj, kk = np.meshgrid(xs, ys, zs, indexing="ij")
    cubes = np.stack((ii.ravel(), jj.ravel(), kk.ravel()), axis=1)
    mid = origin + (cubes.astype(float) + 0.5) * h
    dist = np.linalg.norm(mid - center.reshape(1, 3), axis=1)
    keep = np.abs(dist - float(radius)) <= float(band)
    if not np.any(keep):
        return np.zeros((0, 3), dtype=np.int32)
    return cubes[keep]


def _distance_to_circles(xyz, circles, centers, radii):
    n = int(xyz.shape[0])
    dist = np.full(n, np.inf)
    closest = np.zeros((n, 3), dtype=float)
    radii = np.asarray(radii, dtype=float).reshape(-1)
    for origin, normal, rad, _i, _j, thirds in circles:
        origin = np.asarray(origin, dtype=float)
        normal = np.asarray(normal, dtype=float)
        delta = xyz - origin
        axial = delta @ normal
        radial_vec = delta - axial[:, None] * normal
        radial = np.linalg.norm(radial_vec, axis=1)
        length = np.maximum(radial, 1e-12)
        q = origin + float(rad) * (radial_vec / length[:, None])
        degenerate = radial < 1e-12
        if np.any(degenerate):
            axis = np.array(
                (1.0, 0.0, 0.0) if abs(float(normal[0])) < 0.9 else (0.0, 1.0, 0.0)
            )
            ortho = np.cross(normal, axis)
            ortho_len = float(np.linalg.norm(ortho))
            if ortho_len > 1e-12:
                q[degenerate] = origin + float(rad) * (ortho / ortho_len)
        cand = np.linalg.norm(xyz - q, axis=1)
        if thirds:
            buried = np.zeros(n, dtype=bool)
            for k in thirds:
                buried |= np.linalg.norm(q - centers[k], axis=1) < float(radii[k]) - 1e-6
            cand = np.where(buried, np.inf, cand)
        better = cand < dist
        dist[better] = cand[better]
        closest[better] = q[better]
    return dist, closest


def _distance_to_exposed_spheres(xyz, centers, radii, neighbors=None):
    n = int(xyz.shape[0])
    n_atoms = int(centers.shape[0])
    dist = np.full(n, np.inf)
    closest = np.zeros((n, 3), dtype=float)
    if n == 0 or n_atoms == 0:
        return dist, closest
    if neighbors is None:
        neighbors = _overlap_neighbors(centers, radii)
    offset = xyz[:, None, :] - centers[None, :, :]
    sph_dist = np.linalg.norm(offset, axis=2)
    sph_dist = np.maximum(sph_dist, 1e-12)
    for j in range(n_atoms):
        q = centers[j] + float(radii[j]) * (offset[:, j] / sph_dist[:, j:j + 1])
        buried = np.zeros(n, dtype=bool)
        for k in neighbors[j]:
            buried |= np.linalg.norm(q - centers[k], axis=1) < float(radii[k]) - 1e-8
        cand = np.abs(sph_dist[:, j] - float(radii[j]))
        cand = np.where(buried, np.inf, cand)
        better = cand < dist
        dist[better] = cand[better]
        closest[better] = q[better]
    return dist, closest


def _ball_union_euclidean_sdf(xyz, centers, radii, circles=None, neighbors=None, triples=None):
    """Signed Euclidean distance to the boundary of a union of balls."""
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    n = int(xyz.shape[0])
    if n == 0 or centers.shape[0] == 0:
        return np.zeros((n,), dtype=float), np.zeros((n, 3), dtype=float)
    if circles is None:
        circles = _intersection_circles(centers, radii)
    if neighbors is None:
        neighbors = _overlap_neighbors(centers, radii)
    if triples is None:
        triples = _sas_triple_vertices(centers, radii)
    d_sph, q_sph = _distance_to_exposed_spheres(xyz, centers, radii, neighbors=neighbors)
    d_circ, q_circ = _distance_to_circles(xyz, circles, centers, radii)
    d_vert, q_vert = _distance_to_vertices(xyz, triples)
    dist = d_sph
    closest = q_sph
    use_circ = d_circ < dist
    dist = np.where(use_circ, d_circ, dist)
    closest = np.where(use_circ[:, None], q_circ, closest)
    use_vert = d_vert < dist
    dist = np.where(use_vert, d_vert, dist)
    closest = np.where(use_vert[:, None], q_vert, closest)
    missing = ~np.isfinite(dist)
    if np.any(missing):
        xyz_m = xyz[missing]
        fallback = np.abs(signed_distance(xyz_m, centers, radii))
        dist[missing] = fallback
        owners = _sas_owners(xyz_m, centers, radii)
        delta = xyz_m - centers[owners]
        length = np.linalg.norm(delta, axis=1, keepdims=True)
        length = np.maximum(length, 1e-12)
        closest[missing] = centers[owners] + radii[owners][:, None] * (delta / length)
    inside = signed_distance(xyz, centers, radii) < 0.0
    sdf = np.where(inside, -dist, dist)
    return sdf, closest


def _ses_field(xyz, centers, radii, probe, circles=None, neighbors=None, triples=None):
    """Zero on the rolling-ball surface: Euclidean SDF of SAS volume + probe."""
    sdf, closest = _ball_union_euclidean_sdf(
        xyz, centers, radii, circles=circles, neighbors=neighbors, triples=triples,
    )
    return sdf + float(probe), closest


def _sas_volume_outward_normals(xyz, closest, sdf_eucl):
    delta = xyz - closest
    length = np.linalg.norm(delta, axis=1, keepdims=True)
    length = np.maximum(length, 1e-12)
    normals = delta / length
    inside = np.asarray(sdf_eucl) < 0.0
    if np.any(inside):
        normals = np.array(normals, copy=True)
        normals[inside] *= -1.0
    return normals


def _ses_outward_normals(vertices, closest, sdf_eucl, geom=None):
    normals = _sas_volume_outward_normals(vertices, closest, sdf_eucl)
    if geom is not None and geom.shape[0] == normals.shape[0]:
        flip = np.sum(normals * geom, axis=1) < 0.0
        if np.any(flip):
            normals = np.array(normals, copy=True)
            normals[flip] *= -1.0
    finite = np.isfinite(normals).all(axis=1)
    if not np.all(finite):
        normals = np.array(normals, copy=True)
        normals[~finite] = np.array((0.0, 0.0, 1.0))
    return normals


def _snap_to_ses(vertices, centers, radii, probe, circles=None, neighbors=None, triples=None, max_move=0.55):
    """Place vertices on the rolling-ball surface: probe from the closest SAS point."""
    pts = np.asarray(vertices, dtype=float)
    if pts.shape[0] == 0:
        return pts
    probe = float(probe)
    _field, closest = _ses_field(
        pts, centers, radii, probe, circles=circles, neighbors=neighbors, triples=triples,
    )
    delta = pts - closest
    length = np.linalg.norm(delta, axis=1, keepdims=True)
    too_close = length[:, 0] < 1e-8
    length = np.maximum(length, 1e-12)
    inward = delta / length
    outside = signed_distance(pts, centers, radii) >= 0.0
    if np.any(outside):
        inward = np.array(inward, copy=True)
        inward[outside] *= -1.0
    target = closest + probe * inward
    move = target - pts
    dist = np.linalg.norm(move, axis=1, keepdims=True)
    cap = float(max_move)
    scale = np.minimum(1.0, cap / np.maximum(dist, 1e-12))
    snapped = pts + move * scale
    if np.any(too_close):
        snapped = np.array(snapped, copy=True)
        snapped[too_close] = pts[too_close]
    vdw = np.maximum(np.asarray(radii, dtype=float) - probe, 1e-6)
    sdf_vdw = signed_distance(snapped, centers, vdw)
    inside = sdf_vdw < -0.08
    if np.any(inside):
        owners = _sas_owners(snapped[inside], centers, vdw)
        vec = snapped[inside] - centers[owners]
        ln = np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12)
        snapped = np.array(snapped, copy=True)
        snapped[inside] = centers[owners] + vdw[owners][:, None] * (vec / ln)
    return snapped


def _project_vdw_caps(vertices, centers, vdw, tol=0.04):
    """Put convex-patch vertices on the atom spheres so radial normals match the mesh."""
    pts = np.asarray(vertices, dtype=float)
    if pts.shape[0] == 0:
        return pts
    vdw = np.asarray(vdw, dtype=float).reshape(-1)
    sdf = signed_distance(pts, centers, vdw)
    cap = sdf < float(tol)
    if not np.any(cap):
        return pts
    owners = _sas_owners(pts[cap], centers, vdw)
    vec = pts[cap] - centers[owners]
    length = np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12)
    out = np.array(pts, copy=True)
    out[cap] = centers[owners] + vdw[owners][:, None] * (vec / length)
    return out


def _smooth_on_ses(vertices, faces, centers, radii, probe, circles=None, neighbors=None, triples=None, iterations=3, lam=0.25, pin_vdw=False):
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return vertices
    n = int(vertices.shape[0])
    e0 = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    e1 = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    src = np.concatenate([e0, e1])
    dst = np.concatenate([e1, e0])
    verts = np.asarray(vertices, dtype=float)
    pinned = None
    frozen = None
    if pin_vdw:
        vdw = np.maximum(np.asarray(radii, dtype=float) - float(probe), 1e-6)
        pinned = signed_distance(verts, centers, vdw) <= 0.06
        frozen = np.array(verts, copy=True)
    for _ in range(int(iterations)):
        acc = np.zeros_like(verts)
        np.add.at(acc, src, verts[dst])
        cnt = np.bincount(src, minlength=n).astype(float)
        cnt = np.maximum(cnt, 1.0)
        verts = (1.0 - lam) * verts + lam * (acc / cnt[:, None])
        if pinned is not None:
            verts[pinned] = frozen[pinned]
        verts = _snap_to_ses(
            verts, centers, radii, probe, circles=circles, neighbors=neighbors, triples=triples,
        )
        if pinned is not None:
            verts[pinned] = frozen[pinned]
    return verts


def _circle_is_exposed(origin, normal, rad, thirds, centers, radii, n_samp=12):
    """True if some probe-center on this SAS circle is not inside a third ball."""
    if not thirds:
        return True
    u, v = _orthonormal_axes(normal)
    origin = np.asarray(origin, dtype=float)
    rad = float(rad)
    for s in range(int(n_samp)):
        ang = 2.0 * math.pi * s / float(n_samp)
        q = origin + rad * (math.cos(ang) * u + math.sin(ang) * v)
        buried = False
        for k in thirds:
            if float(np.linalg.norm(q - centers[k])) < float(radii[k]) - 1e-6:
                buried = True
                break
        if not buried:
            return True
    return False


def _torus_cubes(circle_origin, circle_normal, circle_radius, tube_radius, origin, spacing, band):
    h = float(spacing)
    rho = float(tube_radius)
    rad = float(circle_radius)
    pad = rho + float(band)
    normal = np.asarray(circle_normal, dtype=float)
    center = np.asarray(circle_origin, dtype=float)
    sin_n = np.sqrt(np.maximum(1.0 - normal * normal, 0.0))
    half = pad + rad * sin_n
    lo = np.floor((center - half - origin) / h).astype(np.int32)
    hi = np.floor((center + half - origin) / h).astype(np.int32)
    xs = np.arange(int(lo[0]), int(hi[0]) + 1, dtype=np.int32)
    ys = np.arange(int(lo[1]), int(hi[1]) + 1, dtype=np.int32)
    zs = np.arange(int(lo[2]), int(hi[2]) + 1, dtype=np.int32)
    if xs.size == 0 or ys.size == 0 or zs.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    ii, jj, kk = np.meshgrid(xs, ys, zs, indexing="ij")
    cubes = np.stack((ii.ravel(), jj.ravel(), kk.ravel()), axis=1)
    mid = origin + (cubes.astype(float) + 0.5) * h
    delta = mid - center
    axial = delta @ normal
    radial_vec = delta - axial[:, None] * normal
    radial = np.linalg.norm(radial_vec, axis=1)
    dist_c = np.hypot(radial - rad, axial)
    keep = np.abs(dist_c - rho) <= float(band)
    if not np.any(keep):
        return np.zeros((0, 3), dtype=np.int32)
    return cubes[keep]


def _ses_shell_cubes(centers, radii, probe, origin, spacing, circles=None, triples=None):
    """Cubes near VDW patches and near probe-radius fillets of SAS seams."""
    h = float(spacing)
    probe = float(probe)
    vdw = np.maximum(np.asarray(radii, dtype=float) - probe, 1e-6)
    chunks = [_sas_shell_cubes(centers, vdw, origin, h)]
    if probe > 1e-8:
        band = 1.75 * h
        if circles is None:
            circles = _intersection_circles(centers, radii)
        for circle in circles:
            origin_c, normal, rad, _i, _j, thirds = circle
            if thirds and not _circle_is_exposed(
                origin_c, normal, rad, thirds, centers, radii,
            ):
                continue
            chunks.append(
                _torus_cubes(origin_c, normal, rad, probe, origin, h, band)
            )
        if triples is None:
            triples = _sas_triple_vertices(centers, radii)
        for q in np.asarray(triples, dtype=float).reshape(-1, 3):
            chunks.append(_ball_band_cubes(q, probe, origin, h, band))
    nonempty = [c for c in chunks if c.shape[0] > 0]
    if not nonempty:
        return np.zeros((0, 3), dtype=np.int32)
    return np.unique(np.vstack(nonempty), axis=0)


def _orthonormal_axes(normal):
    n = np.asarray(normal, dtype=float)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    axis = np.array((1.0, 0.0, 0.0) if abs(float(n[0])) < 0.9 else (0.0, 1.0, 0.0))
    u = np.cross(n, axis)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    v = np.cross(n, u)
    return u, v


def _slerp(a, b, t):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    omega = math.acos(dot)
    if omega < 1e-8:
        out = (1.0 - t) * a + t * b
        length = float(np.linalg.norm(out))
        return out / max(length, 1e-12)
    so = math.sin(omega)
    out = (math.sin((1.0 - t) * omega) * a + math.sin(t * omega) * b) / so
    length = float(np.linalg.norm(out))
    return out / max(length, 1e-12)


def _probe_collides(probe_center, centers, expanded, skip, eps=1e-6):
    q = np.asarray(probe_center, dtype=float)
    skip = set(int(s) for s in skip)
    for k in range(int(centers.shape[0])):
        if k in skip:
            continue
        if float(np.linalg.norm(q - centers[k])) < float(expanded[k]) - float(eps):
            return True
    return False


def _unit_vec(vector):
    vector = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(vector))
    return vector / max(length, 1e-12)


def _wrap_tau(theta):
    tau = 2.0 * math.pi
    value = theta % tau
    if value < 0.0:
        value += tau
    return value


def _ang_dist(a, b):
    delta = abs(_wrap_tau(a) - _wrap_tau(b))
    tau = 2.0 * math.pi
    return min(delta, tau - delta)


def _circle_point(origin, rad, u, v, theta):
    return np.asarray(origin, dtype=float) + float(rad) * (
        math.cos(theta) * u + math.sin(theta) * v
    )


def _circle_theta(point, origin, u, v):
    delta = np.asarray(point, dtype=float) - np.asarray(origin, dtype=float)
    return _wrap_tau(math.atan2(float(np.dot(delta, v)), float(np.dot(delta, u))))


def _torus_row(probe_center, center_i, center_j, probe, n_phi):
    """Probe-sphere geodesic from the i-contact to the j-contact."""
    q = np.asarray(probe_center, dtype=float)
    ui = _unit_vec(np.asarray(center_i, dtype=float) - q)
    uj = _unit_vec(np.asarray(center_j, dtype=float) - q)
    n_phi = max(1, int(n_phi))
    # Cluster samples at the VDW contacts (Chebyshev) so the first torus step is
    # small. Connolly is only C1 there; a large phi step clips Phong highlights.
    out = []
    for p in range(n_phi + 1):
        u = p / float(n_phi)
        t = 0.5 * (1.0 - math.cos(math.pi * u))
        out.append(q + float(probe) * _slerp(ui, uj, t))
    return np.asarray(out, dtype=float)


def _iter_probe_triples(centers, expanded):
    n = int(centers.shape[0])
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                probes = _sphere_triple_points(
                    centers[i], float(expanded[i]),
                    centers[j], float(expanded[j]),
                    centers[k], float(expanded[k]),
                )
                for q in probes:
                    q = np.asarray(q, dtype=float)
                    if _probe_collides(q, centers, expanded, (i, j, k)):
                        continue
                    out.append((i, j, k, q))
    return out


def _q_key(q, ndigits=5):
    q = np.asarray(q, dtype=float).reshape(3)
    return (
        round(float(q[0]), ndigits),
        round(float(q[1]), ndigits),
        round(float(q[2]), ndigits),
    )


def _rs_face_key(i, j, k, q):
    return (tuple(sorted((int(i), int(j), int(k)))), _q_key(q))


def _rs_edge_key(i, j):
    i, j = int(i), int(j)
    return (i, j) if i < j else (j, i)


def _ses_eaten(point, probes, owner, probe_r, eps=1e-4):
    """True if ``point`` lies inside another jammed probe (nonradial singularity)."""
    if probes is None or len(probes) == 0:
        return False
    p = np.asarray(point, dtype=float).reshape(3)
    probe_r = float(probe_r)
    owner = None if owner is None else np.asarray(owner, dtype=float).reshape(3)
    for q in probes:
        q = np.asarray(q, dtype=float).reshape(3)
        if owner is not None and float(np.linalg.norm(q - owner)) < 1e-5:
            continue
        if float(np.linalg.norm(p - q)) < probe_r - float(eps):
            return True
    return False


def _reduced_surface(centers, expanded, probe):
    """Algorithm 1 (Sanner 1996): reduced surface by rolling a probe.

    Returns dict with ``faces`` ``(i,j,k,q)``, ``edge_faces``, ``free_edges``,
    and ``free_vertices``.
    """
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    expanded = np.asarray(expanded, dtype=float).reshape(-1)
    n = int(centers.shape[0])
    probe = float(probe)
    vdw = np.maximum(expanded - probe, 1e-6)
    empty = {
        "faces": [],
        "edge_faces": {},
        "free_edges": [],
        "free_vertices": list(range(n)),
    }
    if n == 0:
        empty["free_vertices"] = []
        return empty
    tree = cKDTree(centers)
    max_r = float(np.max(expanded)) if n else 0.0
    faces = []
    face_ids = {}
    edge_faces = {}
    assigned = set()

    def add_face(i, j, k, q):
        key = _rs_face_key(i, j, k, q)
        if key in face_ids:
            return None
        idx = len(faces)
        q = np.asarray(q, dtype=float).reshape(3)
        faces.append((int(i), int(j), int(k), q))
        face_ids[key] = idx
        assigned.update((int(i), int(j), int(k)))
        for a, b in ((i, j), (j, k), (k, i)):
            ek = _rs_edge_key(a, b)
            edge_faces.setdefault(ek, []).append(idx)
        return idx

    def neighbor_atoms(i):
        radius = float(expanded[i]) + max_r + 1e-6
        return [int(j) for j in tree.query_ball_point(centers[i], radius) if int(j) != int(i)]

    def leftmost_seed(among, axis):
        among = [int(a) for a in among]
        among_set = set(among)
        if len(among) < 3:
            return None
        left = [float(centers[i, axis]) - float(vdw[i]) for i in among]
        i = among[int(np.argmin(np.asarray(left, dtype=float)))]
        best_j = None
        best_coord = None
        for j in neighbor_atoms(i):
            if j not in among_set:
                continue
            circle = _sphere_intersection_circle(
                centers[i], float(expanded[i]), centers[j], float(expanded[j]),
            )
            if circle is None:
                continue
            origin, normal, rad = circle
            u, v = _orthonormal_axes(normal)
            coord = float(origin[axis]) - float(rad) * math.hypot(float(u[axis]), float(v[axis]))
            if best_j is None or coord < best_coord - 1e-12 or (
                abs(coord - best_coord) < 1e-12 and j < best_j
            ):
                best_coord = coord
                best_j = j
        if best_j is None:
            return None
        j = best_j
        circle = _sphere_intersection_circle(
            centers[i], float(expanded[i]), centers[j], float(expanded[j]),
        )
        if circle is None:
            return None
        origin, _normal, _rad = circle
        search_r = float(np.linalg.norm(origin - centers[i])) + max_r + 1e-6
        cands = []
        for k in tree.query_ball_point(origin, search_r):
            k = int(k)
            if k == i or k == j or k not in among_set:
                continue
            for q in _sphere_triple_points(
                centers[i], float(expanded[i]),
                centers[j], float(expanded[j]),
                centers[k], float(expanded[k]),
            ):
                q = np.asarray(q, dtype=float)
                if _probe_collides(q, centers, expanded, (i, j, k)):
                    continue
                cands.append((float(q[axis]), int(k), q))
        if not cands:
            return None
        cands.sort(key=lambda item: (item[0], item[1]))
        _x, k, q = cands[0]
        return int(i), int(j), int(k), q

    def roll_next(i, j, q, third):
        circle = _sphere_intersection_circle(
            centers[i], float(expanded[i]), centers[j], float(expanded[j]),
        )
        if circle is None:
            return None
        origin, axis, rad = circle
        u, v = _orthonormal_axes(axis)
        q = np.asarray(q, dtype=float).reshape(3)
        theta0 = _circle_theta(q, origin, u, v)
        q_plus = _circle_point(origin, rad, u, v, theta0 + 0.03)
        d0 = float(np.linalg.norm(q - centers[third]))
        d1 = float(np.linalg.norm(q_plus - centers[third]))
        roll_plus = d1 >= d0 - 1e-12
        tau = 2.0 * math.pi
        search_r = float(rad) + max_r + 1e-6
        best = None
        for k in tree.query_ball_point(origin, search_r):
            k = int(k)
            if k == i or k == j:
                continue
            for q2 in _sphere_triple_points(
                centers[i], float(expanded[i]),
                centers[j], float(expanded[j]),
                centers[k], float(expanded[k]),
            ):
                q2 = np.asarray(q2, dtype=float)
                if _probe_collides(q2, centers, expanded, (i, j, k)):
                    continue
                if float(np.linalg.norm(q2 - q)) < 1e-5:
                    continue
                theta = _circle_theta(q2, origin, u, v)
                dth = (theta - theta0) % tau if roll_plus else (theta0 - theta) % tau
                if dth < 1e-4 or dth > tau - 1e-4:
                    continue
                if best is None or dth < best[0] - 1e-12:
                    best = (dth, k, q2)
        if best is None:
            return None
        return best[1], best[2]

    def grow_from_seed(seed):
        i0, j0, k0, q0 = seed
        new_ids = []
        for q2 in _sphere_triple_points(
            centers[i0], float(expanded[i0]),
            centers[j0], float(expanded[j0]),
            centers[k0], float(expanded[k0]),
        ):
            q2 = np.asarray(q2, dtype=float)
            if _probe_collides(q2, centers, expanded, (i0, j0, k0)):
                continue
            fid = add_face(i0, j0, k0, q2)
            if fid is not None:
                new_ids.append(fid)
        if not new_ids:
            fid = add_face(i0, j0, k0, q0)
            if fid is not None:
                new_ids.append(fid)
        queue = deque()
        treated = set()
        for fid in new_ids:
            fi, fj, fk, fq = faces[fid]
            for a, b, c in ((fi, fj, fk), (fj, fk, fi), (fk, fi, fj)):
                queue.append((a, b, c, fq, fid))
        while queue:
            i, j, third, q, _fid = queue.popleft()
            ek = _rs_edge_key(i, j)
            tkey = (ek[0], ek[1], _q_key(q))
            if tkey in treated:
                continue
            treated.add(tkey)
            nxt = roll_next(i, j, q, third)
            if nxt is None:
                continue
            k2, q2 = nxt
            fid2 = add_face(i, j, k2, q2)
            if fid2 is None:
                continue
            fi, fj, fk, fq = faces[fid2]
            for a, b, c in ((fi, fj, fk), (fj, fk, fi), (fk, fi, fj)):
                if _rs_edge_key(a, b) == ek:
                    continue
                queue.append((a, b, c, fq, fid2))

    remaining = set(range(n))
    while remaining:
        seed = None
        for axis in (0, 1, 2):
            seed = leftmost_seed(remaining, axis)
            if seed is not None:
                break
        if seed is None:
            break
        grow_from_seed(seed)
        remaining = set(range(n)) - assigned

    free_edges = []
    for i in range(n):
        for j in neighbor_atoms(i):
            if j <= i:
                continue
            ek = (i, j)
            if len(edge_faces.get(ek, ())) >= 1:
                continue
            circle = _sphere_intersection_circle(
                centers[i], float(expanded[i]), centers[j], float(expanded[j]),
            )
            if circle is None:
                continue
            origin, normal, rad = circle
            u, v = _orthonormal_axes(normal)
            exposed = False
            for s in range(16):
                q = _circle_point(origin, rad, u, v, 2.0 * math.pi * s / 16.0)
                if not _probe_collides(q, centers, expanded, (i, j)):
                    exposed = True
                    break
            if exposed:
                free_edges.append(ek)
                assigned.update((i, j))

    free_vertices = []
    for i in range(n):
        if i in assigned:
            continue
        if not neighbor_atoms(i):
            free_vertices.append(i)

    return {
        "faces": faces,
        "edge_faces": edge_faces,
        "free_edges": free_edges,
        "free_vertices": free_vertices,
    }


def _neighbor_pole(index, centers, expanded):
    center = np.asarray(centers[index], dtype=float)
    pole = np.zeros(3, dtype=float)
    n = int(centers.shape[0])
    for j in range(n):
        if j == index:
            continue
        if float(np.linalg.norm(center - centers[j])) < float(expanded[index]) + float(expanded[j]):
            pole += center - np.asarray(centers[j], dtype=float)
    if float(np.linalg.norm(pole)) < 1e-8:
        return np.array((0.0, 0.0, 1.0))
    return _unit_vec(pole)


def _points_near(a, b, tol=1e-5):
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))) <= float(tol)


def _torus_succ(rails, origin, rad, u, v, centers, expanded, pair, tau, dtheta):
    """Index of the next exposed sample along the probe circle, or -1.

    Adjacent live samples (including exact triples) are stitched across the
    small gaps left when uniforms near a triple are dropped. Long buried
    arcs stay unstitched.
    """
    n_s = len(rails)
    succ = [-1] * n_s
    i, j = pair
    live = [s for s in range(n_s) if rails[s] is not None]
    if len(live) < 2:
        return succ
    gap = max(float(dtheta) * 1.35, 1e-6)
    for idx, s in enumerate(live):
        nxt = live[(idx + 1) % len(live)]
        theta_a = rails[s][0]
        theta_b = rails[nxt][0]
        dth = (theta_b - theta_a) % tau
        if dth < 1e-12:
            continue
        qm = _circle_point(origin, rad, u, v, theta_a + 0.5 * dth)
        if dth > gap and _probe_collides(qm, centers, expanded, (i, j)):
            continue
        succ[s] = nxt
    return succ


def _rail_is_triple(rail):
    return bool(rail is not None and len(rail) > 3 and rail[3])


def _keep_spaced_rails(chain, closed, rails, min_dist, max_dtheta=0.15):
    """Drop uniforms clustered in both space *and* angle.

    Contact-circle Euclidean distance alone collapses a small circle to a
    handful of vertices, which turns the torus into a few huge flat facets.
    Samples that are still spaced in ``theta`` stay, even on a tiny circle.
    """
    n = len(chain)
    if n <= 2:
        return list(chain)
    min_dist = float(min_dist)
    max_dtheta = float(max_dtheta)

    def contact(s):
        return np.asarray(rails[s][2][0], dtype=float)

    def theta_of(s):
        return float(rails[s][0])

    def clustered(a, b):
        dist = float(np.linalg.norm(contact(a) - contact(b)))
        return dist < min_dist and _ang_dist(theta_of(a), theta_of(b)) < max_dtheta

    must = set()
    for i, s in enumerate(chain):
        if _rail_is_triple(rails[s]) or ((not closed) and i in (0, n - 1)):
            must.add(i)

    keep_idx = []
    last_i = None
    for i, s in enumerate(chain):
        if last_i is None or i in must or not clustered(chain[last_i], s):
            keep_idx.append(i)
            last_i = i

    if closed and n >= 8:
        min_keep = max(6, n // 3)
    elif closed:
        min_keep = 3
    else:
        min_keep = 2
    changed = True
    while changed and len(keep_idx) > min_keep:
        changed = False
        n_k = len(keep_idx)
        for pos in range(n_k):
            i = keep_idx[pos]
            if i in must:
                continue
            prev_i = keep_idx[pos - 1]
            has_next = closed or pos + 1 < n_k
            next_i = keep_idx[(pos + 1) % n_k] if has_next else None
            close_prev = clustered(chain[prev_i], chain[i])
            close_nxt = False if next_i is None else clustered(chain[i], chain[next_i])
            if close_prev or close_nxt:
                keep_idx.pop(pos)
                changed = True
                break

    if closed and len(keep_idx) > min_keep:
        if clustered(chain[keep_idx[0]], chain[keep_idx[-1]]):
            if keep_idx[-1] not in must:
                keep_idx.pop()
            elif keep_idx[0] not in must:
                keep_idx.pop(0)

    if len(keep_idx) < min_keep:
        return list(chain)
    return [chain[i] for i in keep_idx]


def _decimate_torus_succ(rails, succ, min_dist=0.08):
    """Drop clustered non-triple rails from both torus strips and contact loops.

    Hopping a successor over a close sample without removing that sample left it
    in the cap polyline, so the VDW patch put a vertex in the middle of a torus
    edge (T-junction cracks and sliver triangles).
    """
    n_s = len(rails)
    new_succ = [-1] * n_s
    min_dist = float(min_dist)
    for chain, closed in _torus_chains(rails, succ):
        kept = _keep_spaced_rails(chain, closed, rails, min_dist)
        if len(kept) < 2:
            continue
        n_k = len(kept)
        limit = n_k if closed else n_k - 1
        for i in range(limit):
            new_succ[kept[i]] = kept[(i + 1) % n_k]
    return new_succ


def _torus_chains(rails, succ):
    n_s = len(rails)
    pred = [-1] * n_s
    for s, nxt in enumerate(succ):
        if nxt >= 0:
            pred[nxt] = s
    used = [False] * n_s
    chains = []
    for s0 in range(n_s):
        if used[s0] or rails[s0] is None:
            continue
        cyclic = False
        cursor = s0
        for _ in range(n_s + 1):
            if pred[cursor] < 0:
                break
            cursor = pred[cursor]
            if cursor == s0:
                cyclic = True
                break
        if pred[s0] >= 0 and not cyclic:
            continue
        chain = []
        s = s0
        for _ in range(n_s + 1):
            if rails[s] is None or used[s]:
                break
            used[s] = True
            chain.append(s)
            nxt = succ[s]
            if nxt < 0 or nxt == s0:
                break
            s = nxt
        if len(chain) >= 2:
            closed = succ[chain[-1]] == chain[0]
            chains.append((chain, closed))
    return chains


def _assemble_loops(parts, tol=0.05):
    """Join open contact polylines at shared triple points into closed loops.

    Leftover open arcs are *not* closed with a chord: that chord cuts through
    the groove and shows up as a flat face between atoms.
    """
    loops = []
    opens = []
    for part in parts:
        if isinstance(part, tuple):
            pts, closed = part
        else:
            pts, closed = part, False
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        if pts.shape[0] < 2:
            continue
        duplicate_ends = pts.shape[0] >= 2 and _points_near(pts[0], pts[-1], 1e-5)
        near_closed = pts.shape[0] >= 3 and _points_near(pts[0], pts[-1], tol)
        if closed:
            if duplicate_ends:
                pts = pts[:-1]
            if pts.shape[0] >= 3:
                loops.append(pts)
            continue
        if duplicate_ends or near_closed:
            if duplicate_ends:
                pts = pts[:-1]
            if pts.shape[0] >= 3:
                loops.append(pts)
            continue
        opens.append(pts)
    merged = True
    while merged and len(opens) > 1:
        merged = False
        for i in range(len(opens)):
            a = opens[i]
            for j in range(len(opens)):
                if i == j:
                    continue
                b = opens[j]
                if _points_near(a[-1], b[0], tol):
                    opens[i] = np.vstack((a, b[1:]))
                    opens.pop(j)
                    merged = True
                    break
                if _points_near(a[-1], b[-1], tol):
                    opens[i] = np.vstack((a, b[-2::-1]))
                    opens.pop(j)
                    merged = True
                    break
                if _points_near(a[0], b[-1], tol):
                    opens[i] = np.vstack((b, a[1:]))
                    opens.pop(j)
                    merged = True
                    break
                if _points_near(a[0], b[0], tol):
                    opens[i] = np.vstack((np.flip(b, axis=0), a[1:]))
                    opens.pop(j)
                    merged = True
                    break
            if merged:
                break
    for pts in opens:
        if pts.shape[0] >= 3 and _points_near(pts[0], pts[-1], tol):
            if _points_near(pts[0], pts[-1], 1e-5):
                pts = pts[:-1]
            if pts.shape[0] >= 3:
                loops.append(pts)
    return loops


def _triple_vdw_contacts(index, centers, probe, triples):
    out = []
    for i, j, k, q in triples:
        if index not in (i, j, k):
            continue
        out.append(np.asarray(q, dtype=float) + float(probe) * _unit_vec(centers[index] - q))
    if not out:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(out, dtype=float)


def _snap_polyline_ends(pts, snap_pts, tol=0.35):
    pts = np.asarray(pts, dtype=float)
    if pts.shape[0] < 2 or snap_pts is None or len(snap_pts) == 0:
        return pts
    snap_pts = np.asarray(snap_pts, dtype=float).reshape(-1, 3)
    out = np.array(pts, copy=True)

    def _snap_one(which):
        delta = snap_pts - out[which]
        dist = np.linalg.norm(delta, axis=1)
        k = int(np.argmin(dist))
        if float(dist[k]) <= float(tol):
            out[which] = snap_pts[k]

    _snap_one(0)
    _snap_one(-1)
    return out


def _loop_n_rings(pole_dir, dirs, frequency):
    """Radial rings so spokes are about as long as the boundary edges."""
    n_k = len(dirs)
    min_rings = max(3, int(frequency) + 1)
    if n_k < 3:
        return min_rings
    pole_dir = _unit_vec(pole_dir)
    omega = 0.0
    step = 0.0
    for k in range(n_k):
        omega = max(omega, math.acos(float(np.clip(np.dot(pole_dir, dirs[k]), -1.0, 1.0))))
        step += math.acos(float(np.clip(np.dot(dirs[k], dirs[(k + 1) % n_k]), -1.0, 1.0)))
    mean_step = step / float(n_k)
    rings = int(round(omega / max(mean_step, 1e-3)))
    return max(min_rings, min(16, rings))


def _saddle_n_phi(centers, expanded, n_theta):
    """Phi samples so torus quads are roughly square."""
    dtheta = 2.0 * math.pi / float(max(int(n_theta), 1))
    max_ang = 4.0 * dtheta
    n = int(centers.shape[0])
    for i in range(n):
        for j in range(i + 1, n):
            circle = _sphere_intersection_circle(
                centers[i], float(expanded[i]), centers[j], float(expanded[j]),
            )
            if circle is None:
                continue
            origin, axis, rad = circle
            u, v = _orthonormal_axes(axis)
            q = None
            for vec in (u, v, -u, -v):
                cand = origin + rad * vec
                if not _probe_collides(cand, centers, expanded, (i, j)):
                    q = cand
                    break
            if q is None:
                continue
            ui = _unit_vec(centers[i] - q)
            uj = _unit_vec(centers[j] - q)
            ang = math.acos(float(np.clip(np.dot(ui, uj), -1.0, 1.0)))
            if ang > max_ang:
                max_ang = ang
    return max(6, int(round(max_ang / max(dtheta, 1e-6))))


def _probe_triangle_tris(probe_center, probe, d0, d1, d2, n_phi):
    """Geodesic tessellation of a probe-sphere triangle; edges match torus slerps."""
    n = max(1, int(n_phi))
    q = np.asarray(probe_center, dtype=float)
    probe = float(probe)
    rows = []
    for i in range(n + 1):
        t = i / float(n)
        left = _slerp(d0, d2, t)
        right = _slerp(d1, d2, t)
        n_j = n - i
        row = []
        for j in range(n_j + 1):
            s = 0.0 if n_j == 0 else j / float(n_j)
            row.append(q + probe * _slerp(left, right, s))
        rows.append(row)
    rows[0] = [q + probe * _slerp(d0, d1, j / float(n)) for j in range(n + 1)]
    tris = []
    for i in range(n):
        for j in range(n - i):
            a, b, c = rows[i][j], rows[i][j + 1], rows[i + 1][j]
            centroid = (a + b + c) / 3.0
            tris.append(_orient((a, b, c), q - centroid))
            if j < n - i - 1:
                d = rows[i + 1][j + 1]
                centroid = (b + d + c) / 3.0
                tris.append(_orient((b, d, c), q - centroid))
    return tris


def _ses_vertex_normals(vertices, faces, centers, vdw, expanded, circles=None, neighbors=None, triples=None):
    """Connolly outward normals from the generating probe, with no cap/torus switch.

    Every SES point is on a probe sphere: ``n = (q - v) / |q - v|``. Caps use the
    owner atom's radial SAS point as ``q``; saddles and reentrants use the
    closest probe-center circle or triple. Those ``q`` coincide at a contact
    circle, so a hard ``sdf`` threshold is not needed and would crease speculars.
    The C1 join is left unsmoothed: averaging across it flattens curvature
    into an indented band that wireframe does not show.
    """
    if vertices.shape[0] == 0:
        return np.zeros((0, 3), dtype=float)
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    vdw = np.asarray(vdw, dtype=float).reshape(-1)
    expanded = np.asarray(expanded, dtype=float).reshape(-1)
    probe = float(np.mean(np.maximum(expanded - vdw, 1e-6)))
    owners = _sas_owners(vertices, centers, vdw)
    dirs = vertices - centers[owners]
    length = np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    dirs = dirs / length
    q = centers[owners] + expanded[owners][:, None] * dirs
    err = np.abs(np.linalg.norm(vertices - q, axis=1) - probe)
    if probe > 1e-8:
        if circles is None:
            circles = _intersection_circles(centers, expanded)
        if circles:
            _d_c, q_c = _distance_to_circles(vertices, circles, centers, expanded)
            err_c = np.abs(np.linalg.norm(vertices - q_c, axis=1) - probe)
            err_c = np.where(np.isfinite(err_c), err_c, np.inf)
            use_c = err_c < err
            q = np.where(use_c[:, None], q_c, q)
            err = np.minimum(err, err_c)
        trip = np.zeros((0, 3), dtype=float)
        if triples is not None:
            trip = np.asarray(triples, dtype=float).reshape(-1, 3)
        if trip.shape[0]:
            _d_t, q_t = _distance_to_vertices(vertices, trip)
            err_t = np.abs(np.linalg.norm(vertices - q_t, axis=1) - probe)
            err_t = np.where(np.isfinite(err_t), err_t, np.inf)
            use_t = err_t < err
            q = np.where(use_t[:, None], q_t, q)
            err = np.minimum(err, err_t)
    delta = q - vertices
    ln = np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-12)
    normals = delta / ln
    bad = (~np.isfinite(normals).all(axis=1)) | (err > 0.5)
    if np.any(bad):
        normals = np.array(normals, copy=True)
        normals[bad] = dirs[bad]
    finite = np.isfinite(normals).all(axis=1)
    if not np.all(finite):
        geom = _face_normals(vertices, faces)
        normals = np.array(normals, copy=True)
        normals[~finite] = geom[~finite]
    return normals


def _refine_phong_faces(
    vertices,
    faces,
    normals,
    centers,
    expanded,
    probe,
    circles=None,
    neighbors=None,
    triples=None,
    min_dot=0.92,
    max_level=3,
    max_vertices=120000,
):
    """Split faces whose vertex normals span too much for CGO Phong.

    PyMOL interpolates the three corner normals in the fragment shader. When
    those vectors are ~40° apart, the lerp dips out of a tight specular lobe
    and shows up as a dark diamond with a hard triangle edge. Midpoints are
    lifted onto the sphere implied by the endpoints so the split stays on the
    Connolly patch.
    """
    from .field_sample import project_edge_midpoint

    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    faces = np.asarray(faces, dtype=int).reshape(-1, 3)
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    if faces.shape[0] == 0 or vertices.shape[0] < 3:
        return vertices, faces, normals
    verts = vertices.tolist()
    norms = normals.tolist()
    faces_list = [tuple(int(i) for i in tri) for tri in faces]
    min_dot = float(min_dot)
    max_vertices = max(int(max_vertices), int(vertices.shape[0]))
    skinny_aspect = 2.8

    def _key(i, j):
        return (i, j) if i < j else (j, i)

    for _ in range(max(0, int(max_level))):
        if len(verts) >= max_vertices:
            break
        nrm = np.asarray(norms, dtype=float)
        lengths = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm = nrm / np.maximum(lengths, 1e-18)
        marked = set()
        for a, b, c in faces_list:
            pairs = ((a, b), (b, c), (c, a))
            dots = [float(np.dot(nrm[i], nrm[j])) for i, j in pairs]
            if min(dots) >= min_dot:
                continue
            edge_len = [
                float(np.linalg.norm(np.subtract(verts[i], verts[j])))
                for i, j in pairs
            ]
            longest = max(edge_len)
            shortest = max(min(edge_len), 1e-18)
            skinny = (longest / shortest) > skinny_aspect
            for (i, j), cdot, length in zip(pairs, dots, edge_len):
                if cdot >= min_dot:
                    continue
                if length < 0.08:
                    continue
                if skinny and length < 0.70 * longest:
                    continue
                marked.add(_key(i, j))
        if not marked:
            break
        mids = {}
        for i, j in marked:
            if len(verts) >= max_vertices and _key(i, j) not in mids:
                continue
            point, normal = project_edge_midpoint(
                verts[i], verts[j], norms[i], norms[j],
            )
            mids[_key(i, j)] = len(verts)
            verts.append(np.asarray(point, dtype=float).reshape(3).tolist())
            ln = float(np.linalg.norm(normal))
            if ln > 1e-18:
                normal = np.asarray(normal, dtype=float).reshape(3) / ln
            else:
                normal = np.asarray(norms[i], dtype=float).reshape(3)
            norms.append(normal.tolist())
        new_faces = []
        for a, b, c in faces_list:
            _subdivide_triangle(
                a, b, c,
                mids.get(_key(a, b)),
                mids.get(_key(b, c)),
                mids.get(_key(c, a)),
                new_faces,
            )
        faces_list = new_faces

    vertices = np.asarray(verts, dtype=float).reshape(-1, 3)
    faces = np.asarray(faces_list, dtype=int).reshape(-1, 3)
    vdw = np.maximum(np.asarray(expanded, dtype=float).reshape(-1) - float(probe), 1e-6)
    normals = _ses_vertex_normals(
        vertices, faces, centers, vdw, expanded,
        circles=circles, neighbors=neighbors, triples=triples,
    )
    return vertices, faces, normals


def _resample_dirs(dirs, count):
    n_k = len(dirs)
    count = max(1, int(count))
    if count >= n_k:
        return list(dirs)
    out = []
    for k in range(count):
        t = k * n_k / float(count)
        i0 = int(math.floor(t)) % n_k
        i1 = (i0 + 1) % n_k
        frac = t - math.floor(t)
        out.append(_slerp(dirs[i0], dirs[i1], frac))
    return out


def _interp_loop_dirs(dirs, count):
    """Resample a closed loop of unit directions to ``count`` vertices."""
    n_k = len(dirs)
    count = max(3, int(count))
    if count == n_k:
        return [np.asarray(d, dtype=float) for d in dirs]
    out = []
    for k in range(count):
        t = (k * n_k) / float(count)
        i0 = int(math.floor(t)) % n_k
        i1 = (i0 + 1) % n_k
        frac = t - math.floor(t)
        out.append(_slerp(dirs[i0], dirs[i1], frac))
    return out


def _zip_spherical_annulus(inner, outer, center):
    """Zip two spherical loops with extra geodesic rings when the gap is wide.

    MSMS algorithm 4 asks for zipper edges of comparable length to the torus
    polyline. A single strip across a large cap produces interior chords.
    """
    inner = np.asarray(inner, dtype=float).reshape(-1, 3)
    outer = np.asarray(outer, dtype=float).reshape(-1, 3)
    if inner.shape[0] < 2 or outer.shape[0] < 2:
        return []
    center = np.asarray(center, dtype=float).reshape(3)
    if np.dot(_loop_area_normal(inner, center), _loop_area_normal(outer, center)) < 0.0:
        inner = inner[::-1]
    inner = _align_loop_start(inner, outer)
    radius = float(np.mean(np.linalg.norm(outer - center, axis=1)))
    inner_d = [_unit_vec(p - center) for p in inner]
    outer_d = [_unit_vec(p - center) for p in outer]
    n_out = len(outer_d)
    edge = 0.0
    for k in range(n_out):
        edge += math.acos(float(np.clip(np.dot(outer_d[k], outer_d[(k + 1) % n_out]), -1.0, 1.0)))
    edge /= float(n_out)
    aligned_inner = _interp_loop_dirs(inner_d, n_out)
    arc = 0.0
    for k in range(n_out):
        arc += math.acos(float(np.clip(np.dot(aligned_inner[k], outer_d[k]), -1.0, 1.0)))
    arc /= float(n_out)
    n_div = max(1, int(round(arc / max(edge, 1e-3))))
    n_div = min(n_div, 12)
    prev = inner
    tris = []
    for i in range(1, n_div + 1):
        t = i / float(n_div)
        if i == n_div:
            nxt = outer
        else:
            nxt = np.asarray(
                [center + radius * _slerp(aligned_inner[k], outer_d[k], t) for k in range(n_out)],
                dtype=float,
            )
        tris.extend(_zip_closed_loops(prev, nxt, center))
        prev = nxt
    return tris


def _zip_closed_loops(inner, outer, hint_origin):
    """Triangle strip between two closed loops (inner may be coarser)."""
    inner = [np.asarray(p, dtype=float) for p in inner]
    outer = [np.asarray(p, dtype=float) for p in outer]
    na, nb = len(inner), len(outer)
    if na < 2 or nb < 2:
        return []
    origin = np.asarray(hint_origin, dtype=float)
    if na == nb:
        tris = []
        for i in range(na):
            i1 = (i + 1) % na
            t0 = (inner[i], inner[i1], outer[i])
            t1 = (inner[i1], outer[i], outer[i1])
            c0 = (t0[0] + t0[1] + t0[2]) / 3.0
            c1 = (t1[0] + t1[1] + t1[2]) / 3.0
            tris.append(_orient(t0, c0 - origin))
            tris.append(_orient(t1, c1 - origin))
        return tris
    ia = ib = 0
    steps_a = steps_b = 0
    tris = []
    while steps_a < na or steps_b < nb:
        a1 = (ia + 1) % na
        b1 = (ib + 1) % nb
        frac_a = (steps_a + 1) / float(na)
        frac_b = (steps_b + 1) / float(nb)
        if steps_a < na and (steps_b >= nb or frac_a <= frac_b):
            tri = (inner[ia], inner[a1], outer[ib])
            ia = a1
            steps_a += 1
        else:
            tri = (inner[ia], outer[ib], outer[b1])
            ib = b1
            steps_b += 1
        centroid = (tri[0] + tri[1] + tri[2]) / 3.0
        tris.append(_orient(tri, centroid - origin))
    return tris


def _loop_area_normal(pts, center):
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    center = np.asarray(center, dtype=float).reshape(3)
    normal = np.zeros(3, dtype=float)
    n = int(pts.shape[0])
    for i in range(n):
        normal += np.cross(pts[i] - center, pts[(i + 1) % n] - center)
    return normal


def _align_loop_start(inner, outer):
    inner = np.asarray(inner, dtype=float).reshape(-1, 3)
    outer0 = np.asarray(outer[0], dtype=float).reshape(3)
    if inner.shape[0] == 0:
        return inner
    k = int(np.argmin(np.linalg.norm(inner - outer0.reshape(1, 3), axis=1)))
    if k == 0:
        return inner
    return np.vstack((inner[k:], inner[:k]))


def _spherical_in_loop(unit, loop_dirs, interior=None):
    """Keep template directions inside a convex spherical polygon."""
    loop_dirs = np.asarray(loop_dirs, dtype=float).reshape(-1, 3)
    unit = np.asarray(unit, dtype=float).reshape(-1, 3)
    n_k = int(loop_dirs.shape[0])
    if n_k < 2 or unit.shape[0] == 0:
        return np.ones((int(unit.shape[0]),), dtype=bool)
    if interior is None:
        interior = np.mean(loop_dirs, axis=0)
    if float(np.linalg.norm(interior)) < 1e-12:
        return np.ones((int(unit.shape[0]),), dtype=bool)
    interior = _unit_vec(interior)
    keep = np.ones((int(unit.shape[0]),), dtype=bool)
    for k in range(n_k):
        nrm = np.cross(loop_dirs[k], loop_dirs[(k + 1) % n_k])
        length = float(np.linalg.norm(nrm))
        if length < 1e-12:
            continue
        nrm = nrm / length
        if float(np.dot(nrm, interior)) < 0.0:
            nrm = -nrm
        keep &= np.dot(unit, nrm) >= -1e-6
    return keep


def _small_circle_cap_mask(unit, loop_dirs, interior, eps=1e-6):
    """Keep directions on the interior side of a planar contact circle.

    Contact loops are small circles. Great-circle half-spaces of the samples
    clip a >90° cap at the antipodal latitude and leave a folded Delaunay
    annulus down to the torus.
    """
    loop_dirs = np.asarray(loop_dirs, dtype=float).reshape(-1, 3)
    unit = np.asarray(unit, dtype=float).reshape(-1, 3)
    if loop_dirs.shape[0] < 3 or unit.shape[0] == 0:
        return np.ones((int(unit.shape[0]),), dtype=bool)
    interior = _unit_vec(interior)
    axis = np.mean(loop_dirs, axis=0)
    if float(np.linalg.norm(axis)) < 1e-12:
        return _spherical_in_loop(unit, loop_dirs, interior=interior)
    axis = _unit_vec(axis)
    dots = unit @ axis
    loop_dots = loop_dirs @ axis
    if float(np.dot(axis, interior)) > 0.0:
        return dots >= (float(np.min(loop_dots)) - eps)
    return dots <= (float(np.max(loop_dots)) + eps)


def _contact_cap_plane(center, r_vdw, loop_dirs, interior):
    """Plane of the contact circle: keep points with ``ge`` relative to offset."""
    loop_dirs = np.asarray(loop_dirs, dtype=float).reshape(-1, 3)
    center = np.asarray(center, dtype=float).reshape(3)
    r_vdw = float(r_vdw)
    interior = _unit_vec(interior)
    axis = _unit_vec(np.mean(loop_dirs, axis=0))
    loop_dots = loop_dirs @ axis
    if float(np.dot(axis, interior)) > 0.0:
        return axis, center, r_vdw * float(np.min(loop_dots)), True
    return axis, center, r_vdw * float(np.max(loop_dots)), False


def _snap_point_to_loop(point, loop_pts, tol=0.20):
    """Project a contact-circle point onto the torus polyline."""
    loop_pts = np.asarray(loop_pts, dtype=float).reshape(-1, 3)
    point = np.asarray(point, dtype=float).reshape(3)
    n_k = int(loop_pts.shape[0])
    if n_k == 0:
        return point
    dist = np.linalg.norm(loop_pts - point.reshape(1, 3), axis=1)
    k = int(np.argmin(dist))
    edge = float(np.linalg.norm(loop_pts[(k + 1) % n_k] - loop_pts[k]))
    prev_edge = float(np.linalg.norm(loop_pts[k] - loop_pts[(k - 1) % n_k]))
    vert_tol = max(0.02, 0.40 * min(edge, prev_edge))
    if float(dist[k]) <= vert_tol:
        return loop_pts[k]
    best = loop_pts[k]
    best_d = float(dist[k])
    for i in range(n_k):
        a = loop_pts[i]
        b = loop_pts[(i + 1) % n_k]
        span = b - a
        length2 = float(np.dot(span, span))
        if length2 < 1e-18:
            continue
        t = float(np.clip(np.dot(point - a, span) / length2, 0.0, 1.0))
        q = a + t * span
        d = float(np.linalg.norm(point - q))
        if d < best_d:
            best_d = d
            best = q
    if best_d <= float(tol):
        return best
    return point


def _sphere_arc_meet_cap_plane(a, b, nrm, origin, offset, center, r_vdw):
    """Intersection of a geodesic VDW edge with the contact plane, on the sphere."""
    center = np.asarray(center, dtype=float).reshape(3)
    nrm = np.asarray(nrm, dtype=float).reshape(3)
    origin = np.asarray(origin, dtype=float).reshape(3)
    r_vdw = float(r_vdw)
    offset = float(offset)
    da = _unit_vec(a - center)
    db = _unit_vec(b - center)
    edge_n = np.cross(da, db)
    length = float(np.linalg.norm(edge_n))
    if length < 1e-18:
        return None
    edge_n = edge_n / length
    d1 = offset + float(np.dot(nrm, center - origin))
    direction = np.cross(nrm, edge_n)
    denom = float(np.dot(direction, direction))
    if denom < 1e-18:
        return None
    u0 = (d1 * np.cross(edge_n, direction) + 0.0 * np.cross(direction, nrm)) / denom
    A = denom
    B = 2.0 * float(np.dot(u0, direction))
    C = float(np.dot(u0, u0)) - r_vdw * r_vdw
    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        return None
    sqrt_d = math.sqrt(disc)
    best = None
    for t in ((-B + sqrt_d) / (2.0 * A), (-B - sqrt_d) / (2.0 * A)):
        u = u0 + t * direction
        un = _unit_vec(u)
        if float(np.dot(np.cross(da, un), edge_n)) < -1e-7:
            continue
        if float(np.dot(np.cross(un, db), edge_n)) < -1e-7:
            continue
        best = center + r_vdw * un
        break
    return best


def _fan_world_tris(center, pts):
    center = np.asarray(center, dtype=float).reshape(3)
    pts = [np.asarray(p, dtype=float).reshape(3) for p in pts]
    if len(pts) < 3:
        return []
    tris = []
    for i in range(1, len(pts) - 1):
        tri = (pts[0], pts[i], pts[i + 1])
        geom = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        if float(np.linalg.norm(geom)) < 1e-16:
            continue
        centroid = (tri[0] + tri[1] + tri[2]) / 3.0
        tris.append(_orient(tri, centroid - center))
    return tris


def _contact_interior_dir(center, expanded_r, index, centers, expanded, loop_dirs):
    """Side of the contact loop whose SAS point is still exposed."""
    loop_dirs = np.asarray(loop_dirs, dtype=float).reshape(-1, 3)
    mean = np.mean(loop_dirs, axis=0)
    if float(np.linalg.norm(mean)) < 1e-12:
        mean = _neighbor_pole(index, centers, expanded)
    mean = _unit_vec(mean)

    def exposed(direction):
        point = np.asarray(center, dtype=float) + float(expanded_r) * direction
        return not _probe_collides(point, centers, expanded, (index,))

    if exposed(mean) and not exposed(-mean):
        return mean
    if exposed(-mean) and not exposed(mean):
        return -mean
    if exposed(mean):
        return mean
    if exposed(-mean):
        return -mean
    pole = _neighbor_pole(index, centers, expanded)
    q_pole = np.asarray(center, dtype=float) + float(expanded_r) * pole
    if _probe_collides(q_pole, centers, expanded, (index,)):
        return -pole
    return pole


def _manifold_disk(tris, n_boundary):
    """True if ``tris`` weld to a manifold disk with ``n_boundary`` rim edges."""
    if not tris:
        return False
    vertices, faces = _weld(tris, ndigits=5)
    faces = _unique_faces(faces)
    faces = _drop_degenerate_faces(vertices, faces)
    if faces.shape[0] == 0:
        return False
    count = {}
    for a, b, c in faces:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (u, v) if u < v else (v, u)
            count[key] = count.get(key, 0) + 1
    bound = 0
    for n in count.values():
        if n == 1:
            bound += 1
        elif n != 2:
            return False
    return bound == int(n_boundary)


def _polygon_edges_present(simplices, n_loop):
    """True if every consecutive loop edge appears in the triangulation."""
    simplices = np.asarray(simplices, dtype=int)
    n_loop = int(n_loop)
    if simplices.size == 0 or n_loop < 3:
        return False
    have = set()
    for a, b, c in simplices:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            if u > v:
                u, v = v, u
            have.add((u, v))
    for i in range(n_loop):
        u, v = i, (i + 1) % n_loop
        if u > v:
            u, v = v, u
        if (u, v) not in have:
            return False
    return True


def _inset_contact_dirs(loop_dirs, interior, step=0.10):
    """Move each loop direction ``step`` radians toward the contact pole."""
    interior = _unit_vec(interior)
    loop_dirs = np.asarray(loop_dirs, dtype=float).reshape(-1, 3)
    out = np.empty_like(loop_dirs)
    for i, direction in enumerate(loop_dirs):
        direction = _unit_vec(direction)
        ang = math.acos(float(np.clip(np.dot(direction, interior), -1.0, 1.0)))
        if ang < 1e-6:
            out[i] = direction
            continue
        t = min(0.45, float(step) / ang)
        out[i] = _unit_vec(_slerp(direction, interior, t))
    return out


def _inside_poly_2d(points, poly):
    """Even-odd ray test for 2D points vs a closed polyline."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    poly = np.asarray(poly, dtype=float).reshape(-1, 2)
    n = int(poly.shape[0])
    if n < 3 or points.shape[0] == 0:
        return np.zeros((int(points.shape[0]),), dtype=bool)
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros((int(points.shape[0]),), dtype=bool)
    x0 = poly[:, 0]
    y0 = poly[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)
    for i in range(n):
        yi0 = y0[i]
        yi1 = y1[i]
        span = (yi0 > y) != (yi1 > y)
        if not np.any(span):
            continue
        denom = yi1 - yi0
        if abs(float(denom)) < 1e-18:
            continue
        xhit = (x1[i] - x0[i]) * (y - yi0) / denom + x0[i]
        inside ^= span & (x < xhit)
    return inside


def _dirs_away_from_loop_edges(dirs, loop_dirs, min_dist=None):
    """Drop Steiner directions that sit on a rim chord (Delaunay sliver seeds)."""
    dirs = np.asarray(dirs, dtype=float).reshape(-1, 3)
    loop_dirs = np.asarray(loop_dirs, dtype=float).reshape(-1, 3)
    if dirs.shape[0] == 0 or loop_dirs.shape[0] < 2:
        return dirs
    keep = np.ones((int(dirs.shape[0]),), dtype=bool)
    n_loop = int(loop_dirs.shape[0])
    if min_dist is None:
        step = 0.0
        for k in range(n_loop):
            step += float(np.linalg.norm(loop_dirs[(k + 1) % n_loop] - loop_dirs[k]))
        min_dist = max(0.02, 0.30 * step / float(n_loop))
    min_dist = float(min_dist)
    for k in range(n_loop):
        a = loop_dirs[k]
        b = loop_dirs[(k + 1) % n_loop]
        span = b - a
        length2 = float(np.dot(span, span))
        if length2 < 1e-18:
            continue
        t = np.clip(((dirs - a) @ span) / length2, 0.0, 1.0)
        proj = a + t[:, None] * span
        dist = np.linalg.norm(dirs - proj, axis=1)
        keep &= dist >= min_dist
    if not np.any(keep):
        return dirs[:0]
    return dirs[keep]


def _delaunay_contact_tris(center, r_vdw, interior_dirs, loop_pts, pole=None):
    """Triangulate a spherical contact face so the rim is exactly the torus loop."""
    loop_pts = np.asarray(loop_pts, dtype=float).reshape(-1, 3)
    center = np.asarray(center, dtype=float).reshape(3)
    r_vdw = float(r_vdw)
    n_loop = int(loop_pts.shape[0])
    if n_loop < 3:
        return []
    loop_dirs = np.asarray([_unit_vec(p - center) for p in loop_pts], dtype=float)
    interior_dirs = np.asarray(interior_dirs, dtype=float).reshape(-1, 3)
    if pole is None:
        pole = np.mean(loop_dirs, axis=0)
    pole = _unit_vec(pole)
    u, v = _orthonormal_axes(pole)

    def stereo(dirs):
        z = dirs @ pole
        den = np.maximum(1.0 + z, 1e-8)
        return np.stack(((dirs @ u) / den, (dirs @ v) / den), axis=1)

    loop_xy = stereo(loop_dirs)
    if float(np.max(np.abs(loop_xy))) > 40.0:
        return []
    if interior_dirs.shape[0] > 0:
        lengths = np.linalg.norm(interior_dirs, axis=1, keepdims=True)
        interior_dirs = interior_dirs / np.maximum(lengths, 1e-12)
        close = np.max(interior_dirs @ loop_dirs.T, axis=1) > 0.9995
        interior_dirs = interior_dirs[~close]
        interior_dirs = _dirs_away_from_loop_edges(interior_dirs, loop_dirs)
    if interior_dirs.shape[0] > 0:
        int_xy = stereo(interior_dirs)
        interior_dirs = interior_dirs[_inside_poly_2d(int_xy, loop_xy)]
    if interior_dirs.shape[0] > 0:
        dirs = np.vstack((loop_dirs, interior_dirs))
    else:
        dirs = loop_dirs
    xy = stereo(dirs)
    if xy.shape[0] < 3:
        return []
    try:
        delaunay = Delaunay(xy)
    except Exception:
        return []
    simplices = np.asarray(delaunay.simplices, dtype=int)
    if simplices.shape[0] == 0:
        return []
    cents = (xy[simplices[:, 0]] + xy[simplices[:, 1]] + xy[simplices[:, 2]]) / 3.0
    keep = _inside_poly_2d(cents, loop_xy)
    simplices = simplices[keep]
    if simplices.shape[0] == 0:
        return []
    if not _polygon_edges_present(simplices, n_loop):
        return []
    if interior_dirs.shape[0] > 0:
        world = np.vstack((loop_pts, center.reshape(1, 3) + interior_dirs * r_vdw))
    else:
        world = loop_pts
    tris = []
    for a, b, c in simplices:
        tri = (world[int(a)], world[int(b)], world[int(c)])
        centroid = (tri[0] + tri[1] + tri[2]) / 3.0
        tris.append(_orient(tri, centroid - center))
    return tris


def _exposed_cap_steiner_dirs(center, expanded_r, index, centers, expanded, unit, loop_dirs, interior):
    """Geodesic directions inside the contact cap whose SAS sample is still exposed."""
    unit = np.asarray(unit, dtype=float).reshape(-1, 3)
    keep = _small_circle_cap_mask(unit, loop_dirs, interior, eps=-0.012)
    if not np.any(keep):
        keep = _small_circle_cap_mask(unit, loop_dirs, interior)
    dirs = unit[keep]
    if dirs.shape[0] == 0:
        return dirs
    center = np.asarray(center, dtype=float).reshape(3)
    expanded_r = float(expanded_r)
    out = []
    for direction in dirs:
        sas = center + expanded_r * direction
        if _probe_collides(sas, centers, expanded, (index,), eps=0.02):
            continue
        out.append(direction)
    if not out:
        return dirs[:0]
    return np.asarray(out, dtype=float)


def _template_contact_tris(center, r_vdw, expanded_r, index, centers, expanded, loop, unit, faces):
    """Algorithm 4: contact cap whose rim is the torus loop.

    Plane-clip of a dense geodesic ico puts extra vertices on torus chords.
    Those T-junctions light as a jagged stitch. Delaunay of exposed ico
    vertices against the exact contact polyline shares the torus rim; clip
    is the fallback when the projected loop is not a convex disk.
    """
    pts = np.asarray(loop, dtype=float).reshape(-1, 3)
    if pts.shape[0] >= 2 and _points_near(pts[0], pts[-1], 1e-5):
        pts = pts[:-1]
    if pts.shape[0] < 3:
        return []
    center = np.asarray(center, dtype=float).reshape(3)
    unit = np.asarray(unit, dtype=float)
    faces = np.asarray(faces, dtype=int).reshape(-1, 3)
    loop_dirs = np.asarray([_unit_vec(p - center) for p in pts], dtype=float)
    interior = _contact_interior_dir(center, expanded_r, index, centers, expanded, loop_dirs)
    vdw = np.maximum(np.asarray(expanded, dtype=float) - (float(expanded_r) - float(r_vdw)), 1e-6)
    nrm, origin, offset, _ge = _contact_cap_plane(center, r_vdw, loop_dirs, interior)
    world = origin.reshape(1, 3) + unit * float(r_vdw)

    def _on_sphere(tris):
        for tri in tris:
            centroid = (tri[0] + tri[1] + tri[2]) / 3.0
            if float(signed_distance(centroid.reshape(1, 3), centers, vdw)[0]) < -0.12:
                return False
        return True

    steiner = _exposed_cap_steiner_dirs(
        center, expanded_r, index, centers, expanded, unit, loop_dirs, interior,
    )
    patch = _delaunay_contact_tris(center, r_vdw, steiner, pts, pole=interior)
    if patch and _manifold_disk(patch, int(pts.shape[0])) and _on_sphere(patch):
        return patch

    tris = []
    verts_inside = _small_circle_cap_mask(unit, loop_dirs, interior)
    edge_clip = {}

    def clipped_edge(i, j):
        key = (i, j) if i < j else (j, i)
        if key in edge_clip:
            return edge_clip[key]
        if bool(verts_inside[i]) == bool(verts_inside[j]):
            edge_clip[key] = None
            return None
        hit = _sphere_arc_meet_cap_plane(
            world[i], world[j], nrm, origin, offset, center, r_vdw,
        )
        if hit is None:
            a, b = world[i], world[j]
            va = float(np.dot(nrm, a - origin))
            vb = float(np.dot(nrm, b - origin))
            denom = vb - va
            if abs(denom) > 1e-18:
                t = min(1.0, max(0.0, (offset - va) / denom))
                point = a + t * (b - a)
                hit = center + float(r_vdw) * _unit_vec(point - center)
        if hit is not None:
            hit = _snap_point_to_loop(hit, pts)
        edge_clip[key] = hit
        return hit

    for a, b, c in faces:
        a, b, c = int(a), int(b), int(c)
        corners = (a, b, c)
        inside = [bool(verts_inside[i]) for i in corners]
        n_in = sum(inside)
        if n_in == 3:
            poly = [world[a], world[b], world[c]]
        elif n_in == 0:
            continue
        else:
            poly = []
            for k in range(3):
                i0 = corners[k]
                i1 = corners[(k + 1) % 3]
                if inside[k]:
                    poly.append(world[i0])
                if inside[k] != inside[(k + 1) % 3]:
                    hit = clipped_edge(i0, i1)
                    if hit is not None:
                        poly.append(hit)
            if len(poly) < 3:
                continue
        for tri in _fan_world_tris(center, poly):
            centroid = (tri[0] + tri[1] + tri[2]) / 3.0
            if float(signed_distance(centroid.reshape(1, 3), centers, vdw)[0]) < -0.12:
                continue
            if n_in == 3:
                direction = _unit_vec(centroid - center)
                sas = center + float(expanded_r) * direction
                if _probe_collides(sas, centers, expanded, (index,), eps=0.02):
                    continue
            tris.append(tri)
    if tris:
        return tris
    keep = _small_circle_cap_mask(unit, loop_dirs, interior)
    if not np.any(keep):
        keep = np.ones((int(unit.shape[0]),), dtype=bool)
    patch = _delaunay_contact_tris(center, r_vdw, unit[keep], pts, pole=interior)
    if not patch:
        return []
    if not _manifold_disk(patch, int(pts.shape[0])):
        return []
    if not _on_sphere(patch):
        return []
    return patch


def _fan_vdw_loop(center, r_vdw, expanded_r, index, centers, expanded, loop, frequency):
    pts = np.asarray(loop, dtype=float).reshape(-1, 3)
    if pts.shape[0] >= 2 and _points_near(pts[0], pts[-1], 1e-5):
        pts = pts[:-1]
    n_k = int(pts.shape[0])
    if n_k < 3:
        return []
    dirs = [_unit_vec(pts[k] - center) for k in range(n_k)]
    pole_dir = _neighbor_pole(index, centers, expanded)
    q_pole = center + float(expanded_r) * pole_dir
    if _probe_collides(q_pole, centers, expanded, (index,)):
        pole_dir = -pole_dir
    mean = np.mean(np.asarray(dirs, dtype=float), axis=0)
    mean_n = float(np.linalg.norm(mean))
    if mean_n > 0.15:
        fan_pole = mean / mean_n
        q_fan = center + float(expanded_r) * fan_pole
        if not _probe_collides(q_fan, centers, expanded, (index,)):
            pole_dir = fan_pole
    n_rings = _loop_n_rings(pole_dir, dirs, frequency)
    n1 = n_k if n_k <= 8 else 6
    counts = [1]
    for r in range(1, n_rings):
        count = max(n1, int(round(n1 + (n_k - n1) * r / float(n_rings))))
        counts.append(min(n_k, count))
    counts.append(n_k)
    for r in range(1, n_rings):
        counts[r] = max(counts[r], counts[r - 1] if counts[r - 1] > 1 else n1)
        counts[r] = min(counts[r], n_k)
    if n_rings >= 1:
        counts[n_rings - 1] = n_k
    grid = []
    pole = center + r_vdw * pole_dir
    grid.append([pole])
    for r in range(1, n_rings):
        t = r / float(n_rings)
        sample = _resample_dirs(dirs, counts[r])
        grid.append([center + r_vdw * _slerp(pole_dir, sample[k], t) for k in range(len(sample))])
    grid.append([np.asarray(pts[k], dtype=float) for k in range(n_k)])
    tris = []
    first = grid[1]
    for k in range(len(first)):
        tris.append(_orient((pole, first[k], first[(k + 1) % len(first)]), pole - center))
    for r in range(1, n_rings):
        tris.extend(_zip_closed_loops(grid[r], grid[r + 1], center))
    return tris


def _pinch_radial_row(row, centers, vdw, singular):
    """Algorithm 3 radial: snap spindle-torus samples onto the RS-edge cusps."""
    row = np.asarray(row, dtype=float)
    sdf = signed_distance(row, centers, vdw)
    out = np.array(row, copy=True)
    s0 = np.asarray(singular[0], dtype=float)
    s1 = np.asarray(singular[1], dtype=float)
    for i, val in enumerate(sdf):
        if i == 0 or i == int(out.shape[0]) - 1:
            continue
        if float(val) < -0.02:
            d0 = float(np.linalg.norm(out[i] - s0))
            d1 = float(np.linalg.norm(out[i] - s1))
            out[i] = s0 if d0 <= d1 else s1
    return out


def _ses_torus_tris(
    centers, expanded, probe, n_theta, n_phi, triples=None, pairs=None, eat_probes=None, vdw=None,
):
    """Algorithm 2: toric reentrant faces on reduced-surface edges.

    Algorithm 3: pinch radial spindle tori onto RS-edge cusps, and drop
    rolling samples eaten by another jammed probe (nonradial).
    """
    tris = []
    n = int(centers.shape[0])
    contacts = [[] for _ in range(n)]
    if float(probe) <= 1e-8:
        return tris, contacts
    if triples is None:
        triples = _iter_probe_triples(centers, expanded)
    if vdw is None:
        vdw = np.maximum(np.asarray(expanded, dtype=float) - float(probe), 1e-6)
    pair_qs = {}
    for i, j, k, q in triples:
        pair_qs.setdefault(_rs_edge_key(i, j), []).append((k, q))
        pair_qs.setdefault(_rs_edge_key(i, k), []).append((j, q))
        pair_qs.setdefault(_rs_edge_key(j, k), []).append((i, q))
    if pairs is None:
        pair_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        pair_list = [_rs_edge_key(i, j) for i, j in pairs]
    tau = 2.0 * math.pi
    n_theta = max(3, int(n_theta))
    n_phi = max(1, int(n_phi))
    dtheta = tau / float(n_theta)
    probe = float(probe)
    for i, j in pair_list:
        circle = _sphere_intersection_circle(
            centers[i], float(expanded[i]), centers[j], float(expanded[j]),
        )
        if circle is None:
            continue
        origin, axis, rad = circle
        u, v = _orthonormal_axes(axis)
        singular = None
        if float(rad) + 1e-8 < probe:
            height = math.sqrt(max(probe * probe - float(rad) * float(rad), 0.0))
            ax = _unit_vec(axis)
            singular = (origin - height * ax, origin + height * ax)
        samples = []
        triple_thetas = []
        for third, q in pair_qs.get((i, j), ()):
            q = np.asarray(q, dtype=float)
            theta = _circle_theta(q, origin, u, v)
            samples.append((theta, q, (i, j, int(third))))
            triple_thetas.append(theta)
        for s in range(n_theta):
            theta = tau * s / float(n_theta)
            if any(_ang_dist(theta, tt) < 0.45 * dtheta for tt in triple_thetas):
                continue
            samples.append((theta, _circle_point(origin, rad, u, v, theta), None))
        samples.sort(key=lambda item: item[0])
        rails = []
        for theta, q, triple_atoms in samples:
            skip = triple_atoms if triple_atoms is not None else (i, j)
            if _probe_collides(q, centers, expanded, skip):
                rails.append(None)
                continue
            row = _torus_row(q, centers[i], centers[j], probe, n_phi)
            if singular is not None:
                row = _pinch_radial_row(row, centers, vdw, singular)
            if triple_atoms is None and _ses_eaten(row[len(row) // 2], eat_probes, q, probe):
                rails.append(None)
                continue
            rails.append((theta, q, row, triple_atoms is not None))
        succ = _torus_succ(
            rails, origin, rad, u, v, centers, expanded, (i, j), tau, dtheta,
        )
        succ = _decimate_torus_succ(rails, succ)
        for s, nxt in enumerate(succ):
            if nxt < 0:
                continue
            theta_a, qa, ra = rails[s][:3]
            theta_b, qb, rb = rails[nxt][:3]
            q_mid = 0.5 * (qa + qb)
            for p in range(n_phi):
                for tri in ((ra[p], rb[p], rb[p + 1]), (ra[p], rb[p + 1], ra[p + 1])):
                    centroid = (tri[0] + tri[1] + tri[2]) / 3.0
                    tris.append(_orient(tri, q_mid - centroid))
        for chain, closed in _torus_chains(rails, succ):
            pts_i = np.stack([rails[idx][2][0] for idx in chain])
            pts_j = np.stack([rails[idx][2][-1] for idx in chain])
            contacts[i].append((pts_i, closed))
            contacts[j].append((pts_j, closed))
    return tris, contacts


def _ses_concave_tris(centers, expanded, probe, n_phi, triples=None, eat_probes=None, vdw=None):
    """Algorithm 2: spheric reentrant triangles on each RS-face probe."""
    tris = []
    if float(probe) <= 1e-8:
        return tris
    if triples is None:
        triples = _iter_probe_triples(centers, expanded)
    if vdw is None:
        vdw = np.maximum(np.asarray(expanded, dtype=float) - float(probe), 1e-6)
    n_phi = max(1, int(n_phi))
    probe = float(probe)
    for i, j, k, q in triples:
        q = np.asarray(q, dtype=float)
        d0 = _unit_vec(centers[i] - q)
        d1 = _unit_vec(centers[j] - q)
        d2 = _unit_vec(centers[k] - q)
        tris.extend(_probe_triangle_tris(q, probe, d0, d1, d2, n_phi))
    return tris


def _ses_convex_tris(centers, vdw, expanded, frequency, contacts, probe=0.0, triples=None):
    cap_freq = _convex_cap_frequency(frequency)
    unit, faces, _edges = geodesic_icosphere(int(cap_freq))
    unit = np.asarray(unit, dtype=float)
    faces = np.asarray(faces, dtype=int)
    tmpl_unit, tmpl_faces = unit, faces
    tris = []
    n = int(centers.shape[0])
    for i in range(n):
        center = np.asarray(centers[i], dtype=float)
        r_vdw = float(vdw[i])
        if r_vdw < 1e-8:
            continue

        def _full_sphere():
            world = center.reshape(1, 3) + unit * r_vdw
            for a, b, c in faces:
                tri = (world[a], world[b], world[c])
                centroid = (tri[0] + tri[1] + tri[2]) / 3.0
                tris.append(_orient(tri, centroid - center))

        if not contacts[i]:
            _full_sphere()
            continue
        parts = contacts[i]
        if triples:
            snapped = []
            for part in parts:
                if isinstance(part, tuple):
                    pts, closed = part
                else:
                    pts, closed = part, False
                pts = np.asarray(pts, dtype=float).reshape(-1, 3)
                cleaned = [pts[0]]
                for p in pts[1:]:
                    if not _points_near(p, cleaned[-1], 1e-5):
                        cleaned.append(p)
                pts = np.asarray(cleaned, dtype=float)
                if pts.shape[0] < 2:
                    continue
                if (not closed) and pts.shape[0] < 4 and _points_near(pts[0], pts[-1], 0.05):
                    continue
                snapped.append((pts, closed))
            parts = snapped
        loops = _assemble_loops(parts)
        if not loops:
            if n == 1:
                _full_sphere()
            continue
        for loop in loops:
            patch = _template_contact_tris(
                center, r_vdw, float(expanded[i]), i, centers, expanded, loop,
                tmpl_unit, tmpl_faces,
            )
            if not patch:
                patch = _fan_vdw_loop(
                    center, r_vdw, float(expanded[i]), i, centers, expanded, loop, cap_freq,
                )
            tris.extend(patch)
    return tris


def _quality_from_frequency(frequency: int) -> int:
    freq = int(frequency)
    for quality, value in ASA_FREQUENCY.items():
        if int(value) == freq:
            return int(quality)
    if freq <= 2:
        return 1
    if freq >= 8:
        return 5
    return DEFAULT_QUALITY


def _overlap_components(centers, radii):
    n = int(centers.shape[0])
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    neighbors = _overlap_neighbors(centers, radii)
    for i, js in enumerate(neighbors):
        for j in js:
            union(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


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


def _trilinear_sample(grid, origin, h, xyz):
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    n = int(xyz.shape[0])
    vector = grid.ndim == 4
    nx, ny, nz = grid.shape[:3]
    if n == 0:
        return np.zeros((0, 3), dtype=float) if vector else np.zeros((0,), dtype=float)
    p = (xyz - np.asarray(origin, dtype=float)) / float(h)
    p[:, 0] = np.clip(p[:, 0], 0.0, max(nx - 1.000001, 0.0))
    p[:, 1] = np.clip(p[:, 1], 0.0, max(ny - 1.000001, 0.0))
    p[:, 2] = np.clip(p[:, 2], 0.0, max(nz - 1.000001, 0.0))
    i0 = np.floor(p).astype(np.int32)
    i1 = np.stack(
        (
            np.minimum(i0[:, 0] + 1, nx - 1),
            np.minimum(i0[:, 1] + 1, ny - 1),
            np.minimum(i0[:, 2] + 1, nz - 1),
        ),
        axis=1,
    )
    f = p - i0.astype(float)
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    if vector:
        fx, fy, fz = fx[:, None], fy[:, None], fz[:, None]

    def corner(ix, iy, iz):
        return grid[ix, iy, iz]

    c000 = corner(i0[:, 0], i0[:, 1], i0[:, 2])
    c100 = corner(i1[:, 0], i0[:, 1], i0[:, 2])
    c010 = corner(i0[:, 0], i1[:, 1], i0[:, 2])
    c110 = corner(i1[:, 0], i1[:, 1], i0[:, 2])
    c001 = corner(i0[:, 0], i0[:, 1], i1[:, 2])
    c101 = corner(i1[:, 0], i0[:, 1], i1[:, 2])
    c011 = corner(i0[:, 0], i1[:, 1], i1[:, 2])
    c111 = corner(i1[:, 0], i1[:, 1], i1[:, 2])
    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


def _cubes_with_sign_change(field):
    nx, ny, nz = field.shape
    if nx < 2 or ny < 2 or nz < 2:
        return np.zeros((0, 3), dtype=np.int32)
    corners = (
        field[0:nx - 1, 0:ny - 1, 0:nz - 1],
        field[1:nx, 0:ny - 1, 0:nz - 1],
        field[0:nx - 1, 1:ny, 0:nz - 1],
        field[1:nx, 1:ny, 0:nz - 1],
        field[0:nx - 1, 0:ny - 1, 1:nz],
        field[1:nx, 0:ny - 1, 1:nz],
        field[0:nx - 1, 1:ny, 1:nz],
        field[1:nx, 1:ny, 1:nz],
    )
    vmin = corners[0]
    vmax = corners[0]
    for values in corners[1:]:
        vmin = np.minimum(vmin, values)
        vmax = np.maximum(vmax, values)
    keep = (vmin < 0.0) & (vmax >= 0.0)
    ii, jj, kk = np.nonzero(keep)
    if ii.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    return np.stack((ii, jj, kk), axis=1).astype(np.int32)


def _isosurface_from_signed_field(origin, h, field, grad):
    empty = (
        np.zeros((0, 3), dtype=float),
        np.zeros((0, 3), dtype=float),
        np.zeros((0, 3), dtype=int),
    )
    cubes = _cubes_with_sign_change(field)
    if cubes.shape[0] == 0:
        return empty
    tris = march_cubes(origin, h, field, cubes)
    if not tris:
        return empty
    vertices, faces = _weld(tris, ndigits=5)
    faces = _unique_faces(faces)
    faces = _drop_degenerate_faces(vertices, faces)
    vertices, faces = _compact_mesh(vertices, faces)
    if faces.shape[0] == 0:
        return empty
    normals = _trilinear_sample(grad, origin, h, vertices)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    tiny = lengths[:, 0] < 1e-12
    lengths = np.maximum(lengths, 1e-12)
    normals = normals / lengths
    if np.any(tiny):
        geom = _face_normals(vertices, faces)
        normals = np.array(normals, copy=True)
        normals[tiny] = geom[tiny]
    faces = _orient_faces_to_vertex_normals(vertices, faces, normals)
    return vertices, normals, faces


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
    occ = _paint_ball_union(shape, origin, h, centers, expanded)
    if not np.any(occ):
        return None
    edt = distance_transform_edt(occ, sampling=(h, h, h))
    field = float(probe) - edt
    gx, gy, gz = np.gradient(edt, h, h, h)
    grad = np.stack((-gx, -gy, -gz), axis=-1)
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


def _connolly_ses_component(centers, expanded, vdw, probe, frequency):
    """SES via the four MSMS algorithms (Sanner, Olson & Spehner, 1996)."""
    empty = (
        np.zeros((0, 3), dtype=float),
        np.zeros((0, 3), dtype=float),
        np.zeros((0, 3), dtype=int),
    )
    freq = max(1, int(frequency))
    probe = float(probe)
    rs = _reduced_surface(centers, expanded, probe)
    triples = list(rs["faces"])
    if not triples and int(centers.shape[0]) >= 3:
        triples = _iter_probe_triples(centers, expanded)
    triples = [
        (int(i), int(j), int(k), np.array(_q_key(q), dtype=float))
        for i, j, k, q in triples
    ]
    pairs = list(rs["free_edges"])
    pairs.extend(rs["edge_faces"].keys())
    pair_set = {_rs_edge_key(i, j) for i, j in pairs}
    eat = [q for _i, _j, _k, q in triples]
    n_theta = _torus_n_theta(freq)
    n_phi = max(16, _saddle_n_phi(centers, expanded, n_theta))
    torus, contacts = _ses_torus_tris(
        centers, expanded, probe, n_theta, n_phi,
        triples=triples, pairs=(list(pair_set) if pair_set else None),
        eat_probes=eat, vdw=vdw,
    )
    tris = list(torus)
    tris.extend(
        _ses_concave_tris(
            centers, expanded, probe, n_phi, triples=triples, eat_probes=eat, vdw=vdw,
        )
    )
    tris.extend(
        _ses_convex_tris(
            centers, vdw, expanded, freq, contacts, probe=probe, triples=triples,
        )
    )
    if not tris:
        return empty
    vertices, faces = _weld(tris, ndigits=5)
    faces = _unique_faces(faces)
    faces = _drop_degenerate_faces(vertices, faces)
    faces = _drop_overcovered_edge_faces(vertices, faces)
    vertices, faces = _compact_mesh(vertices, faces)
    if faces.shape[0] == 0:
        return empty
    vertices, faces = _split_t_junctions(vertices, faces)
    vertices, faces = _weld_contact_seam_stubs(vertices, faces, centers, vdw)
    vertices, faces = _split_t_junctions(vertices, faces)
    faces = _drop_overcovered_edge_faces(vertices, faces)
    vertices, faces = _compact_mesh(vertices, faces)
    vertices, faces = _fill_boundary_holes(vertices, faces)
    circles = _intersection_circles(centers, expanded)
    neighbors = _overlap_neighbors(centers, expanded)
    triples = _sas_triple_vertices(centers, expanded)
    vertices = _smooth_on_ses(
        vertices, faces, centers, expanded, probe,
        circles=circles, neighbors=neighbors, triples=triples,
        iterations=3, lam=0.22, pin_vdw=True,
    )
    normals = _ses_vertex_normals(
        vertices, faces, centers, vdw, expanded,
        circles=circles, neighbors=neighbors, triples=triples,
    )
    vertices, faces, normals = _refine_phong_faces(
        vertices, faces, normals, centers, expanded, probe,
        circles=circles, neighbors=neighbors, triples=triples,
    )
    vertices, faces = _split_t_junctions(vertices, faces, cleanup=True)
    vertices, faces = _weld_contact_seam_stubs(vertices, faces, centers, vdw)
    vertices, faces = _fill_boundary_holes(vertices, faces)
    faces = _drop_overcovered_edge_faces(vertices, faces)
    vertices, faces = _compact_mesh(vertices, faces)
    vertices, faces = _fill_boundary_holes(vertices, faces)
    normals = _ses_vertex_normals(
        vertices, faces, centers, vdw, expanded,
        circles=circles, neighbors=neighbors, triples=triples,
    )
    faces = _orient_faces_to_vertex_normals(vertices, faces, normals)
    return vertices, normals, faces


def _sas_mesh(centers, radii, frequency: int, probe_radius: float = 0.0):
    empty = (
        np.zeros((0, 3), dtype=float),
        np.zeros((0, 3), dtype=float),
        np.zeros((0, 3), dtype=int),
    )
    probe = float(probe_radius)
    expanded = np.asarray(radii, dtype=float).reshape(-1)
    vdw = np.maximum(expanded - probe, 1e-6)
    freq = max(1, int(frequency))
    n = int(np.asarray(centers).reshape(-1, 3).shape[0])
    if n == 0:
        return empty
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    parts_v = []
    parts_n = []
    parts_f = []
    offset = 0
    for group in _overlap_components(centers, expanded):
        idx = np.asarray(group, dtype=int)
        vertices, normals, faces = _connolly_ses_component(
            centers[idx], expanded[idx], vdw[idx], probe, freq,
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


def _gauss_pad_radius(elements, resolution, b_floor) -> float:
    blur = gaussian_blur_factor(resolution)
    extent = max(float(resolution), 1.0)
    for elem in elements:
        _amps, _kappas, rcut = atom_gaussian_terms(elem, b_floor, 1.0, blur)
        real = rcut / blur if blur > 1e-12 else rcut
        if real > extent:
            extent = real
    return float(extent)


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
    density = paint_gaussian_density(
        shape, origin, h, centers, elements,
        resolution=resolution, b_floor=b_floor,
    )
    if not np.any(np.abs(density) > 1e-12):
        return None
    signed = float(isolevel) - normalize_gaussian_map(density)
    gx, gy, gz = np.gradient(signed, h, h, h)
    grad = np.stack((gx, gy, gz), axis=-1)
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


_ASA_INSIDE_EPS = 1e-7
_ASA_ON_SPHERE = 1e-3


def _asa_outside(point, center, radius) -> bool:
    return float(np.linalg.norm(point - center)) >= float(radius) - _ASA_INSIDE_EPS


def _on_sphere(point, center, radius, tol=_ASA_ON_SPHERE) -> bool:
    return abs(float(np.linalg.norm(point - center)) - float(radius)) <= float(tol)


def _project_on_sphere(point, center, radius):
    center = np.asarray(center, dtype=float)
    vector = np.asarray(point, dtype=float) - center
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return center + np.array([0.0, 0.0, float(radius)])
    return center + float(radius) * vector / norm


def _sphere_intersection_circle(center_i, radius_i, center_j, radius_j):
    """Return ``(center, unit_normal, radius)`` of the i–j intersection, or None."""
    delta = np.asarray(center_j, dtype=float) - np.asarray(center_i, dtype=float)
    dist = float(np.linalg.norm(delta))
    ri = float(radius_i)
    rj = float(radius_j)
    if dist < 1e-12 or dist >= ri + rj - 1e-10:
        return None
    if dist + min(ri, rj) <= max(ri, rj) + 1e-10:
        return None
    normal = delta / dist
    axis = (ri * ri - rj * rj + dist * dist) / (2.0 * dist)
    rad2 = ri * ri - axis * axis
    if rad2 <= 1e-16:
        return None
    return np.asarray(center_i, dtype=float) + axis * normal, normal, float(np.sqrt(rad2))


def _snap_to_intersection_circle(point, center_i, radius_i, center_j, radius_j):
    circle = _sphere_intersection_circle(center_i, radius_i, center_j, radius_j)
    if circle is None:
        return _project_on_sphere(point, center_i, radius_i)
    origin, normal, rad = circle
    radial = np.asarray(point, dtype=float) - origin
    radial = radial - normal * float(np.dot(radial, normal))
    length = float(np.linalg.norm(radial))
    if length < 1e-12:
        axis = np.array((1.0, 0.0, 0.0) if abs(normal[0]) < 0.9 else (0.0, 1.0, 0.0))
        radial = np.cross(normal, axis)
        length = float(np.linalg.norm(radial))
    if length < 1e-12 or rad <= 1e-12:
        return _project_on_sphere(point, center_i, radius_i)
    return origin + rad * radial / length


def _sphere_triple_points(center_i, radius_i, center_j, radius_j, center_k, radius_k):
    """0–2 points on spheres i, j, and k."""
    circle = _sphere_intersection_circle(center_i, radius_i, center_j, radius_j)
    if circle is None:
        return []
    origin, normal, rad = circle
    offset = origin - np.asarray(center_k, dtype=float)
    rhs = 0.5 * (float(radius_k) * float(radius_k) - rad * rad - float(np.dot(offset, offset)))
    planar = offset - normal * float(np.dot(offset, normal))
    span = float(np.linalg.norm(planar))
    if span < 1e-12:
        return []
    along = (rhs / (span * span)) * planar
    height2 = rad * rad - float(np.dot(along, along))
    if height2 < -1e-10:
        return []
    if height2 < 0.0:
        height2 = 0.0
    lateral = np.cross(normal, planar / span)
    height = float(np.sqrt(height2))
    points = [origin + along]
    if height > 1e-10:
        points = [origin + along + height * lateral, origin + along - height * lateral]
    return [_project_on_sphere(point, center_i, radius_i) for point in points]


def _dist_to_segment(point, start, end):
    span = end - start
    length2 = float(np.dot(span, span))
    if length2 < 1e-18:
        return float(np.linalg.norm(point - start))
    t = float(np.dot(point - start, span) / length2)
    t = min(max(t, 0.0), 1.0)
    return float(np.linalg.norm(point - (start + t * span)))


def _radical_plane(center_i, radius_i, center_j, radius_j):
    ci = np.asarray(center_i, dtype=float)
    cj = np.asarray(center_j, dtype=float)
    normal = cj - ci
    if float(np.linalg.norm(normal)) < 1e-12:
        return None
    offset = 0.5 * (
        float(np.dot(cj, cj)) - float(np.dot(ci, ci))
        + float(radius_i) * float(radius_i) - float(radius_j) * float(radius_j)
    )
    return normal, offset


def _plane_keep(point, normal, offset) -> bool:
    return float(np.dot(point, normal) - offset) <= 1e-8


def _intersect_plane(p0, p1, normal, offset):
    f0 = float(np.dot(p0, normal) - offset)
    f1 = float(np.dot(p1, normal) - offset)
    denom = f0 - f1
    if abs(denom) < 1e-18:
        return np.asarray(p0, dtype=float)
    t = min(max(f0 / denom, 0.0), 1.0)
    return np.asarray(p0, dtype=float) + t * (np.asarray(p1, dtype=float) - p0)


def _clip_vertex(p0, p1, i, j, centers, radii):
    """Boundary point of an accessible/buried edge against neighbor *j*."""
    center_i = centers[i]
    radius_i = float(radii[i])
    plane = _radical_plane(center_i, radius_i, centers[j], float(radii[j]))
    if plane is None:
        return _snap_to_intersection_circle(
            0.5 * (np.asarray(p0) + np.asarray(p1)),
            center_i, radius_i, centers[j], float(radii[j]),
        )
    raw = _intersect_plane(p0, p1, plane[0], plane[1])
    triples = []
    for k, center_k in enumerate(centers):
        if k == i or k == j:
            continue
        if _on_sphere(p0, center_k, radii[k]) or _on_sphere(p1, center_k, radii[k]):
            triples.extend(_sphere_triple_points(
                center_i, radius_i, centers[j], float(radii[j]),
                center_k, float(radii[k]),
            ))
    if triples:
        return min(triples, key=lambda point: _dist_to_segment(point, p0, p1))
    return _snap_to_intersection_circle(
        raw, center_i, radius_i, centers[j], float(radii[j]),
    )


def _clip_poly_outside_neighbor(polygon, i, j, centers, radii):
    """Sutherland–Hodgman: keep the part of *polygon* outside neighbor *j*."""
    if len(polygon) < 3:
        return []
    center_i = centers[i]
    radius_i = float(radii[i])
    plane = _radical_plane(center_i, radius_i, centers[j], float(radii[j]))
    if plane is None:
        if radius_i <= float(radii[j]) + 1e-9:
            return []
        return list(polygon)
    normal, offset = plane
    previous = polygon[-1]
    prev_keep = _plane_keep(previous, normal, offset)
    clipped = []
    for current in polygon:
        curr_keep = _plane_keep(current, normal, offset)
        if curr_keep:
            if not prev_keep:
                clipped.append(_clip_vertex(previous, current, i, j, centers, radii))
            clipped.append(np.asarray(current, dtype=float))
        elif prev_keep:
            clipped.append(_clip_vertex(previous, current, i, j, centers, radii))
        previous = current
        prev_keep = curr_keep
    cleaned = []
    for point in clipped:
        if not cleaned or float(np.linalg.norm(point - cleaned[-1])) > 1e-8:
            cleaned.append(point)
    if len(cleaned) > 1 and float(np.linalg.norm(cleaned[-1] - cleaned[0])) <= 1e-8:
        cleaned.pop()
    return cleaned if len(cleaned) >= 3 else []


def _triangle_inside_one_neighbor(pa, pb, pc, neighbors, centers, radii) -> bool:
    for j in neighbors:
        center = centers[j]
        radius = float(radii[j])
        if (
            not _asa_outside(pa, center, radius)
            and not _asa_outside(pb, center, radius)
            and not _asa_outside(pc, center, radius)
        ):
            return True
    return False


def _sphere_contained(index, neighbors, centers, radii) -> bool:
    center = centers[index]
    radius = float(radii[index])
    for j in neighbors:
        dist = float(np.linalg.norm(center - centers[j]))
        if dist + radius <= float(radii[j]) + 1e-9:
            return True
    return False


def _asa_mesh(centers, radii, frequency: int):
    unit, faces, _edges = geodesic_icosphere(frequency)
    unit = np.asarray(unit, dtype=float)
    faces = np.asarray(faces, dtype=int)
    n_dir = unit.shape[0]
    n_spheres = int(centers.shape[0])
    neighbors = [[] for _ in range(n_spheres)]
    for i in range(n_spheres):
        for j in range(n_spheres):
            if i == j:
                continue
            dist = float(np.linalg.norm(centers[i] - centers[j]))
            if dist < float(radii[i]) + float(radii[j]) - _ASA_INSIDE_EPS:
                neighbors[i].append(j)
    tris = []
    for i, center in enumerate(centers):
        radius = float(radii[i])
        neigh = neighbors[i]
        if _sphere_contained(i, neigh, centers, radii):
            continue
        world = center.reshape(1, 3) + unit * radius
        keep = np.ones(n_dir, dtype=bool)
        for j in neigh:
            keep &= np.linalg.norm(world - centers[j].reshape(1, 3), axis=1) >= (
                float(radii[j]) - _ASA_INSIDE_EPS
            )
        for a, b, c in faces:
            pa, pb, pc = world[a], world[b], world[c]
            if bool(keep[a]) and bool(keep[b]) and bool(keep[c]):
                tris.append((pa, pb, pc))
                continue
            if neigh and _triangle_inside_one_neighbor(pa, pb, pc, neigh, centers, radii):
                continue
            polygon = [pa, pb, pc]
            for j in neigh:
                polygon = _clip_poly_outside_neighbor(polygon, i, j, centers, radii)
                if len(polygon) < 3:
                    break
            if len(polygon) < 3:
                continue
            origin = polygon[0]
            for k in range(1, len(polygon) - 1):
                p1, p2 = polygon[k], polygon[k + 1]
                if float(np.linalg.norm(np.cross(p1 - origin, p2 - origin))) < 1e-16:
                    continue
                tris.append((origin, p1, p2))
    vertices, faces_out = _weld(tris)
    return vertices, _face_normals(vertices, faces_out), faces_out


def build_solvent_surface(
    points: Sequence[Sequence[float]],
    atom_radius=DEFAULT_ATOM_RADIUS,
    probe_radius: float = DEFAULT_PROBE_RADIUS,
    algorithm: str = DEFAULT_ALGORITHM,
    quality: int = DEFAULT_QUALITY,
    elements=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(vertices, normals, faces)`` for the chosen algorithm."""
    empty = (
        np.zeros((0, 3), dtype=float),
        np.zeros((0, 3), dtype=float),
        np.zeros((0, 3), dtype=int),
    )
    centers = np.asarray(points, dtype=float)
    if centers.size == 0:
        return empty
    centers = np.reshape(centers, (-1, 3))
    radii = expanded_radii(atom_radius, centers.shape[0], probe_radius)
    kind = normalize_algorithm(algorithm)
    if kind == "ASA":
        return _asa_mesh(centers, radii, asa_frequency(quality))
    if kind == "MC":
        return _mc_mesh(centers, radii, quality, float(probe_radius))
    if kind == "GAUSS":
        return _gauss_mesh(centers, elements, quality)
    return _sas_mesh(centers, radii, asa_frequency(quality), float(probe_radius))
