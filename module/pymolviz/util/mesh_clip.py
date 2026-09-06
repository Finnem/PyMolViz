"""Clip a triangle mesh by infinite planes.

Keep the half-space ``dot(v - origin, normal) >= 0``. The normal points toward
the kept side (the eye). Split edges interpolate position and normals.
"""

from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .view import camera_to_model_offset, screen_center

_KEEP_EPS = 1e-8
_WELD = 1e-6
_AREA_EPS = 1e-16


def normalize_clip_planes(planes: Optional[Iterable[Mapping]] = None) -> List[dict]:
    """JSON-safe plane dicts: origin, unit normal, gizmo scale."""
    out: List[dict] = []
    for raw in planes or ():
        if not isinstance(raw, Mapping):
            continue
        origin = [float(x) for x in (raw.get("origin") or (0.0, 0.0, 0.0))]
        if len(origin) < 3:
            continue
        origin = origin[:3]
        normal = np.asarray(raw.get("normal", (0.0, 0.0, 1.0)), dtype=float).reshape(-1)
        if normal.size < 3:
            continue
        ln = float(np.linalg.norm(normal[:3]))
        if ln < 1e-12:
            continue
        normal = (normal[:3] / ln).tolist()
        try:
            scale = float(raw.get("scale", 5.0) or 5.0)
        except (TypeError, ValueError):
            scale = 5.0
        if scale <= 0.0:
            scale = 5.0
        out.append({"origin": origin, "normal": normal, "scale": scale})
    return out


def clip_planes_match(a, b, *, atol: float = 1e-6, ignore_scale: bool = False) -> bool:
    left = normalize_clip_planes(a)
    right = normalize_clip_planes(b)
    if len(left) != len(right):
        return False
    for pa, pb in zip(left, right):
        if not ignore_scale and abs(float(pa["scale"]) - float(pb["scale"])) > atol:
            return False
        if any(abs(x - y) > atol for x, y in zip(pa["origin"], pb["origin"])):
            return False
        if any(abs(x - y) > atol for x, y in zip(pa["normal"], pb["normal"])):
            return False
    return True


def _unit3(v):
    arr = np.asarray(v, dtype=float).reshape(3)
    ln = float(np.linalg.norm(arr))
    if ln < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return arr / ln


def rotate_around(vec, axis, degrees):
    """Rotate ``vec`` around ``axis`` by ``degrees`` (Rodrigues)."""
    vec = np.asarray(vec, dtype=float).reshape(3)
    axis = _unit3(axis)
    rad = float(np.deg2rad(degrees))
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    return vec * c + np.cross(axis, vec) * s + axis * float(np.dot(axis, vec)) * (1.0 - c)


def oriented_clip_normal(base_normal, tilt_deg=0.0, turn_deg=0.0):
    """Tilt around the in-plane U axis, then turn around V, from ``base_normal``."""
    from .math import get_perp

    n = _unit3(base_normal)
    u = get_perp(n)
    u = u / float(np.linalg.norm(u))
    v = np.cross(n, u)
    v = v / float(np.linalg.norm(v))
    n = rotate_around(n, u, tilt_deg)
    n = rotate_around(n, v, turn_deg)
    return _unit3(n)


def clip_plane_from_view(view, points_xyz=None) -> dict:
    """Plane through the screen center, keep-side toward the camera."""
    look = np.asarray(camera_to_model_offset(view, (0.0, 0.0, -1.0)), dtype=float)
    ln = float(np.linalg.norm(look))
    if ln < 1e-12:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        normal = -look / ln
    origin = np.asarray(screen_center(view), dtype=float).reshape(3)
    scale = 5.0
    if points_xyz is not None:
        pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
        if pts.shape[0]:
            center = pts.mean(axis=0)
            extent = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
            span = max(extent, 1.0)
            if float(np.linalg.norm(origin - center)) > 2.0 * span:
                origin = center
            scale = max(span * 0.65, 4.0)
    return {
        "origin": [float(origin[0]), float(origin[1]), float(origin[2])],
        "normal": [float(normal[0]), float(normal[1]), float(normal[2])],
        "scale": float(scale),
    }


def clip_triangle_mesh(vertices, normals, faces, origin, normal):
    """Clip one plane. Returns ``(vertices, normals, faces)``."""
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    faces = np.asarray(faces, dtype=int).reshape(-1, 3)
    empty_v = np.zeros((0, 3), dtype=float)
    empty_f = np.zeros((0, 3), dtype=int)
    if vertices.size == 0 or faces.size == 0:
        return empty_v, empty_v.copy(), empty_f
    if normals is None:
        normals = np.zeros_like(vertices)
    else:
        normals = np.asarray(normals, dtype=float).reshape(-1, 3)
        if normals.shape[0] != vertices.shape[0]:
            normals = np.zeros_like(vertices)
    origin = np.asarray(origin, dtype=float).reshape(3)
    nrm = np.asarray(normal, dtype=float).reshape(3)
    ln = float(np.linalg.norm(nrm))
    if ln < 1e-12:
        return (
            np.array(vertices, copy=True),
            np.array(normals, copy=True),
            np.array(faces, copy=True),
        )
    nrm = nrm / ln

    out_v: List[np.ndarray] = []
    out_n: List[np.ndarray] = []
    out_f: List[Sequence[int]] = []
    index = {}

    def emit_vertex(point, vert_n):
        key = (
            round(float(point[0]) / _WELD),
            round(float(point[1]) / _WELD),
            round(float(point[2]) / _WELD),
        )
        found = index.get(key)
        if found is not None:
            return found
        i = len(out_v)
        index[key] = i
        out_v.append(np.asarray(point, dtype=float).reshape(3))
        out_n.append(np.asarray(vert_n, dtype=float).reshape(3))
        return i

    def intersect(p, q, dp, dq, np_, nq):
        denom = float(dp - dq)
        t = 0.5 if abs(denom) < 1e-18 else float(dp / denom)
        t = min(1.0, max(0.0, t))
        pos = p + t * (q - p)
        blended = (1.0 - t) * np_ + t * nq
        length = float(np.linalg.norm(blended))
        if length > 1e-18:
            blended = blended / length
        else:
            blended = np_
        return pos, blended

    dist = (vertices - origin) @ nrm
    for face in faces:
        i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])
        vs = (vertices[i0], vertices[i1], vertices[i2])
        ns = (normals[i0], normals[i1], normals[i2])
        ds = (float(dist[i0]), float(dist[i1]), float(dist[i2]))
        inside = (ds[0] >= -_KEEP_EPS, ds[1] >= -_KEEP_EPS, ds[2] >= -_KEEP_EPS)
        n_in = int(inside[0]) + int(inside[1]) + int(inside[2])
        if n_in == 0:
            continue
        poly_v = []
        poly_n = []
        if n_in == 3:
            poly_v = list(vs)
            poly_n = list(ns)
        else:
            for k in range(3):
                nxt = (k + 1) % 3
                pk, qk = vs[k], vs[nxt]
                nk, nq = ns[k], ns[nxt]
                dk, dq = ds[k], ds[nxt]
                if inside[k]:
                    poly_v.append(pk)
                    poly_n.append(nk)
                    if not inside[nxt]:
                        hit, hn = intersect(pk, qk, dk, dq, nk, nq)
                        poly_v.append(hit)
                        poly_n.append(hn)
                elif inside[nxt]:
                    hit, hn = intersect(pk, qk, dk, dq, nk, nq)
                    poly_v.append(hit)
                    poly_n.append(hn)
        if len(poly_v) < 3:
            continue
        for k in range(1, len(poly_v) - 1):
            a, b, c = poly_v[0], poly_v[k], poly_v[k + 1]
            area = float(np.linalg.norm(np.cross(b - a, c - a)))
            if area < _AREA_EPS:
                continue
            out_f.append((
                emit_vertex(a, poly_n[0]),
                emit_vertex(b, poly_n[k]),
                emit_vertex(c, poly_n[k + 1]),
            ))

    if not out_f:
        return empty_v, empty_v.copy(), empty_f
    return (
        np.vstack(out_v),
        np.vstack(out_n),
        np.asarray(out_f, dtype=int).reshape(-1, 3),
    )


def clip_mesh_by_planes(vertices, faces, normals, planes):
    """Apply planes in order. Returns ``(vertices, normals, faces)``."""
    v = np.asarray(vertices, dtype=float).reshape(-1, 3)
    f = np.asarray(faces, dtype=int).reshape(-1, 3)
    if normals is None:
        n = np.zeros_like(v)
    else:
        n = np.asarray(normals, dtype=float).reshape(-1, 3)
    for plane in normalize_clip_planes(planes):
        v, n, f = clip_triangle_mesh(v, n, f, plane["origin"], plane["normal"])
        if f.size == 0:
            break
    return v, n, f
