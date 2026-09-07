"""Lorensen marching cubes: table completeness and a closed spherical isosurface."""

from __future__ import annotations

import numpy as np
import pytest

from pymolviz.util.marching_cubes import (
    _EDGES,
    _TRI_TABLE,
    cube_triangles,
    march_cubes,
)


def _sign_edges(idx):
    used = set()
    for edge, (a, b) in enumerate(_EDGES):
        if ((idx >> a) & 1) != ((idx >> b) & 1):
            used.add(edge)
    return used


def _weld(tris, ndigits=6):
    index = {}
    verts = []
    faces = []
    for a, b, c in tris:
        ids = []
        for p in (a, b, c):
            key = (
                round(float(p[0]), ndigits),
                round(float(p[1]), ndigits),
                round(float(p[2]), ndigits),
            )
            vid = index.get(key)
            if vid is None:
                vid = len(verts)
                index[key] = vid
                verts.append((float(p[0]), float(p[1]), float(p[2])))
            ids.append(vid)
        if ids[0] != ids[1] and ids[1] != ids[2] and ids[2] != ids[0]:
            faces.append(ids)
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=int)


def _boundary_edge_count(faces):
    count = {}
    for a, b, c in np.asarray(faces, dtype=int):
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (u, v) if u < v else (v, u)
            count[key] = count.get(key, 0) + 1
    return sum(1 for n in count.values() if n == 1)


def test_tri_table_uses_every_sign_change_edge():
    assert _TRI_TABLE.shape == (256, 15)
    for idx in range(256):
        sign = _sign_edges(idx)
        used = [int(e) for e in _TRI_TABLE[idx] if int(e) >= 0]
        assert len(used) % 3 == 0
        assert set(used) == sign


def test_cube_one_corner_one_triangle():
    pts = [
        np.array(c, dtype=float)
        for c in (
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
            (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
        )
    ]
    tris = cube_triangles(pts, [-1, 1, 1, 1, 1, 1, 1, 1])
    assert len(tris) == 1
    verts = np.vstack(tris[0])
    assert verts.shape == (3, 3)
    assert {tuple(np.round(v, 6)) for v in verts} == {
        (0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 0.5),
    }


def test_march_cubes_sphere_is_closed():
    origin = np.array([-1.0, -1.0, -1.0], dtype=float)
    h = 0.125
    n = 17
    xs = origin[0] + np.arange(n) * h
    ys = origin[1] + np.arange(n) * h
    zs = origin[2] + np.arange(n) * h
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    radius = 0.55
    field = np.sqrt(xx * xx + yy * yy + zz * zz) - radius
    nx, ny, nz = field.shape
    cubes = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                corners = (
                    field[i, j, k], field[i + 1, j, k],
                    field[i, j + 1, k], field[i + 1, j + 1, k],
                    field[i, j, k + 1], field[i + 1, j, k + 1],
                    field[i, j + 1, k + 1], field[i + 1, j + 1, k + 1],
                )
                if min(corners) < 0.0 <= max(corners):
                    cubes.append((i, j, k))
    cubes = np.asarray(cubes, dtype=np.int32)
    tris = march_cubes(origin, h, field, cubes)
    assert len(tris) > 20
    verts, faces = _weld(tris)
    assert faces.shape[0] > 20
    assert _boundary_edge_count(faces) == 0
    dist = np.linalg.norm(verts, axis=1)
    assert dist == pytest.approx(radius, abs=h)
