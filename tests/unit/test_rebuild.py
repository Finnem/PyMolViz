"""Unit tests for mesh rebuild and Displayable ids."""

from __future__ import annotations

import numpy as np

import pytest

from pymolviz.Displayable import Displayable
from pymolviz.meshes.Sphere import Sphere
from pymolviz.meshes.Cylinder import Cylinder
from pymolviz.points import FixedPoint, AtomPoint


class _DummyDisplayable(Displayable):
    def _script_string(self):
        return ""


def test_displayable_id_stable_and_not_name():
    obj = _DummyDisplayable(name="my_name", obj_id="deadbeef")
    assert obj.id == "deadbeef"
    assert obj.name == "my_name"
    obj.name = "other_name"
    assert obj.id == "deadbeef"


def test_sphere_rebuild_changes_vertices():
    sphere = Sphere((0.0, 0.0, 0.0), 1.0, bypass_colormap=True)
    before = np.array(sphere.vertices, copy=True)
    sphere.position = FixedPoint((5.0, 0.0, 0.0))
    sphere.rebuild(None)
    after = np.array(sphere.vertices)
    assert not np.allclose(before, after)
    center = after.mean(axis=0)
    assert abs(center[0] - 5.0) < 0.5


def test_sphere_cgo_token_shape():
    sphere = Sphere((0.0, 0.0, 0.0), 1.0, bypass_colormap=True)
    tokens = sphere._create_CGO_list()
    kinds = [t for t in tokens if isinstance(t, str)]
    assert "BEGIN" in kinds
    assert "TRIANGLES" in kinds
    assert "END" in kinds


def test_cylinder_construct_and_rebuild():
    cyl = Cylinder((0, 0, 0), (0, 0, 5), 0.2, bypass_colormap=True)
    assert len(cyl.vertices) > 0
    cyl.start = FixedPoint((1.0, 0.0, 0.0))
    cyl.end = FixedPoint((1.0, 0.0, 6.0))
    cyl.rebuild(None)
    zvals = cyl.vertices[:, 2]
    assert zvals.min() < 1.0
    assert zvals.max() > 5.0


def test_sphere_with_atom_point_last_xyz_bakes():
    src = AtomPoint("prot", 1, last_xyz=(2.0, 3.0, 4.0))
    sphere = Sphere(src, 0.5, bypass_colormap=True)
    sphere.rebuild(None)
    center = np.asarray(sphere.vertices).mean(axis=0)
    assert abs(center[0] - 2.0) < 0.2
    assert abs(center[1] - 3.0) < 0.2
    assert abs(center[2] - 4.0) < 0.2


def test_sphere_shift_vertices_keeps_mesh_and_patches_cgo():
    sphere = Sphere((0.0, 0.0, 0.0), 1.0, bypass_colormap=True)
    tokens = sphere._create_CGO_list()
    before = np.array(sphere.vertices, copy=True)
    vertex_idx = next(i for i, t in enumerate(tokens) if str(t) == "VERTEX")
    x0 = float(tokens[vertex_idx + 1])
    sphere.shift_vertices((5.0, 0.0, 0.0))
    after = np.array(sphere.vertices)
    assert np.allclose(after, before + np.array([5.0, 0.0, 0.0]))
    assert tokens is sphere._cached_cgo
    assert float(tokens[vertex_idx + 1]) == pytest.approx(x0 + 5.0)
    again = sphere._create_CGO_list()
    assert again is tokens


def test_sphere_shift_vertices_patches_resolved_float_cgo(monkeypatch):
    """replace_cgo loads _cached_resolved; real PyMOL opcodes are floats."""
    import pymol.cgo as cgo_mod

    monkeypatch.setattr(cgo_mod, "BEGIN", 2.0)
    monkeypatch.setattr(cgo_mod, "END", 3.0)
    monkeypatch.setattr(cgo_mod, "VERTEX", 4.0)
    monkeypatch.setattr(cgo_mod, "TRIANGLES", 4.0)
    monkeypatch.setattr(cgo_mod, "NORMAL", 5.0)
    monkeypatch.setattr(cgo_mod, "COLOR", 6.0)
    monkeypatch.setattr(cgo_mod, "SPHERE", 7.0)
    monkeypatch.setattr(cgo_mod, "CYLINDER", 9.0)
    monkeypatch.setattr(cgo_mod, "CONE", 27.0)
    monkeypatch.setattr(cgo_mod, "LINEWIDTH", 10.0)
    monkeypatch.setattr(cgo_mod, "ALPHA", 25.0)
    monkeypatch.setattr(cgo_mod, "POINTS", 0.0)
    monkeypatch.setattr(cgo_mod, "LINES", 1.0)

    from pymolviz.runtime.renderer import resolved_cgo_tokens

    sphere = Sphere((0.0, 0.0, 0.0), 1.0, bypass_colormap=True)
    resolved = resolved_cgo_tokens(sphere, None)
    assert resolved[0] == 2.0
    assert resolved[1] == 4.0
    vertex_xs = []
    i = 0
    n = len(resolved)
    after_begin = False
    while i < n:
        tok = resolved[i]
        if after_begin:
            after_begin = False
            i += 1
            continue
        if tok == 2.0:
            after_begin = True
            i += 1
            continue
        if tok == 4.0 and i + 3 < n:
            vertex_xs.append(float(resolved[i + 1]))
            i += 4
            continue
        if tok == 6.0:
            i += 4
            continue
        if tok == 5.0:
            i += 4
            continue
        i += 1
    assert vertex_xs
    x0_mean = sum(vertex_xs) / len(vertex_xs)
    sphere.shift_vertices((5.0, 0.0, 0.0))
    assert sphere._cached_resolved is resolved
    vertex_xs_after = []
    i = 0
    after_begin = False
    while i < n:
        tok = resolved[i]
        if after_begin:
            after_begin = False
            i += 1
            continue
        if tok == 2.0:
            after_begin = True
            i += 1
            continue
        if tok == 4.0 and i + 3 < n:
            vertex_xs_after.append(float(resolved[i + 1]))
            i += 4
            continue
        if tok == 6.0:
            i += 4
            continue
        if tok == 5.0:
            i += 4
            continue
        i += 1
    x1_mean = sum(vertex_xs_after) / len(vertex_xs_after)
    assert x1_mean == pytest.approx(x0_mean + 5.0)
    assert resolved[1] == 4.0
