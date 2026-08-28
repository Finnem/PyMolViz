"""Unit tests for mesh rebuild and Displayable ids."""

from __future__ import annotations

import numpy as np

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
