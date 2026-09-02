"""Rebuild builder points from a persisted collection."""

from __future__ import annotations

from pymolviz.meshes.Arrows import Arrows
from pymolviz.meshes.CenteredBox import CenteredBox
from pymolviz.meshes.CGOCollection import CGOCollection
from pymolviz.meshes.Sphere import Sphere
from pymolviz.points import AtomPoint, FixedPoint
from pymolviz.wizards.builders.load_visual import (
    pairs_from_mesh,
    points_from_mesh,
    sphere_options,
)


def test_points_from_sphere_collection():
    src = AtomPoint("prot", 7, chain="A", resi="12", name="CA", last_xyz=(4.0, 5.0, 6.0))
    sphere = Sphere(src, 1.25, color=(1.0, 0.0, 0.0), bypass_colormap=True, wireframe=True, frequency=4)
    coll = CGOCollection([sphere], name="pmv_spheres", obj_id="c1")
    points = points_from_mesh(coll)
    assert len(points) == 1
    assert points[0].source == "selection"
    assert points[0].xyz() == (4.0, 5.0, 6.0)
    assert points[0].point_source.atom_id == 7
    opts = sphere_options(coll)
    assert opts["radius"] == 1.25
    assert opts["wireframe"] is True


def test_points_from_box_uses_center():
    box = CenteredBox(FixedPoint((1.0, 0.0, -1.0)), (2.0, 3.0, 4.0), bypass_colormap=True)
    coll = CGOCollection([box], name="pmv_boxes")
    points = points_from_mesh(coll)
    assert len(points) == 1
    assert points[0].source == "manual"
    assert points[0].xyz() == (1.0, 0.0, -1.0)


def test_pairs_from_arrows():
    arrows = Arrows(
        starts=[FixedPoint((0.0, 0.0, 0.0))],
        ends=[FixedPoint((1.0, 0.0, 0.0))],
        color=[(0.2, 0.4, 0.8)],
        bypass_colormap=True,
        quality=2,
    )
    coll = CGOCollection([arrows], name="pmv_arrows")
    pairs = pairs_from_mesh(coll)
    assert len(pairs) == 1
    assert pairs[0].start.xyz() == (0.0, 0.0, 0.0)
    assert pairs[0].end.xyz() == (1.0, 0.0, 0.0)
