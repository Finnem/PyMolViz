"""Rebuild builder points from a persisted collection."""

from __future__ import annotations

import pytest

from pymolviz.meshes.Arrows import Arrows
from pymolviz.meshes.CenteredBox import CenteredBox
from pymolviz.meshes.CGOCollection import CGOCollection
from pymolviz.meshes.Sphere import Sphere
from pymolviz.points import AtomPoint, FixedPoint
from pymolviz.wizards.builders.load_visual import (
    pairs_from_mesh,
    points_from_mesh,
    sphere_options,
    surface_options,
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
    assert pairs[0].pair_id.startswith("arrow-")
    assert pairs[0].width == arrows.shaft_radius
    assert pairs[0].style.ends == "Arrow"
    assert pairs[0].style.start_head == "None"
    assert pairs[0].style.end_head == "Arrow"


def test_points_from_surface_collection():
    from pymolviz.meshes.Surface import Surface

    src = AtomPoint("prot", 7, chain="A", resi="12", name="CA", last_xyz=(4.0, 5.0, 6.0))
    surface = Surface(
        [src, FixedPoint((1.0, 0.0, 0.0))],
        atom_radius=1.1,
        probe_radius=1.4,
        algorithm="ASA",
        quality=1,
        color=(0.2, 0.4, 0.8),
        bypass_colormap=True,
        wireframe=True,
        point_radii=[1.25, None],
        radius_mode="uniform",
        vdw_scale=1.0,
    )
    coll = CGOCollection([surface], name="pmv_surface")
    points = points_from_mesh(coll)
    assert len(points) == 2
    assert points[0].source == "selection"
    assert points[0].xyz() == (4.0, 5.0, 6.0)
    assert points[0].point_source.atom_id == 7
    assert points[1].source == "manual"
    assert points[1].xyz() == (1.0, 0.0, 0.0)
    assert points[0].radius == pytest.approx(1.25)
    assert points[1].radius is None
    opts = surface_options(coll)
    assert opts["radius"] == pytest.approx(1.1)
    assert opts["probe_radius"] == pytest.approx(1.4)
    assert opts["algorithm"] == "ASA"
    assert opts["quality"] == 1
    assert opts["wireframe"] is True
    assert opts["radius_mode"] == "uniform"
    assert opts["vdw_scale"] == pytest.approx(1.0)
    assert opts["color"] == pytest.approx((0.2, 0.4, 0.8))
    assert opts["clip_planes"] == []


def test_surface_options_include_clip_planes():
    from pymolviz.meshes.Surface import Surface

    planes = [{"origin": [1.0, 0.0, 0.0], "normal": [1.0, 0.0, 0.0], "scale": 4.0}]
    surface = Surface(
        [FixedPoint((0.0, 0.0, 0.0))],
        algorithm="ASA",
        quality=1,
        clip_planes=planes,
        bypass_colormap=True,
    )
    coll = CGOCollection([surface], name="pmv_surface")
    opts = surface_options(coll)
    assert len(opts["clip_planes"]) == 1
    assert opts["clip_planes"][0]["origin"] == pytest.approx([1.0, 0.0, 0.0])
    assert opts["clip_planes"][0]["normal"] == pytest.approx([1.0, 0.0, 0.0])


