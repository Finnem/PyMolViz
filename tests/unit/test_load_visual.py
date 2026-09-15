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
    assert opts["specular"] is True
    coll.specular = False
    assert sphere_options(coll)["specular"] is False


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
        algorithm="GAUSS",
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
    assert opts["algorithm"] == "GAUSS"
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
        algorithm="GAUSS",
        quality=1,
        clip_planes=planes,
        bypass_colormap=True,
    )
    coll = CGOCollection([surface], name="pmv_surface")
    opts = surface_options(coll)
    assert len(opts["clip_planes"]) == 1
    assert opts["clip_planes"][0]["origin"] == pytest.approx([1.0, 0.0, 0.0])
    assert opts["clip_planes"][0]["normal"] == pytest.approx([1.0, 0.0, 0.0])


def test_points_from_mesh_restore_field_id():
    sphere = Sphere(
        FixedPoint((0.0, 0.0, 0.0)),
        0.5,
        color=(0.1, 0.2, 0.3),
        bypass_colormap=True,
    )
    sphere.field_id = "pymol_map:density"
    sphere.field_colormap = "plasma"
    sphere.field_clims = [0.0, 1.0]
    coll = CGOCollection([sphere], name="pmv_spheres")
    points = points_from_mesh(coll)
    assert points[0].field_id == "pymol_map:density"
    assert points[0].field_colormap == "plasma"
    assert points[0].field_clims == pytest.approx((0.0, 1.0))
    assert points[0].color_choice().is_field is True


def test_points_from_mesh_restore_enabled():
    sphere = Sphere(
        FixedPoint((0.0, 0.0, 0.0)), 0.5,
        bypass_colormap=True, enabled=False, frequency=2,
    )
    coll = CGOCollection([sphere], name="pmv_spheres")
    points = points_from_mesh(coll)
    assert len(points) == 1
    assert points[0].enabled is False


def test_surface_points_restore_point_enabled():
    from pymolviz.meshes.Surface import Surface

    surface = Surface(
        [FixedPoint((0.0, 0.0, 0.0)), FixedPoint((1.0, 0.0, 0.0))],
        algorithm="GAUSS", quality=1, bypass_colormap=True,
        point_enabled=[True, False],
    )
    coll = CGOCollection([surface], name="pmv_surface")
    points = points_from_mesh(coll)
    assert [pt.enabled for pt in points] == [True, False]


def test_box_and_arrow_options_include_clip_planes():
    from pymolviz.wizards.builders.load_visual import arrow_options, box_options

    planes = [{"origin": [0.0, 1.0, 0.0], "normal": [0.0, 1.0, 0.0], "scale": 3.0}]
    box = CenteredBox(
        FixedPoint((0.0, 0.0, 0.0)), (1.0, 1.0, 1.0),
        bypass_colormap=True, clip_planes=planes,
    )
    assert box_options(CGOCollection([box], name="pmv_boxes"))["clip_planes"][0]["scale"] == pytest.approx(3.0)
    arrows = Arrows(
        starts=[FixedPoint((0.0, 0.0, 0.0))],
        ends=[FixedPoint((1.0, 0.0, 0.0))],
        color=[(1.0, 0.0, 0.0)],
        bypass_colormap=True,
        clip_planes=planes,
        head_radius=0.2,
    )
    opts = arrow_options(CGOCollection([arrows], name="pmv_arrows"))
    assert opts["clip_planes"][0]["origin"] == pytest.approx([0.0, 1.0, 0.0])
    assert opts["head_radius"] == pytest.approx(0.2)


def test_lines_mesh_options_match_arrow_line_style():
    from pymolviz.meshes.Arrows import Arrows
    from pymolviz.util.line_style import LineStyle
    from pymolviz.wizards.builders.load_visual import arrow_options, pairs_from_mesh

    lines = Arrows(
        starts=[FixedPoint((0.0, 0.0, 0.0))],
        ends=[FixedPoint((1.0, 0.0, 0.0))],
        color=[(0.2, 0.4, 0.8)],
        shaft_radius=0.09,
        quality=0,
        line_style=LineStyle(ends="None"),
        bypass_colormap=True,
    )
    coll = CGOCollection([lines], name="pmv_lines")
    opts = arrow_options(coll)
    assert opts["shaft_radius"] == pytest.approx(0.09)
    assert opts["quality"] == 0
    assert opts["line_style"].n_arrow_heads() == 0
    assert opts["end_head"] == "None"
    assert opts["start_head"] == "None"
    assert opts["dash"] == "Solid"
    pairs = pairs_from_mesh(coll)
    assert len(pairs) == 1
    assert pairs[0].width == pytest.approx(0.09)
    assert pairs[0].style.ends == "None"
    styled = Arrows(
        starts=[FixedPoint((0.0, 0.0, 0.0))],
        ends=[FixedPoint((1.0, 0.0, 0.0))],
        color=[(1.0, 0.0, 0.0)],
        shaft_radius=0.05,
        line_style=LineStyle(ends="Circles"),
        bypass_colormap=True,
    )
    caps = arrow_options(CGOCollection([styled], name="ends"))
    assert isinstance(caps["line_style"], LineStyle)
    assert caps["start_head"] == "Circles"
    assert caps["end_head"] == "Circles"



