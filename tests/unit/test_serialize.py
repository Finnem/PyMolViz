"""Unit tests for schema-1 serialization."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pymolviz.points import AtomPoint, FixedPoint
from pymolviz.serialization import (
    SCHEMA_VERSION,
    SerializationError,
    assert_plain,
    displayable_from_dict,
    displayable_to_dict,
    point_source_from_dict,
    session_document,
    session_from_document,
)
from pymolviz.meshes.Sphere import Sphere
from pymolviz.meshes.CGOCollection import CGOCollection


def test_point_source_roundtrip():
    data = AtomPoint(
        "prot", 7, chain="A", resi="12", name="CA", elem="C",
        last_xyz=[1.0, 2.0, 3.0], last_vdw=1.7,
    ).to_dict()
    restored = point_source_from_dict(data)
    assert isinstance(restored, AtomPoint)
    assert restored.object == "prot"
    assert restored.atom_id == 7
    assert restored.last_xyz == (1.0, 2.0, 3.0)
    assert restored.elem == "C"
    assert restored.last_vdw == pytest.approx(1.7)


def test_fixed_point_roundtrip():
    data = FixedPoint((0.5, 1.5, 2.5)).to_dict()
    restored = point_source_from_dict(data)
    assert restored.resolve(None) == (0.5, 1.5, 2.5)


def test_sphere_collection_roundtrip():
    sphere = Sphere((0.0, 0.0, 0.0), 1.0, color=(1.0, 0.0, 0.0), bypass_colormap=True, obj_id="abc123")
    collection = CGOCollection([sphere], name="pmv_spheres", obj_id="coll001")
    data = displayable_to_dict(collection)
    assert_plain(data)
    json.dumps(data)
    assert data["schema"] == SCHEMA_VERSION
    assert data["type"] == "CGOCollection"
    assert data["id"] == "coll001"
    restored = displayable_from_dict(data)
    assert isinstance(restored, CGOCollection)
    assert restored.id == "coll001"
    assert len(restored) == 1
    assert isinstance(restored[0], Sphere)
    assert restored.specular is True


def test_collection_specular_off_roundtrip():
    sphere = Sphere((0.0, 0.0, 0.0), 1.0, bypass_colormap=True, obj_id="abc123")
    collection = CGOCollection(
        [sphere], name="pmv_spheres", obj_id="collSpec", specular=False,
    )
    data = displayable_to_dict(collection)
    assert data["specular"] is False
    restored = displayable_from_dict(data)
    assert restored.specular is False


def test_session_document_roundtrip():
    sphere = Sphere((1.0, 0.0, 0.0), 0.5, bypass_colormap=True, obj_id="s1")
    coll = CGOCollection([sphere], name="test_coll", obj_id="c1")
    doc = session_document([coll])
    assert doc["schema"] == SCHEMA_VERSION
    objects = session_from_document(doc)
    assert len(objects) == 1
    assert objects[0].id == "c1"


def test_assert_plain_rejects_numpy():
    with pytest.raises(SerializationError):
        assert_plain({"x": __import__("numpy").array([1.0])})


def test_arrows_independent_heads_roundtrip():
    from pymolviz.meshes.Arrows import Arrows
    from pymolviz.util.line_style import LineStyle

    arrows = Arrows(
        starts=[FixedPoint((0.0, 0.0, 0.0))],
        ends=[FixedPoint((1.0, 0.0, 0.0))],
        color=[(0.2, 0.4, 0.8)],
        bypass_colormap=True,
        line_style=LineStyle(start_head="Circles", end_head="Arrow"),
        use_styled_cgo=True,
        obj_id="arr001",
    )
    collection = CGOCollection([arrows], name="pmv_arrows", obj_id="collA")
    data = displayable_to_dict(collection)
    assert_plain(data)
    json.dumps(data)
    style = data["objects"][0]["line_style"]
    assert style["start_head"] == "Circles"
    assert style["end_head"] == "Arrow"
    restored = displayable_from_dict(data)
    got = restored[0].line_style
    assert got.start_head == "Circles"
    assert got.end_head == "Arrow"
    child = data["objects"][0]
    assert child["type"] == "Arrows"
    assert child["quality"] == arrows.quality
    assert child["shaft_radius"] == pytest.approx(arrows.shaft_radius)


def test_legacy_lines_session_loads_as_arrows():
    from pymolviz.meshes.Arrows import Arrows

    payload = {
        "type": "CGOCollection",
        "id": "coll_legacy_lines",
        "name": "pmv_lines",
        "state": 1,
        "transparency": 0.0,
        "schema": SCHEMA_VERSION,
        "objects": [{
            "type": "Lines",
            "id": "ln1",
            "name": "lines",
            "state": 1,
            "transparency": 0.0,
            "color": [[1.0, 0.0, 0.0]],
            "starts": [FixedPoint((0.0, 0.0, 0.0)).to_dict()],
            "ends": [FixedPoint((1.0, 0.0, 0.0)).to_dict()],
            "linewidth": 0.09,
            "render_as": "lines",
            "render_ends": False,
        }],
    }
    restored = displayable_from_dict(payload)
    child = restored[0]
    assert isinstance(child, Arrows)
    assert type(child).__name__ == "Arrows"
    assert child.quality == 0
    assert child.shaft_radius == pytest.approx(0.09)
    opts = child.options()
    assert opts["start_head"] == "None"
    assert opts["end_head"] == "None"
    dumped = displayable_to_dict(restored)
    assert dumped["objects"][0]["type"] == "Arrows"
    style = dumped["objects"][0]["line_style"]
    assert style["start_head"] == "None"
    assert style["end_head"] == "None"


def test_arrows_independent_margins_roundtrip():
    from pymolviz.meshes.Arrows import Arrows
    from pymolviz.util.line_style import LineStyle

    arrows = Arrows(
        starts=[FixedPoint((0.0, 0.0, 0.0))],
        ends=[FixedPoint((1.0, 0.0, 0.0))],
        color=[(0.2, 0.4, 0.8)],
        bypass_colormap=True,
        line_style=LineStyle(start_margin=1.25, end_margin=0.4),
        use_styled_cgo=True,
        obj_id="arr002",
    )
    collection = CGOCollection([arrows], name="pmv_arrows", obj_id="collB")
    data = displayable_to_dict(collection)
    style = data["objects"][0]["line_style"]
    assert style["start_margin"] == 1.25
    assert style["end_margin"] == 0.4
    restored = displayable_from_dict(data)
    got = restored[0].line_style
    assert got.start_margin == 1.25
    assert got.end_margin == 0.4


def test_surface_collection_roundtrip():
    from pymolviz.meshes.Surface import Surface

    surface = Surface(
        [FixedPoint((0.0, 0.0, 0.0)), FixedPoint((2.0, 0.0, 0.0))],
        atom_radius=1.2,
        probe_radius=1.4,
        algorithm="GAUSS",
        quality=1,
        radius_mode="vdw",
        vdw_scale=0.8,
        point_radii=[None, 2.1],
        color=(0.1, 0.2, 0.3),
        bypass_colormap=True,
        obj_id="surf001",
    )
    collection = CGOCollection([surface], name="pmv_surface", obj_id="collS")
    data = displayable_to_dict(collection)
    assert_plain(data)
    json.dumps(data)
    child = data["objects"][0]
    assert child["type"] == "Surface"
    assert child["algorithm"] == "GAUSS"
    assert child["probe_radius"] == pytest.approx(1.4)
    assert child["atom_radius"] == pytest.approx(1.2)
    assert child["radius_mode"] == "vdw"
    assert child["vdw_scale"] == pytest.approx(0.8)
    assert child["point_radii"][1] == pytest.approx(2.1)
    assert child["point_radii"][0] is None
    assert len(child["points"]) == 2
    restored = displayable_from_dict(data)
    got = restored[0]
    assert isinstance(got, Surface)
    assert got.algorithm == "GAUSS"
    assert got.probe_radius == pytest.approx(1.4)
    assert got.radius_mode == "vdw"
    assert got.vdw_scale == pytest.approx(0.8)
    assert got.point_radii[1] == pytest.approx(2.1)
    assert len(got.point_sources) == 2
    assert got.point_sources[1].resolve(None) == (2.0, 0.0, 0.0)


def test_surface_clip_planes_roundtrip():
    from pymolviz.meshes.Surface import Surface

    planes = [{"origin": [0.0, 0.0, 0.5], "normal": [0.0, 0.0, 1.0], "scale": 7.0}]
    surface = Surface(
        [FixedPoint((0.0, 0.0, 0.0))],
        algorithm="GAUSS",
        quality=1,
        clip_planes=planes,
        bypass_colormap=True,
        obj_id="surfclip",
    )
    collection = CGOCollection([surface], name="pmv_surface", obj_id="collC")
    data = displayable_to_dict(collection)
    assert_plain(data)
    json.dumps(data)
    child = data["objects"][0]
    assert child["clip_planes"][0]["origin"] == pytest.approx([0.0, 0.0, 0.5])
    assert child["clip_planes"][0]["normal"] == pytest.approx([0.0, 0.0, 1.0])
    restored = displayable_from_dict(data)
    got = restored[0]
    assert got.clip_planes[0]["scale"] == pytest.approx(7.0)
    assert np.all(np.asarray(got.vertices)[:, 2] >= 0.5 - 1e-5)


def test_field_coloring_roundtrip():
    from pymolviz.util.field_sample import _NATIVE_GRIDS, paint_mesh_by_field
    from pymolviz.volumetric.GridData import GridData

    _NATIVE_GRIDS.clear()
    values = np.linspace(0.0, 1.0, 8).reshape(2, 2, 2)
    grid = GridData(
        values.reshape(-1),
        step_sizes=(1.0, 1.0, 1.0),
        step_counts=(1, 1, 1),
        origin=(0.0, 0.0, 0.0),
        name="density",
    )
    _NATIVE_GRIDS[grid.id] = grid
    sphere = Sphere(
        FixedPoint((0.5, 0.5, 0.5)), 0.4,
        color=(1.0, 0.0, 0.0), bypass_colormap=True, frequency=2, obj_id="fldsph",
    )
    assert paint_mesh_by_field(sphere, grid.id, colormap="viridis", refine=False)
    collection = CGOCollection([sphere], name="pmv_spheres", obj_id="collF")
    data = displayable_to_dict(collection)
    assert_plain(data)
    json.dumps(data)
    child = data["objects"][0]
    assert child["field_id"] == str(grid.id)
    assert child["field_colormap"] == "viridis"
    restored = displayable_from_dict(data)
    got = restored[0]
    assert got.field_id == str(grid.id)
    assert got.field_colormap == "viridis"
    colors = np.asarray(got.color, dtype=float).reshape(-1, 3)
    assert colors.shape[0] == got.vertices.shape[0]
    assert len({tuple(np.round(row, 4)) for row in colors}) > 1


def test_field_colormap_spec_roundtrip():
    from pymolviz.util.colormap_spec import ColorStop, ColormapDefinition
    from pymolviz.util.field_sample import _NATIVE_GRIDS, paint_mesh_by_field
    from pymolviz.volumetric.GridData import GridData

    _NATIVE_GRIDS.clear()
    values = np.linspace(0.0, 1.0, 8).reshape(2, 2, 2)
    grid = GridData(
        values.reshape(-1),
        step_sizes=(1.0, 1.0, 1.0),
        step_counts=(1, 1, 1),
        origin=(0.0, 0.0, 0.0),
        name="density_spec",
    )
    _NATIVE_GRIDS[grid.id] = grid
    spec = ColormapDefinition(
        preset="custom",
        stops=(
            ColorStop(0.0, (0.0, 0.0, 1.0, 1.0)),
            ColorStop(1.0, (1.0, 0.0, 0.0, 1.0)),
        ),
        customized=True,
    ).to_dict()
    sphere = Sphere(
        FixedPoint((0.5, 0.5, 0.5)), 0.4,
        color=(1.0, 0.0, 0.0), bypass_colormap=True, frequency=2, obj_id="fldspec",
    )
    assert paint_mesh_by_field(
        sphere, grid.id, colormap="custom", colormap_spec=spec, refine=False,
    )
    collection = CGOCollection([sphere], name="pmv_spheres", obj_id="collSpec")
    data = displayable_to_dict(collection)
    assert_plain(data)
    child = data["objects"][0]
    assert child["field_colormap_spec"]["customized"] is True
    restored = displayable_from_dict(data)
    got = restored[0]
    assert got.field_colormap_spec["customized"] is True
    colors = np.asarray(got.color, dtype=float).reshape(-1, 3)
    assert colors.shape[0] == got.vertices.shape[0]


def test_sphere_enabled_and_clip_roundtrip():
    planes = [{"origin": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0], "scale": 4.0}]
    sphere = Sphere(
        FixedPoint((0.0, 0.0, 0.0)), 1.0,
        color=(1.0, 0.0, 0.0), bypass_colormap=True,
        clip_planes=planes, enabled=False, frequency=2, obj_id="sphclip",
    )
    collection = CGOCollection([sphere], name="pmv_spheres", obj_id="collClip")
    data = displayable_to_dict(collection)
    child = data["objects"][0]
    assert child["enabled"] is False
    assert child["clip_planes"][0]["normal"] == pytest.approx([0.0, 0.0, 1.0])
    restored = displayable_from_dict(data)
    got = restored[0]
    assert got.enabled is False
    assert got.clip_planes[0]["scale"] == pytest.approx(4.0)


def test_surface_point_enabled_roundtrip():
    from pymolviz.meshes.Surface import Surface

    surface = Surface(
        [FixedPoint((0.0, 0.0, 0.0)), FixedPoint((2.0, 0.0, 0.0))],
        algorithm="GAUSS", quality=1, bypass_colormap=True,
        point_enabled=[True, False], obj_id="surfen",
    )
    collection = CGOCollection([surface], name="pmv_surface", obj_id="collEn")
    data = displayable_to_dict(collection)
    child = data["objects"][0]
    assert child["point_enabled"] == [True, False]
    restored = displayable_from_dict(data)
    got = restored[0]
    assert got.point_enabled == [True, False]
    assert len(got.point_sources) == 2


def test_surface_point_colors_roundtrip():
    from pymolviz.meshes.Surface import Surface
    from pymolviz.wizards.builders.preview import apply_surface_color_to_mesh
    from pymolviz.wizards.builders.points import VisualPoint

    surface = Surface(
        [FixedPoint((0.0, 0.0, 0.0)), FixedPoint((6.0, 0.0, 0.0))],
        algorithm="GAUSS", quality=1, bypass_colormap=True, obj_id="surfpc",
    )
    points = [
        VisualPoint("a", "manual", 0, 0, 0, color=(1.0, 0.0, 0.0), point_source=FixedPoint((0.0, 0.0, 0.0))),
        VisualPoint("b", "manual", 6, 0, 0, color=(0.0, 0.0, 1.0), point_source=FixedPoint((6.0, 0.0, 0.0))),
    ]
    apply_surface_color_to_mesh(surface, points)
    collection = CGOCollection([surface], name="pmv_surface", obj_id="collPC")
    data = displayable_to_dict(collection)
    child = data["objects"][0]
    assert child["point_colors"][0] == pytest.approx([1.0, 0.0, 0.0])
    assert child["point_colors"][1] == pytest.approx([0.0, 0.0, 1.0])
    restored = displayable_from_dict(data)
    got = restored[0]
    assert got.point_colors[0] == pytest.approx((1.0, 0.0, 0.0))
    assert got.point_colors[1] == pytest.approx((0.0, 0.0, 1.0))
    colors = np.asarray(got.color, dtype=float).reshape(-1, 3)
    assert colors.shape[0] == got.vertices.shape[0]
    assert len({tuple(np.round(row, 3)) for row in colors}) > 1


def test_arrows_clip_and_head_radius_roundtrip():
    from pymolviz.meshes.Arrows import Arrows

    planes = [{"origin": [0.0, 0.0, 0.0], "normal": [1.0, 0.0, 0.0], "scale": 5.0}]
    arrows = Arrows(
        starts=[FixedPoint((0.0, 0.0, 0.0))],
        ends=[FixedPoint((1.0, 0.0, 0.0))],
        color=[(0.2, 0.4, 0.8)],
        bypass_colormap=True,
        head_radius=0.12,
        clip_planes=planes,
        obj_id="arrclip",
    )
    collection = CGOCollection([arrows], name="pmv_arrows", obj_id="collArrClip")
    data = displayable_to_dict(collection)
    child = data["objects"][0]
    assert child["head_radius"] == pytest.approx(0.12)
    assert child["clip_planes"][0]["normal"] == pytest.approx([1.0, 0.0, 0.0])
    restored = displayable_from_dict(data)
    got = restored[0]
    assert got.head_radius == pytest.approx(0.12)
    assert got.clip_planes[0]["scale"] == pytest.approx(5.0)


def test_field_recipe_and_visual_refs_roundtrip():
    from pymolviz.fields import Domain, Field, ensure_brick
    from pymolviz.fields.domain import BOUNDS_AROUND_SELECTION
    from pymolviz.fields.identity import GEN_DISTANCE
    from pymolviz.runtime import session as session_mod
    from pymolviz.wizards.builders.field_visual import make_field_visual

    session_mod.clear()
    field = Field(
        name="dist",
        units="Å",
        generator={
            "type": GEN_DISTANCE,
            "atoms": [{"xyz": [0.0, 0.0, 0.0], "elem": "C"}],
        },
        domain=Domain(bounds_mode=BOUNDS_AROUND_SELECTION, padding=1.0, spacing=1.0),
        obj_id="fld-dist",
    )
    ensure_brick(field)
    visual = make_field_visual(
        "IsoSurface",
        field,
        "iso",
        level=0.5,
        geometry_field_id=field.id,
        isovalues=[{"value": 0.5, "side": 1, "enabled": True}],
        clip_aabb=[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        selection="sele",
        carve=2.0,
        obj_id="iso-ref",
    )
    doc = session_document([field, visual])
    assert_plain(doc)
    json.dumps(doc)
    assert doc["objects"][0]["type"] == "Field"
    assert doc["objects"][0]["generator"]["type"] == GEN_DISTANCE
    assert "brick" not in doc["objects"][0]
    vis = doc["objects"][1]
    assert vis["geometry_field_id"] == str(field.id)
    assert vis["isovalues"][0]["value"] == pytest.approx(0.5)
    assert vis["clip_aabb"] is not None
    assert vis["selection"] == "sele"
    assert vis["carve"] == pytest.approx(2.0)
    restored = session_from_document(doc)
    kinds = [type(obj).__name__ for obj in restored]
    assert kinds[0] == "Field"
    assert kinds[1] == "IsoSurface"
    assert restored[1].geometry_field_id == str(field.id)
    assert restored[1].selection == "sele"
    assert restored[1].carve == pytest.approx(2.0)
    session_mod.clear()


def test_converted_surface_provenance_roundtrip():
    from pymolviz.meshes.Surface import Surface

    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    faces = np.array([[0, 1, 2]], dtype=int)
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=float)
    surface = Surface(
        [],
        color=(0.2, 0.6, 0.9),
        bypass_colormap=True,
        created_from={"field_id": "fld-1", "isovalue": 0.4},
        source_vertices=verts,
        source_normals=normals,
        source_faces=faces,
        obj_id="surf-from",
    )
    collection = CGOCollection([surface], name="pmv_surface", obj_id="collFrom")
    data = displayable_to_dict(collection)
    child = data["objects"][0]
    assert child["created_from"]["field_id"] == "fld-1"
    assert child["created_from"]["isovalue"] == pytest.approx(0.4)
    restored = displayable_from_dict(data)
    got = restored[0]
    assert got.created_from["field_id"] == "fld-1"
    assert np.asarray(got.vertices).shape[0] == 3


def test_color_field_recipe_and_default_id_roundtrip():
    from pymolviz.fields import Domain, KIND_SCALAR
    from pymolviz.fields.identity import GEN_DISTANCE, GEN_SIGNED_VDW
    from pymolviz.runtime import session as session_mod
    from pymolviz.wizards.builders.load_field import (
        default_color_field_id,
        field_from_selection,
        remember_default_color_field,
    )
    from pymolviz.wizards.builders.points import AtomRef, VisualPoint
    from tests.fakes.cmd import FakeCmd

    session_mod.clear()
    pts = [
        VisualPoint(
            "C1", "manual", 0.0, 0.0, 0.0,
            color=(1.0, 0.0, 0.0),
            point_source=FixedPoint((0.0, 0.0, 0.0)),
            atom_ref=AtomRef("m", 1, elem="C"),
        ),
        VisualPoint(
            "O1", "manual", 1.5, 0.0, 0.0,
            color=(0.0, 0.0, 1.0),
            point_source=FixedPoint((1.5, 0.0, 0.0)),
            atom_ref=AtomRef("m", 2, elem="O"),
        ),
    ]
    cmd = FakeCmd()
    domain = Domain(padding=1.0, spacing=1.0)
    geom = field_from_selection(cmd, pts, "dist", algorithm=GEN_DISTANCE, domain=domain)
    color = field_from_selection(
        cmd, pts, "vdw_color", algorithm=GEN_SIGNED_VDW, domain=domain,
    )
    remember_default_color_field(geom, color)
    doc = session_document([geom, color])
    assert_plain(doc)
    json.dumps(doc)
    color_dump = next(row for row in doc["objects"] if row["generator"]["type"] == GEN_SIGNED_VDW)
    assert color_dump["kind"] == KIND_SCALAR
    assert "brick" not in color_dump
    geom_dump = next(row for row in doc["objects"] if row["id"] == str(geom.id))
    assert geom_dump["default_color_field_id"] == str(color.id)
    restored = session_from_document(doc)
    geom_r = next(obj for obj in restored if obj.id == geom.id)
    color_r = next(obj for obj in restored if obj.id == color.id)
    assert default_color_field_id(geom_r) == str(color_r.id)
    assert color_r.kind == KIND_SCALAR
    session_mod.clear()


