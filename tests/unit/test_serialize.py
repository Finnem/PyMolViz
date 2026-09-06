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


def test_surface_collection_roundtrip():
    from pymolviz.meshes.Surface import Surface

    surface = Surface(
        [FixedPoint((0.0, 0.0, 0.0)), FixedPoint((2.0, 0.0, 0.0))],
        atom_radius=1.2,
        probe_radius=1.4,
        algorithm="ASA",
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
    assert child["algorithm"] == "ASA"
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
    assert got.algorithm == "ASA"
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
        algorithm="ASA",
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

