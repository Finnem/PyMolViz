"""Unit tests for schema-1 serialization."""

from __future__ import annotations

import json

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
    data = AtomPoint("prot", 7, chain="A", resi="12", name="CA", last_xyz=[1.0, 2.0, 3.0]).to_dict()
    restored = point_source_from_dict(data)
    assert isinstance(restored, AtomPoint)
    assert restored.object == "prot"
    assert restored.atom_id == 7
    assert restored.last_xyz == (1.0, 2.0, 3.0)


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
