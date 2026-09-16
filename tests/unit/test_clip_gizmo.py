"""Clip-gizmo visibility helpers persist independently of clip geometry."""

from pymolviz.meshes.CGOCollection import CGOCollection
from pymolviz.meshes.Sphere import Sphere
from pymolviz.util.clip_gizmo import (
    normalize_clip_gizmo_state,
    read_clip_gizmo_state,
    stamp_clip_gizmo_state,
)


def test_normalize_clip_gizmo_state_defaults_on():
    state = normalize_clip_gizmo_state(None)
    assert state == {"shown": True, "axes": [True, True, True]}
    hidden = normalize_clip_gizmo_state(False)
    assert hidden["shown"] is False
    assert hidden["axes"] == [True, True, True]


def test_stamp_and_read_on_collection_and_child():
    mesh = Sphere((0.0, 0.0, 0.0), 1.0, name="s")
    collection = CGOCollection([mesh], name="coll")
    stamp_clip_gizmo_state(collection, {"shown": False, "axes": [True, False, True]})
    assert collection.clip_gizmos["shown"] is False
    assert mesh.clip_gizmos["shown"] is False
    assert read_clip_gizmo_state(collection)["axes"][1] is False
    assert read_clip_gizmo_state(mesh)["shown"] is False
