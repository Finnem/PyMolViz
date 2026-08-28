"""Runtime tests for materialize / sync / reconcile and session store."""

from __future__ import annotations

import pytest

from pymolviz.meshes.CGOCollection import CGOCollection
from pymolviz.meshes.Sphere import Sphere
from pymolviz.points import AtomPoint, FixedPoint
from pymolviz.runtime import session as pmv_session
from pymolviz.runtime.runtime import binding_name
from tests.fakes.cmd import FakeAtom


def _hooked_sphere_collection(cmd, obj_id="coll001"):
    cmd.add_atom(FakeAtom("prot", 10, 1.0, 0.0, 0.0, chain="A", resi="1", name="CA"))
    sphere = Sphere(
        AtomPoint("prot", 10, chain="A", resi="1", name="CA", last_xyz=(1.0, 0.0, 0.0)),
        0.5,
        bypass_colormap=True,
        obj_id="sph1",
    )
    return CGOCollection([sphere], name="pmv_test", obj_id=obj_id)


def test_materialize_creates_object(runtime, fake_cmd):
    coll = _hooked_sphere_collection(fake_cmd)
    name = runtime.materialize(coll)
    assert name == "pmv_test"
    assert name in fake_cmd.objects
    assert runtime.bindings.get(coll.id) is not None


def test_sync_reuses_binding(runtime, fake_cmd):
    coll = _hooked_sphere_collection(fake_cmd)
    runtime.materialize(coll)
    binding = runtime.bindings.get(coll.id)
    name = runtime.sync(coll)
    assert name == binding.pymol_name
    assert name in fake_cmd.objects


def test_reconcile_reuses_existing_cgo(runtime, fake_cmd):
    coll = _hooked_sphere_collection(fake_cmd, obj_id="coll002")
    name = binding_name(coll)
    fake_cmd.load_cgo([1.0, 2.0], name)
    runtime.reconcile([coll])
    binding = runtime.bindings.get(coll.id)
    assert binding is not None
    assert binding.pymol_name == name


def test_reconcile_materializes_missing(runtime, fake_cmd):
    coll = _hooked_sphere_collection(fake_cmd, obj_id="coll003")
    runtime.reconcile([coll])
    assert binding_name(coll) in fake_cmd.objects


def test_sync_moves_with_atom(runtime, fake_cmd, resolve_context):
    fake_cmd.add_atom(FakeAtom("prot", 10, 1.0, 0.0, 0.0, chain="A", resi="1", name="CA"))
    coll = _hooked_sphere_collection(fake_cmd, obj_id="coll004")
    runtime.materialize(coll)
    sphere = coll[0]
    before = sphere.vertices.mean(axis=0).copy()

    fake_cmd.atoms[0].x = 5.0
    sphere.rebuild(resolve_context)
    runtime.sync(coll)
    after = coll[0].vertices.mean(axis=0)
    assert after[0] > before[0] + 3.0


def test_session_add_and_persist(runtime, fake_cmd):
    coll = _hooked_sphere_collection(fake_cmd, obj_id="coll005")
    pmv_session.add(coll)
    blob = pmv_session.persist()
    assert blob["schema"] == 1
    assert len(blob["objects"]) == 1
    assert blob["objects"][0]["id"] == "coll005"


def test_session_restore_roundtrip(runtime, fake_cmd):
    coll = _hooked_sphere_collection(fake_cmd, obj_id="coll006")
    pmv_session.add(coll)
    pmv_session.persist()

    pmv_session.clear()
    restored = pmv_session.restore_from_session()
    assert len(restored) == 1
    assert restored[0].id == "coll006"
    runtime.reconcile(restored)
    assert binding_name(restored[0]) in fake_cmd.objects


def test_ephemeral_preview_not_persisted():
    sphere = Sphere((0, 0, 0), 1.0, bypass_colormap=True, obj_id="preview_abc123")
    coll = CGOCollection([sphere], name="_pmv_prev_test", obj_id="preview_deadbeef")
    pmv_session.add(coll)
    assert pmv_session.all_objects() == []
