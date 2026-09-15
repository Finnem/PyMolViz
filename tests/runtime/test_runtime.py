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


def test_materialize_sets_object_cgo_lighting(runtime, fake_cmd):
    coll = _hooked_sphere_collection(fake_cmd)
    coll.specular = False
    name = runtime.materialize(coll)
    stored = fake_cmd.settings.get(name, {})
    assert stored.get("cgo_lighting") == 0
    assert "specular" not in stored
    assert "spec_reflect" not in stored
    coll.specular = True
    runtime.replace_cgo(coll)
    stored = fake_cmd.settings.get(name, {})
    assert stored.get("cgo_lighting") == 1


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


def test_persist_collection_replaces_same_id(fake_cmd, monkeypatch):
    monkeypatch.setattr("pymolviz.runtime.integration.install", lambda *_a, **_k: None)
    from pymolviz.wizards.builders.preview import persist_collection

    first = _hooked_sphere_collection(fake_cmd, obj_id="keep-me")
    persist_collection(fake_cmd, first)
    assert pmv_session.get("keep-me") is first
    replacement = CGOCollection(
        [
            Sphere(
                AtomPoint("prot", 10, chain="A", resi="1", name="CA", last_xyz=(2.0, 0.0, 0.0)),
                0.5,
                bypass_colormap=True,
            )
        ],
        name="pmv_updated",
        obj_id="other",
    )
    persist_collection(fake_cmd, replacement, obj_id="keep-me")
    stored = pmv_session.get("keep-me")
    assert stored is replacement
    assert stored.id == "keep-me"
    assert stored._name == "pmv_updated"


def test_visual_is_enabled_follows_pymol_enable(fake_cmd):
    from pymolviz.runtime.runtime import get_runtime
    from pymolviz.wizards.builders.preview import set_visual_enabled, visual_is_enabled

    coll = _hooked_sphere_collection(fake_cmd, obj_id="vis1")
    get_runtime(fake_cmd).materialize(coll)
    assert visual_is_enabled(fake_cmd, coll) is True
    set_visual_enabled(fake_cmd, coll, False)
    assert visual_is_enabled(fake_cmd, coll) is False
    set_visual_enabled(fake_cmd, coll, True)
    assert visual_is_enabled(fake_cmd, coll) is True


def test_delete_visual_removes_session_and_pymol(fake_cmd):
    from pymolviz.runtime.runtime import get_runtime
    from pymolviz.wizards.builders.preview import delete_visual

    coll = _hooked_sphere_collection(fake_cmd, obj_id="gone")
    pmv_session.add(coll)
    runtime = get_runtime(fake_cmd)
    name = runtime.materialize(coll)
    delete_visual(fake_cmd, coll)
    assert pmv_session.get("gone") is None
    assert name not in fake_cmd.objects
    assert runtime.bindings.get(coll.id) is None


def _spy_cgo_list(obj):
    calls = {"n": 0}

    def boom(*_a, **_k):
        calls["n"] += 1
        raise AssertionError("%s must not produce CGO" % type(obj).__name__)

    obj._create_CGO_list = boom
    return calls


def _distance_field():
    from pymolviz.fields import Domain, Field
    from pymolviz.fields.domain import BOUNDS_AROUND_SELECTION
    from pymolviz.fields.identity import GEN_DISTANCE

    return Field(
        name="dist",
        generator={
            "type": GEN_DISTANCE,
            "atoms": [{"xyz": [0.0, 0.0, 0.0], "elem": "C"}],
        },
        domain=Domain(bounds_mode=BOUNDS_AROUND_SELECTION, padding=1.0, spacing=1.0),
        obj_id="fld-dist",
    )


def test_field_is_not_loaded_as_cgo(runtime, fake_cmd):
    from pymolviz.runtime.renderer import resolved_cgo_tokens
    from pymolviz.runtime.runtime import binding_name

    field = _distance_field()
    calls = _spy_cgo_list(field)
    before = set(fake_cmd.objects)

    assert resolved_cgo_tokens(field, None) == []
    assert runtime.materialize(field) == binding_name(field)
    runtime.reconcile([field])
    runtime.sync(field)
    runtime.replace_cgo(field)

    assert calls["n"] == 0
    assert runtime.bindings.get(field.id) is None
    assert set(fake_cmd.objects) == before
    assert binding_name(field) not in fake_cmd.objects


def test_reconcile_skips_field_and_loads_field_visual(runtime, fake_cmd):
    from pymolviz.fields import ensure_brick
    from pymolviz.runtime.runtime import binding_name
    from pymolviz.wizards.builders.field_visual import make_field_visual

    field = _distance_field()
    ensure_brick(field)
    visual = make_field_visual(
        "IsoSurface",
        field,
        "iso",
        level=0.5,
        geometry_field_id=field.id,
        obj_id="iso-ref",
    )
    field_calls = _spy_cgo_list(field)
    visual_calls = _spy_cgo_list(visual)
    loaded = []

    def fake_load():
        loaded.append(visual.name)
        fake_cmd.isosurface(visual.name, "dist", 0.5)

    visual.load = fake_load
    coll = _hooked_sphere_collection(fake_cmd, obj_id="coll-field")
    runtime.reconcile([field, visual, coll])

    assert field_calls["n"] == 0
    assert visual_calls["n"] == 0
    assert loaded == ["iso"]
    assert binding_name(field) not in fake_cmd.objects
    assert visual.name in fake_cmd.objects
    assert fake_cmd.object_types.get(visual.name) == "object:isosurface"
    assert binding_name(coll) in fake_cmd.objects
    assert runtime.bindings.get(field.id) is None
    assert runtime.bindings.get(visual.id) is not None
    assert runtime.bindings.get(visual.id).pymol_name == visual.name
    assert runtime.bindings.get(coll.id) is not None
    assert visual.geometry_field_id == str(field.id)


def test_session_restore_field_is_not_cgo(runtime, fake_cmd):
    from pymolviz.fields import ensure_brick
    from pymolviz.runtime.runtime import binding_name
    from pymolviz.serialization import session_document, session_from_document
    from pymolviz.wizards.builders.field_visual import make_field_visual

    field = _distance_field()
    ensure_brick(field)
    visual = make_field_visual(
        "IsoSurface",
        field,
        "iso",
        level=0.5,
        geometry_field_id=field.id,
        obj_id="iso-ref",
    )
    pmv_session.add(field)
    pmv_session.add(visual)
    pmv_session.persist()
    pmv_session.clear()
    restored = pmv_session.restore_from_session()
    kinds = [type(obj).__name__ for obj in restored]
    assert kinds[0] == "Field"
    assert "IsoSurface" in kinds
    field_obj = next(obj for obj in restored if type(obj).__name__ == "Field")
    vis_obj = next(obj for obj in restored if type(obj).__name__ == "IsoSurface")
    assert vis_obj.geometry_field_id == str(field_obj.id)
    assert vis_obj.geometry_field_id == str(field.id)

    field_calls = _spy_cgo_list(field_obj)
    vis_obj.load = lambda: fake_cmd.isosurface(vis_obj.name, "dist", 0.5)
    runtime.reconcile(restored)

    assert field_calls["n"] == 0
    assert binding_name(field_obj) not in fake_cmd.objects
    assert vis_obj.name in fake_cmd.objects
    assert runtime.bindings.get(field_obj.id) is None

    again = session_from_document(session_document(restored))
    assert type(again[0]).__name__ == "Field"
    assert again[1].geometry_field_id == str(field.id)


def test_from_selection_uniform_commit_binds_native_isosurface(runtime, fake_cmd, monkeypatch):
    import pymol
    from pymolviz.fields import Domain
    from pymolviz.fields.field import Field
    from pymolviz.fields.identity import GEN_GAUSSIAN
    from pymolviz.points import FixedPoint
    from pymolviz.volumetric.ColorRamp import ColorRamp
    from pymolviz.volumetric.GridData import GridData
    from pymolviz.wizards.builders.load_field import commit_from_selection_preset
    from pymolviz.wizards.builders.points import AtomRef, VisualPoint

    def _field_must_not_load(self):
        raise AssertionError("Field.load must not run on preset commit")

    monkeypatch.setattr(Field, "load", _field_must_not_load)
    monkeypatch.setattr(GridData, "load", lambda self: setattr(self, "is_loaded", True))
    monkeypatch.setattr(pymol, "cmd", fake_cmd)
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
    domain = Domain(padding=2.0, spacing=1.0)
    geom, color, visual = commit_from_selection_preset(
        fake_cmd, pts, "dist", algorithm=GEN_GAUSSIAN, domain=domain,
        quality=1, iso_level=0.5,
    )
    field_calls = _spy_cgo_list(geom)
    visual_calls = _spy_cgo_list(visual)
    runtime.reconcile([geom, visual])
    assert color is None
    assert visual.color_field_id is None
    assert visual.geometry_field_id == str(geom.id)
    assert not isinstance(visual.color, ColorRamp)
    assert visual.name in fake_cmd.objects
    assert fake_cmd.object_types.get(visual.name) == "object:isosurface"
    assert field_calls["n"] == 0
    assert visual_calls["n"] == 0
    assert runtime.bindings.get(geom.id) is None
    assert runtime.bindings.get(visual.id) is not None
    assert runtime.bindings.get(visual.id).pymol_name == visual.name
