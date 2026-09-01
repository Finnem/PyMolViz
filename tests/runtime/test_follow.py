"""Follow callback: apply once after a source object or atom finishes moving."""

import pytest

from pymolviz.meshes.CGOCollection import CGOCollection
from pymolviz.meshes.Sphere import Sphere
from pymolviz.points import AtomPoint
from pymolviz.runtime import follow as follow_mod
from pymolviz.runtime import session as sess
from pymolviz.runtime.runtime import get_runtime, reset_runtime
from pymolviz.util.view import translation_ttt
from tests.fakes.cmd import FakeAtom


def _hooked_sphere(cmd, obj_id="coll001"):
    cmd.add_atom(FakeAtom("prot", 10, 1.0, 0.0, 0.0, chain="A", resi="1", name="CA"))
    sphere = Sphere(
        AtomPoint("prot", 10, chain="A", resi="1", name="CA", last_xyz=(1.0, 0.0, 0.0)),
        0.5,
        bypass_colormap=True,
        obj_id="sph1",
    )
    return CGOCollection([sphere], name="pmv_test", obj_id=obj_id)


def _sphere_center(coll):
    return coll[0].vertices.mean(axis=0)


def _pump(n=1):
    for _ in range(n):
        follow_mod.pymolviz_follow_callback()


def _armed(fake_cmd):
    reset_runtime()
    runtime = get_runtime(fake_cmd)
    follow_mod.reset_follow_state()
    coll = _hooked_sphere(fake_cmd)
    runtime.materialize(coll)
    sess.add(coll)
    _pump(1)
    return runtime, coll


def test_follow_idle_does_not_keep_scanning_session(monkeypatch, fake_cmd):
    reset_runtime()
    get_runtime(fake_cmd)
    follow_mod.reset_follow_state()
    calls = {"n": 0}

    def counting():
        calls["n"] += 1
        return []

    monkeypatch.setattr(sess, "all_objects", counting)
    _pump(1)
    _pump(3)
    assert calls["n"] == 1


def test_follow_updates_after_atom_move_even_if_view_changed(fake_cmd):
    runtime, coll = _armed(fake_cmd)
    fake_cmd._view[12] = 9.0
    fake_cmd.atoms[0].x = 5.0
    _pump(1)
    assert abs(_sphere_center(coll)[0] - 5.0) < 0.5


def test_follow_applies_when_pointer_is_not_over_viewer(monkeypatch, fake_cmd):
    runtime, coll = _armed(fake_cmd)
    fake_cmd.atoms[0].x = 5.0
    import pymolviz.wizards.pick as pick_mod
    monkeypatch.setattr(pick_mod, "pointer_over_viewer", lambda: False)
    _pump(1)
    assert abs(_sphere_center(coll)[0] - 5.0) < 0.5


def test_follow_does_not_rebuild_during_object_motion(fake_cmd):
    runtime, coll = _armed(fake_cmd)
    syncs = {"n": 0}
    orig = runtime.sync

    def counted(obj):
        syncs["n"] += 1
        return orig(obj)

    runtime.sync = counted
    fake_cmd.set_object_ttt("prot", translation_ttt((1.0, 0.0, 0.0)))
    _pump(1)
    fake_cmd.set_object_ttt("prot", translation_ttt((2.0, 0.0, 0.0)))
    _pump(1)
    fake_cmd.set_object_ttt("prot", translation_ttt((3.0, 0.0, 0.0)))
    _pump(1)
    assert syncs["n"] == 0


def test_follow_translates_visual_after_atom_move(fake_cmd):
    runtime, coll = _armed(fake_cmd)
    syncs = {"n": 0}
    orig = runtime.sync

    def counted(obj):
        syncs["n"] += 1
        return orig(obj)

    runtime.sync = counted
    loads = {"n": 0}
    orig_load = fake_cmd.load_cgo

    def counted_load(*args, **kwargs):
        loads["n"] += 1
        return orig_load(*args, **kwargs)

    fake_cmd.load_cgo = counted_load
    loads["n"] = 0
    fake_cmd.atoms[0].x = 5.0
    _pump(1)
    assert syncs["n"] == 0
    assert loads["n"] >= 1
    assert abs(_sphere_center(coll)[0] - 5.0) < 0.5
    _pump(3)
    assert syncs["n"] == 0


def test_follow_applies_first_atom_move_after_create(fake_cmd):
    reset_runtime()
    runtime = get_runtime(fake_cmd)
    follow_mod.reset_follow_state()
    coll = _hooked_sphere(fake_cmd)
    runtime.materialize(coll)
    sess.add(coll)
    fake_cmd.atoms[0].x = 5.0
    follow_mod.pymolviz_follow_callback()
    assert abs(_sphere_center(coll)[0] - 5.0) < 0.5


def test_follow_copies_object_ttt_after_move(fake_cmd):
    runtime, coll = _armed(fake_cmd)
    loads = {"n": 0}
    orig_load = fake_cmd.load_cgo

    def counted_load(*args, **kwargs):
        loads["n"] += 1
        return orig_load(*args, **kwargs)

    fake_cmd.load_cgo = counted_load
    loads["n"] = 0
    moved = translation_ttt((4.0, 1.0, 0.0))
    fake_cmd.set_object_ttt("prot", moved)
    _pump(1)
    visual = runtime.bindings.get(coll.id).pymol_name
    assert fake_cmd.settings[visual]["_ttt"] == moved
    assert loads["n"] == 0


def test_follow_shifts_vertices_when_anchored_atoms_move_independently(fake_cmd):
    reset_runtime()
    runtime = get_runtime(fake_cmd)
    follow_mod.reset_follow_state()
    fake_cmd.add_atom(FakeAtom("prot", 10, 1.0, 0.0, 0.0, chain="A", resi="1", name="CA"))
    fake_cmd.add_atom(FakeAtom("prot", 11, 2.0, 0.0, 0.0, chain="A", resi="2", name="CA"))
    spheres = [
        Sphere(
            AtomPoint("prot", 10, chain="A", resi="1", name="CA", last_xyz=(1.0, 0.0, 0.0)),
            0.5,
            bypass_colormap=True,
            obj_id="sph1",
        ),
        Sphere(
            AtomPoint("prot", 11, chain="A", resi="2", name="CA", last_xyz=(2.0, 0.0, 0.0)),
            0.5,
            bypass_colormap=True,
            obj_id="sph2",
        ),
    ]
    coll = CGOCollection(spheres, name="pmv_test", obj_id="coll_multi")
    runtime.materialize(coll)
    sess.add(coll)
    _pump(1)
    syncs = {"n": 0}
    orig = runtime.sync

    def counted(obj):
        syncs["n"] += 1
        return orig(obj)

    runtime.sync = counted
    replaces = {"n": 0}
    orig_replace = runtime.replace_cgo

    def counted_replace(obj):
        replaces["n"] += 1
        return orig_replace(obj)

    runtime.replace_cgo = counted_replace
    fake_cmd.atoms[1].x = 8.0
    _pump(1)
    assert syncs["n"] == 0
    assert replaces["n"] == 1
    c1 = spheres[0].vertices.mean(axis=0)
    c2 = spheres[1].vertices.mean(axis=0)
    assert abs(c1[0] - 1.0) < 0.5
    assert abs(c2[0] - 8.0) < 0.5
    assert spheres[1].position.last_xyz[0] == pytest.approx(8.0)


def _cgo_vertex_xs(tokens):
    """X of each VERTEX, skipping the primitive mode after BEGIN (aliases VERTEX in PyMOL)."""
    from pymol import cgo

    xs = []
    i = 0
    n = len(tokens)
    after_begin = False
    while i < n:
        if after_begin:
            after_begin = False
            i += 1
            continue
        tok = tokens[i]
        if tok == cgo.BEGIN:
            after_begin = True
            i += 1
            continue
        if tok == cgo.VERTEX and i + 3 < n:
            xs.append(float(tokens[i + 1]))
            i += 4
            continue
        i += 1
    if not xs:
        raise AssertionError("no VERTEX in loaded CGO")
    return xs


def test_follow_reloads_shifted_cgo_tokens(fake_cmd):
    runtime, coll = _armed(fake_cmd)
    visual = runtime.bindings.get(coll.id).pymol_name
    xs0 = _cgo_vertex_xs(fake_cmd.objects[visual])
    x0 = sum(xs0) / len(xs0)
    fake_cmd.atoms[0].x = 5.0
    _pump(1)
    xs = _cgo_vertex_xs(fake_cmd.objects[visual])
    x1 = sum(xs) / len(xs)
    assert x1 == pytest.approx(x0 + 4.0, abs=1e-5)


def test_follow_atom_move_does_not_copy_source_ttt(fake_cmd):
    """Atom edits shift object-space vertices; they must not apply the molecule TTT."""
    runtime, coll = _armed(fake_cmd)
    visual = runtime.bindings.get(coll.id).pymol_name
    moved = translation_ttt((4.0, 1.0, 0.0))
    fake_cmd.set_object_ttt("prot", moved)
    _pump(1)
    assert fake_cmd.settings[visual]["_ttt"] == moved
    x0 = _sphere_center(coll)[0]
    fake_cmd.atoms[0].x = 5.0
    _pump(1)
    assert fake_cmd.settings[visual]["_ttt"] == moved
    assert fake_cmd.settings[visual]["_ttt"] != fake_cmd.get_object_matrix("prot")
    assert _sphere_center(coll)[0] == pytest.approx(x0 + 4.0, abs=0.5)
    xs = _cgo_vertex_xs(fake_cmd.objects[visual])
    assert sum(xs) / len(xs) == pytest.approx(x0 + 4.0, abs=0.5)


def test_follow_atom_only_does_not_set_visual_ttt(fake_cmd):
    runtime, coll = _armed(fake_cmd)
    visual = runtime.bindings.get(coll.id).pymol_name
    fake_cmd.atoms[0].x = 5.0
    _pump(1)
    stored = fake_cmd.settings.get(visual, {}).get("_ttt")
    assert stored in (None, [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ])
    assert abs(_sphere_center(coll)[0] - 5.0) < 0.5


def test_follow_skips_load_cgo_while_visual_disabled(fake_cmd):
    runtime, coll = _armed(fake_cmd)
    visual = runtime.bindings.get(coll.id).pymol_name
    fake_cmd.disable(visual)
    loads = {"n": 0}
    orig_load = fake_cmd.load_cgo

    def counted_load(*args, **kwargs):
        loads["n"] += 1
        return orig_load(*args, **kwargs)

    fake_cmd.load_cgo = counted_load
    loads["n"] = 0
    xs0 = _cgo_vertex_xs(fake_cmd.objects[visual])
    x0 = sum(xs0) / len(xs0)
    fake_cmd.atoms[0].x = 5.0
    _pump(1)
    assert loads["n"] == 0
    assert abs(_sphere_center(coll)[0] - 5.0) < 0.5
    xs_hidden = _cgo_vertex_xs(fake_cmd.objects[visual])
    assert sum(xs_hidden) / len(xs_hidden) == pytest.approx(x0, abs=1e-5)
    fake_cmd.enable(visual)
    _pump(1)
    assert loads["n"] >= 1
    xs1 = _cgo_vertex_xs(fake_cmd.objects[visual])
    assert sum(xs1) / len(xs1) == pytest.approx(x0 + 4.0, abs=1e-5)
