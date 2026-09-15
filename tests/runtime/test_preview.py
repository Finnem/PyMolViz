"""Preview reuses baked meshes instead of remeshing on adopt / move."""

from __future__ import annotations

import numpy as np
import pytest

from pymolviz.meshes.Arrows import Arrows
from pymolviz.meshes.CGOCollection import CGOCollection
from pymolviz.meshes.Sphere import Sphere
from pymolviz.points import FixedPoint
from pymolviz.runtime.runtime import get_runtime, reset_runtime
from pymolviz.runtime import session as pmv_session
from pymolviz.util.line_style import LineStyle
from pymolviz.wizards.builders.pairs import VisualPair
from pymolviz.wizards.builders.points import VisualPoint
from pymolviz.wizards.builders.preview import (
    PREVIEW_SPHERE_NAME,
    PREVIEW_SURFACE_CLIP_NAME,
    PREVIEW_SURFACE_NAME,
    ArrowPreview,
    SpherePreview,
    SurfacePreview,
    persist_live_preview,
    retarget_arrow_collection,
    retarget_point_collection,
)


def _point(name, xyz, color=(1.0, 0.85, 0.15)):
    return VisualPoint(
        name, "manual", xyz[0], xyz[1], xyz[2],
        color=color, point_source=FixedPoint(xyz),
    )


def _count_sphere_inits(monkeypatch):
    n = {"init": 0}
    orig = Sphere.__init__

    def wrapped(self, *args, **kwargs):
        n["init"] += 1
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(Sphere, "__init__", wrapped)
    return n


@pytest.fixture
def preview_runtime(fake_cmd):
    reset_runtime()
    get_runtime(fake_cmd)
    yield fake_cmd
    reset_runtime()


def test_sphere_preview_adopt_does_not_construct_spheres(preview_runtime, monkeypatch):
    counts = _count_sphere_inits(monkeypatch)
    s1 = Sphere((0.0, 0.0, 0.0), 1.0, bypass_colormap=True, frequency=4)
    s2 = Sphere((2.0, 0.0, 0.0), 1.0, bypass_colormap=True, frequency=4)
    s1._create_CGO_list()
    s2._create_CGO_list()
    source = CGOCollection([s1, s2], name="src")
    built = counts["init"]
    preview = SpherePreview(preview_runtime)
    preview.adopt(source)
    assert counts["init"] == built
    assert preview._preview._obj is not None
    assert len(preview._preview._obj) == 2
    assert preview._preview._obj[0] is not s1
    assert preview._preview._obj[0].faces is s1.faces


def test_sphere_preview_move_reuses_meshes(preview_runtime, monkeypatch):
    counts = _count_sphere_inits(monkeypatch)
    s1 = Sphere((0.0, 0.0, 0.0), 1.0, bypass_colormap=True, frequency=4)
    s2 = Sphere((2.0, 0.0, 0.0), 1.0, bypass_colormap=True, frequency=4)
    source = CGOCollection([s1, s2], name="src")
    preview = SpherePreview(preview_runtime)
    preview.adopt(source)
    built = counts["init"]
    points = [_point("a", (0.0, 0.0, 0.0)), _point("b", (2.0, 0.0, 0.0))]
    preview.update(points, 1.0, False, 3)
    assert counts["init"] == built
    moved = [_point("a", (0.0, 0.0, 0.0)), _point("b", (5.0, 0.0, 0.0))]
    preview.update(moved, 1.0, False, 3)
    assert counts["init"] == built
    center = np.asarray(preview._preview._obj[1].vertices).mean(axis=0)
    assert center[0] == pytest.approx(5.0, abs=0.5)
    original = np.asarray(s2.vertices).mean(axis=0)
    assert original[0] == pytest.approx(2.0, abs=0.5)


def test_sphere_preview_radius_change_remeshes(preview_runtime, monkeypatch):
    counts = _count_sphere_inits(monkeypatch)
    s1 = Sphere((0.0, 0.0, 0.0), 1.0, bypass_colormap=True, frequency=4)
    source = CGOCollection([s1], name="src")
    preview = SpherePreview(preview_runtime)
    preview.adopt(source)
    built = counts["init"]
    points = [_point("a", (0.0, 0.0, 0.0))]
    preview.update(points, 2.0, False, 3)
    assert counts["init"] == built + 1


def test_sphere_preview_add_point_keeps_existing_meshes(preview_runtime, monkeypatch):
    counts = _count_sphere_inits(monkeypatch)
    s1 = Sphere((0.0, 0.0, 0.0), 1.0, bypass_colormap=True, frequency=4)
    s2 = Sphere((2.0, 0.0, 0.0), 1.0, bypass_colormap=True, frequency=4)
    source = CGOCollection([s1, s2], name="src")
    preview = SpherePreview(preview_runtime)
    preview.adopt(source)
    child0 = preview.collection[0]
    child1 = preview.collection[1]
    built = counts["init"]
    creates = {"n": 0}
    orig = child0._create_CGO_list

    def wrapped():
        creates["n"] += 1
        return orig()

    child0._create_CGO_list = wrapped
    assert preview.add_points([_point("c", (4.0, 0.0, 0.0))], 1.0, False, 3) is True
    assert counts["init"] == built + 1
    assert len(preview.collection) == 3
    assert preview.collection[0] is child0
    assert preview.collection[1] is child1
    assert creates["n"] == 0


def test_sphere_preview_remove_point_keeps_other_meshes(preview_runtime, monkeypatch):
    counts = _count_sphere_inits(monkeypatch)
    s1 = Sphere((0.0, 0.0, 0.0), 1.0, bypass_colormap=True, frequency=4)
    s2 = Sphere((2.0, 0.0, 0.0), 1.0, bypass_colormap=True, frequency=4)
    s3 = Sphere((4.0, 0.0, 0.0), 1.0, bypass_colormap=True, frequency=4)
    source = CGOCollection([s1, s2, s3], name="src")
    preview = SpherePreview(preview_runtime)
    preview.adopt(source)
    child1 = preview.collection[1]
    child2 = preview.collection[2]
    center1 = np.asarray(child1.vertices).mean(axis=0).copy()
    built = counts["init"]
    creates = {"n": 0}
    orig = child1._create_CGO_list

    def wrapped():
        creates["n"] += 1
        return orig()

    child1._create_CGO_list = wrapped
    preview.remove_rows([0])
    assert counts["init"] == built
    assert creates["n"] == 0
    assert len(preview.collection) == 2
    assert preview.collection[0] is child1
    assert preview.collection[1] is child2
    assert np.allclose(np.asarray(child1.vertices).mean(axis=0), center1)


def test_persist_live_preview_promotes_meshes_without_remesh(preview_runtime, monkeypatch):
    monkeypatch.setattr("pymolviz.runtime.integration.install", lambda *_a, **_k: None)
    counts = _count_sphere_inits(monkeypatch)
    s1 = Sphere((0.0, 0.0, 0.0), 1.0, bypass_colormap=True, frequency=4)
    source = CGOCollection([s1], name="src")
    preview = SpherePreview(preview_runtime)
    preview.adopt(source)
    points = [_point("a", (1.0, 0.0, 0.0))]
    preview.update(points, 1.0, False, 3)
    built = counts["init"]
    mesh = preview.collection[0]
    persist_live_preview(
        preview_runtime,
        preview,
        "pmv_spheres",
        retarget=lambda coll: retarget_point_collection(coll, points),
        fallback=lambda: pytest.fail("should reuse preview meshes"),
    )
    assert counts["init"] == built
    stored = pmv_session.all_objects()
    assert len(stored) == 1
    assert stored[0][0] is mesh
    assert stored[0].id[:8] != "preview_"
    assert "pmv_spheres" in preview_runtime.objects
    assert PREVIEW_SPHERE_NAME not in preview_runtime.objects


def test_persist_live_preview_replaces_existing_id(preview_runtime, monkeypatch):
    monkeypatch.setattr("pymolviz.runtime.integration.install", lambda *_a, **_k: None)
    original = Sphere((0.0, 0.0, 0.0), 1.0, bypass_colormap=True, frequency=4)
    existing = CGOCollection([original], name="pmv_spheres", obj_id="keep-me")
    get_runtime(preview_runtime).materialize(existing, rebuild=False)
    pmv_session.add(existing)

    counts = _count_sphere_inits(monkeypatch)
    preview = SpherePreview(preview_runtime)
    preview.adopt(existing)
    points = [_point("a", (4.0, 0.0, 0.0))]
    preview.update(points, 1.0, False, 3)
    built = counts["init"]
    mesh = preview.collection[0]
    persist_live_preview(
        preview_runtime,
        preview,
        "pmv_spheres",
        obj_id="keep-me",
        retarget=lambda coll: retarget_point_collection(coll, points),
        fallback=lambda: pytest.fail("should reuse preview meshes"),
    )
    assert counts["init"] == built
    stored = pmv_session.get("keep-me")
    assert stored[0] is mesh
    assert stored[0] is not original
    assert "pmv_spheres" in preview_runtime.objects
    assert PREVIEW_SPHERE_NAME not in preview_runtime.objects


def _pair(start_xyz, end_xyz):
    return VisualPair(_point("s", start_xyz), _point("e", end_xyz))


def test_arrow_preview_add_pair_does_not_rebuild_existing(preview_runtime):
    style = LineStyle()
    pairs = [_pair((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), _pair((0.0, 1.0, 0.0), (1.0, 1.0, 0.0))]
    source = CGOCollection(
        [Arrows(
            starts=[pair.start.point_source for pair in pairs],
            ends=[pair.end.point_source for pair in pairs],
            color=[pair.color for pair in pairs],
            transparency=[0.0, 0.0],
            quality=3,
            line_style=style,
            use_styled_cgo=True,
            bypass_colormap=True,
        )],
        name="src",
    )
    preview = ArrowPreview(preview_runtime)
    preview.adopt(source)
    mesh = preview.collection[0]
    extra = _pair((0.0, 2.0, 0.0), (1.0, 2.0, 0.0))
    assert preview.add_pairs([extra], 3, style, current_pairs=pairs) is True
    assert preview.collection[0] is mesh
    assert mesh.vertices.reshape(-1, 2, 3).shape[0] == 3


def test_arrow_preview_remove_pair_keeps_other_geometry(preview_runtime):
    style = LineStyle()
    pairs = [
        _pair((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        _pair((0.0, 1.0, 0.0), (1.0, 1.0, 0.0)),
        _pair((0.0, 2.0, 0.0), (1.0, 2.0, 0.0)),
    ]
    source = CGOCollection(
        [Arrows(
            starts=[pair.start.point_source for pair in pairs],
            ends=[pair.end.point_source for pair in pairs],
            color=[pair.color for pair in pairs],
            transparency=[0.0, 0.0, 0.0],
            quality=3,
            line_style=style,
            use_styled_cgo=True,
            bypass_colormap=True,
        )],
        name="src",
    )
    preview = ArrowPreview(preview_runtime)
    preview.adopt(source)
    mesh = preview.collection[0]
    kept = np.array(mesh.vertices[2:4], copy=True)
    preview.remove_rows([0], 3, style, pairs)
    assert preview.collection[0] is mesh
    assert mesh.vertices.reshape(-1, 2, 3).shape[0] == 2
    assert np.allclose(mesh.vertices[0:2], kept)


def test_arrow_preview_skips_incomplete_pair(preview_runtime):
    style = LineStyle()
    complete = _pair((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    pending = VisualPair(_point("s", (0.0, 2.0, 0.0)), None)
    preview = ArrowPreview(preview_runtime)
    preview.update([complete, pending], 3, style)
    mesh = preview.collection[0]
    assert mesh.vertices.reshape(-1, 2, 3).shape[0] == 1
    preview.update([complete, pending], 3, style, highlight_id=complete.pair_id)
    assert preview.collection[0].vertices.reshape(-1, 2, 3).shape[0] == 1


def test_persist_arrow_width_reloads_pymol_cgo(preview_runtime, monkeypatch):
    from dataclasses import replace

    monkeypatch.setattr("pymolviz.runtime.integration.install", lambda *_a, **_k: None)
    style = LineStyle()
    thin = [_pair((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))]
    preview = ArrowPreview(preview_runtime)
    preview.update(thin, 3, style, head_radius=None)
    persist_live_preview(
        preview_runtime,
        preview,
        "pmv_arrows",
        retarget=lambda coll: retarget_arrow_collection(coll, thin, head_radius=None),
        fallback=lambda: pytest.fail("should reuse preview meshes"),
    )
    stored = pmv_session.all_objects()[0]
    before = list(preview_runtime.objects["pmv_arrows"])
    assert stored[0].pair_radii[0] == pytest.approx(0.045)

    fat = [replace(thin[0], width=0.2)]
    preview = ArrowPreview(preview_runtime)
    preview.adopt(stored)
    preview.update(fat, 3, style, head_radius=None)
    persist_live_preview(
        preview_runtime,
        preview,
        "pmv_arrows",
        obj_id=stored.id,
        retarget=lambda coll: retarget_arrow_collection(coll, fat, head_radius=None),
        fallback=lambda: pytest.fail("should reuse preview meshes"),
    )
    updated = pmv_session.get(stored.id)
    after = list(preview_runtime.objects["pmv_arrows"])
    assert updated[0].pair_radii[0] == pytest.approx(0.2)
    assert updated[0].head_radius is None
    assert after != before


def test_surface_preview_adopt_does_not_remesh(preview_runtime, monkeypatch):
    from pymolviz.meshes.Surface import Surface

    n = {"init": 0}
    orig = Surface.__init__

    def wrapped(self, *args, **kwargs):
        n["init"] += 1
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(Surface, "__init__", wrapped)
    mesh = Surface([(0.0, 0.0, 0.0)], quality=1, algorithm="GAUSS", bypass_colormap=True)
    mesh._create_CGO_list()
    built = n["init"]
    preview = SurfacePreview(preview_runtime)
    preview.adopt(CGOCollection([mesh], name="src"))
    assert n["init"] == built
    assert preview.collection is not None
    assert preview.collection[0] is not mesh
    assert preview.collection[0].faces is mesh.faces


def test_surface_preview_update_remeshes(preview_runtime, monkeypatch):
    from pymolviz.meshes.Surface import Surface

    n = {"init": 0}
    orig = Surface.__init__

    def wrapped(self, *args, **kwargs):
        n["init"] += 1
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(Surface, "__init__", wrapped)
    mesh = Surface([(0.0, 0.0, 0.0)], quality=1, algorithm="GAUSS", bypass_colormap=True)
    preview = SurfacePreview(preview_runtime)
    preview.adopt(CGOCollection([mesh], name="src"))
    built = n["init"]
    preview.update([_point("a", (0.0, 0.0, 0.0))], 1.5, 1.4, "GAUSS", 1, False)
    assert n["init"] > built
    assert PREVIEW_SURFACE_NAME in preview_runtime.objects


def test_surface_preview_reuses_mesh_for_wireframe_and_color(preview_runtime, monkeypatch):
    from pymolviz.meshes.Surface import Surface
    from pymolviz.util.solvent_surface import DEFAULT_ATOM_RADIUS, DEFAULT_PROBE_RADIUS

    n = {"init": 0}
    orig = Surface.__init__

    def wrapped(self, *args, **kwargs):
        n["init"] += 1
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(Surface, "__init__", wrapped)
    preview = SurfacePreview(preview_runtime)
    pt = _point("a", (0.0, 0.0, 0.0), color=(1.0, 0.0, 0.0))
    preview.update([pt], DEFAULT_ATOM_RADIUS, DEFAULT_PROBE_RADIUS, "SASA", 1, False)
    built = n["init"]
    mesh = preview.collection[0]
    preview.update([pt], DEFAULT_ATOM_RADIUS, DEFAULT_PROBE_RADIUS, "SASA", 1, True)
    assert n["init"] == built
    assert preview.collection[0] is mesh
    assert mesh.wireframe is True
    kinds = [t for t in mesh._create_CGO_list() if isinstance(t, str)]
    assert "CONE" in kinds
    assert "TRIANGLES" not in kinds
    recolored = _point("a", (0.0, 0.0, 0.0), color=(0.0, 1.0, 0.0))
    preview.update([recolored], DEFAULT_ATOM_RADIUS, DEFAULT_PROBE_RADIUS, "SASA", 1, True)
    assert n["init"] == built
    assert np.allclose(
        np.asarray(preview.collection[0].color, dtype=float).reshape(-1)[:3],
        (0.0, 1.0, 0.0),
    )


def test_surface_preview_gizmos_are_separate_from_mesh(preview_runtime):
    from pymolviz.util.clip_drag import CLIP_DRAG_NAME
    from pymolviz.util.solvent_surface import DEFAULT_ATOM_RADIUS, DEFAULT_PROBE_RADIUS

    preview = SurfacePreview(preview_runtime)
    pt = _point("a", (0.0, 0.0, 0.0))
    planes = [{
        "origin": [0.0, 0.0, 0.0],
        "normal": [0.0, 0.0, 1.0],
        "scale": 5.0,
        "committed": False,
    }]
    preview.update(
        [pt], DEFAULT_ATOM_RADIUS, DEFAULT_PROBE_RADIUS, "GAUSS", 1, False,
        gizmo_planes=planes, gizmo_selected=0,
    )
    assert len(preview.collection) == 1
    assert type(preview.collection[0]).__name__ == "Surface"
    assert PREVIEW_SURFACE_NAME in preview_runtime.objects
    assert PREVIEW_SURFACE_CLIP_NAME in preview_runtime.objects
    preview.cleanup()
    assert PREVIEW_SURFACE_CLIP_NAME not in preview_runtime.objects
    assert CLIP_DRAG_NAME not in preview_runtime.objects
    assert preview_runtime.get_drag_object_name() == ""


def test_surface_preview_attaches_native_drag_widget(preview_runtime):
    from pymolviz.util.clip_drag import CLIP_DRAG_NAME
    from pymolviz.util.solvent_surface import DEFAULT_ATOM_RADIUS, DEFAULT_PROBE_RADIUS
    from pymolviz.util.view import translation_ttt

    preview = SurfacePreview(preview_runtime)
    pt = _point("a", (0.0, 0.0, 0.0))
    planes = [{
        "origin": [0.0, 0.0, 0.0],
        "normal": [0.0, 0.0, 1.0],
        "scale": 5.0,
        "committed": False,
    }]
    preview.update(
        [pt], DEFAULT_ATOM_RADIUS, DEFAULT_PROBE_RADIUS, "GAUSS", 1, False,
        gizmo_planes=planes, gizmo_selected=0,
    )
    assert CLIP_DRAG_NAME in preview_runtime.objects
    assert preview_runtime.get_drag_object_name() == CLIP_DRAG_NAME
    assert preview.poll_clip_drag() is None
    preview_runtime.set_object_ttt(CLIP_DRAG_NAME, translation_ttt((0.0, 0.0, 2.0)))
    pose = preview.poll_clip_drag()
    assert pose is not None
    origin, normal = pose
    assert origin[2] == pytest.approx(2.0)
    assert normal == pytest.approx([0.0, 0.0, 1.0])
    preview.cleanup()
    assert CLIP_DRAG_NAME not in preview_runtime.objects
    assert preview_runtime.get_drag_object_name() == ""
    assert preview_runtime.get("button_mode") == 0


def test_surface_preview_keeps_drag_matrix_during_live_update(preview_runtime):
    from pymolviz.util.clip_drag import CLIP_DRAG_NAME
    from pymolviz.util.solvent_surface import DEFAULT_ATOM_RADIUS, DEFAULT_PROBE_RADIUS
    from pymolviz.util.view import translation_ttt

    preview = SurfacePreview(preview_runtime)
    pt = _point("a", (0.0, 0.0, 0.0))
    planes = [{
        "origin": [0.0, 0.0, 0.0],
        "normal": [0.0, 0.0, 1.0],
        "scale": 5.0,
        "committed": False,
    }]
    preview.update(
        [pt], DEFAULT_ATOM_RADIUS, DEFAULT_PROBE_RADIUS, "GAUSS", 1, False,
        gizmo_planes=planes, gizmo_selected=0,
    )
    preview_runtime.set_object_ttt(CLIP_DRAG_NAME, translation_ttt((0.0, 0.0, 2.0)))
    moved = [{
        "origin": [0.0, 0.0, 2.0],
        "normal": [0.0, 0.0, 1.0],
        "scale": 5.0,
        "committed": False,
    }]
    preview.update(
        [pt], DEFAULT_ATOM_RADIUS, DEFAULT_PROBE_RADIUS, "GAUSS", 1, False,
        gizmo_planes=moved, gizmo_selected=0,
    )
    assert preview_runtime.get_drag_object_name() == CLIP_DRAG_NAME
    pose = preview.poll_clip_drag()
    assert pose is not None
    origin, _normal = pose
    assert origin[2] == pytest.approx(2.0)
    preview.set_gizmos(moved, selected_index=0, attach_drag=True)
    rest = preview._gizmos._clip_drag_rest
    assert rest is not None
    assert rest["origin"][2] == pytest.approx(0.0)
    preview._gizmos._clip_drag_matrix = None
    pose = preview.poll_clip_drag()
    assert pose is not None
    origin, _normal = pose
    assert origin[2] == pytest.approx(2.0)
    preview.cleanup()
    assert preview_runtime.get("button_mode") == 0

