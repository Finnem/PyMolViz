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
    ArrowPreview,
    SpherePreview,
    persist_live_preview,
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
