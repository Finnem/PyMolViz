"""Unit tests for PointSource and wizard hook helper."""

from __future__ import annotations

import numpy as np
import pytest

from pymolviz.points import AtomPoint, FixedPoint, PointUnresolvedError, as_point_source
from pymolviz.wizards.builders.points import (
    AtomRef,
    VisualPoint,
    _point_source_for_atom,
    apply_global_color,
    camera_center_point,
    export_points_to_selection,
    hide_exported_point_labels,
    nearest_atom_at_view_center,
    nearest_atom_within,
    update_points_from_camera,
    update_points_from_selection,
    POINTS_EXPORT_PREFIX,
    POINTS_EXPORT_SELE,
)


def test_as_point_source_from_tuple():
    src = as_point_source((1.0, 2.0, 3.0))
    assert isinstance(src, FixedPoint)
    assert src.resolve(None) == (1.0, 2.0, 3.0)


def test_as_point_source_from_ndarray():
    src = as_point_source(np.array([0.0, 1.5, -2.0]))
    assert src.resolve(None) == (0.0, 1.5, -2.0)


def test_as_point_source_passthrough():
    fixed = FixedPoint((4.0, 5.0, 6.0))
    assert as_point_source(fixed) is fixed


def test_fixed_point_resolve_without_context():
    pt = FixedPoint([1.0, 2.0, 3.0])
    assert pt.resolve(None) == (1.0, 2.0, 3.0)
    assert pt.has_dynamic_source() is False


def test_hook_helper_atom_when_checked():
    src = _point_source_for_atom(True, "prot", 42, "A", "15", "CA", (1.0, 2.0, 3.0))
    assert isinstance(src, AtomPoint)
    assert src.object == "prot"
    assert src.atom_id == 42
    assert src.last_xyz == (1.0, 2.0, 3.0)


def test_hook_helper_fixed_when_unchecked():
    src = _point_source_for_atom(False, "prot", 42, "A", "15", "CA", (1.0, 2.0, 3.0))
    assert isinstance(src, FixedPoint)
    assert src.resolve(None) == (1.0, 2.0, 3.0)


def test_atom_point_unresolved_without_context_or_last_xyz():
    pt = AtomPoint("prot", 1)
    with pytest.raises(PointUnresolvedError):
        pt.resolve(None)


def test_atom_point_uses_last_xyz_without_context():
    pt = AtomPoint("prot", 1, last_xyz=(7.0, 8.0, 9.0))
    assert pt.resolve(None) == (7.0, 8.0, 9.0)


def test_atom_point_remember_false_leaves_last_xyz(fake_cmd, resolve_context):
    from tests.fakes.cmd import FakeAtom

    fake_cmd.add_atom(FakeAtom("prot", 1, 4.0, 5.0, 6.0))
    pt = AtomPoint("prot", 1, last_xyz=(1.0, 2.0, 3.0))
    assert pt.resolve(resolve_context, remember=False) == (4.0, 5.0, 6.0)
    assert pt.last_xyz == (1.0, 2.0, 3.0)
    assert pt.resolve(resolve_context) == (4.0, 5.0, 6.0)
    assert pt.last_xyz == (4.0, 5.0, 6.0)


def test_atom_point_lookup_vdw(fake_cmd, resolve_context):
    from tests.fakes.cmd import FakeAtom

    fake_cmd.add_atom(FakeAtom("prot", 1, 4.0, 5.0, 6.0, elem="O", vdw=1.52))
    pt = AtomPoint("prot", 1, name="O", elem="O")
    assert pt.lookup_vdw(resolve_context) == pytest.approx(1.52)
    assert pt.last_vdw == pytest.approx(1.52)


def test_visual_point_anchor_toggle():
    ref = AtomRef("prot", 42, "A", "15", "CA")
    src = AtomPoint("prot", 42, chain="A", resi="15", name="CA", last_xyz=(1.0, 2.0, 3.0))
    pt = VisualPoint("a", "selection", 1.0, 2.0, 3.0, point_source=src, atom_ref=ref)
    assert pt.is_anchored() is True
    assert pt.can_anchor() is True

    static = pt.with_anchored(False)
    assert static.is_anchored() is False
    assert isinstance(static.point_source, FixedPoint)
    assert static.atom_ref == ref

    reanchored = static.with_anchored(True)
    assert reanchored.is_anchored() is True
    assert isinstance(reanchored.point_source, AtomPoint)
    assert reanchored.point_source.atom_id == 42
    assert reanchored.point_source.elem == ""


def test_visual_point_reanchor_keeps_elem():
    ref = AtomRef("prot", 42, "A", "15", "CA", elem="C")
    src = AtomPoint(
        "prot", 42, chain="A", resi="15", name="CA", elem="C", last_xyz=(1.0, 2.0, 3.0),
    )
    pt = VisualPoint("a", "selection", 1.0, 2.0, 3.0, point_source=src, atom_ref=ref)
    reanchored = pt.with_anchored(False).with_anchored(True)
    assert reanchored.point_source.elem == "C"


def test_visual_point_anchor_intent_defers_source_swap():
    ref = AtomRef("prot", 42, "A", "15", "CA")
    src = AtomPoint("prot", 42, chain="A", resi="15", name="CA", last_xyz=(1.0, 2.0, 3.0))
    pt = VisualPoint("a", "selection", 1.0, 2.0, 3.0, point_source=src, atom_ref=ref)
    pending = pt.with_anchor_intent(False)
    assert pending.wants_anchor() is False
    assert pending.is_anchored() is True
    assert pending.point_source is src
    committed = pending.commit_anchor()
    assert committed.is_anchored() is False
    assert isinstance(committed.point_source, FixedPoint)
    assert committed.wants_anchor() is False


def test_visual_point_manual_edit_keeps_atom_ref():
    ref = AtomRef("prot", 7, "", "1", "N")
    pt = VisualPoint(
        "a", "selection", 0.0, 0.0, 0.0,
        point_source=AtomPoint("prot", 7, last_xyz=(0.0, 0.0, 0.0)),
        atom_ref=ref,
    )
    edited = pt.with_xyz((5.0, 6.0, 7.0))
    assert edited.is_anchored() is False
    assert edited.can_anchor() is True
    assert edited.with_anchored(True).is_anchored() is True


def test_visual_point_camera_center_cannot_anchor():
    pt = VisualPoint("cam", "manual", 1.0, 2.0, 3.0)
    assert pt.can_anchor() is False
    assert pt.with_anchored(True) is pt


def test_nearest_atom_within_does_not_need_far_atoms():
    from tests.fakes.cmd import FakeAtom, FakeCmd

    cmd = FakeCmd()
    cmd.add_atom(FakeAtom("prot", 1, 0.0, 0.0, 0.0, name="CA"))
    cmd.add_atom(FakeAtom("prot", 2, 80.0, 0.0, 0.0, name="CB"))
    hit = nearest_atom_within(cmd, (0.0, 0.0, 0.0), radius=1.0)
    assert hit is not None
    assert hit["index"] == 1
    assert nearest_atom_within(cmd, (10.0, 0.0, 0.0), radius=1.0) is None


def _look_at_origin_view():
    return [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, -50.0,
        0.0, 0.0, 0.0,
        2.0, 200.0, 0.0,
    ]


def test_nearest_atom_at_view_center_requires_two_angstroms():
    from tests.fakes.cmd import FakeAtom, FakeCmd

    cmd = FakeCmd()
    cmd.set_view(_look_at_origin_view())
    cmd.add_atom(FakeAtom("prot", 1, 0.0, 0.0, 1.5, name="CA"))
    cmd.add_atom(FakeAtom("prot", 2, 80.0, 0.0, 0.0, name="CB"))
    hit = nearest_atom_at_view_center(cmd)
    assert hit is not None
    assert hit["index"] == 1
    assert hit["z"] == 1.5


def test_nearest_atom_at_view_center_ignores_atoms_beyond_two_angstroms():
    from tests.fakes.cmd import FakeAtom, FakeCmd

    cmd = FakeCmd()
    cmd.set_view(_look_at_origin_view())
    cmd.add_atom(FakeAtom("prot", 1, 0.0, 0.0, 2.5, name="CA"))
    assert nearest_atom_at_view_center(cmd) is None


def test_camera_center_snap_uses_nearby_atom():
    from pymolviz.points import AtomPoint
    from tests.fakes.cmd import FakeAtom, FakeCmd

    cmd = FakeCmd()
    cmd.set_view(_look_at_origin_view())
    cmd.add_atom(FakeAtom("prot", 1, 0.0, 0.0, 1.5, name="CA"))
    snapped = camera_center_point(cmd, snap_to_atom=True)
    assert snapped.source == "selection"
    assert snapped.xyz() == (0.0, 0.0, 1.5)
    assert isinstance(snapped.point_source, AtomPoint)
    look_at = camera_center_point(cmd, snap_to_atom=False)
    assert look_at.source == "manual"
    assert look_at.xyz() == (0.0, 0.0, 0.0)


def test_camera_center_snap_skips_when_atom_is_too_far():
    from tests.fakes.cmd import FakeAtom, FakeCmd

    cmd = FakeCmd()
    cmd.set_view(_look_at_origin_view())
    cmd.add_atom(FakeAtom("prot", 1, 0.0, 0.0, 2.5, name="CA"))
    pt = camera_center_point(cmd, snap_to_atom=True)
    assert pt.source == "manual"
    assert pt.xyz() == (0.0, 0.0, 0.0)


def test_update_points_from_camera_keeps_color():
    from tests.fakes.cmd import FakeCmd

    cmd = FakeCmd()
    cmd.set_view(_look_at_origin_view())
    pt = VisualPoint("old", "manual", 9.0, 8.0, 7.0, color=(1.0, 0.0, 0.0))
    out = update_points_from_camera(cmd, [pt], [0], snap_to_atom=False)
    assert out[0].xyz() == (0.0, 0.0, 0.0)
    assert out[0].color == (1.0, 0.0, 0.0)
    assert out[0].source == "manual"


def test_update_points_from_selection_replaces_xyz():
    from tests.fakes.cmd import FakeAtom, FakeCmd

    cmd = FakeCmd()
    cmd.add_atom(FakeAtom("prot", 1, 4.0, 5.0, 6.0, name="CA"))
    cmd.select("sele", 'object "prot" and id 1')
    pt = VisualPoint("old", "manual", 0.0, 0.0, 0.0, color=(0.0, 1.0, 0.0))
    out = update_points_from_selection(cmd, [pt], [0])
    assert out is not None
    assert out[0].xyz() == (4.0, 5.0, 6.0)
    assert out[0].color == (0.0, 1.0, 0.0)
    assert out[0].source == "selection"


def test_update_points_from_selection_empty_returns_none():
    from tests.fakes.cmd import FakeCmd

    cmd = FakeCmd()
    cmd.select("sele", "none")
    pt = VisualPoint("old", "manual", 1.0, 2.0, 3.0)
    assert update_points_from_selection(cmd, [pt], [0]) is None


def test_apply_global_color_sets_rgb_and_alpha():
    pts = [
        VisualPoint("a", "manual", 0.0, 0.0, 0.0, color=(1.0, 0.0, 0.0), alpha=1.0),
        VisualPoint("b", "manual", 1.0, 0.0, 0.0, color=(0.0, 1.0, 0.0), alpha=1.0),
    ]
    apply_global_color(pts, (0.2, 0.3, 0.4, 0.5))
    assert pts[0].color == pytest.approx((0.2, 0.3, 0.4))
    assert pts[1].color == pytest.approx((0.2, 0.3, 0.4))
    assert pts[0].alpha == pytest.approx(0.5)
    assert pts[1].alpha == pytest.approx(0.5)


def test_export_points_to_selection_shows_labels():
    from tests.fakes.cmd import FakeCmd

    cmd = FakeCmd()
    pts = [
        VisualPoint("A/12/CA", "sele", 1.0, 2.0, 3.0),
        VisualPoint("A/13/CB", "sele", 4.0, 5.0, 6.0),
    ]
    sele = export_points_to_selection(cmd, pts)
    assert sele == POINTS_EXPORT_SELE
    first = "%s_0" % POINTS_EXPORT_PREFIX
    second = "%s_1" % POINTS_EXPORT_PREFIX
    assert first in cmd.objects
    assert second in cmd.objects
    assert cmd.settings[first]["label"] == "A/12/CA"
    assert ("labels", first) in cmd._shown
    assert ("labels", second) in cmd._shown


def test_hide_exported_point_labels_only_touches_overlay():
    from tests.fakes.cmd import FakeCmd

    cmd = FakeCmd()
    cmd.objects["protein"] = []
    pts = [VisualPoint("CA", "sele", 0.0, 0.0, 0.0)]
    export_points_to_selection(cmd, pts)
    hide_exported_point_labels(cmd)
    overlay = "%s_0" % POINTS_EXPORT_PREFIX
    hidden_targets = {sel for rep, sel in cmd._hidden if rep == "labels"}
    assert overlay in hidden_targets
    assert POINTS_EXPORT_SELE in hidden_targets
    assert "protein" not in hidden_targets
    assert overlay in cmd.objects


def test_enabled_points_filter_and_roundtrip():
    from pymolviz.wizards.builders.points import (
        PointDefinition,
        definition_from_visual_point,
        enabled_points,
        visual_point_from_definition,
    )

    pts = [
        VisualPoint("a", "manual", 0, 0, 0, enabled=True),
        VisualPoint("b", "manual", 1, 0, 0, enabled=False),
    ]
    assert len(enabled_points(pts)) == 1
    assert enabled_points(pts)[0].name == "a"
    toggled = pts[1].with_enabled(True)
    assert toggled.enabled is True
    defn = definition_from_visual_point(pts[0])
    assert isinstance(defn, PointDefinition)
    restored = visual_point_from_definition(defn)
    assert restored.name == "a"
    assert restored.xyz() == (0.0, 0.0, 0.0)


def test_export_skips_disabled_points():
    from tests.fakes.cmd import FakeCmd

    cmd = FakeCmd()
    pts = [
        VisualPoint("on", "manual", 0, 0, 0, enabled=True),
        VisualPoint("off", "manual", 1, 0, 0, enabled=False),
    ]
    sele = export_points_to_selection(cmd, pts)
    assert sele == POINTS_EXPORT_SELE
    assert "%s_0" % POINTS_EXPORT_PREFIX in cmd.objects
    assert "%s_1" % POINTS_EXPORT_PREFIX not in cmd.objects


def test_active_selection_ignores_disabled_sele(fake_cmd):
    from pymolviz.wizards.builders.points import (
        _active_selection,
        resolve_insertion_points,
        selection_points,
        INSERT_SOURCE_SELECTION,
    )
    from tests.fakes.cmd import FakeAtom

    for atom_id in range(1, 23):
        fake_cmd.add_atom(FakeAtom(
            "prot", atom_id, float(atom_id), 0.0, 0.0, name="CA",
        ))
    fake_cmd.select("sele", 'object "prot"')
    assert _active_selection(fake_cmd, interactive_only=True) == "(sele)"
    assert len(selection_points(fake_cmd, interactive_only=True)) == 22

    fake_cmd.disable("sele")
    assert fake_cmd.count_atoms("sele") == 22
    assert _active_selection(fake_cmd, interactive_only=True) is None
    assert selection_points(fake_cmd, interactive_only=True) == []
    assert resolve_insertion_points(
        fake_cmd, INSERT_SOURCE_SELECTION, existing=(), snap=False, hook=True,
    ) == []

    fake_cmd.enable("sele")
    assert _active_selection(fake_cmd, interactive_only=True) == "(sele)"
    assert len(selection_points(fake_cmd, interactive_only=True)) == 22


def test_resolve_insertion_points_selection_vs_camera(fake_cmd):
    from pymolviz.wizards.builders.points import (
        INSERT_SOURCE_CAMERA,
        INSERT_SOURCE_SELECTION,
        atom_insertion_preview_label,
        insertion_preview_text,
        insertion_selection_summary,
        resolve_insertion_points,
    )
    from tests.fakes.cmd import FakeAtom

    fake_cmd.add_atom(FakeAtom("prot", 1, 1.0, 2.0, 3.0, name="CA"))
    fake_cmd.select("sele", 'object "prot" and id 1')
    selected = resolve_insertion_points(
        fake_cmd, INSERT_SOURCE_SELECTION, existing=(), snap=False, hook=True,
    )
    assert len(selected) == 1
    assert selected[0].xyz() == (1.0, 2.0, 3.0)
    assert selected[0].source == "selection"
    count, summary = insertion_selection_summary(fake_cmd)
    assert count == 1
    assert "1.00" in summary
    assert insertion_preview_text(fake_cmd, INSERT_SOURCE_SELECTION) == summary
    fake_cmd.set_view([
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, -50.0,
        1.0, 2.0, 3.0,
        2.0, 200.0, 0.0,
    ])
    camera = resolve_insertion_points(
        fake_cmd, INSERT_SOURCE_CAMERA, existing=(), snap=False, hook=True,
    )
    assert len(camera) == 1
    assert camera[0].source == "manual"
    assert insertion_preview_text(fake_cmd, INSERT_SOURCE_CAMERA).startswith("Camera center")
    snapped = resolve_insertion_points(
        fake_cmd, INSERT_SOURCE_CAMERA, existing=(), snap=True, hook=True,
    )
    assert snapped[0].source == "selection"
    assert snapped[0].xyz() == (1.0, 2.0, 3.0)
    snap_text = insertion_preview_text(fake_cmd, INSERT_SOURCE_CAMERA, snap=True)
    assert "CA" in snap_text or atom_insertion_preview_label({
        "model": "prot",
        "name": "CA",
        "elem": "C",
        "chain": "",
        "resn": "",
        "resi": "",
        "x": 1.0,
        "y": 2.0,
        "z": 3.0,
    }) in snap_text


