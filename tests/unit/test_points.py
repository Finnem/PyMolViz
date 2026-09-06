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
    nearest_atom_at_view_center,
    nearest_atom_within,
    update_points_from_camera,
    update_points_from_selection,
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
