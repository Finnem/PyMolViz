"""Unit tests for PointSource and wizard hook helper."""

from __future__ import annotations

import numpy as np
import pytest

from pymolviz.points import AtomPoint, FixedPoint, PointUnresolvedError, as_point_source
from pymolviz.wizards.builders.points import (
    AtomRef,
    VisualPoint,
    _point_source_for_atom,
    nearest_atom_within,
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
