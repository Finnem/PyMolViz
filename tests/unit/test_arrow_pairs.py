"""Arrow pair labels, status, and selection-first creation."""

from __future__ import annotations

import pytest

from pymolviz.points import AtomPoint, FixedPoint
from pymolviz.runtime.context import ResolveContext
from pymolviz.wizards.builders.pairs import (
    PENDING_END,
    STATUS_MISSING,
    STATUS_OK,
    STATUS_PICKING,
    VisualPair,
    commit_pair_anchors,
    complete_pairs,
    endpoint_label,
    flatten_pair_points,
    free_point_display_names,
    pair_row_text,
    pair_status,
    pair_status_glyph,
    take_selection_endpoints,
)
from pymolviz.wizards.builders.points import AtomRef, VisualPoint, atom_anchor_label
from pymolviz.wizards.builders.zoom_selection import focus_visual_point, points_from_pair_rows
from tests.fakes.cmd import FakeAtom


def _atom_point(chain, resi, name, xyz=(0.0, 0.0, 0.0), atom_id=1, model="prot"):
    src = AtomPoint(
        model, atom_id, chain=chain, resi=resi, name=name, last_xyz=xyz,
    )
    return VisualPoint(
        "long_internal_name",
        "selection",
        xyz[0], xyz[1], xyz[2],
        point_source=src,
        atom_ref=AtomRef(model, atom_id, chain, resi, name),
    )


def _free_point(xyz, name="cam_1"):
    return VisualPoint(name, "manual", xyz[0], xyz[1], xyz[2], point_source=FixedPoint(xyz))


def test_atom_anchor_label_compact():
    assert atom_anchor_label(AtomRef("prot", 7, "A", "42", "CA")) == "A/42/CA"
    assert atom_anchor_label(AtomRef("prot", 7, "", "15", "N")) == "15/N"
    assert atom_anchor_label(None) == ""


def test_endpoint_label_prefers_anchor_over_internal_name():
    pt = _atom_point("A", "42", "CA")
    assert endpoint_label(pt) == "A/42/CA"
    assert "long_internal" not in endpoint_label(pt)


def test_endpoint_label_pending_and_free_points():
    assert endpoint_label(None) == PENDING_END
    start = _free_point((1.0, 2.0, 3.0))
    end = _atom_point("B", "17", "N", atom_id=2)
    pair = VisualPair(start, end)
    names = free_point_display_names([pair])
    assert endpoint_label(start, free_names=names) == "Point 1"
    assert endpoint_label(end, free_names=names) == "B/17/N"
    assert pair_row_text(pair, names) == "Point 1  →  B/17/N"


def test_pair_row_text_uses_title_when_set():
    pair = VisualPair(_atom_point("A", "42", "CA"), _atom_point("A", "87", "CA", atom_id=2))
    pair = pair.with_title("Helix 1")
    assert pair_row_text(pair) == "Helix 1"


def test_pair_status_picking_ok_missing(fake_cmd):
    complete = VisualPair(_atom_point("A", "42", "CA"), _atom_point("A", "87", "CA", atom_id=2))
    assert pair_status(complete) == STATUS_OK
    assert pair_status_glyph(STATUS_OK) == "✓"

    pending = VisualPair(_atom_point("A", "51", "N"), None)
    assert pair_status(pending) == STATUS_PICKING
    assert pair_status_glyph(STATUS_PICKING) == "●"

    missing_src = AtomPoint("gone", 1, chain="A", resi="92", name="CA")
    missing_start = VisualPoint(
        "gone", "selection", 0, 0, 0,
        point_source=missing_src,
        atom_ref=AtomRef("gone", 1, "A", "92", "CA"),
    )
    missing = VisualPair(missing_start, _atom_point("A", "10", "CA", atom_id=3))
    context = ResolveContext(fake_cmd)
    assert pair_status(missing, context) == STATUS_MISSING
    assert pair_status_glyph(STATUS_MISSING) == "!"


def test_with_methods_preserve_pair_id():
    pair = VisualPair(_free_point((0, 0, 0)), _free_point((1, 0, 0)))
    pid = pair.pair_id
    other = _free_point((2, 0, 0))
    assert pair.with_end(other).pair_id == pid
    assert pair.with_color((1.0, 0.0, 0.0)).pair_id == pid
    assert pair.with_width(0.2).width == 0.2
    assert pair.with_width(0.2).head == pytest.approx(1.6)
    assert pair.swapped().start.xyz() == (1.0, 0.0, 0.0)


def test_complete_pairs_and_flatten_skip_pending_end():
    done = VisualPair(_free_point((0, 0, 0)), _free_point((1, 0, 0)))
    pending = VisualPair(_free_point((2, 0, 0)), None)
    pairs = [done, pending]
    assert complete_pairs(pairs) == [done]
    flat = flatten_pair_points(pairs)
    assert len(flat) == 3
    committed = commit_pair_anchors(pairs)
    assert len(committed) == 1
    assert committed[0].pair_id == done.pair_id


def test_points_from_pair_rows_skips_pending_end():
    pending = VisualPair(_free_point((0, 0, 0)), None)
    pts = points_from_pair_rows([pending], [0])
    assert len(pts) == 1


def test_take_selection_endpoints_ignores_pk1_when_interactive_only(fake_cmd):
    from pymolviz.wizards.builders.pairs import take_selection_endpoints
    from tests.fakes.cmd import FakeAtom

    fake_cmd.add_atom(FakeAtom("prot", 1, 0.0, 0.0, 0.0, chain="A", resi="42", name="CA"))
    fake_cmd.select("pk1", 'object "prot" and id 1')
    fake_cmd.select("sele", "none")

    start, end, status = take_selection_endpoints(fake_cmd, interactive_only=True)
    assert status == "empty"
    assert start is None and end is None

    start, end, status = take_selection_endpoints(fake_cmd, interactive_only=False)
    assert status == "one"


def test_take_selection_endpoints_counts(fake_cmd):
    fake_cmd.add_atom(FakeAtom("prot", 1, 0.0, 0.0, 0.0, chain="A", resi="42", name="CA", elem="CA"))
    fake_cmd.add_atom(FakeAtom("prot", 2, 1.0, 0.0, 0.0, chain="A", resi="87", name="CA", elem="CA"))
    fake_cmd.add_atom(FakeAtom("prot", 3, 2.0, 0.0, 0.0, chain="B", resi="15", name="N", elem="N"))

    start, end, status = take_selection_endpoints(fake_cmd)
    assert status == "empty"

    fake_cmd.select("sele", 'object "prot" and id 1')
    start, end, status = take_selection_endpoints(fake_cmd)
    assert status == "one"
    assert end is None
    assert endpoint_label(start) == "A/42/CA"

    fake_cmd.select("sele", 'object "prot" and id 1 or object "prot" and id 2')
    start, end, status = take_selection_endpoints(fake_cmd)
    assert status == "pair"
    assert endpoint_label(start) == "A/42/CA"
    assert endpoint_label(end) == "A/87/CA"

    fake_cmd.select(
        "sele",
        'object "prot" and id 1 or object "prot" and id 2 or object "prot" and id 3',
    )
    start, end, status = take_selection_endpoints(fake_cmd)
    assert status == "multiple"
    assert start is None


def test_focus_visual_point_selects_atom(fake_cmd):
    fake_cmd.add_atom(FakeAtom("prot", 7, 4.0, 5.0, 6.0, chain="A", resi="42", name="CA"))
    pt = _atom_point("A", "42", "CA", xyz=(4.0, 5.0, 6.0), atom_id=7)
    focus_visual_point(fake_cmd, pt)
    selected = fake_cmd.selections.get("sele") or []
    assert len(selected) == 1
    assert selected[0].atom_id == 7
    assert fake_cmd._last_zoom["center"] == (4.0, 5.0, 6.0)
