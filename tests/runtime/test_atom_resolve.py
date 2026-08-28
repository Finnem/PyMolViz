"""Runtime tests for AtomPoint resolution and selection hook helper."""

from __future__ import annotations

import pytest

from pymolviz.points import AtomPoint, FixedPoint, PointUnresolvedError
from pymolviz.runtime.context import ResolveContext
from pymolviz.wizards.builders.points import selection_points
from tests.fakes.cmd import FakeAtom


def _add_water(cmd, name="1jxv", chain="H", resi="185", atom_id=100, xyz=(1.0, 2.0, 3.0)):
    atom = FakeAtom(
        model=name,
        atom_id=atom_id,
        x=xyz[0],
        y=xyz[1],
        z=xyz[2],
        chain=chain,
        resn="HOH",
        resi=resi,
        name="O",
        elem="O",
    )
    cmd.add_atom(atom)
    cmd.select("sele", 'object "%s" and id %d' % (name, atom_id))
    return atom


def test_atom_point_resolve_by_id(fake_cmd, resolve_context):
    _add_water(fake_cmd, atom_id=42, xyz=(5.0, 6.0, 7.0))
    pt = AtomPoint("1jxv", 42, last_xyz=(0.0, 0.0, 0.0))
    xyz = pt.resolve(resolve_context)
    assert xyz == pytest.approx((5.0, 6.0, 7.0))
    assert pt.last_xyz == pytest.approx((5.0, 6.0, 7.0))


def test_atom_point_fallback_by_residue(fake_cmd, resolve_context):
    fake_cmd.add_atom(
        FakeAtom("1jxv", 999, 8.0, 9.0, 10.0, chain="H", resi="185", name="O", elem="O")
    )
    pt = AtomPoint("1jxv", 1, chain="H", resi="185", name="O", last_xyz=(1.0, 1.0, 1.0))
    xyz = pt.resolve(resolve_context)
    assert xyz == pytest.approx((8.0, 9.0, 10.0))


def test_atom_point_unresolved_keeps_last_xyz(fake_cmd, resolve_context):
    pt = AtomPoint("missing", 1, last_xyz=(2.0, 3.0, 4.0))
    with pytest.raises(PointUnresolvedError):
        pt.resolve(resolve_context)
    assert pt.last_xyz == (2.0, 3.0, 4.0)


def test_selection_points_hook_on(fake_cmd):
    _add_water(fake_cmd)
    pts = selection_points(fake_cmd, hook_to_selection=True)
    assert len(pts) == 1
    src = pts[0].point_source
    assert isinstance(src, AtomPoint)
    assert src.object == "1jxv"


def test_selection_points_hook_off(fake_cmd):
    _add_water(fake_cmd)
    pts = selection_points(fake_cmd, hook_to_selection=False)
    assert len(pts) == 1
    assert isinstance(pts[0].point_source, FixedPoint)


def test_selection_points_empty(fake_cmd):
    fake_cmd.select("sele", "none")
    assert selection_points(fake_cmd) == []
