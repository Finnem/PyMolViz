"""Unit tests for PointSource and wizard hook helper."""

from __future__ import annotations

import numpy as np
import pytest

from pymolviz.points import AtomPoint, FixedPoint, PointUnresolvedError, as_point_source
from pymolviz.wizards.builders.points import _point_source_for_atom


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
