"""Carve-around target normalization for field visuals."""

from __future__ import annotations

import pytest

from pymolviz.wizards.builders.carve_around import normalize_carve_args
from pymolviz.wizards.builders.field_visual import field_visual_options, make_field_visual


def test_normalize_carve_args_requires_target_and_positive_radius():
    assert normalize_carve_args(None, 2.0) == (None, None)
    assert normalize_carve_args("protein", None) == ("protein", None)
    assert normalize_carve_args("protein", 0.0) == (None, None)
    assert normalize_carve_args("protein", 2.5) == ("protein", 2.5)
    assert normalize_carve_args("(choose target)", 2.0) == (None, None)
    assert normalize_carve_args("", 2.0) == (None, None)


def test_placeholder_carve_is_not_stored_on_volume():
    from tests.unit.test_field_visuals import _tiny_grid

    grid = _tiny_grid()
    vol = make_field_visual(
        "Volume", grid, "vol", selection="(choose target)", carve=2.0,
    )
    assert vol.selection is None
    assert vol.carve is None


def test_make_field_visual_passes_carve_to_iso_and_volume():
    from tests.unit.test_field_visuals import _tiny_grid

    grid = _tiny_grid()
    iso = make_field_visual(
        "IsoSurface", grid, "iso", level=1.0, selection="sele", carve=3.0,
    )
    assert iso.selection == "sele"
    assert iso.carve == pytest.approx(3.0)
    vol = make_field_visual(
        "Volume", grid, "vol", selection="obj1", carve=1.5,
    )
    assert vol.selection == "obj1"
    assert vol.carve == pytest.approx(1.5)


def test_field_visual_options_includes_carve():
    from pymolviz.volumetric.IsoSurface import IsoSurface

    from tests.unit.test_field_visuals import _tiny_grid

    grid = _tiny_grid()
    iso = IsoSurface(grid, 1.0, name="iso", selection="ligand", carve=2.0)
    opts = field_visual_options(iso)
    assert opts["selection"] == "ligand"
    assert opts["carve"] == pytest.approx(2.0)
