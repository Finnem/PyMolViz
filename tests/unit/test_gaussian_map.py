"""PyMOL map_new gaussian density (Cromer–Mann splat)."""

from __future__ import annotations

import numpy as np
import pytest

from pymolviz.util.gaussian_map import (
    DEFAULT_GAUSSIAN_B_FLOOR,
    gaussian_density_at_origin,
    normalize_gaussian_map,
    paint_gaussian_density,
)


def test_carbon_b20_peak_matches_pymol_formula():
    peak = gaussian_density_at_origin("C", b_factor=20.0, occupancy=1.0, resolution=2.0)
    assert peak == pytest.approx(1.59742, abs=1e-4)


def test_paint_gaussian_density_peaks_at_atom():
    origin = np.array([-2.0, -2.0, -2.0], dtype=float)
    h = 0.5
    shape = (9, 9, 9)
    field = paint_gaussian_density(
        shape, origin, h, [(0.0, 0.0, 0.0)], ["C"],
        b_factors=[DEFAULT_GAUSSIAN_B_FLOOR],
    )
    assert float(field.max()) == pytest.approx(
        gaussian_density_at_origin("C"), abs=0.08,
    )
    ijk = np.unravel_index(int(np.argmax(field)), field.shape)
    xyz = origin + np.array(ijk, dtype=float) * h
    assert np.linalg.norm(xyz) < h * 1.5


def test_normalize_gaussian_map_is_zero_mean_unit_stdev():
    raw = np.array([0.0, 0.0, 1.0, 3.0], dtype=float)
    out = normalize_gaussian_map(raw)
    assert float(out.mean()) == pytest.approx(0.0, abs=1e-12)
    assert float(out.std(ddof=1)) == pytest.approx(1.0, abs=1e-12)
