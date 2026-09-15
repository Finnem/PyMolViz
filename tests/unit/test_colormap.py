"""ColorMap range modes and reverse helpers (no wizard)."""

from __future__ import annotations

import numpy as np
import pytest

from pymolviz.ColorMap import (
    RANGE_MODE_AUTO,
    RANGE_MODE_CUSTOM,
    RANGE_MODE_SYMMETRIC,
    ColorMap,
    apply_colormap_reverse,
    normalize_range_mode,
    parse_colormap_reverse,
    preview_ramp_rgba,
    resolve_colormap_clims,
)


def test_normalize_range_mode():
    assert normalize_range_mode("auto") == RANGE_MODE_AUTO
    assert normalize_range_mode("manual") == RANGE_MODE_CUSTOM
    assert normalize_range_mode("about-zero") == RANGE_MODE_SYMMETRIC


def test_apply_and_parse_colormap_reverse():
    assert apply_colormap_reverse("viridis", False) == "viridis"
    assert apply_colormap_reverse("viridis", True) == "viridis_r"
    assert apply_colormap_reverse("RdYlBu_r", True) == "RdYlBu"
    preset, reverse = parse_colormap_reverse("viridis_r", presets=("viridis", "RdYlBu_r"))
    assert preset == "viridis"
    assert reverse is True
    preset, reverse = parse_colormap_reverse("RdYlBu_r", presets=("viridis", "RdYlBu_r"))
    assert preset == "RdYlBu_r"
    assert reverse is False


def test_resolve_colormap_clims_auto_custom_symmetric():
    values = np.array([-2.0, 1.0, 4.0])
    auto = resolve_colormap_clims("auto", values)
    assert auto == pytest.approx([-2.0, 4.0])
    custom = resolve_colormap_clims("custom", values, custom=(1.0, -0.5))
    assert custom == pytest.approx([ -0.5, 1.0])
    symmetric = resolve_colormap_clims("symmetric", values)
    assert symmetric == pytest.approx([-4.0, 4.0])


def test_colormap_range_modes_on_instance():
    values = [-2.0, 1.0, 4.0]
    auto = ColorMap(values, "viridis", values_are_single_color=False, range_mode="auto")
    assert auto.clims[0] == pytest.approx(-2.0)
    assert auto.clims[1] == pytest.approx(4.0)
    custom = ColorMap(
        values, "viridis", values_are_single_color=False, range_mode="custom", clims=(-1.0, 1.0),
    )
    assert custom.clims == pytest.approx([-1.0, 1.0])
    symmetric = ColorMap(values, "viridis", values_are_single_color=False, range_mode="symmetric")
    assert symmetric.clims[0] == pytest.approx(-4.0)
    assert symmetric.clims[1] == pytest.approx(4.0)


def test_colormap_reverse_flips_ramp_ends():
    fwd = ColorMap([0.0, 1.0], "viridis", values_are_single_color=False, reverse=False)
    rev = ColorMap([0.0, 1.0], "viridis", values_are_single_color=False, reverse=True)
    a0 = np.asarray(fwd.get_color([0.0]))[0, :3]
    b1 = np.asarray(rev.get_color([1.0]))[0, :3]
    assert a0 == pytest.approx(b1, abs=0.05)


def test_preview_ramp_rgba_shape():
    rgba = preview_ramp_rgba("viridis", reverse=False, n=8)
    assert rgba.shape[0] == 8
    assert rgba.shape[1] >= 3
