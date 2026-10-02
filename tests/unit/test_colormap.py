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


def test_guess_structure_factor_map_prefers_2fo_before_fofc():
    from pymolviz.util.colormap_spec import (
        MAP_KIND_AMPLITUDE,
        MAP_KIND_DIFFERENCE,
        guess_structure_factor_map,
    )

    assert guess_structure_factor_map(title="2Fo-Fc", factor="FWT") == MAP_KIND_AMPLITUDE
    assert guess_structure_factor_map(name="refine_2fofc") == MAP_KIND_AMPLITUDE
    assert guess_structure_factor_map(title="Fo-Fc", factor="DELFWT") == MAP_KIND_DIFFERENCE
    assert guess_structure_factor_map(factor="FOFCWT") == MAP_KIND_DIFFERENCE
    assert guess_structure_factor_map(name="electrostatic") is None


def test_resolve_limits_sigma_uses_baked_full_map_stats():
    from pymolviz.util.colormap_spec import RANGE_MODE_SIGMA, Normalization, resolve_limits

    crop = np.array([10.0, 11.0, 12.0])
    # Crop is high; baked stats are mean 0, std/rms 2.
    amp = Normalization(
        mode=RANGE_MODE_SIGMA,
        sigma_low=-2.0,
        sigma_high=2.0,
        link_center_zero=False,
        stats_mean=0.0,
        stats_std=2.0,
        stats_rms=2.0,
    )
    lo, hi = resolve_limits(amp, crop)
    assert lo == pytest.approx(-4.0)
    assert hi == pytest.approx(4.0)
    diff = Normalization(
        mode=RANGE_MODE_SIGMA,
        sigma_low=-3.0,
        sigma_high=3.0,
        link_center_zero=True,
        stats_mean=1.0,
        stats_std=2.0,
        stats_rms=4.0,
    )
    lo, hi = resolve_limits(diff, crop)
    assert lo == pytest.approx(-12.0)
    assert hi == pytest.approx(12.0)


def test_sigma_tick_marks_and_labels():
    from pymolviz.util.colormap_spec import (
        Normalization,
        format_sigma_label,
        sigma_annotation_frame,
        sigma_tick_marks,
    )

    assert format_sigma_label(0.0) == "0"
    assert format_sigma_label(-2.0) == "-2σ"
    assert format_sigma_label(3.0) == "+3σ"
    marks = sigma_tick_marks(0.0, 2.0, -5.0, 5.0)
    labels = [label for _value, label in marks]
    assert "0" in labels
    assert "-2σ" in labels
    assert "+2σ" in labels
    assert all(abs(value) <= 5.0 + 1e-9 for value, _label in marks)
    norm = Normalization(
        mode="sigma",
        link_center_zero=True,
        stats_mean=1.0,
        stats_std=2.0,
        stats_rms=4.0,
    )
    center, scale = sigma_annotation_frame(norm)
    assert center == pytest.approx(0.0)
    assert scale == pytest.approx(4.0)


def test_default_normalization_for_fo_fc_field():
    from types import SimpleNamespace

    from pymolviz.util.colormap_spec import (
        RANGE_MODE_SIGMA,
        default_normalization_for_field,
    )

    field = SimpleNamespace(
        values=np.array([-3.0, 0.0, 3.0, 0.0], dtype=float),
        name="map",
        generator={"map_title": "Fo-Fc", "factor_column": "DELFWT"},
        provenance={},
    )
    norm = default_normalization_for_field(field)
    assert norm is not None
    assert norm.mode == RANGE_MODE_SIGMA
    assert norm.link_center_zero is True
    assert norm.sigma_low == pytest.approx(-3.0)
    assert norm.sigma_high == pytest.approx(3.0)


def test_range_mode_for_endpoint_edit_promotes_auto_keeps_symmetric():
    from pymolviz.util.colormap_spec import (
        RANGE_MODE_AUTO,
        RANGE_MODE_CUSTOM,
        RANGE_MODE_PERCENTILE,
        RANGE_MODE_SYMMETRIC,
        clamp_range,
        range_mode_for_endpoint_edit,
    )

    assert range_mode_for_endpoint_edit(RANGE_MODE_AUTO) == RANGE_MODE_CUSTOM
    assert range_mode_for_endpoint_edit(RANGE_MODE_PERCENTILE) == RANGE_MODE_CUSTOM
    assert range_mode_for_endpoint_edit(RANGE_MODE_CUSTOM) == RANGE_MODE_CUSTOM
    assert range_mode_for_endpoint_edit(RANGE_MODE_SYMMETRIC) == RANGE_MODE_SYMMETRIC
    lo, hi = clamp_range(3.0, 1.0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(3.0)


def test_pad_span_around_leaves_margin_inside_data():
    from pymolviz.util.colormap_spec import LIMIT_VIEW_PAD, pad_span_around

    lo, hi = pad_span_around(10.0, 20.0, 0.0, 40.0)
    width = 10.0
    assert lo == pytest.approx(10.0 - width * LIMIT_VIEW_PAD)
    assert hi == pytest.approx(20.0 + width * LIMIT_VIEW_PAD)


def test_pad_span_around_caps_at_data_limits():
    from pymolviz.util.colormap_spec import pad_span_around

    lo, hi = pad_span_around(0.0, 10.0, 0.0, 10.0)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(10.0)
    lo, hi = pad_span_around(0.0, 2.0, 0.0, 10.0)
    assert lo == pytest.approx(0.0)
    assert hi > 2.0
    assert hi < 10.0


def test_include_handle_in_view_caps_at_data_and_grows_only_to_handle():
    from pymolviz.util.colormap_spec import include_handle_in_view

    view = (2.0, 8.0)
    same = include_handle_in_view(2.0, 8.0, 5.0, "vmin", 0.0, 10.0)
    assert same[0] == pytest.approx(view[0])
    assert same[1] == pytest.approx(view[1])
    left = include_handle_in_view(2.0, 8.0, 0.5, "vmin", 0.0, 10.0)
    assert left[0] == pytest.approx(0.5)
    assert left[1] == pytest.approx(8.0)
    capped = include_handle_in_view(2.0, 8.0, -5.0, "vmin", 0.0, 10.0)
    assert capped[0] == pytest.approx(0.0)
    right = include_handle_in_view(2.0, 8.0, 9.5, "vmax", 0.0, 10.0)
    assert right[0] == pytest.approx(2.0)
    assert right[1] == pytest.approx(9.5)
    capped_r = include_handle_in_view(2.0, 8.0, 50.0, "vmax", 0.0, 10.0)
    assert capped_r[1] == pytest.approx(10.0)


def test_histogram_axis_delta_matches_view_scale():
    from pymolviz.util.colormap_spec import (
        histogram_axis_delta_for_pixels,
        shift_histogram_value,
        value_to_histogram_axis,
    )

    assert histogram_axis_delta_for_pixels(10.0, 0.0, 10.0, 100.0) == pytest.approx(1.0)
    assert shift_histogram_value(5.0, 1.0) == pytest.approx(6.0)
    assert value_to_histogram_axis(0.0, 10.0, 20.0) == pytest.approx(0.0)
    assert value_to_histogram_axis(0.0, 10.0, 20.0, clip=False) == pytest.approx(-1.0)


def test_histogram_counts_for_span_rebins_window_without_overflow_dump():
    from pymolviz.util.colormap_spec import histogram_counts_for_span, histogram_from_values

    counts, edges = histogram_counts_for_span(
        [0.0, 1.0, 1.0, 1.0, 5.0, 10.0], 0.5, 1.5, "full", bins=8,
    )
    assert edges[0] == pytest.approx(0.5)
    assert edges[-1] == pytest.approx(1.5)
    assert int(np.sum(counts)) == 3
    hist = histogram_from_values(
        np.concatenate([np.linspace(0.0, 1.0, 200), np.array([1000.0])]),
        view="percentile",
    )
    assert hist["overflow"] >= 1
    assert int(np.sum(hist["counts"])) == int(hist["n"]) - int(hist["underflow"]) - int(hist["overflow"])
    assert len(hist["samples"]) == int(hist["n"])
