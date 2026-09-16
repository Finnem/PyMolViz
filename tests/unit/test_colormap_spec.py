"""Colormap definition, range, histogram, and colorbar export (no Qt)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from pymolviz.util.colormap_spec import (
    INTERP_HSV,
    MAP_DISCRETE,
    OOR_CUSTOM,
    OOR_TRANSPARENT,
    RANGE_MODE_CUSTOM,
    RANGE_MODE_PERCENTILE,
    RANGE_MODE_SYMMETRIC,
    ColorStop,
    ColormapDefinition,
    ColorbarExportSettings,
    FieldColorMapping,
    Normalization,
    colorbar_caption,
    definition_from_preset,
    format_number,
    histogram_from_values,
    map_scalars,
    ramp_rgba,
    resolve_limits,
    reverse_definition,
    sample_unit,
    subsample_values,
    tick_values,
)


def test_stop_neighbor_limits_pin_endpoints():
    from pymolviz.wizards.builders.colormap_dialog import (
        _clamp_stop_position,
        _stop_neighbor_limits,
    )

    stops = (
        ColorStop(0.0, (0.0, 0.0, 1.0, 1.0)),
        ColorStop(0.5, (0.0, 1.0, 0.0, 1.0)),
        ColorStop(1.0, (1.0, 0.0, 0.0, 1.0)),
    )
    assert _stop_neighbor_limits(0, stops) == (0.0, 0.0)
    assert _stop_neighbor_limits(2, stops) == (1.0, 1.0)
    assert _clamp_stop_position(0, stops, 0.25) == pytest.approx(0.0)
    assert _clamp_stop_position(2, stops, 0.8) == pytest.approx(1.0)
    assert _stop_neighbor_limits(1, stops) == (pytest.approx(0.004), pytest.approx(0.996))
    assert _clamp_stop_position(1, stops, 0.25) == pytest.approx(0.25)


def test_stop_drag_inverts_paint_on_log_axis():
    from pymolviz.util.colormap_spec import HISTOGRAM_VIEW_LOG
    from pymolviz.wizards.builders.colormap_plot import (
        _unit_position_for_histogram_x,
        _x_for_stop_position,
    )

    class _Rect:
        def left(self):
            return 20.0

        def right(self):
            return 220.0

    plot = _Rect()
    dmin, dmax = 1e-4, 10.0
    vmin, vmax = 1e-3, 1.0
    for pos in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = _x_for_stop_position(plot, pos, vmin, vmax, dmin, dmax, HISTOGRAM_VIEW_LOG)
        back = _unit_position_for_histogram_x(plot, x, vmin, vmax, dmin, dmax, HISTOGRAM_VIEW_LOG)
        assert back == pytest.approx(pos, rel=1e-6, abs=1e-6)


def test_moving_endpoint_stop_does_not_spawn_sentinels():
    defn = ColormapDefinition(
        stops=(
            ColorStop(0.0, (0.0, 0.0, 1.0, 1.0)),
            ColorStop(0.5, (0.0, 1.0, 0.0, 1.0)),
            ColorStop(1.0, (1.0, 0.0, 0.0, 1.0)),
        ),
        customized=True,
    )
    moved = ColormapDefinition(
        stops=(
            ColorStop(0.25, defn.stops[0].rgba),
            defn.stops[1],
            ColorStop(0.8, defn.stops[2].rgba),
        ),
        customized=True,
    )
    assert len(moved.stops) == 3
    assert moved.stops[0].position == pytest.approx(0.25)
    assert moved.stops[-1].position == pytest.approx(0.8)
    assert sample_unit(moved, 0.0)[:3] == pytest.approx(moved.stops[0].rgba[:3])
    assert sample_unit(moved, 1.0)[:3] == pytest.approx(moved.stops[-1].rgba[:3])


def test_color_stops_sit_on_range_handles():
    from pymolviz.wizards.builders.colormap_plot import _x_for_stop_position, _x_for_value

    class _Rect:
        def left(self):
            return 0.0

        def right(self):
            return 100.0

    plot = _Rect()
    dmin, dmax = 0.0, 10.0
    vmin, vmax = 2.0, 8.0
    x_min = _x_for_value(plot, vmin, dmin, dmax)
    x_max = _x_for_value(plot, vmax, dmin, dmax)
    assert _x_for_stop_position(plot, 0.0, vmin, vmax, dmin, dmax) == pytest.approx(x_min)
    assert _x_for_stop_position(plot, 1.0, vmin, vmax, dmin, dmax) == pytest.approx(x_max)
    assert _x_for_stop_position(plot, 0.5, vmin, vmax, dmin, dmax) == pytest.approx(
        0.5 * (x_min + x_max)
    )
    vmin, vmax = 1.0, 4.0
    assert _x_for_stop_position(plot, 0.0, vmin, vmax, dmin, dmax) == pytest.approx(
        _x_for_value(plot, vmin, dmin, dmax)
    )
    assert _x_for_stop_position(plot, 1.0, vmin, vmax, dmin, dmax) == pytest.approx(
        _x_for_value(plot, vmax, dmin, dmax)
    )


def test_preset_stops_are_normalized_0_to_1():
    defn = definition_from_preset("RdYlBu_r")
    assert defn.preset == "RdYlBu_r"
    assert not defn.customized
    assert defn.stops[0].position == pytest.approx(0.0)
    assert defn.stops[-1].position == pytest.approx(1.0)
    ramp = ramp_rgba(defn, n=16)
    assert ramp.shape == (16, 4)
    assert np.all(ramp[:, 3] == pytest.approx(1.0))


def test_reverse_named_preset_uses_mpl_suffix():
    flipped = reverse_definition(definition_from_preset("viridis"))
    assert flipped.preset == "viridis_r"
    original = ramp_rgba(definition_from_preset("viridis"), n=8)
    reversed_ramp = ramp_rgba(flipped, n=8)
    assert np.allclose(original[0, :3], reversed_ramp[-1, :3], atol=0.08)


def test_reverse_custom_stops_flips_positions():
    defn = ColormapDefinition(
        preset="custom",
        stops=(
            ColorStop(0.0, (0.0, 0.0, 1.0, 1.0)),
            ColorStop(1.0, (1.0, 0.0, 0.0, 1.0)),
        ),
        customized=True,
    )
    flipped = reverse_definition(defn)
    assert flipped.stops[0].rgba[:3] == pytest.approx((1.0, 0.0, 0.0))
    assert flipped.stops[-1].rgba[:3] == pytest.approx((0.0, 0.0, 1.0))


def test_hsv_interpolation_shortest_hue_arc():
    defn = ColormapDefinition(
        stops=(
            ColorStop(0.0, (1.0, 0.0, 0.0, 1.0)),
            ColorStop(1.0, (0.0, 1.0, 0.0, 0.5)),
        ),
        interpolation=INTERP_HSV,
        customized=True,
    )
    mid = sample_unit(defn, 0.5)
    assert mid[3] == pytest.approx(0.75)
    assert mid[1] > mid[2]


def test_discrete_map_quantizes_unit_interval():
    defn = ColormapDefinition(
        stops=(
            ColorStop(0.0, (0.0, 0.0, 0.0, 1.0)),
            ColorStop(1.0, (1.0, 1.0, 1.0, 1.0)),
        ),
        map_type=MAP_DISCRETE,
        levels=2,
        customized=True,
    )
    lo = sample_unit(defn, 0.1)
    hi = sample_unit(defn, 0.9)
    assert lo[0] == pytest.approx(0.25, abs=0.05)
    assert hi[0] == pytest.approx(0.75, abs=0.05)


def test_nan_and_out_of_range_handling():
    defn = ColormapDefinition(
        stops=(
            ColorStop(0.0, (0.0, 0.0, 1.0, 1.0)),
            ColorStop(1.0, (1.0, 0.0, 0.0, 1.0)),
        ),
        customized=True,
        nan_transparent=True,
        out_of_range=OOR_CUSTOM,
        below_rgba=(0.0, 1.0, 0.0, 1.0),
        above_rgba=(1.0, 1.0, 0.0, 1.0),
    )
    mapped = map_scalars([-2.0, 0.5, 3.0, np.nan], defn, 0.0, 1.0)
    assert mapped[0, :3] == pytest.approx((0.0, 1.0, 0.0))
    assert mapped[2, :3] == pytest.approx((1.0, 1.0, 0.0))
    assert mapped[3, 3] == pytest.approx(0.0)
    hidden = ColormapDefinition(
        stops=defn.stops, customized=True, out_of_range=OOR_TRANSPARENT,
    )
    out = map_scalars([-1.0, 0.5], hidden, 0.0, 1.0)
    assert out[0, 3] == pytest.approx(0.0)


def test_normalization_modes():
    values = np.array([-8.0, -1.0, 0.0, 1.0, 4.0])
    auto = resolve_limits(Normalization(mode="auto"), values)
    assert auto == pytest.approx((-8.0, 4.0))
    custom = resolve_limits(Normalization(mode=RANGE_MODE_CUSTOM, vmin=-2, vmax=2), values)
    assert custom == pytest.approx((-2.0, 2.0))
    sym = resolve_limits(Normalization(mode=RANGE_MODE_SYMMETRIC), values)
    assert sym[0] == pytest.approx(-8.0)
    assert sym[1] == pytest.approx(8.0)
    pct = resolve_limits(
        Normalization(mode=RANGE_MODE_PERCENTILE, percentile_low=0.0, percentile_high=100.0),
        values,
    )
    assert pct[0] == pytest.approx(-8.0)
    assert pct[1] == pytest.approx(4.0)


def test_histogram_zeros_not_dumped_into_first_bin():
    zeros = np.zeros(9000)
    pos = np.logspace(-4, -1, 1000)
    hist = histogram_from_values(np.concatenate([zeros, pos]), bins=24, view="auto")
    counts = np.asarray(hist["counts"], dtype=float)
    assert counts.max() < 0.5 * counts.sum()
    assert np.count_nonzero(counts) >= 5
    values = np.linspace(-3, 3, 100000)
    hist = histogram_from_values(values, bins=40, max_samples=2000)
    assert hist["n"] <= 2000
    assert len(hist["counts"]) == 40
    assert hist["vmin"] == pytest.approx(-3.0, abs=0.02)
    assert hist["n_total"] == 100000


def test_limits_sane_for_field_rejects_stale_wide_limits():
    from pymolviz.util.colormap_spec import limits_sane_for_field

    values = np.linspace(0.0, 9.4, 100)
    assert limits_sane_for_field((0.0, 511.0), values) == (0.0, 9.4)


def test_histogram_auto_zooms_electron_density_like_tail():
    bulk = np.random.default_rng(1).exponential(0.0008, size=12000)
    bulk = np.clip(bulk, 0.0, 0.006)
    spikes = np.array([9.4, 8.0, 7.2])
    values = np.concatenate([bulk, spikes])
    hist = histogram_from_values(values, bins=40, view="auto")
    assert hist["display_max"] < 0.02
    assert hist["vmax"] == pytest.approx(9.4)
    assert len([c for c in hist["counts"] if c > 0]) >= 5


def test_histogram_auto_focuses_heavy_tailed_positive_density():
    from pymolviz.util.colormap_spec import (
        HISTOGRAM_VIEW_LOG,
        analyze_value_distribution,
        histogram_axis_to_value,
        value_to_histogram_axis,
    )

    rng = np.random.default_rng(0)
    bulk = rng.exponential(0.02, size=8000)
    spikes = rng.uniform(50.0, 200.0, size=40)
    values = np.concatenate([bulk, spikes])
    analysis = analyze_value_distribution(values)
    assert analysis["suggest_percentile_range"] is True
    hist = histogram_from_values(values, bins=32, max_samples=5000, view="auto")
    assert hist["display_max"] < analysis["vmax"] * 0.5
    assert max(hist["counts"]) > 0
    assert len([c for c in hist["counts"] if c > 0]) >= 3
    span = hist["display_max"] - hist["display_min"]
    assert span > 0
    assert max(hist["counts"]) > 0
    dlo, dhi = hist["display_min"], hist["display_max"]
    axis = hist["axis"]
    mid = histogram_axis_to_value(0.5, dlo, dhi, axis)
    assert dlo <= mid <= dhi
    assert value_to_histogram_axis(mid, dlo, dhi, axis) == pytest.approx(0.5, abs=0.05)
    tiny = subsample_values(np.array([1.0, np.nan, 2.0]), max_samples=10)
    assert list(tiny) == pytest.approx([1.0, 2.0])


def test_histogram_tick_values_log_and_linear():
    from pymolviz.wizards.builders.colormap_plot import _histogram_tick_values

    linear = _histogram_tick_values(0.0, 1.0, "full", 5)
    assert linear[0] == pytest.approx(0.0)
    assert linear[-1] == pytest.approx(1.0)
    logs = _histogram_tick_values(1e-4, 1.0, "log", 5)
    assert logs[0] == pytest.approx(1e-4, rel=1e-6)
    assert logs[-1] == pytest.approx(1.0, rel=1e-6)
    ratios = [logs[i + 1] / logs[i] for i in range(len(logs) - 1)]
    assert max(ratios) / min(ratios) == pytest.approx(1.0, rel=1e-5)


def test_distribution_axis_mapping():
    from pymolviz.util.colormap_spec import (
        alpha_to_screen_y,
        axis_to_unit,
        clamp_range,
        data_to_unit,
        screen_y_to_alpha,
        unit_to_axis,
        unit_to_data,
    )

    assert axis_to_unit(5, 0, 10) == pytest.approx(0.5)
    assert unit_to_axis(0.25, 10, 20) == pytest.approx(12.5)
    assert data_to_unit(0.0, -2, 2) == pytest.approx(0.5)
    assert unit_to_data(0.25, -2, 2) == pytest.approx(-1.0)
    assert screen_y_to_alpha(10, 0, 100) == pytest.approx(0.9)
    assert alpha_to_screen_y(0.25, 0, 100) == pytest.approx(75.0)
    lo, hi = clamp_range(3, -1)
    assert lo == pytest.approx(-1.0)
    assert hi == pytest.approx(3.0)
    lo, hi = clamp_range(1.0, 1.0)
    assert hi > lo
    assert format_number(0.30000000000004, "auto") in ("0.3", "0.30")
    assert format_number(1.0, "fixed", decimals=2) == "1"
    assert "e" in format_number(0.00123, "scientific", sig_digits=3)
    ticks = tick_values(-2, 2, 5)
    assert ticks[0] == pytest.approx(-2.0)
    assert ticks[-1] == pytest.approx(2.0)
    assert colorbar_caption("Electrostatic potential", "kT/e") == "Electrostatic potential (kT/e)"


def test_definition_roundtrip_dict():
    defn = definition_from_preset("coolwarm")
    again = ColormapDefinition.from_dict(defn.to_dict())
    assert again.preset == "coolwarm"
    assert again.stops[0].position == pytest.approx(0.0)
    mapping = FieldColorMapping(
        field_id="f1",
        colormap=again,
        normalization=Normalization(mode="custom", vmin=-1, vmax=1),
    )
    loaded = FieldColorMapping.from_dict(mapping.to_dict())
    assert loaded.field_id == "f1"
    assert loaded.normalization.vmax == pytest.approx(1.0)


def test_colorbar_export_png_and_svg(tmp_path):
    from pymolviz.util.colorbar_export import export_colorbar

    mapping = FieldColorMapping(
        colormap=definition_from_preset("RdYlBu_r"),
        normalization=Normalization(mode=RANGE_MODE_CUSTOM, vmin=-2, vmax=2),
        title="Electrostatic potential",
        units="kT/e",
    )
    png = export_colorbar(
        os.path.join(str(tmp_path), "bar.png"),
        mapping,
        ColorbarExportSettings(fmt="png", width=400, height=80, dpi=72),
        limits=(-2.0, 2.0),
    )
    svg = export_colorbar(
        os.path.join(str(tmp_path), "bar.svg"),
        mapping,
        ColorbarExportSettings(fmt="svg", orientation="vertical", width=80, height=400, dpi=72),
        limits=(-2.0, 2.0),
    )
    assert os.path.isfile(png)
    assert os.path.getsize(png) > 50
    with open(svg, "r", encoding="utf-8") as handle:
        body = handle.read()
    assert "<svg" in body.lower()


def test_persist_colormap_attrs_only_stores_custom_maps():
    from pymolviz.util.colormap_spec import persist_colormap_attrs, uses_stop_sampling

    named = definition_from_preset("viridis")
    assert uses_stop_sampling(named) is False
    name, spec = persist_colormap_attrs(named)
    assert name == "viridis"
    assert spec is None
    custom = ColormapDefinition(
        preset="custom",
        stops=(ColorStop(0.0, (0.0, 0.0, 1.0, 1.0)), ColorStop(1.0, (1.0, 0.0, 0.0, 1.0))),
        customized=True,
    )
    name, spec = persist_colormap_attrs(custom)
    assert name == "custom"
    assert spec["customized"] is True
    assert persist_colormap_attrs("plasma") == ("plasma", None)


def test_stored_colormap_spec_keeps_custom_range():
    from pymolviz.util.colormap_spec import (
        Normalization,
        apply_colormap_alpha_to_volume_ramp,
        persist_colormap_attrs,
        stored_colormap_spec,
        normalization_from_stored_spec,
    )

    named = definition_from_preset("viridis")
    spec = stored_colormap_spec(
        named, Normalization(mode=RANGE_MODE_CUSTOM, vmin=0.0, vmax=0.005),
    )
    assert spec is not None
    assert spec["preset"] == "viridis"
    norm = normalization_from_stored_spec(spec)
    assert norm.mode == RANGE_MODE_CUSTOM
    assert norm.vmax == pytest.approx(0.005)
    name, stored = persist_colormap_attrs(named, spec)
    assert name == "viridis"
    assert stored["normalization"]["vmax"] == pytest.approx(0.005)
    fades = ColormapDefinition(
        preset="viridis",
        stops=(
            ColorStop(0.0, (0.0, 0.0, 1.0, 0.0)),
            ColorStop(1.0, (1.0, 0.0, 0.0, 0.5)),
        ),
        customized=True,
    )
    alphas = apply_colormap_alpha_to_volume_ramp((0.0, 1.0), (1.0, 1.0), fades)
    assert alphas[0] == pytest.approx(0.0)
    assert alphas[-1] == pytest.approx(0.5)


def test_unused_custom_preset_name_skips_existing(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from pymolviz.util.colormap_spec import (
        custom_preset_definition,
        is_custom_preset_name,
        save_custom_preset,
        unused_custom_preset_name,
    )

    assert unused_custom_preset_name() == "Custom 1"
    assert is_custom_preset_name("Custom 1") is False
    save_custom_preset("Custom 1", definition_from_preset("viridis"))
    assert unused_custom_preset_name() == "Custom 2"
    assert is_custom_preset_name("Custom 1") is True
    loaded = custom_preset_definition("Custom 1")
    assert loaded is not None
    assert loaded.preset == "Custom 1"


def test_create_and_delete_custom_preset(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from pymolviz.util.colormap_spec import (
        create_custom_preset,
        custom_colormap_rows,
        delete_custom_preset,
        mapping_for_custom_preset,
    )

    name = create_custom_preset()
    assert name == "Custom 1"
    rows = custom_colormap_rows()
    assert [row["name"] for row in rows] == ["Custom 1"]
    mapping = mapping_for_custom_preset(name)
    assert mapping is not None
    assert mapping.colormap.customized is True
    assert delete_custom_preset(name) is True
    assert custom_colormap_rows() == []
    assert delete_custom_preset(name) is False


def test_similar_custom_colormap_detects_near_duplicates(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from dataclasses import replace

    from pymolviz.util.colormap_spec import (
        ColorStop,
        closest_similar_custom_colormap,
        colormap_similarity,
        definition_from_preset,
        save_custom_preset,
    )

    base = definition_from_preset("viridis")
    save_custom_preset("Custom 1", base)
    same = colormap_similarity(base, definition_from_preset("viridis"))
    assert same.similar is True
    assert same.color_max < 0.02
    assert same.alpha_max < 0.02
    match = closest_similar_custom_colormap(base)
    assert match is not None
    assert match.name == "Custom 1"
    assert closest_similar_custom_colormap(base, exclude_name="Custom 1") is None

    other = definition_from_preset("plasma")
    far = colormap_similarity(base, other)
    assert far.similar is False
    assert closest_similar_custom_colormap(other) is None

    tweaked_stops = []
    for stop in base.stops:
        rgba = list(stop.rgba)
        rgba[3] = max(0.0, rgba[3] - 0.04)
        tweaked_stops.append(ColorStop(stop.position, tuple(rgba)))
    near = replace(base, stops=tuple(tweaked_stops), customized=True)
    close = colormap_similarity(near, base)
    assert close.similar is True
    assert any(row.highlight and row.alpha_delta > 0 for row in close.stop_diffs)


def test_create_custom_preset_from_default_matches_saved(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from pymolviz.util.colormap_spec import closest_similar_custom_colormap, create_custom_preset, definition_from_preset
    from pymolviz.util.field_sample import DEFAULT_SURFACE_COLORMAP

    create_custom_preset(definition_from_preset(DEFAULT_SURFACE_COLORMAP))
    match = closest_similar_custom_colormap(definition_from_preset(DEFAULT_SURFACE_COLORMAP))
    assert match is not None
    assert match.name == "Custom 1"


def test_maybe_reuse_leaves_named_custom_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from pymolviz.util.colormap_spec import custom_preset_definition, definition_from_preset, save_custom_preset
    from pymolviz.wizards.builders.colormap_similar import maybe_reuse_similar_colormap

    save_custom_preset("Custom 1", definition_from_preset("viridis"))
    defn = custom_preset_definition("Custom 1")
    applied = maybe_reuse_similar_colormap(None, defn, "Custom 1")
    assert applied.cancelled is False
    assert applied.name == "Custom 1"
    assert applied.definition is defn


def test_mpl_colormap_anchors_inner_stops_for_matplotlib():
    from pymolviz.ColorMap import ColorMap
    from pymolviz.util.colormap_spec import ColormapDefinition, ColorStop, mpl_colormap

    defn = ColormapDefinition(
        stops=(
            ColorStop(0.15, (1.0, 0.0, 0.0, 1.0)),
            ColorStop(0.85, (0.0, 0.0, 1.0, 1.0)),
        ),
        customized=True,
    )
    mpl = mpl_colormap(defn)
    clims = [float(x) for x in __import__("numpy").linspace(-2.0, 2.0, 17)]
    wrapped = ColorMap(clims, mpl, values_are_single_color=False)
    rgba = wrapped.get_color(0.0)
    assert len(rgba) >= 3


def test_volume_colormap_arg_resolves_saved_custom_preset(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from matplotlib.colors import Colormap as MplColormap

    from pymolviz.ColorMap import ColorMap
    from pymolviz.util.colormap_spec import save_custom_preset, volume_colormap_arg
    from pymolviz.util.colors import get_colormap

    save_custom_preset("Custom 2", definition_from_preset("viridis"))
    arg = volume_colormap_arg("Custom 2")
    assert isinstance(arg, MplColormap)
    cmap = ColorMap([0.0, 1.0], "Custom 2", values_are_single_color=False)
    color = np.asarray(cmap.get_color(0.0), dtype=float).reshape(-1)
    assert color.size >= 3
    resolved = get_colormap("Custom 2")
    assert isinstance(resolved, MplColormap)
    unknown = get_colormap("not_a_real_cmap_name_xyz")
    assert isinstance(unknown, MplColormap)
