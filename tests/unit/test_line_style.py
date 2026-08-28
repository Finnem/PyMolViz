"""Unit tests for dash / margin line-style helpers."""

from __future__ import annotations

from pymolviz.util.line_style import LineStyle, apply_margin, dash_on_segments


def test_solid_dash_single_segment():
    segs = dash_on_segments((0, 0, 0), (4, 0, 0), (1.0, 0.0), 1.0)
    assert len(segs) == 1
    assert abs(segs[0][1][0] - 4.0) < 1e-6


def test_dashed_multiple_on_segments():
    segs = dash_on_segments((0, 0, 0), (10, 0, 0), (1.0, 1.0), 1.0)
    assert len(segs) == 5
    assert abs(segs[0][1][0] - 1.0) < 1e-6


def test_apply_margin():
    a, b = apply_margin((0, 0, 0), (10, 0, 0), 1.0)
    assert abs(a[0] - 1.0) < 1e-6
    assert abs(b[0] - 9.0) < 1e-6


def test_line_style_pattern():
    style = LineStyle(dash="Dashed")
    assert style.pattern() == (0.45, 0.28)
