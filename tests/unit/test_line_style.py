"""Unit tests for dash / margin line-style helpers."""

from __future__ import annotations

import pytest

from pymolviz.util.line_style import (
    LineStyle,
    MIN_ARROW_SHAFT,
    absolute_head_length,
    apply_margin,
    dash_on_segments,
    default_head_length,
    max_margin_for_length,
)


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


def test_line_style_migrates_legacy_ends():
    assert LineStyle(ends="None").start_head == "None"
    assert LineStyle(ends="None").end_head == "None"
    classic = LineStyle(ends="Arrow")
    assert classic.start_head == "None"
    assert classic.end_head == "Arrow"
    assert classic.ends == "Arrow"
    both = LineStyle(ends="Double arrow")
    assert both.start_head == "Arrow"
    assert both.end_head == "Arrow"
    assert both.ends == "Double arrow"
    circles = LineStyle(ends="Circles")
    assert circles.start_head == "Circles"
    assert circles.end_head == "Circles"


def test_line_style_independent_heads_roundtrip():
    mixed = LineStyle(start_head="Circles", end_head="Arrow")
    assert mixed.ends == "Circles/Arrow"
    assert mixed.start_head == "Circles"
    assert mixed.end_head == "Arrow"
    data = mixed.to_dict()
    assert data["start_head"] == "Circles"
    assert data["end_head"] == "Arrow"
    restored = LineStyle.from_dict(data)
    assert restored.start_head == "Circles"
    assert restored.end_head == "Arrow"
    legacy = LineStyle.from_dict({"dash": "Solid", "ends": "Double arrow"})
    assert legacy.start_head == "Arrow"
    assert legacy.end_head == "Arrow"
    copied = mixed.copy().updated(start_head="Arrow")
    assert copied.start_head == "Arrow"
    assert copied.end_head == "Arrow"
    assert mixed.start_head == "Circles"


def test_head_length_follows_width_not_shaft():
    assert default_head_length(0.05) == pytest.approx(0.4)
    short = absolute_head_length(0.05, 2.0)
    long = absolute_head_length(0.05, 20.0)
    assert short == pytest.approx(0.4)
    assert long == pytest.approx(0.4)
    tiny = absolute_head_length(0.05, 0.2)
    assert tiny == pytest.approx(0.2 - MIN_ARROW_SHAFT)


def test_max_margin_leaves_room_for_head():
    head = 0.4
    length = 2.0
    cap = max_margin_for_length(length, head, double_head=False)
    assert cap == pytest.approx((length - head - MIN_ARROW_SHAFT) / 2.0)
    a, b = apply_margin((0, 0, 0), (length, 0, 0), 10.0, head_length=head)
    remaining = b[0] - a[0]
    assert remaining == pytest.approx(head + MIN_ARROW_SHAFT)
    assert remaining + 2.0 * cap == pytest.approx(length)


def test_apply_margin_oversize_without_head():
    a, b = apply_margin((0, 0, 0), (1, 0, 0), 10.0)
    assert abs((b[0] - a[0]) - MIN_ARROW_SHAFT) < 1e-6
