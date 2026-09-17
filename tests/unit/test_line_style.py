"""Unit tests for dash / margin line-style helpers."""

from __future__ import annotations

import pytest

from pymolviz.util.line_style import (
    LineStyle,
    MIN_ARROW_SHAFT,
    MAX_ARROW_CONE_SEGMENTS,
    absolute_head_length,
    apply_margin,
    arrow_cone_segments,
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


def test_arrow_cone_segments_stay_bounded():
    from pymolviz.util.line_style import ARROW_QUALITY_SEGMENTS

    n_seg = ARROW_QUALITY_SEGMENTS[3]
    default_head = 0.045 * 2.4
    assert arrow_cone_segments(n_seg, 0.045, default_head) == 24
    skinny = arrow_cone_segments(n_seg, 0.001, default_head)
    assert skinny <= MAX_ARROW_CONE_SEGMENTS
    assert skinny >= n_seg
    huge_head = arrow_cone_segments(n_seg, 0.045, 50.0)
    assert huge_head <= MAX_ARROW_CONE_SEGMENTS


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


def test_apply_margin_independent_ends():
    a, b = apply_margin(
        (0, 0, 0), (10, 0, 0),
        start_margin=2.0,
        end_margin=1.0,
    )
    assert abs(a[0] - 2.0) < 1e-6
    assert abs(b[0] - 9.0) < 1e-6


def test_line_style_independent_margins_roundtrip():
    mixed = LineStyle(start_margin=1.5, end_margin=0.25)
    assert mixed.start_margin == 1.5
    assert mixed.end_margin == 0.25
    assert mixed.margin == 1.5
    data = mixed.to_dict()
    assert data["start_margin"] == 1.5
    assert data["end_margin"] == 0.25
    restored = LineStyle.from_dict(data)
    assert restored.start_margin == 1.5
    assert restored.end_margin == 0.25
    legacy = LineStyle.from_dict({"dash": "Solid", "margin": 2.0, "ends": "Arrow"})
    assert legacy.start_margin == 2.0
    assert legacy.end_margin == 2.0
    both = LineStyle(margin=3.0)
    assert both.start_margin == 3.0
    assert both.end_margin == 3.0
    updated = mixed.updated(start_margin=0.5)
    assert updated.start_margin == 0.5
    assert updated.end_margin == 0.25
    symmetric = mixed.updated(margin=1.0)
    assert symmetric.start_margin == 1.0
    assert symmetric.end_margin == 1.0


def test_arrow_shafts_use_cylinder_not_cone():
    """PyMOL ray tracing draws CYLINDER, not equal-radius CONE."""
    from pymolviz.meshes.Arrows import Arrows
    from pymolviz.util.line_style import LineStyle

    arrows = Arrows(
        starts=[(0.0, 0.0, 0.0)],
        ends=[(2.0, 0.0, 0.0)],
        color="red",
        name="ray_shaft",
        quality=3,
        line_style=LineStyle(ends="Arrow"),
    )
    kinds = [t for t in arrows._create_CGO_list() if isinstance(t, str)]
    assert "CYLINDER" in kinds
    assert "CONE" not in kinds
    assert "TRIANGLES" in kinds

