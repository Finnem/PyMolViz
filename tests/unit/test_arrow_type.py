"""Hit-testing for the stylized arrow-type control (no Qt)."""

from __future__ import annotations

from pymolviz.wizards.builders.arrow_type import (
    arrow_type_geometry,
    hit_arrow_type_part,
    margin_from_tick0,
    margin_from_tick1,
    part_highlight_rect,
)


def test_hit_regions_circle_shaft_head_ticks():
    width, height, margin = 200.0, 32.0, 0.0
    g = arrow_type_geometry(width, height, margin)
    assert hit_arrow_type_part(g["circle_x"], g["y"], width, height, margin) == "circle"
    assert hit_arrow_type_part(g["end_circle_x"], g["y"], width, height, margin) == "circle"
    mid = 0.5 * (g["shaft0"] + g["shaft1"])
    assert hit_arrow_type_part(mid, g["y"], width, height, margin) == "shaft"
    start_body = 0.5 * (g["start_tip"] + g["start_base"])
    end_body = 0.5 * (g["end_base"] + g["end_tip"])
    assert hit_arrow_type_part(start_body, g["y"], width, height, margin) == "head_start"
    assert hit_arrow_type_part(end_body, g["y"], width, height, margin) == "head_end"
    assert hit_arrow_type_part(g["tick0"], g["tick_top"] + 2.0, width, height, margin) == "tick0"
    assert hit_arrow_type_part(g["tick1"], g["tick_top"] + 2.0, width, height, margin) == "tick1"
    assert g["tick_top"] < g["y"] - g["circle_r"]


def test_circles_stay_put_ticks_move_with_margin():
    width, height = 200.0, 32.0
    g0 = arrow_type_geometry(width, height, 0.0)
    g = arrow_type_geometry(width, height, 4.0)
    assert g["circle_x"] == g0["circle_x"]
    assert g["end_circle_x"] == g0["end_circle_x"]
    assert g["tick0"] > g0["tick0"]
    assert g["tick1"] < g0["tick1"]
    assert g["shaft0"] > g0["shaft0"]
    assert g["head_tip"] < g0["head_tip"]


def test_margin_round_trip_from_ticks():
    width, height = 200.0, 32.0
    g0 = arrow_type_geometry(width, height, 0.0)
    assert margin_from_tick0(g0["tick0"], width, height) == 0.0
    assert margin_from_tick1(g0["tick1"], width, height) == 0.0
    g = arrow_type_geometry(width, height, 4.0)
    assert abs(margin_from_tick0(g["tick0"], width, height) - 4.0) < 0.05
    assert abs(margin_from_tick1(g["tick1"], width, height) - 4.0) < 0.05


def test_section_highlights_are_distinct():
    g = arrow_type_geometry(200.0, 46.0, 0.0)
    shaft = part_highlight_rect(g, "shaft")
    head_start = part_highlight_rect(g, "head_start")
    head_end = part_highlight_rect(g, "head_end")
    tick = part_highlight_rect(g, "tick0")
    assert shaft is not None and head_start is not None and head_end is not None and tick is not None
    assert head_start[0] < shaft[0] < head_end[0]
    assert tick[1] < g["y"] - g["circle_r"]
