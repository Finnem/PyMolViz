"""Unit tests for table zoom-to-selection."""

from __future__ import annotations

from pymolviz.wizards.builders.points import VisualPoint
from pymolviz.wizards.builders.zoom_selection import (
    points_from_pair_rows,
    points_from_rows,
    zoom_to_visual_points,
)
from tests.fakes.cmd import FakeCmd


def test_points_from_rows():
    pts = [VisualPoint("a", "manual", 0, 0, 0), VisualPoint("b", "manual", 1, 1, 1)]
    assert len(points_from_rows(pts, [1])) == 1
    assert points_from_rows(pts, [1])[0].name == "b"


def test_zoom_to_visual_points_creates_and_cleans_tmp():
    cmd = FakeCmd()
    pts = [
        VisualPoint("a", "manual", 1.0, 2.0, 3.0),
        VisualPoint("b", "manual", 4.0, 5.0, 6.0),
    ]
    zoom_to_visual_points(cmd, pts)
    assert hasattr(cmd, "_last_zoom")
    assert cmd._last_zoom["center"] == (2.5, 3.5, 4.5)
    assert not any(n.startswith("_pmv_zoom_tmp") for n in cmd.objects)
