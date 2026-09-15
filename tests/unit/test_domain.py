"""Domain AABB helpers and 2D schematic layout (no Qt, no wizard)."""

from __future__ import annotations

import pytest

from pymolviz.fields.domain import (
    BOUNDS_AROUND_SELECTION,
    BOUNDS_CUSTOM_BOX,
    BOUNDS_OBJECT,
    Domain,
    aabb_center_extent,
    aabb_from_points,
    aabb_has_extent,
    domain_preview_aabbs,
)
from pymolviz.wizards.builders.domain_schematic import (
    best_projection_axes,
    domain_schematic_scene,
    layout_schematic,
)


def test_aabb_from_points_applies_padding():
    box = aabb_from_points([(0.0, 0.0, 0.0), (2.0, 1.0, 0.0)], padding=1.0)
    assert box is not None
    assert box[0] == pytest.approx([-1.0, -1.0, -1.0])
    assert box[1] == pytest.approx([3.0, 2.0, 1.0])


def test_aabb_from_points_empty_or_none():
    assert aabb_from_points(None) is None
    assert aabb_from_points([]) is None
    assert aabb_from_points([(1.0, 2.0)]) is None


def test_domain_resolve_aabb_around_selection_uses_padding():
    domain = Domain(bounds_mode=BOUNDS_AROUND_SELECTION, padding=0.5, spacing=0.25)
    box = domain.resolve_aabb([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    assert box[0] == pytest.approx([-0.5, -0.5, -0.5])
    assert box[1] == pytest.approx([1.5, 0.5, 0.5])


def test_domain_resolve_aabb_custom_box_uses_typed_aabb():
    domain = Domain(
        bounds_mode=BOUNDS_CUSTOM_BOX,
        padding=9.0,
        aabb=[[0.0, 0.0, 0.0], [4.0, 2.0, 1.0]],
    )
    box = domain.resolve_aabb([(10.0, 10.0, 10.0)])
    assert box[0] == pytest.approx([0.0, 0.0, 0.0])
    assert box[1] == pytest.approx([4.0, 2.0, 1.0])


def test_domain_preview_aabbs_inner_hull_vs_padded_outer():
    domain = Domain(padding=2.0, spacing=0.5)
    outer, inner = domain_preview_aabbs(domain, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    assert inner[0] == pytest.approx([0.0, 0.0, 0.0])
    assert inner[1] == pytest.approx([1.0, 0.0, 0.0])
    assert outer[0] == pytest.approx([-2.0, -2.0, -2.0])
    assert outer[1] == pytest.approx([3.0, 2.0, 2.0])


def test_object_bounds_use_stored_aabb():
    domain = Domain(
        bounds_mode=BOUNDS_OBJECT,
        padding=1.0,
        aabb=[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        object_name="mol",
    )
    outer, inner = domain_preview_aabbs(domain, [(0.0, 0.0, 0.0)])
    assert outer[0] == pytest.approx([-1.0, -1.0, -1.0])
    assert outer[1] == pytest.approx([1.0, 1.0, 1.0])
    assert inner[0] == pytest.approx([0.0, 0.0, 0.0])


def test_aabb_center_extent_and_has_extent():
    pose = aabb_center_extent([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    assert pose is not None
    center, extent = pose
    assert center == pytest.approx((1.0, 2.0, 3.0))
    assert extent == pytest.approx((2.0, 4.0, 6.0))
    assert aabb_has_extent([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) is True
    assert aabb_has_extent([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]) is False


def test_best_projection_prefers_largest_spans():
    assert best_projection_axes([[0.0, 0.0, 0.0], [10.0, 4.0, 0.1]]) == (0, 1)
    assert best_projection_axes([[0.0, 0.0, 0.0], [0.1, 8.0, 9.0]]) == (1, 2)


def test_schematic_layout_outer_contains_inner_when_padded():
    domain = Domain(padding=2.0, spacing=1.0)
    scene = domain_schematic_scene(domain, [(0.0, 0.0, 0.0), (2.0, 1.0, 0.0)])
    layout = layout_schematic(scene, 160, 160)
    assert layout.empty is False
    assert layout.domain_rect is not None
    assert layout.selection_rect is not None
    dx, dy, dw, dh = layout.domain_rect
    sx, sy, sw, sh = layout.selection_rect
    assert dw + 1e-6 >= sw
    assert dh + 1e-6 >= sh
    assert dx <= sx + 1e-6
    assert dy <= sy + 1e-6
    assert dx + dw + 1e-6 >= sx + sw
    assert dy + dh + 1e-6 >= sy + sh
    assert layout.points
    assert len(layout.grid_lines) <= 16


def test_schematic_layout_empty_without_points_or_box():
    scene = domain_schematic_scene(Domain(), centers=None)
    layout = layout_schematic(scene, 160, 140)
    assert layout.empty is True


def test_custom_box_schematic_uses_typed_aabb():
    domain = Domain(
        bounds_mode=BOUNDS_CUSTOM_BOX,
        padding=0.0,
        spacing=1.0,
        aabb=[[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]],
    )
    scene = domain_schematic_scene(domain, [(0.0, 0.0, 0.0)])
    assert scene.domain_aabb[0] == pytest.approx([-2.0, -2.0, -2.0])
    assert scene.domain_aabb[1] == pytest.approx([2.0, 2.0, 2.0])
    layout = layout_schematic(scene, 160, 160)
    assert layout.empty is False
    assert layout.domain_rect is not None


def test_color_field_domain_is_one_third_geometry_spacing():
    from pymolviz.fields.domain import COLOR_FIELD_SPACING_FACTOR, color_field_domain

    geom = Domain(padding=2.0, spacing=0.30)
    color = color_field_domain(geom)
    assert COLOR_FIELD_SPACING_FACTOR == pytest.approx(1.0 / 3.0)
    assert color.spacing == pytest.approx(0.10)
    assert color.padding == pytest.approx(2.0)
    assert color.bounds_mode == geom.bounds_mode
    assert geom.spacing == pytest.approx(0.30)
    finer = color_field_domain(Domain(spacing=0.16))
    assert finer.spacing < 0.16
    assert finer.spacing == pytest.approx(0.16 / 3.0)
