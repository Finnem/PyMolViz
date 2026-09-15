"""Axis-aligned crop faces and axis-locked clip drag."""

from __future__ import annotations

import numpy as np
import pytest

from pymolviz.fields.clip import (
    aabb_corners,
    aabb_to_axis_planes,
    apply_axis_origin_to_aabb,
    retarget_clip_aabb,
)
from pymolviz.fields.domain import field_display_aabb
from pymolviz.util.clip_drag import apply_axis_drag_matrix, apply_drag_matrix
from pymolviz.wizards.builders.aabb_clip import AabbClipGizmoController


def test_aabb_to_axis_planes_six_inward_cardinals():
    planes = aabb_to_axis_planes([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    assert len(planes) == 6
    normals = [tuple(p["normal"]) for p in planes]
    assert (1.0, 0.0, 0.0) in normals
    assert (-1.0, 0.0, 0.0) in normals
    xmin = next(p for p in planes if p["axis"] == 0 and not p["hi"])
    xmax = next(p for p in planes if p["axis"] == 0 and p["hi"])
    assert xmin["origin"][0] == pytest.approx(0.0)
    assert xmax["origin"][0] == pytest.approx(2.0)
    assert xmin["normal"] == pytest.approx([1.0, 0.0, 0.0])
    assert xmax["normal"] == pytest.approx([-1.0, 0.0, 0.0])


def test_aabb_corners_are_eight_unique_points():
    corners = aabb_corners([[0, 0, 0], [1, 2, 3]])
    assert corners.shape == (8, 3)
    assert len({tuple(p) for p in corners}) == 8


def test_apply_axis_origin_clamps_against_opposite_face():
    box = apply_axis_origin_to_aabb([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], 0, False, (5.0, 0.0, 0.0))
    assert box[0][0] < box[1][0]
    assert box[0][0] == pytest.approx(2.0 - 1e-4)
    widened = apply_axis_origin_to_aabb(box, 0, True, (8.0, 0.0, 0.0))
    assert widened[1][0] == pytest.approx(8.0)


def test_apply_axis_drag_matrix_drops_off_axis_and_rotation():
    origin, normal, center = apply_axis_drag_matrix(
        (0.0, 1.0, 5.0),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 5.0),
        [
            0.0, -1.0, 0.0, 2.0,
            1.0, 0.0, 0.0, 3.0,
            0.0, 0.0, 1.0, -1.5,
            0.0, 0.0, 0.0, 1.0,
        ],
    )
    assert origin == pytest.approx([0.0, 1.0, 3.5])
    assert center == pytest.approx([0.0, 1.0, 3.5])
    assert normal == pytest.approx([0.0, 0.0, 1.0])
    free_o, free_n, _c = apply_drag_matrix(
        (0.0, 1.0, 5.0),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 5.0),
        [
            0.0, -1.0, 0.0, 2.0,
            1.0, 0.0, 0.0, 3.0,
            0.0, 0.0, 1.0, -1.5,
            0.0, 0.0, 0.0, 1.0,
        ],
    )
    assert free_n != pytest.approx([0.0, 0.0, 1.0]) or free_o[0] != pytest.approx(0.0)


class _DragPreview:
    def __init__(self, pose):
        self._pose = pose
        self.gizmo_calls = []

    def poll_clip_drag(self):
        pose = self._pose
        self._pose = None
        return pose

    def set_gizmos(self, planes, selected_index=None, span_points=None, attach_drag=True, axis_lock=None):
        self.gizmo_calls.append({
            "n": len(planes or []),
            "attach_drag": attach_drag,
            "axis_lock": axis_lock,
            "selected": selected_index,
            "origin": list(planes[0]["origin"]) if planes else None,
        })

    def drag_is_live(self):
        return False

    def release_clip_drag(self):
        pass


def test_aabb_clip_controller_drag_updates_face_without_immediate_commit():
    aabb = [[[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]]

    def get_aabb():
        return aabb[0]

    def set_aabb(box):
        aabb[0] = box

    preview = _DragPreview(([1.25, 1.0, 1.0], [1.0, 0.0, 0.0]))
    changed = []
    ctrl = AabbClipGizmoController(
        page=None,
        preview=preview,
        get_aabb=get_aabb,
        set_aabb=set_aabb,
        span_points=lambda: None,
        on_changed=lambda: changed.append("preview"),
    )
    ctrl.selected_index = 0
    ctrl._on_drag_tick()
    assert changed == []
    assert aabb[0][0][0] == pytest.approx(1.25)
    assert preview.gizmo_calls[-1]["attach_drag"] is False
    assert preview.gizmo_calls[-1]["axis_lock"] is True
    assert ctrl._mesh_clip_pending is True
    ctrl._commit_pending_clip_mesh()
    assert changed == ["preview"]


def test_retarget_clip_full_brick_follows_new_cell():
    old = [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]
    new = [[10.0, 0.0, 0.0], [20.0, 10.0, 10.0]]
    moved = retarget_clip_aabb(old, old, new, origin_delta=[10.0, 0.0, 0.0], copied=False)
    assert moved[0][0] == pytest.approx(10.0)
    assert moved[1][0] == pytest.approx(20.0)


def test_retarget_clip_keeps_custom_box_already_on_selection():
    old_brick = [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]
    new_brick = [[10.0, 0.0, 0.0], [20.0, 10.0, 10.0]]
    crop = [[12.0, 1.0, 1.0], [18.0, 8.0, 8.0]]
    kept = retarget_clip_aabb(crop, old_brick, new_brick, origin_delta=[10.0, 0.0, 0.0], copied=False)
    assert kept[0][0] == pytest.approx(12.0)


def test_field_display_aabb_prefers_brick_over_stale_domain():
    from pymolviz.fields.domain import Domain
    from pymolviz.volumetric.GridData import GridData

    values = np.arange(8, dtype=float)
    grid = GridData(values, step_sizes=(1.0, 1.0, 1.0), step_counts=(1, 1, 1), origin=(10.0, 0.0, 0.0))

    class _Field:
        domain = Domain(bounds_mode="custom_box", aabb=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        grid_data = grid

    box = field_display_aabb(_Field(), grid)
    assert box[0][0] == pytest.approx(10.0)
