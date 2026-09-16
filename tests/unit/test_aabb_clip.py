"""Axis-aligned crop faces and axis-locked clip drag."""

from __future__ import annotations

import numpy as np
import pytest

from pymolviz.fields.clip import (
    aabb_corners,
    aabb_to_axis_planes,
    aabb_to_cardinal_planes,
    apply_axis_origin_to_aabb,
    apply_origin_to_cardinal_planes,
    cardinal_planes_to_aabb,
    cardinal_planes_to_gizmos,
    default_cardinal_plane,
    retarget_clip_aabb,
    normalize_cardinal_planes,
    opposite_cardinal_plane,
)
from pymolviz.fields.domain import field_display_aabb
from pymolviz.util.clip_drag import apply_axis_drag_matrix, apply_drag_matrix
from pymolviz.wizards.builders.aabb_clip import CardinalClipGizmoController


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


def test_cardinal_planes_one_axis_does_not_enable_the_others():
    domain = [[0.0, 0.0, 0.0], [10.0, 8.0, 6.0]]
    box = cardinal_planes_to_aabb(
        [{"axis": 0, "position": 3.0, "hi": False}],
        domain,
    )
    assert box[0][0] == pytest.approx(3.0)
    assert box[1][0] == pytest.approx(10.0)
    assert box[0][1] == pytest.approx(0.0)
    assert box[1][1] == pytest.approx(8.0)
    assert box[0][2] == pytest.approx(0.0)
    assert box[1][2] == pytest.approx(6.0)
    gizmos = cardinal_planes_to_gizmos(
        [{"axis": 0, "position": 3.0, "hi": False}],
        domain,
    )
    assert len(gizmos) == 1
    assert gizmos[0]["normal"] == pytest.approx([1.0, 0.0, 0.0])
    assert gizmos[0]["origin"][0] == pytest.approx(3.0)


def test_cardinal_flip_keeps_the_other_half():
    domain = [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]
    lo_keep = cardinal_planes_to_aabb(
        [{"axis": 2, "position": 4.0, "hi": False}],
        domain,
    )
    hi_keep = cardinal_planes_to_aabb(
        [{"axis": 2, "position": 4.0, "hi": True}],
        domain,
    )
    assert lo_keep[0][2] == pytest.approx(4.0)
    assert lo_keep[1][2] == pytest.approx(10.0)
    assert hi_keep[0][2] == pytest.approx(0.0)
    assert hi_keep[1][2] == pytest.approx(4.0)


def test_aabb_to_cardinal_planes_ignores_domain_faces():
    domain = [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]
    assert aabb_to_cardinal_planes(domain, domain) == []
    planes = aabb_to_cardinal_planes([[2.0, 0.0, 0.0], [10.0, 10.0, 7.0]], domain)
    by_axis = {p["axis"]: p for p in planes}
    assert set(by_axis) == {0, 2}
    assert by_axis[0]["hi"] is False
    assert by_axis[0]["position"] == pytest.approx(2.0)
    assert by_axis[2]["hi"] is True
    assert by_axis[2]["position"] == pytest.approx(7.0)


def test_two_planes_on_one_axis_make_a_slab():
    domain = [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]
    first = {"axis": 0, "position": 2.0, "hi": False}
    second = opposite_cardinal_plane(first, domain)
    assert second["axis"] == 0
    assert second["hi"] is True
    assert second["position"] > 2.0
    box = cardinal_planes_to_aabb([first, second], domain)
    assert box[0][0] == pytest.approx(2.0)
    assert box[1][0] == pytest.approx(second["position"])
    assert box[0][1] == pytest.approx(0.0)
    gizmos = cardinal_planes_to_gizmos([first, second], domain)
    assert len(gizmos) == 2
    kept = normalize_cardinal_planes([first, second, {"axis": 0, "position": 9.0, "hi": True}])
    assert len([p for p in kept if p["axis"] == 0]) == 2
    inferred = aabb_to_cardinal_planes([[2.0, 0.0, 0.0], [8.0, 10.0, 10.0]], domain)
    by_hi = {(p["axis"], p["hi"]): p for p in inferred}
    assert (0, False) in by_hi and (0, True) in by_hi
    assert by_hi[(0, False)]["position"] == pytest.approx(2.0)
    assert by_hi[(0, True)]["position"] == pytest.approx(8.0)


def test_apply_origin_moves_only_the_locked_face():
    domain = [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]
    planes = [
        {"axis": 0, "position": 2.0, "hi": False},
        {"axis": 0, "position": 8.0, "hi": True},
        {"axis": 2, "position": 5.0, "hi": True},
    ]
    moved = apply_origin_to_cardinal_planes(planes, 0, (3.0, 0.0, 0.0), domain, hi=False)
    by_key = {(p["axis"], p["hi"]): p for p in moved}
    assert by_key[(0, False)]["position"] == pytest.approx(3.0)
    assert by_key[(0, True)]["position"] == pytest.approx(8.0)
    assert by_key[(2, True)]["position"] == pytest.approx(5.0)
    blocked = apply_origin_to_cardinal_planes(planes, 0, (9.5, 0.0, 0.0), domain, hi=False)
    by_key = {(p["axis"], p["hi"]): p for p in blocked}
    assert by_key[(0, False)]["position"] < by_key[(0, True)]["position"]
    assert by_key[(0, True)]["position"] == pytest.approx(8.0)


def test_default_cardinal_plane_is_domain_midpoint():
    plane = default_cardinal_plane(1, [[0.0, -4.0, 0.0], [10.0, 6.0, 2.0]])
    assert plane["axis"] == 1
    assert plane["hi"] is False
    assert plane["position"] == pytest.approx(1.0)


def test_apply_origin_moves_only_the_dragged_axis():
    planes = [
        {"axis": 0, "position": 1.0, "hi": False},
        {"axis": 2, "position": 5.0, "hi": True},
    ]
    moved = apply_origin_to_cardinal_planes(
        planes, 0, (2.5, 9.0, 9.0), [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]],
    )
    by_axis = {p["axis"]: p for p in moved}
    assert by_axis[0]["position"] == pytest.approx(2.5)
    assert by_axis[2]["position"] == pytest.approx(5.0)


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


def test_cardinal_clip_controller_drag_updates_plane_without_immediate_commit():
    state = [[{"axis": 0, "position": 0.0, "hi": False}]]
    domain = [[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]]

    def get_planes():
        return state[0]

    def set_planes(planes):
        state[0] = planes

    preview = _DragPreview(([1.25, 1.0, 1.0], [1.0, 0.0, 0.0]))
    changed = []
    ctrl = CardinalClipGizmoController(
        page=None,
        preview=preview,
        get_planes=get_planes,
        set_planes=set_planes,
        span_points=lambda: None,
        domain_aabb=lambda: domain,
        on_changed=lambda: changed.append("preview"),
    )
    ctrl.selected_axis = 0
    ctrl._on_drag_tick()
    assert changed == []
    assert state[0][0]["position"] == pytest.approx(1.25)
    assert preview.gizmo_calls[-1]["attach_drag"] is False
    assert preview.gizmo_calls[-1]["axis_lock"] is True
    assert preview.gizmo_calls[-1]["n"] == 1
    assert ctrl._mesh_clip_pending is True
    ctrl._commit_pending_clip_mesh()
    assert changed == ["preview"]


def test_cardinal_clip_drag_moves_only_selected_face():
    state = [[
        {"axis": 0, "position": 1.0, "hi": False},
        {"axis": 0, "position": 3.0, "hi": True},
    ]]
    domain = [[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]]
    preview = _DragPreview(([1.5, 1.0, 1.0], [1.0, 0.0, 0.0]))
    ctrl = CardinalClipGizmoController(
        page=None,
        preview=preview,
        get_planes=lambda: state[0],
        set_planes=lambda planes: state.__setitem__(0, planes),
        span_points=lambda: None,
        domain_aabb=lambda: domain,
        on_changed=lambda: None,
    )
    ctrl.selected_axis = 0
    ctrl.selected_hi = True
    ctrl._on_drag_tick()
    by_hi = {p["hi"]: p for p in state[0] if p["axis"] == 0}
    assert by_hi[False]["position"] == pytest.approx(1.0)
    assert by_hi[True]["position"] == pytest.approx(1.5)
    assert preview.gizmo_calls[-1]["n"] == 2


def test_hidden_cardinal_gizmo_is_omitted_without_commit():
    state = [[{"axis": 0, "position": 0.0, "hi": False}]]
    domain = [[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]]
    visible = {0: True}

    preview = _DragPreview(None)
    changed = []
    ctrl = CardinalClipGizmoController(
        page=None,
        preview=preview,
        get_planes=lambda: state[0],
        set_planes=lambda planes: state.__setitem__(0, planes),
        span_points=lambda: None,
        domain_aabb=lambda: domain,
        on_changed=lambda: changed.append("preview"),
        gizmos_visible=lambda axis: visible.get(int(axis), True),
    )
    ctrl.selected_axis = 0
    ctrl.refresh_gizmos()
    assert preview.gizmo_calls[-1]["n"] == 1

    visible[0] = False
    ctrl.refresh_gizmos()
    assert changed == []
    assert preview.gizmo_calls[-1]["n"] == 0
    assert state[0][0]["position"] == pytest.approx(0.0)
    assert ctrl.selected_axis == 0

    visible[0] = True
    ctrl.refresh_gizmos()
    assert preview.gizmo_calls[-1]["n"] == 1
    assert preview.gizmo_calls[-1]["selected"] == 0


def test_axis_locked_gizmo_has_one_keep_side_arrow():
    from pymolviz.util.clip_gizmo import build_clip_gizmo_cgo

    tokens = build_clip_gizmo_cgo(
        (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 4.0, axis_arrow=True,
    )
    assert tokens.count("CONE") == 1
    plain = build_clip_gizmo_cgo((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 4.0)
    assert "CONE" not in plain


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
