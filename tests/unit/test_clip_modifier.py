"""Clip-plane drag updates the gizmo immediately and remeshes after settle."""

from __future__ import annotations

import pytest

from pymolviz.wizards.builders.clip_modifier import ClipModifierController


class _RowList:
    def __init__(self, row=0):
        self._row = int(row)

    def currentRow(self):
        return self._row

    def setCurrentRow(self, row):
        self._row = int(row)


class _DragPreview:
    def __init__(self, pose):
        self._pose = pose
        self.gizmo_calls = []
        self.released = 0

    def poll_clip_drag(self):
        pose = self._pose
        self._pose = None
        return pose

    def set_gizmos(self, planes, selected_index=None, span_points=None, attach_drag=True):
        origin = list(planes[0]["origin"]) if planes else None
        self.gizmo_calls.append({
            "origin": origin,
            "attach_drag": attach_drag,
            "selected": selected_index,
            "n": len(planes or []),
        })

    def release_clip_drag(self):
        self.released += 1

    def span_points(self):
        return None


def _controller(preview, on_changed):
    ctrl = ClipModifierController(
        cmd=None,
        page=None,
        context="test",
        preview=preview,
        span_points=lambda: None,
        on_changed=on_changed,
    )
    ctrl.clip_planes = [{
        "origin": [0.0, 0.0, 0.0],
        "normal": [0.0, 0.0, 1.0],
        "scale": 5.0,
    }]
    ctrl._list = _RowList()
    return ctrl


def test_drag_tick_moves_plane_gizmo_without_remeshing():
    changed = []
    preview = _DragPreview(([0.0, 0.0, 2.0], [0.0, 0.0, 1.0]))
    ctrl = _controller(preview, lambda: changed.append("mesh"))

    ctrl._on_drag_tick()

    assert changed == []
    assert ctrl.clip_planes[0]["origin"][2] == pytest.approx(2.0)
    assert preview.gizmo_calls
    assert preview.gizmo_calls[-1]["attach_drag"] is False
    assert preview.gizmo_calls[-1]["origin"][2] == pytest.approx(2.0)
    assert ctrl._mesh_clip_pending is True

    ctrl._commit_pending_clip_mesh()
    assert changed == ["mesh"]
    assert ctrl._mesh_clip_pending is False


def test_drag_tick_skips_noop_pose():
    changed = []
    preview = _DragPreview(([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]))
    ctrl = _controller(preview, lambda: changed.append("mesh"))

    ctrl._on_drag_tick()

    assert changed == []
    assert preview.gizmo_calls == []
    assert ctrl._mesh_clip_pending is False


def test_resolve_list_row_allows_no_selection():
    ctrl = _controller(_DragPreview(None), lambda: None)
    ctrl.clip_planes = [
        {"origin": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0], "scale": 5.0},
        {"origin": [1.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0], "scale": 5.0},
    ]
    ctrl._list = _RowList(row=-1)
    assert ctrl.resolve_list_row() == -1
    assert ctrl.resolve_list_row(keep_row=1) == 1
    assert ctrl.resolve_list_row(keep_row=9) == 1
    ctrl.clip_planes = []
    assert ctrl.resolve_list_row() == -1


def test_clear_selection_keeps_planes_and_drops_drag():
    changed = []
    preview = _DragPreview(None)
    ctrl = _controller(preview, lambda: changed.append("mesh"))
    ctrl._list = _RowList(row=0)

    ctrl.clear_selection()

    assert ctrl.selected_index() is None
    assert preview.released == 1
    assert preview.gizmo_calls[-1]["selected"] is None
    assert preview.gizmo_calls[-1]["n"] == 1
    assert changed == []
    assert ctrl.clip_planes


def test_hidden_plane_gizmo_is_omitted_without_remesh():
    changed = []
    preview = _DragPreview(None)
    ctrl = _controller(preview, lambda: changed.append("mesh"))
    ctrl.clip_planes.append({
        "origin": [1.0, 0.0, 0.0],
        "normal": [0.0, 0.0, 1.0],
        "scale": 5.0,
        "gizmo": True,
    })

    ctrl.set_plane_gizmo_visible(0, False)
    assert changed == []
    assert ctrl.clip_planes[0]["gizmo"] is False
    planes, selected = ctrl.preview_gizmos()
    assert len(planes) == 1
    assert planes[0]["origin"][0] == pytest.approx(1.0)
    assert selected is None
    assert preview.gizmo_calls[-1]["n"] == 1
    assert preview.gizmo_calls[-1]["selected"] is None

    ctrl.set_gizmos_shown(False)
    assert ctrl.preview_gizmos() == ([], None)
    assert preview.gizmo_calls[-1]["n"] == 0
    assert preview.released >= 1
    assert ctrl.clip_planes

    ctrl.set_gizmos_shown(True)
    planes, _selected = ctrl.preview_gizmos()
    assert len(planes) == 1
    assert preview.gizmo_calls[-1]["n"] == 1


def test_hiding_selected_gizmo_drops_drag_poll_index():
    preview = _DragPreview(None)
    ctrl = _controller(preview, lambda: None)
    ctrl._list = _RowList(row=0)
    ctrl.set_plane_gizmo_visible(0, False)
    _planes, selected = ctrl.preview_gizmos()
    assert selected is None
    assert preview.released == 1
