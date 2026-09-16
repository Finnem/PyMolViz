"""Axis-locked gizmos for independently enabled X/Y/Z field clip planes."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from ...fields.clip import (
    aabb_corners,
    apply_origin_to_cardinal_planes,
    cardinal_planes_to_gizmos,
    normalize_cardinal_planes,
)
from ..pick import qt_modules, qt_widget_alive

_CLIP_MESH_SETTLE_MS = 150


class CardinalClipGizmoController:
    """One gizmo per enabled cardinal plane; drag is locked to that axis."""

    def __init__(
        self,
        *,
        page,
        preview,
        get_planes: Callable[[], Sequence],
        set_planes: Callable[[Sequence], None],
        span_points: Callable[[], Optional[Sequence]],
        domain_aabb: Callable[[], Optional[list]],
        on_changed: Callable[[], None],
        gizmos_visible: Optional[Callable[[int], bool]] = None,
    ):
        self._page = page
        self._preview = preview
        self._get_planes = get_planes
        self._set_planes = set_planes
        self._span_points = span_points
        self._domain_aabb = domain_aabb
        self._on_changed = on_changed
        self._gizmos_visible = gizmos_visible
        self.selected_axis = None
        self.selected_hi = None
        self._drag_timer = None
        self._settle_timer = None
        self._mesh_clip_pending = False
        self._suspend = False

    def planes(self):
        return cardinal_planes_to_gizmos(self._get_planes(), self._domain())

    def _axis_gizmo_visible(self, axis: int) -> bool:
        if self._gizmos_visible is None:
            return True
        try:
            return bool(self._gizmos_visible(int(axis)))
        except Exception:
            return True

    def visible_gizmos(self):
        return [
            plane for plane in self.planes()
            if self._axis_gizmo_visible(int(plane.get("axis", 0)))
        ]

    def _domain(self):
        try:
            return self._domain_aabb()
        except Exception:
            return None

    def selected_index(self):
        gizmos = self.visible_gizmos()
        if not gizmos:
            return None
        axis = self.selected_axis
        if axis is None:
            return None
        matches = [
            i for i, plane in enumerate(gizmos)
            if int(plane.get("axis", -1)) == int(axis)
        ]
        if not matches:
            return None
        hi = self.selected_hi
        if hi is not None:
            for i in matches:
                if bool(gizmos[i].get("hi")) == bool(hi):
                    return i
        return matches[0]

    def clear(self) -> None:
        self.stop_drag_poll()
        self._cancel_mesh_commit()
        self.selected_axis = None
        self.selected_hi = None
        if self._preview is not None and hasattr(self._preview, "set_gizmos"):
            self._preview.set_gizmos([], attach_drag=False, axis_lock=True)
        if self._preview is not None and hasattr(self._preview, "release_clip_drag"):
            self._preview.release_clip_drag()

    def relatch(self) -> None:
        if self._preview is not None and hasattr(self._preview, "release_clip_drag"):
            self._preview.release_clip_drag()

    def select_axis(self, axis, hi=None) -> None:
        try:
            self.selected_axis = int(axis)
        except (TypeError, ValueError):
            self.selected_axis = None
        if hi is None:
            if self.selected_axis is None:
                self.selected_hi = None
        else:
            self.selected_hi = bool(hi)
        self.relatch()
        self.refresh_gizmos()

    def clip_drag_live(self) -> bool:
        preview = self._preview
        if preview is None:
            return False
        fn = getattr(preview, "drag_is_live", None)
        if callable(fn):
            return bool(fn())
        return False

    def refresh_gizmos(self, attach_drag=None) -> None:
        if self._preview is None or not hasattr(self._preview, "set_gizmos"):
            return
        enabled = self.planes()
        if not enabled:
            self.clear()
            return
        gizmos = self.visible_gizmos()
        if not gizmos:
            self._preview.set_gizmos([], attach_drag=False, axis_lock=True)
            if hasattr(self._preview, "release_clip_drag"):
                self._preview.release_clip_drag()
            self.stop_drag_poll()
            return
        if self.selected_axis is None:
            self.selected_axis = int(gizmos[0].get("axis", 0))
            self.selected_hi = bool(gizmos[0].get("hi"))
        if attach_drag is None:
            attach_drag = not self.clip_drag_live()
        if self.selected_index() is None:
            attach_drag = False
        span = None
        try:
            span = self._span_points()
        except Exception:
            span = None
        if span is None:
            span = aabb_corners(self._domain())
        self._preview.set_gizmos(
            gizmos,
            selected_index=self.selected_index(),
            span_points=span,
            attach_drag=attach_drag,
            axis_lock=True,
        )
        self.sync_drag_poll()

    def sync_drag_poll(self) -> None:
        QtCore, _, _ = qt_modules()
        if self.selected_index() is None:
            self.stop_drag_poll()
            self._commit_pending_clip_mesh()
            return
        if QtCore is None or not hasattr(QtCore, "QTimer") or self._page is None:
            return
        if self._drag_timer is None:
            timer = QtCore.QTimer(self._page)
            timer.setInterval(50)
            timer.timeout.connect(self._on_drag_tick)
            self._drag_timer = timer
        if not self._drag_timer.isActive():
            self._drag_timer.start()

    def stop_drag_poll(self) -> None:
        timer = self._drag_timer
        if timer is not None:
            try:
                timer.stop()
            except RuntimeError:
                pass

    def _ensure_settle_timer(self):
        QtCore, _, _ = qt_modules()
        if QtCore is None or not hasattr(QtCore, "QTimer") or self._page is None:
            return None
        if self._settle_timer is None:
            timer = QtCore.QTimer(self._page)
            timer.setSingleShot(True)
            timer.setInterval(_CLIP_MESH_SETTLE_MS)
            timer.timeout.connect(self._commit_pending_clip_mesh)
            self._settle_timer = timer
        return self._settle_timer

    def _cancel_mesh_commit(self) -> None:
        self._mesh_clip_pending = False
        timer = self._settle_timer
        if timer is not None:
            try:
                timer.stop()
            except RuntimeError:
                pass

    def _schedule_mesh_commit(self) -> None:
        self._mesh_clip_pending = True
        timer = self._ensure_settle_timer()
        if timer is not None:
            timer.start()

    def _commit_pending_clip_mesh(self) -> None:
        if not self._mesh_clip_pending:
            return
        if self._page is not None and not qt_widget_alive(self._page):
            self._mesh_clip_pending = False
            return
        self._mesh_clip_pending = False
        self._on_changed()

    def _on_drag_tick(self):
        if self._suspend:
            return
        gizmos = self.visible_gizmos()
        idx = self.selected_index()
        if not gizmos or idx is None or idx >= len(gizmos):
            self.stop_drag_poll()
            self._commit_pending_clip_mesh()
            return
        if self._preview is None or not hasattr(self._preview, "poll_clip_drag"):
            return
        pose = self._preview.poll_clip_drag()
        if pose is None:
            return
        origin, _normal = pose
        axis = int(gizmos[idx].get("axis", 0))
        hi = bool(gizmos[idx].get("hi"))
        self.selected_hi = hi
        current = normalize_cardinal_planes(self._get_planes())
        updated = apply_origin_to_cardinal_planes(
            current, axis, origin, self._domain(), hi=hi,
        )
        if updated == current:
            return
        self._suspend = True
        try:
            self._set_planes(updated)
        finally:
            self._suspend = False
        self.refresh_gizmos(attach_drag=False)
        self._schedule_mesh_commit()
