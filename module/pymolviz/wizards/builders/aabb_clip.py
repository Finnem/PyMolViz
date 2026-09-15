"""Axis-locked crop-face gizmos for field visual AABB clip."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from ...fields.clip import aabb_corners, aabb_to_axis_planes, apply_axis_origin_to_aabb, normalize_clip_aabb
from ..pick import qt_modules, qt_widget_alive

_CLIP_MESH_SETTLE_MS = 150


class AabbClipGizmoController:
    """Six cardinal crop planes; drag is locked to each face's axis."""

    def __init__(
        self,
        *,
        page,
        preview,
        get_aabb: Callable[[], Optional[list]],
        set_aabb: Callable[[Optional[list]], None],
        span_points: Callable[[], Optional[Sequence]],
        on_changed: Callable[[], None],
    ):
        self._page = page
        self._preview = preview
        self._get_aabb = get_aabb
        self._set_aabb = set_aabb
        self._span_points = span_points
        self._on_changed = on_changed
        self.selected_index = 0
        self._drag_timer = None
        self._settle_timer = None
        self._mesh_clip_pending = False
        self._suspend = False

    def planes(self):
        return aabb_to_axis_planes(self._get_aabb())

    def clear(self) -> None:
        self.stop_drag_poll()
        self._cancel_mesh_commit()
        if self._preview is not None and hasattr(self._preview, "set_gizmos"):
            self._preview.set_gizmos([], attach_drag=False, axis_lock=True)
        if self._preview is not None and hasattr(self._preview, "release_clip_drag"):
            self._preview.release_clip_drag()

    def select_face(self, axis, is_hi) -> None:
        axis = int(axis)
        for i, plane in enumerate(self.planes()):
            if int(plane.get("axis", -1)) == axis and bool(plane.get("hi")) is bool(is_hi):
                self.selected_index = i
                self.refresh_gizmos()
                return

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
        planes = self.planes()
        if not planes:
            self.clear()
            return
        if self.selected_index is None or self.selected_index >= len(planes):
            self.selected_index = 0
        if attach_drag is None:
            attach_drag = not self.clip_drag_live()
        span = None
        try:
            span = self._span_points()
        except Exception:
            span = None
        self._preview.set_gizmos(
            planes,
            selected_index=self.selected_index,
            span_points=span if span is not None else aabb_corners(self._get_aabb()),
            attach_drag=attach_drag,
            axis_lock=True,
        )
        self.sync_drag_poll()

    def sync_drag_poll(self) -> None:
        QtCore, _, _ = qt_modules()
        if not self.planes():
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
        planes = self.planes()
        idx = self.selected_index
        if not planes or idx is None or idx >= len(planes):
            self.stop_drag_poll()
            self._commit_pending_clip_mesh()
            return
        if self._preview is None or not hasattr(self._preview, "poll_clip_drag"):
            return
        pose = self._preview.poll_clip_drag()
        if pose is None:
            return
        origin, _normal = pose
        plane = planes[idx]
        box = apply_axis_origin_to_aabb(
            self._get_aabb(), plane.get("axis", 0), plane.get("hi"), origin,
        )
        current = normalize_clip_aabb(self._get_aabb())
        if box is None or box == current:
            return
        self._suspend = True
        try:
            self._set_aabb(box)
        finally:
            self._suspend = False
        self.refresh_gizmos(attach_drag=False)
        self._schedule_mesh_commit()
