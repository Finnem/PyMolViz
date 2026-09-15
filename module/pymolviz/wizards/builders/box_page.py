"""Box mesh builder page inside the objects menu."""

from __future__ import annotations

from typing import Tuple

from ..pick import qt_widget_alive
from ..widgets.breadcrumb import CRUMB_BOXES
from ..widgets.log_slider import LogSegmentRadiusWidget
from ..widgets.section import make_section
from .point_table_page import PointTableBuilderPage
from .points import commit_point_anchors, enabled_points
from .preview import (
    BoxPreview,
    build_box_cgo_collection,
    retarget_point_collection,
)
from .preview_mode import preview_is_on, preview_wireframe


class BoxBuilderPage(PointTableBuilderPage):
    """Full editor for multi-point box CGOs."""

    DEFAULT_NAME = "pmv_boxes"
    CRUMB_LEAF = CRUMB_BOXES
    CONTEXT = "BoxBuilderPage"

    def _appearance_config(self) -> dict:
        return {"show_wireframe": True, "show_quality": False}

    def _make_preview(self):
        return BoxPreview(self.cmd)

    def _init_options_state(self):
        self._extent_x = None
        self._extent_y = None
        self._extent_z = None

    def _reset_options(self):
        for widget, value in (
            (self._extent_x, 1.0),
            (self._extent_y, 1.0),
            (self._extent_z, 1.0),
        ):
            if widget is not None:
                widget.set_value(value)
        if self._appearance is not None:
            self._appearance.set_wireframe(False)

    def _load_options(self, obj):
        from .load_visual import box_options

        opts = box_options(obj)
        extent = opts["extent"]
        if self._extent_x is not None:
            self._extent_x.set_value(extent[0])
        if self._extent_y is not None:
            self._extent_y.set_value(extent[1])
        if self._extent_z is not None:
            self._extent_z.set_value(extent[2])
        if self._appearance is not None:
            self._appearance.set_wireframe(opts["wireframe"])

    def _mount_geometry(self, root, QtCore, QtGui, QtWidgets):
        geom = make_section("Geometry", form=True)
        layout = geom.layout
        self._extent_x = LogSegmentRadiusWidget(initial=1.0)
        self._extent_y = LogSegmentRadiusWidget(initial=1.0)
        self._extent_z = LogSegmentRadiusWidget(initial=1.0)
        for widget in (self._extent_x, self._extent_y, self._extent_z):
            widget.connect_changed(self._schedule_preview)
        layout.addRow("Extent X (Å)", self._extent_x.widget)
        layout.addRow("Extent Y (Å)", self._extent_y.widget)
        layout.addRow("Extent Z (Å)", self._extent_z.widget)
        root.addWidget(geom.widget)
        return (
            (self._extent_x, "Full box width along X in Ångströms (centered on the point)."),
            (self._extent_y, "Full box width along Y in Ångströms (centered on the point)."),
            (self._extent_z, "Full box width along Z in Ångströms (centered on the point)."),
        )

    def _extent(self) -> Tuple[float, float, float]:
        return (
            self._extent_x.value(),
            self._extent_y.value(),
            self._extent_z.value(),
        )

    def _refresh_preview(self):
        if not qt_widget_alive(self._page):
            return
        try:
            if not enabled_points(self._points):
                self._preview.cleanup()
                return
            mode = self._current_preview_mode()
            if not preview_is_on(mode):
                self._preview.clear_meshes()
                if self._modifiers is not None:
                    self._modifiers.clip.refresh_gizmos()
                return
            wire = preview_wireframe(
                self._appearance.wireframe() if self._appearance else False, mode,
            )
            self._preview.update(
                self._points,
                self._extent(),
                wire,
                clip_planes=self._clip_planes(),
            )
            if self._modifiers is not None:
                self._modifiers.clip.refresh_gizmos()
        except RuntimeError:
            pass

    def _add_preview_points(self, new_pts):
        mode = self._current_preview_mode()
        if not preview_is_on(mode):
            return
        wire = preview_wireframe(
            self._appearance.wireframe() if self._appearance else False, mode,
        )
        added = self._preview.add_points(new_pts, self._extent(), wire)
        self._sync_table(preview=not added)

    def _remove_preview_rows(self, rows):
        self._preview.remove_rows(rows)

    def _preview_after_delete(self) -> bool:
        return False

    def _collection(self, name: str):
        wire = self._appearance.wireframe() if self._appearance else False
        return self._with_look(build_box_cgo_collection(
            commit_point_anchors(self._points),
            self._extent(),
            wire,
            name=name,
            clip_planes=self._clip_planes(),
        ))

    def _retarget(self, collection):
        return retarget_point_collection(
            collection,
            commit_point_anchors(self._points),
            clip_planes=self._clip_planes(),
        )
