"""Sphere mesh builder page inside the objects menu."""

from __future__ import annotations

from ..pick import qt_widget_alive
from ..widgets.breadcrumb import CRUMB_SPHERES
from ..widgets.log_slider import LogSegmentRadiusWidget
from ..widgets.section import make_section
from .point_table_page import PointTableBuilderPage
from .points import commit_point_anchors, enabled_points
from .preview import (
    SpherePreview,
    build_cgo_collection,
    retarget_point_collection,
)
from .wireframe_quality import (
    DEFAULT_WIREFRAME_QUALITY,
    effective_wireframe_quality,
    max_allowed_wireframe_quality,
)


class SphereBuilderPage(PointTableBuilderPage):
    """Full editor for multi-point sphere CGOs."""

    MESH_INDEX = 1
    DEFAULT_NAME = "pmv_spheres"
    CRUMB_LEAF = CRUMB_SPHERES
    CONTEXT = "SphereBuilderPage"

    def _appearance_config(self) -> dict:
        return {
            "show_wireframe": True,
            "show_quality": True,
            "quality_range": (1, 5),
            "quality_tooltip": (
                "Mesh detail: 1=80, 2=180, 3=320, 4=720, 5=1280 triangles. "
                "Automatically limited when many points are present."
            ),
        }

    def _make_preview(self):
        return SpherePreview(self.cmd)

    def _init_options_state(self):
        self._radius_widget = None

    def _after_build(self):
        self._update_wireframe_quality_limits()
        super()._after_build()

    def _reset_options(self):
        if self._radius_widget is not None:
            self._radius_widget.set_value(1.0)
        if self._appearance is not None:
            self._appearance.set_wireframe(False)
            self._appearance.set_quality(DEFAULT_WIREFRAME_QUALITY)
            self._appearance.set_specular(True)

    def _load_options(self, obj):
        from .load_visual import sphere_options

        opts = sphere_options(obj)
        if self._radius_widget is not None:
            self._radius_widget.set_value(opts["radius"])
        if self._appearance is not None:
            self._appearance.set_wireframe(opts["wireframe"])
            self._appearance.set_quality(int(opts["quality"]))
            self._appearance.set_specular(opts.get("specular", True))

    def _mount_geometry(self, root, QtCore, QtGui, QtWidgets):
        geom = make_section("Geometry", form=True)
        layout = geom.layout
        self._radius_widget = LogSegmentRadiusWidget(initial=1.0)
        self._radius_widget.connect_changed(self._schedule_preview)
        layout.addRow("Radius (Å)", self._radius_widget.widget)
        root.addWidget(geom.widget)
        return ((self._radius_widget, "Radius of each sphere in Ångströms."),)

    def _update_wireframe_quality_limits(self):
        if self._appearance is None or self._appearance.quality_spin is None:
            return
        spin = self._appearance.quality_spin
        n = len(enabled_points(self._points))
        allowed = max_allowed_wireframe_quality(
            n,
            wireframe=self._appearance.wireframe(),
        )
        current = spin.value()
        spin.blockSignals(True)
        try:
            spin.setMaximum(allowed)
            if current > allowed:
                spin.setValue(allowed)
        finally:
            spin.blockSignals(False)
        effective = effective_wireframe_quality(spin.value(), n)
        if effective.level < spin.value():
            spin.setToolTip(
                "Capped to level %d for %d points (lower = faster)."
                % (effective.level, n)
            )

    def _wireframe_quality_level(self) -> int:
        if self._appearance is None:
            return DEFAULT_WIREFRAME_QUALITY
        return self._appearance.quality()

    def _on_table_sync_begin(self):
        self._update_wireframe_quality_limits()

    def _refresh_preview(self):
        if not qt_widget_alive(self._page):
            return
        try:
            active = enabled_points(self._points)
            if not active:
                self._preview.cleanup()
                return
            wire = self._appearance.wireframe() if self._appearance else False
            self._preview.update(
                self._points,
                self._radius_widget.value(),
                wire,
                self._wireframe_quality_level(),
                clip_planes=self._clip_planes(),
            )
            if self._modifiers is not None:
                self._modifiers.clip.refresh_gizmos()
        except RuntimeError:
            pass

    def _add_preview_points(self, new_pts):
        before = self._wireframe_quality_level()
        self._update_wireframe_quality_limits()
        if self._appearance is not None and self._appearance.quality() != before:
            self._sync_table(preview=True)
            return
        wire = self._appearance.wireframe() if self._appearance else False
        added = self._preview.add_points(
            new_pts,
            self._radius_widget.value(),
            wire,
            self._wireframe_quality_level(),
        )
        self._sync_table(preview=not added)

    def _remove_preview_rows(self, rows):
        self._preview.remove_rows(rows)

    def _preview_after_delete(self) -> bool:
        return False

    def _collection(self, name: str):
        wire = self._appearance.wireframe() if self._appearance else False
        return self._with_look(build_cgo_collection(
            commit_point_anchors(self._points),
            self._radius_widget.value(),
            wire,
            name=name,
            wireframe_quality=self._wireframe_quality_level(),
            clip_planes=self._clip_planes(),
        ))

    def _retarget(self, collection):
        return retarget_point_collection(
            collection,
            commit_point_anchors(self._points),
            clip_planes=self._clip_planes(),
        )
