"""Surface mesh builder: SASA / MC / GAUSS around selected points."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ...util.solvent_surface import (
    DEFAULT_ALGORITHM,
    DEFAULT_ATOM_RADIUS,
    DEFAULT_PROBE_RADIUS,
    DEFAULT_QUALITY,
    DEFAULT_RADIUS_MODE,
    DEFAULT_VDW_SCALE,
    SURFACE_ALGORITHM_LABELS,
    SURFACE_ALGORITHM_UI_ORDER,
    estimate_surface_job,
    lookup_source_vdw,
    normalize_algorithm,
)
from ..pick import overlay_warning, qt_modules, qt_widget_alive
from .heavy_job import ask_heavy_job
from ..widgets.ascii_locale import apply_ascii_float_locale
from ..widgets.breadcrumb import CRUMB_SURFACE
from ..widgets.log_slider import LogSegmentRadiusWidget
from ..widgets.section import make_section
from ..widgets.switch import make_switch
from .anchor_table import POINT_COLOR_COL, SURFACE_RADIUS_COL, SURFACE_RADIUS_TIP, surface_point_columns
from .load_visual import surface_options
from .point_table_page import PointTableBuilderPage
from .points import VisualPoint, commit_point_anchors, enabled_points
from .preview import SurfacePreview, retarget_surface_collection
from .preview_mode import preview_is_on, preview_is_simple, preview_mesh_quality, preview_wireframe
from .runtime_helper import build_surface_collection
from .surface_params import (
    SurfaceParamSnapshots,
    algorithm_shows_atom_radius,
    algorithm_shows_probe,
    algorithm_shows_vdw,
)


class SurfaceBuilderPage(PointTableBuilderPage):
    """Editor for a solvent-accessible surface around table points."""

    DEFAULT_NAME = "pmv_surface"
    CRUMB_LEAF = CRUMB_SURFACE
    CONTEXT = "SurfaceBuilderPage"
    COLS = surface_point_columns()
    TABLE_STRONG_FOCUS = True

    def _appearance_config(self) -> dict:
        return {
            "show_wireframe": True,
            "show_quality": True,
            "quality_range": (1, 5),
            "quality_tooltip": (
                "Mesh detail from 1 (draft) to 5 (fine). "
                "Solvent Accessible Surface coarsens geodesic caps; "
                "Marching Cubes and Gaussian Spheres share a finer voxel brick."
            ),
        }

    def _make_preview(self):
        return SurfacePreview(self.cmd)

    def _init_options_state(self):
        self._radius_widget = None
        self._probe_widget = None
        self._use_vdw = None
        self._vdw_scale = None
        self._algorithm = None
        self._param_snapshots = SurfaceParamSnapshots()
        self._atom_radius_row = None
        self._probe_row = None
        self._vdw_row = None
        self._geom_form = None
        self._heavy_ok = None
        self._heavy_denied = None

    def _reset_options(self):
        if self._radius_widget is not None:
            self._radius_widget.set_value(DEFAULT_ATOM_RADIUS)
        if self._probe_widget is not None:
            self._probe_widget.set_value(DEFAULT_PROBE_RADIUS)
        if self._use_vdw is not None:
            self._use_vdw.setChecked(DEFAULT_RADIUS_MODE == "vdw")
        if self._vdw_scale is not None:
            self._vdw_scale.setValue(DEFAULT_VDW_SCALE)
            self._vdw_scale.setEnabled(DEFAULT_RADIUS_MODE == "vdw")
        if self._algorithm is not None:
            index = self._algorithm.findData(DEFAULT_ALGORITHM)
            if index >= 0:
                self._algorithm.setCurrentIndex(index)
        if self._appearance is not None:
            self._appearance.set_quality(DEFAULT_QUALITY)
            self._appearance.set_wireframe(False)
        self._param_snapshots = SurfaceParamSnapshots()
        self._heavy_ok = None
        self._heavy_denied = None
        self._sync_algorithm_visibility()

    def _load_options(self, obj):
        opts = surface_options(obj)
        self._param_snapshots.snapshot_from(
            opts["algorithm"],
            atom_radius=opts["radius"],
            probe=opts["probe_radius"],
            radius_mode=opts.get("radius_mode") or DEFAULT_RADIUS_MODE,
            vdw_scale=opts.get("vdw_scale") or DEFAULT_VDW_SCALE,
        )
        if self._algorithm is not None:
            algo = normalize_algorithm(opts["algorithm"])
            index = self._algorithm.findData(algo)
            if index >= 0:
                self._algorithm.setCurrentIndex(index)
        self._apply_param_snapshot(opts["algorithm"])
        if self._appearance is not None:
            self._appearance.set_quality(int(opts["quality"]))
            self._appearance.set_wireframe(opts["wireframe"])

    def _after_load(self):
        self._schedule_preview()

    def _configure_table(self, QtCore, QtGui, QtWidgets):
        return

    def point_editor_extras(self, index: int, pt: VisualPoint) -> Sequence:
        QtCore, _, QtWidgets = qt_modules()
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)
        caption = QtWidgets.QLabel("Atom radius (Å)")
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(2)
        spin.setRange(0.0, 100.0)
        spin.setSpecialValueText("inherit")
        inherited = self._inherited_radius(pt)
        if pt.radius is None:
            spin.setValue(0.0)
            spin.setToolTip(SURFACE_RADIUS_TIP + " Current inherit: %.2f Å." % inherited)
        else:
            spin.setValue(float(pt.radius))
            spin.setToolTip(SURFACE_RADIUS_TIP)
        apply_ascii_float_locale(spin, QtCore)
        spin.valueChanged.connect(
            lambda value, i=index: self._on_point_radius_changed(i, value)
        )
        layout.addWidget(caption)
        layout.addWidget(spin)
        return (box,)

    def _on_point_radius_changed(self, index: int, value: float):
        if index < 0 or index >= len(self._points):
            return
        if float(value) <= 0.0:
            self._points[index] = self._points[index].with_radius(None)
        else:
            self._points[index] = self._points[index].with_radius(float(value))
        self._schedule_preview()

    def _mount_geometry(self, root, QtCore, QtGui, QtWidgets):
        geom = make_section("Geometry", form=True)
        layout = geom.layout
        self._algorithm = QtWidgets.QComboBox()
        for algo in SURFACE_ALGORITHM_UI_ORDER:
            self._algorithm.addItem(SURFACE_ALGORITHM_LABELS[algo], algo)
        default_idx = self._algorithm.findData(DEFAULT_ALGORITHM)
        if default_idx >= 0:
            self._algorithm.setCurrentIndex(default_idx)
        self._algorithm.currentIndexChanged.connect(self._on_algorithm_changed)
        self._radius_widget = LogSegmentRadiusWidget(initial=DEFAULT_ATOM_RADIUS)
        self._radius_widget.connect_changed(self._on_radius_policy_changed)
        self._probe_widget = LogSegmentRadiusWidget(initial=DEFAULT_PROBE_RADIUS)
        self._probe_widget.connect_changed(self._schedule_preview)
        self._use_vdw = make_switch("Scale by VDW")
        self._use_vdw.setChecked(DEFAULT_RADIUS_MODE == "vdw")
        self._vdw_scale = QtWidgets.QDoubleSpinBox()
        self._vdw_scale.setDecimals(2)
        self._vdw_scale.setRange(0.10, 5.00)
        self._vdw_scale.setSingleStep(0.05)
        self._vdw_scale.setValue(DEFAULT_VDW_SCALE)
        self._vdw_scale.setEnabled(DEFAULT_RADIUS_MODE == "vdw")
        apply_ascii_float_locale(self._vdw_scale, QtCore)
        self._use_vdw.toggled.connect(self._on_vdw_toggled)
        self._vdw_scale.valueChanged.connect(lambda *_: self._on_radius_policy_changed())
        vdw_row = QtWidgets.QWidget()
        vdw_layout = QtWidgets.QHBoxLayout(vdw_row)
        vdw_layout.setContentsMargins(0, 0, 0, 0)
        vdw_layout.addWidget(self._use_vdw)
        vdw_layout.addWidget(self._vdw_scale)
        vdw_layout.addStretch(1)
        self._geom_form = layout
        layout.addRow("Algorithm", self._algorithm)
        self._atom_radius_row = layout.rowCount()
        layout.addRow("Atom radius (Å)", self._radius_widget.widget)
        self._probe_row = layout.rowCount()
        layout.addRow("Probe radius (Å)", self._probe_widget.widget)
        self._vdw_row = layout.rowCount()
        layout.addRow("", vdw_row)
        root.addWidget(geom.widget)
        self._sync_algorithm_visibility()
        return (
            (
                self._algorithm,
                "Gaussian Spheres, Marching Cubes, or Solvent Accessible Surface.",
            ),
            (self._radius_widget, "Default atom radius in Ångströms for points without a custom R value."),
            (
                self._probe_widget,
                "Rolling solvent sphere radius in Ångströms (GAUSS ignores probe).",
            ),
            (
                self._use_vdw,
                "When checked, atom-anchored points use VDW radius times the scale.",
            ),
            (self._vdw_scale, "Multiplier for van der Waals radii when Scale by VDW is on."),
        )

    def _current_algorithm(self) -> str:
        if self._algorithm is None:
            return DEFAULT_ALGORITHM
        algo = self._algorithm.currentData()
        if not algo:
            algo = normalize_algorithm(self._algorithm.currentText())
        return normalize_algorithm(algo)

    def _snapshot_current_params(self):
        if self._radius_widget is None:
            return
        self._param_snapshots.snapshot_from(
            self._current_algorithm(),
            atom_radius=self._radius_widget.value(),
            probe=self._probe_widget.value(),
            radius_mode=self._radius_mode(),
            vdw_scale=float(self._vdw_scale.value()) if self._vdw_scale is not None else DEFAULT_VDW_SCALE,
        )

    def _apply_param_snapshot(self, algorithm: str):
        snap = self._param_snapshots.restore(algorithm)
        if self._radius_widget is not None:
            self._radius_widget.set_value(snap["atom_radius"])
        if self._probe_widget is not None:
            self._probe_widget.set_value(snap["probe"])
        use_vdw = str(snap.get("radius_mode") or DEFAULT_RADIUS_MODE).lower() == "vdw"
        if self._use_vdw is not None:
            self._use_vdw.setChecked(use_vdw)
        if self._vdw_scale is not None:
            self._vdw_scale.setValue(float(snap.get("vdw_scale") or DEFAULT_VDW_SCALE))
            self._vdw_scale.setEnabled(use_vdw)
        self._sync_algorithm_visibility()

    def _on_algorithm_changed(self, *_args):
        new_algo = self._current_algorithm()
        old_algo = self._param_snapshots.algorithm
        if self._radius_widget is not None and new_algo != old_algo:
            self._param_snapshots.snapshot_from(
                old_algo,
                atom_radius=self._radius_widget.value(),
                probe=self._probe_widget.value(),
                radius_mode=self._radius_mode(),
                vdw_scale=float(self._vdw_scale.value()) if self._vdw_scale is not None else DEFAULT_VDW_SCALE,
            )
        self._param_snapshots.set_algorithm(new_algo)
        self._apply_param_snapshot(new_algo)
        self._schedule_preview()

    def _set_form_row_visible(self, row, visible: bool):
        layout = getattr(self, "_geom_form", None)
        if layout is None or row is None:
            return
        setter = getattr(layout, "setRowVisible", None)
        if setter is not None:
            setter(row, bool(visible))
            return
        _, _, QtWidgets = qt_modules()
        if QtWidgets is None:
            return
        for role in (QtWidgets.QFormLayout.LabelRole, QtWidgets.QFormLayout.FieldRole):
            item = layout.itemAt(int(row), role)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.setVisible(bool(visible))

    def _sync_algorithm_visibility(self):
        algo = self._current_algorithm()
        self._set_form_row_visible(self._atom_radius_row, algorithm_shows_atom_radius(algo))
        self._set_form_row_visible(self._probe_row, algorithm_shows_probe(algo))
        self._set_form_row_visible(self._vdw_row, algorithm_shows_vdw(algo))

    def _params(self):
        use_vdw = bool(self._use_vdw.isChecked()) if self._use_vdw is not None else False
        vdw_scale = float(self._vdw_scale.value()) if self._vdw_scale is not None else DEFAULT_VDW_SCALE
        wire = self._appearance.wireframe() if self._appearance else False
        quality = self._appearance.quality() if self._appearance else DEFAULT_QUALITY
        return (
            self._radius_widget.value(),
            self._probe_widget.value(),
            self._current_algorithm(),
            int(quality),
            wire,
            use_vdw,
            vdw_scale,
        )

    def _radius_mode(self) -> str:
        return "vdw" if self._params()[5] else "uniform"

    def _inherited_radius(self, pt) -> float:
        atom_radius, _probe, _alg, _qual, _wire, use_vdw, vdw_scale = self._params()
        if use_vdw:
            vdw = lookup_source_vdw(pt.point_source, None)
            if vdw is not None:
                return float(vdw) * float(vdw_scale)
        return float(atom_radius)

    def _on_vdw_toggled(self, checked: bool):
        if self._vdw_scale is not None:
            self._vdw_scale.setEnabled(bool(checked))
        self._on_radius_policy_changed()

    def _on_radius_policy_changed(self):
        if self._list is not None or self._table is not None:
            self._sync_table(preview=True)
        else:
            self._schedule_preview()

    def _context_color_action(self) -> bool:
        return True

    def _extend_points_context_menu(self, menu, rows):
        super()._extend_points_context_menu(menu, rows)
        reset_r = menu.addAction("Reset radius to default")
        reset_r.setEnabled(bool(rows))
        reset_r.triggered.connect(self._reset_selected_radii)

    def _extra_row_values(self, pt: VisualPoint):
        return (
            (SURFACE_RADIUS_COL, "%.2f" % (
                float(pt.radius) if pt.radius is not None else self._inherited_radius(pt)
            )),
        )

    def _style_cell(self, item, col: int, pt: VisualPoint):
        item.setToolTip(SURFACE_RADIUS_TIP if col == SURFACE_RADIUS_COL else "")
        if col == SURFACE_RADIUS_COL:
            font = item.font()
            font.setItalic(pt.radius is None)
            item.setFont(font)

    def _apply_extra_cell(self, pt: VisualPoint, col: int, text: str):
        if col != SURFACE_RADIUS_COL:
            return None
        stripped = text.strip()
        if not stripped:
            return pt.with_radius(None), True
        return pt.with_radius(float(stripped)), True

    def _remove_preview_rows(self, rows):
        self._deferred.cancel()

    def _reset_selected_radii(self):
        rows = self._selected_rows()
        if not rows:
            return
        for row in rows:
            if 0 <= row < len(self._points):
                self._points[row] = self._points[row].with_radius(None)
        self._sync_table()

    def _refresh_preview(self):
        mode = self._preview_if_on()
        if self._modifiers is not None:
            self._modifiers.clip._sync_drag_poll()
        if mode is None:
            return
        try:
            if not preview_is_simple(mode) and not self._confirm_heavy_surface():
                return
            radius, probe, algorithm, quality, wireframe, _use_vdw, vdw_scale = self._params()
            self._preview.update(
                self._points, radius, probe, algorithm,
                preview_mesh_quality(quality, mode),
                preview_wireframe(wireframe, mode),
                radius_mode=self._radius_mode(), vdw_scale=vdw_scale,
                clip_planes=self._clip_planes(),
                gizmo_planes=self._modifiers.clip.planes if self._modifiers else [],
                gizmo_selected=self._modifiers.clip._selected_index() if self._modifiers else None,
            )
        except Exception as exc:
            self._warn_surface_failed(exc)
        if self._modifiers is not None:
            self._modifiers.clip.refresh_gizmos()

    def _current_job(self):
        active = enabled_points(self._points)
        if not active:
            return estimate_surface_job([], self._current_algorithm(), DEFAULT_QUALITY)
        xyz = np.array([pt.xyz() for pt in active], dtype=float)
        radius, probe, algorithm, quality, _wire, _use_vdw, _vdw_scale = self._params()
        return estimate_surface_job(
            xyz, algorithm, quality, atom_radius=radius, probe_radius=probe,
        )

    def _confirm_heavy_surface(self) -> bool:
        job = self._current_job()
        allowed, ok_fp, denied_fp = ask_heavy_job(
            self._page,
            job,
            title="Large surface",
            previous_ok=self._heavy_ok,
            previous_denied=self._heavy_denied,
            no_qt_default=False,
        )
        if ok_fp is not None:
            self._heavy_ok = ok_fp
            self._heavy_denied = None
        if denied_fp is not None:
            self._heavy_denied = denied_fp
        return allowed

    def _warn_surface_failed(self, exc):
        _, _, QtWidgets = qt_modules()
        if QtWidgets is None or self._page is None:
            return
        overlay_warning(
            self._page,
            "Surface preview failed",
            "Could not build the surface (%s)." % exc,
        )

    def _create_cgo(self):
        if not self._confirm_heavy_surface():
            return
        super()._create_cgo()

    def _export_cgo(self):
        if not self._confirm_heavy_surface():
            return
        super()._export_cgo()

    def _collection(self, name: str):
        radius, probe, algorithm, quality, wireframe, _use_vdw, vdw_scale = self._params()
        return self._with_look(build_surface_collection(
            commit_point_anchors(self._points),
            radius, probe, algorithm, quality, wireframe, name,
            radius_mode=self._radius_mode(), vdw_scale=vdw_scale,
            clip_planes=self._clip_planes(),
        ))

    def _retarget(self, collection):
        radius, probe, algorithm, quality, wireframe, _use_vdw, vdw_scale = self._params()
        return retarget_surface_collection(
            collection, commit_point_anchors(self._points),
            radius, probe, algorithm, quality, wireframe,
            radius_mode=self._radius_mode(), vdw_scale=vdw_scale,
            clip_planes=self._clip_planes(),
        )
