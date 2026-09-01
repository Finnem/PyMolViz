"""Sphere mesh builder page inside Add Visual."""

from __future__ import annotations

from typing import Callable, List, Optional

from ..pick import DeferredCallback, qt_modules, qt_widget_alive
from ..tooltips import (
    HOOK_TO_SELECTION_TIP,
    SNAP_TO_ATOM_TIP,
    apply_required_tooltips,
    warn_missing_setting_tooltips,
)
from ..widgets.log_slider import LogSegmentRadiusWidget
from .anchor_table import (
    POINT_NAME_COL,
    POINT_SOURCE_COL,
    POINT_X_COL,
    POINT_Y_COL,
    POINT_Z_COL,
    anchor_col_index,
    block_table_selection_signals,
    point_columns,
    sync_anchor_cell,
    unblock_table_selection_signals,
)
from .colors import (
    colors_for_new_points,
    pick_rgb,
    readable_text_color,
)
from .points import (
    VisualPoint,
    camera_center_point,
    export_points_to_selection,
    selection_points,
)
from .preview import SpherePreview, build_cgo_collection, persist_collection
from .zoom_selection import ZOOM_TO_SELECTION_TIP, points_from_rows, wire_zoom_to_selection
from .wireframe_quality import (
    DEFAULT_WIREFRAME_QUALITY,
    effective_wireframe_quality,
    max_allowed_wireframe_quality,
)

COLS = point_columns()


class SphereBuilderPage:
    """Full editor for multi-point sphere CGOs."""

    MESH_INDEX = 1

    def __init__(
        self,
        cmd_,
        on_back: Callable[[], None],
        on_create: Optional[Callable[[], None]] = None,
        parent=None,
    ):
        self.cmd = cmd_
        self._on_back = on_back
        self._on_create = on_create
        self._points: List[VisualPoint] = []
        self._preview = SpherePreview(cmd_)
        self._deferred = DeferredCallback()
        self._page = None
        self._table = None
        self._radius_widget = None
        self._wireframe = None
        self._wireframe_quality = None
        self._snap_atom = None
        self._hook_selection = None
        self._zoom_selection = None
        self._object_name = None
        self._table_filter = None
        self._build(parent)

    @property
    def widget(self):
        return self._page

    def cleanup_preview(self):
        self._deferred.cancel()
        self._preview.cleanup()

    def _build(self, parent):
        QtCore, QtGui, QtWidgets = qt_modules()
        if QtWidgets is None:
            raise RuntimeError("PyMOL Qt UI required")

        page = QtWidgets.QWidget(parent)
        root = QtWidgets.QVBoxLayout(page)

        header = QtWidgets.QHBoxLayout()
        back = QtWidgets.QPushButton("← Back")
        back.setFlat(True)
        back.clicked.connect(self._go_back)
        title = QtWidgets.QLabel("Sphere")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        header.addWidget(back)
        header.addWidget(title)
        header.addStretch(1)
        root.addLayout(header)

        opts = QtWidgets.QGroupBox("Options")
        opts_layout = QtWidgets.QFormLayout(opts)
        self._radius_widget = LogSegmentRadiusWidget(initial=1.0)
        self._radius_widget.connect_changed(self._schedule_preview)
        self._wireframe = QtWidgets.QCheckBox("Wireframe")
        self._wireframe.toggled.connect(self._on_wireframe_toggled)
        self._wireframe_quality = QtWidgets.QSpinBox()
        self._wireframe_quality.setRange(1, 5)
        self._wireframe_quality.setValue(DEFAULT_WIREFRAME_QUALITY)
        self._wireframe_quality.setEnabled(True)
        self._wireframe_quality.valueChanged.connect(lambda *_: self._schedule_preview())
        opts_layout.addRow("Radius", self._radius_widget.widget)
        opts_layout.addRow("", self._wireframe)
        opts_layout.addRow("Mesh quality", self._wireframe_quality)
        root.addWidget(opts)

        pts_box = QtWidgets.QGroupBox("Points")
        pts_layout = QtWidgets.QVBoxLayout(pts_box)
        btn_row = QtWidgets.QHBoxLayout()
        self._snap_atom = QtWidgets.QCheckBox("Snap to atom")
        self._hook_selection = QtWidgets.QCheckBox("Anchor new points")
        self._hook_selection.setChecked(True)
        self._zoom_selection = QtWidgets.QCheckBox("Zoom to selection")
        add_cam = QtWidgets.QPushButton("Add camera center")
        add_cam.clicked.connect(self._add_camera_center)
        add_sel = QtWidgets.QPushButton("Add selection")
        add_sel.clicked.connect(self._add_selection)
        export_sel = QtWidgets.QPushButton("Export points to selection")
        export_sel.clicked.connect(self._export_selection)
        btn_row.addWidget(add_cam)
        btn_row.addWidget(add_sel)
        btn_row.addWidget(export_sel)
        flags = QtWidgets.QHBoxLayout()
        flags.addWidget(self._snap_atom)
        flags.addWidget(self._hook_selection)
        flags.addWidget(self._zoom_selection)
        flags.addStretch(1)
        pts_layout.addLayout(flags)
        pts_layout.addLayout(btn_row)

        self._table = QtWidgets.QTableWidget(0, len(COLS))
        self._table.setHorizontalHeaderLabels(list(COLS))
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.cellChanged.connect(self._on_cell_changed)
        self._table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._show_points_context_menu)
        wire_zoom_to_selection(
            self._table,
            self._zoom_selection,
            self.cmd,
            lambda rows: points_from_rows(self._points, rows),
        )

        class _TableKeyFilter(QtCore.QObject):
            def __init__(self, owner):
                QtCore.QObject.__init__(self)
                self._owner = owner

            def eventFilter(self, obj, event):
                if event.type() != QtCore.QEvent.KeyPress:
                    return False
                if event.key() not in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
                    return False
                table = self._owner._table
                _, _, QtWidgets = qt_modules()
                if QtWidgets is not None and table.state() == QtWidgets.QAbstractItemView.EditingState:
                    return False
                self._owner._delete_selected()
                return True

        self._table_filter = _TableKeyFilter(self)
        self._table.installEventFilter(self._table_filter)
        self._table.viewport().installEventFilter(self._table_filter)
        pts_layout.addWidget(self._table)

        root.addWidget(pts_box, stretch=1)

        actions = QtWidgets.QHBoxLayout()
        self._object_name = QtWidgets.QLineEdit()
        self._object_name.setPlaceholderText("Object name")
        self._object_name.setText("pmv_spheres")
        create_btn = QtWidgets.QPushButton("Create CGO")
        create_btn.clicked.connect(self._create_cgo)
        export_btn = QtWidgets.QPushButton("Export CGO")
        export_btn.clicked.connect(self._export_cgo)
        actions.addWidget(self._object_name, stretch=2)
        actions.addWidget(create_btn)
        actions.addWidget(export_btn)
        root.addLayout(actions)

        apply_required_tooltips(
            [
                (back, "Return to the mesh type list."),
                (self._radius_widget, "Radius of each sphere in Ångströms."),
                (self._wireframe, "Draw each sphere as a mesh cage instead of a filled surface."),
                (
                    self._wireframe_quality,
                    "Mesh detail: 1=80, 2=180, 3=320, 4=720, 5=1280 triangles. "
                    "Automatically limited when many points are present.",
                    "Mesh quality",
                ),
                (self._snap_atom, SNAP_TO_ATOM_TIP),
                (self._hook_selection, HOOK_TO_SELECTION_TIP),
                (self._zoom_selection, ZOOM_TO_SELECTION_TIP),
                (
                    add_cam,
                    "Add a point at the current camera/screen center. "
                    "With Snap to atom, uses the nearest atom within 1 Å.",
                ),
                (add_sel, "Add one point per atom in the current PyMOL selection (sele)."),
                (
                    export_sel,
                    "Create a PyMOL selection covering the table points as pseudoatoms.",
                ),
                (self._object_name, "Name of the PyMOL CGO object created or exported."),
                (create_btn, "Commit the spheres to the session as a named CGO object."),
                (export_btn, "Write a Python script that rebuilds this CGO."),
            ],
            context="SphereBuilderPage",
        )

        self._page = page
        self._update_wireframe_quality_limits()
        warn_missing_setting_tooltips(page, context="SphereBuilderPage")

    def _on_wireframe_toggled(self, checked):
        self._update_wireframe_quality_limits()
        self._schedule_preview()

    def _update_wireframe_quality_limits(self):
        allowed = max_allowed_wireframe_quality(
            len(self._points),
            wireframe=self._wireframe.isChecked(),
        )
        current = self._wireframe_quality.value()
        self._wireframe_quality.blockSignals(True)
        try:
            self._wireframe_quality.setMaximum(allowed)
            if current > allowed:
                self._wireframe_quality.setValue(allowed)
        finally:
            self._wireframe_quality.blockSignals(False)
        effective = effective_wireframe_quality(self._wireframe_quality.value(), len(self._points))
        if effective.level < self._wireframe_quality.value():
            self._wireframe_quality.setToolTip(
                "Capped to level %d for %d points (lower = faster)."
                % (effective.level, len(self._points))
            )
        else:
            self._wireframe_quality.setToolTip(
                "Mesh detail: 1=80, 2=180, 3=320, 4=720, 5=1280 triangles. "
                "Automatically limited when many points are present."
            )

    def _wireframe_quality_level(self) -> int:
        return self._wireframe_quality.value()

    def _selected_rows(self) -> List[int]:
        return sorted({i.row() for i in self._table.selectedIndexes()})

    def _pick_sphere_color(self):
        rows = self._selected_rows()
        if not rows:
            return
        initial = self._points[rows[0]].rgba()
        targets = rows

        original = {row: self._points[row].rgba() for row in targets}

        def on_preview(rgba):
            for row in targets:
                if 0 <= row < len(self._points):
                    self._points[row] = self._points[row].with_color(rgba)
            self._refresh_preview()
            for row in targets:
                item = self._table.item(row, POINT_NAME_COL)
                if item is not None:
                    self._style_name_cell(item, self._points[row])

        def on_done(rgba):
            if rgba is None:
                for row, color in original.items():
                    self._points[row] = self._points[row].with_color(color)
            self._sync_table()

        pick_rgb(self._page, initial, on_change=on_preview, on_done=on_done)

    def _show_points_context_menu(self, pos):
        _, _, QtWidgets = qt_modules()
        index = self._table.indexAt(pos)
        if not index.isValid():
            return
        row = index.row()
        if row not in self._selected_rows():
            self._table.selectRow(row)

        rows = self._selected_rows()
        menu = QtWidgets.QMenu(self._table)
        color_act = menu.addAction("Color selection…")
        color_act.setEnabled(bool(rows))
        color_act.triggered.connect(self._pick_sphere_color)
        del_act = menu.addAction("Delete selected")
        del_act.setEnabled(bool(rows))
        del_act.triggered.connect(self._delete_selected)
        menu.exec_(self._table.viewport().mapToGlobal(pos))

    def _stamp_new_points(self, new_pts: List[VisualPoint]) -> List[VisualPoint]:
        palette = colors_for_new_points(len(new_pts), start_index=len(self._points))
        return [pt.with_color(palette[i]) for i, pt in enumerate(new_pts)]

    def _style_name_cell(self, item, pt):
        _, QtGui, _ = qt_modules()
        text_rgb = readable_text_color(pt.color)
        bg = QtGui.QColor(
            int(pt.color[0] * 255),
            int(pt.color[1] * 255),
            int(pt.color[2] * 255),
        )
        fg = QtGui.QColor(
            int(text_rgb[0] * 255),
            int(text_rgb[1] * 255),
            int(text_rgb[2] * 255),
        )
        item.setBackground(bg)
        item.setForeground(fg)

    def _go_back(self):
        self._deferred.cancel()
        self._preview.cleanup()
        self._on_back()

    def _schedule_preview(self):
        self._deferred.schedule(self._refresh_preview, page=self._page)

    def _refresh_preview(self):
        if not qt_widget_alive(self._page):
            return
        try:
            if not self._points:
                self._preview.cleanup()
                return
            self._preview.update(
                self._points,
                self._radius_widget.value(),
                self._wireframe.isChecked(),
                self._wireframe_quality_level(),
            )
        except RuntimeError:
            pass

    def _anchor_col(self) -> int:
        return anchor_col_index(COLS)

    def _on_anchor_toggled(self, row: int, checked: bool):
        if row < 0 or row >= len(self._points):
            return
        pt = self._points[row]
        if not pt.can_anchor():
            return
        self._points[row] = pt.with_anchored(checked)
        self._schedule_preview()

    def _sync_table(self):
        self._update_wireframe_quality_limits()
        QtCore, QtGui, QtWidgets = qt_modules()
        anchor_col = self._anchor_col()
        sel_blocked = block_table_selection_signals(self._table)
        self._table.blockSignals(True)
        try:
            self._table.setRowCount(len(self._points))
            for row, pt in enumerate(self._points):
                sync_anchor_cell(
                    self._table, row, anchor_col, pt,
                    self._on_anchor_toggled, QtWidgets, QtCore,
                )
                values = (
                    (POINT_NAME_COL, pt.name),
                    (POINT_SOURCE_COL, pt.source),
                    (POINT_X_COL, "%.3f" % pt.x),
                    (POINT_Y_COL, "%.3f" % pt.y),
                    (POINT_Z_COL, "%.3f" % pt.z),
                )
                for col, text in values:
                    item = self._table.item(row, col)
                    if item is None:
                        item = QtWidgets.QTableWidgetItem()
                        self._table.setItem(row, col, item)
                    item.setText(text)
                    if col == POINT_NAME_COL:
                        self._style_name_cell(item, pt)
                    else:
                        item.setBackground(QtGui.QBrush())
                        item.setForeground(QtGui.QBrush())
        finally:
            self._table.blockSignals(False)
            unblock_table_selection_signals(self._table, sel_blocked)
        self._schedule_preview()

    def _on_cell_changed(self, row, col):
        if row < 0 or row >= len(self._points):
            return
        item = self._table.item(row, col)
        if item is None:
            return
        text = item.text()
        pt = self._points[row]
        if col == anchor_col_index(COLS):
            return
        try:
            if col == POINT_NAME_COL:
                pt = pt.with_name(text)
            elif col == POINT_SOURCE_COL:
                pt = pt.with_source(text)
            elif col == POINT_X_COL:
                pt = pt.with_xyz((float(text), pt.y, pt.z))
            elif col == POINT_Y_COL:
                pt = pt.with_xyz((pt.x, float(text), pt.z))
            elif col == POINT_Z_COL:
                pt = pt.with_xyz((pt.x, pt.y, float(text)))
            else:
                return
            self._points[row] = pt
        except ValueError:
            self._sync_table()
            return
        self._schedule_preview()

    def _add_camera_center(self):
        pt = camera_center_point(
            self.cmd,
            self._snap_atom.isChecked(),
            self._points,
            hook_to_selection=self._hook_selection.isChecked(),
        )
        self._points.extend(self._stamp_new_points([pt]))
        self._sync_table()

    def _add_selection(self):
        new_pts = selection_points(
            self.cmd, self._points,
            hook_to_selection=self._hook_selection.isChecked(),
        )
        if not new_pts:
            _, _, QtWidgets = qt_modules()
            if QtWidgets is not None:
                QtWidgets.QMessageBox.information(
                    self._page,
                    "Add selection",
                    "No atoms in the current PyMOL selection.\n"
                    "Pick atoms first (they go into sele), then try again.",
                )
            return
        self._points.extend(self._stamp_new_points(new_pts))
        self._sync_table()

    def _delete_selected(self):
        rows = sorted({i.row() for i in self._table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        for row in rows:
            if 0 <= row < len(self._points):
                del self._points[row]
        self._sync_table()

    def _export_selection(self):
        export_points_to_selection(self.cmd, self._points)

    def _collection(self, name: str):
        return build_cgo_collection(
            self._points,
            self._radius_widget.value(),
            self._wireframe.isChecked(),
            name=name,
            wireframe_quality=self._wireframe_quality_level(),
        )

    def _create_cgo(self):
        if not self._points:
            return
        name = self._object_name.text().strip() or "pmv_spheres"
        persist_collection(self.cmd, self._collection(name))
        self._preview.cleanup()
        if self._on_create is not None:
            self._on_create()

    def _export_cgo(self):
        if not self._points:
            return
        QtCore, _, QtWidgets = qt_modules()
        name = self._object_name.text().strip() or "pmv_spheres"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self._page,
            "Export CGO script",
            "%s.py" % name,
            "Python (*.py)",
        )
        if not path:
            return
        collection = self._collection(name)
        collection.write(path)
