"""Sphere mesh builder page inside Add Visual."""

from __future__ import annotations

from typing import Callable, List, Optional

from ..pick import qt_modules
from ..widgets.log_slider import LogSegmentRadiusWidget
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
from .preview import (
    SpherePreview,
    build_cgo_collection,
    build_sphere_cgo_list,
    _commit_cgo,
)
from .wireframe_quality import (
    DEFAULT_WIREFRAME_QUALITY,
    effective_wireframe_quality,
    max_allowed_wireframe_quality,
)

COLS = ("Name", "Source", "X", "Y", "Z")


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
        self._page = None
        self._table = None
        self._radius_widget = None
        self._wireframe = None
        self._wireframe_quality = None
        self._snap_atom = None
        self._object_name = None
        self._table_filter = None
        self._build(parent)

    @property
    def widget(self):
        return self._page

    def cleanup_preview(self):
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
        back.clicked.connect(self._on_back)
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
        self._snap_atom.setToolTip("Snap camera-center point to the nearest atom within 1 Å")
        add_cam = QtWidgets.QPushButton("Add camera center")
        add_cam.clicked.connect(self._add_camera_center)
        add_sel = QtWidgets.QPushButton("Add selection")
        add_sel.clicked.connect(self._add_selection)
        export_sel = QtWidgets.QPushButton("Export points to selection")
        export_sel.clicked.connect(self._export_selection)
        btn_row.addWidget(add_cam)
        btn_row.addWidget(add_sel)
        btn_row.addWidget(export_sel)
        pts_layout.addWidget(self._snap_atom)
        pts_layout.addLayout(btn_row)

        self._table = QtWidgets.QTableWidget(0, len(COLS))
        self._table.setHorizontalHeaderLabels(list(COLS))
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.cellChanged.connect(self._on_cell_changed)
        self._table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._show_points_context_menu)

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

        self._page = page
        self._update_wireframe_quality_limits()

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
                item = self._table.item(row, 0)
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

    def _schedule_preview(self):
        QtCore, _, _ = qt_modules()
        if QtCore is None:
            return
        QtCore.QTimer.singleShot(0, self._refresh_preview)

    def _refresh_preview(self):
        if not self._points:
            self._preview.cleanup()
            return
        self._preview.update(
            self._points,
            self._radius_widget.value(),
            self._wireframe.isChecked(),
            self._wireframe_quality_level(),
        )

    def _sync_table(self):
        self._update_wireframe_quality_limits()
        _, QtGui, QtWidgets = qt_modules()
        self._table.blockSignals(True)
        try:
            self._table.setRowCount(len(self._points))
            for row, pt in enumerate(self._points):
                values = (pt.name, pt.source, "%.3f" % pt.x, "%.3f" % pt.y, "%.3f" % pt.z)
                for col, text in enumerate(values):
                    item = self._table.item(row, col)
                    if item is None:
                        item = QtWidgets.QTableWidgetItem()
                        self._table.setItem(row, col, item)
                    item.setText(text)
                    if col == 0:
                        self._style_name_cell(item, pt)
                    else:
                        item.setBackground(QtGui.QBrush())
                        item.setForeground(QtGui.QBrush())
        finally:
            self._table.blockSignals(False)
        self._schedule_preview()

    def _on_cell_changed(self, row, col):
        if row < 0 or row >= len(self._points):
            return
        item = self._table.item(row, col)
        if item is None:
            return
        text = item.text()
        pt = self._points[row]
        try:
            if col == 0:
                pt = pt.with_name(text)
            elif col == 1:
                pt = pt.with_source(text)
            elif col == 2:
                pt = pt.with_xyz((float(text), pt.y, pt.z))
            elif col == 3:
                pt = pt.with_xyz((pt.x, float(text), pt.z))
            elif col == 4:
                pt = pt.with_xyz((pt.x, pt.y, float(text)))
            else:
                return
            self._points[row] = pt
        except ValueError:
            self._sync_table()
            return
        self._schedule_preview()

    def _add_camera_center(self):
        pt = camera_center_point(self.cmd, self._snap_atom.isChecked(), self._points)
        self._points.extend(self._stamp_new_points([pt]))
        self._sync_table()

    def _add_selection(self):
        new_pts = selection_points(self.cmd, self._points)
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
        radius = self._radius_widget.value()
        wireframe = self._wireframe.isChecked()
        quality = self._wireframe_quality_level()
        _commit_cgo(
            self.cmd,
            name,
            self._points,
            lambda: build_sphere_cgo_list(self._points, radius, wireframe, wireframe_quality=quality),
            lambda pt: build_sphere_cgo_list([pt], radius, wireframe, wireframe_quality=quality),
        )
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
