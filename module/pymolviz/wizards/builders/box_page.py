"""Box mesh builder page inside the objects menu."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from ..pick import DeferredCallback, qt_modules, qt_widget_alive
from ..tooltips import (
    EMPTY_PYMOL_SELECTION_MSG,
    HOOK_TO_SELECTION_TIP,
    SNAP_TO_ATOM_TIP,
    UPDATE_TO_CAMERA_TIP,
    UPDATE_TO_SELECTION_TIP,
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
from .colors import colors_for_new_points, pick_rgb, readable_text_color
from .points import (
    VisualPoint,
    camera_center_point,
    commit_point_anchors,
    export_points_to_selection,
    selection_points,
    update_points_from_camera,
    update_points_from_selection,
)
from .preview import (
    BoxPreview,
    build_box_cgo_collection,
    persist_live_preview,
    retarget_point_collection,
)
from .object_names import unused_object_name
from .zoom_selection import ZOOM_TO_SELECTION_TIP, points_from_rows, wire_zoom_to_selection
from ..widgets.sticky_add import StickyAddOverlay

COLS = point_columns()


class BoxBuilderPage:
    """Full editor for multi-point box CGOs."""

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
        self._preview = BoxPreview(cmd_)
        self._deferred = DeferredCallback()
        self._page = None
        self._table = None
        self._extent_x = None
        self._extent_y = None
        self._extent_z = None
        self._wireframe = None
        self._snap_atom = None
        self._hook_selection = None
        self._zoom_selection = None
        self._object_name = None
        self._create_btn = None
        self._table_filter = None
        self._editing_id = None
        self._loaded_name = None
        self._suspend_preview = False
        self._sticky_add = None
        self._build(parent)

    @property
    def widget(self):
        return self._page

    def cleanup_preview(self):
        self._deferred.cancel()
        self._preview.cleanup()

    def reset_for_create(self):
        self._editing_id = None
        self._loaded_name = None
        self._points = []
        if self._object_name is not None:
            self._object_name.setText(unused_object_name("pmv_boxes", self.cmd))
        if self._create_btn is not None:
            self._create_btn.setText("Create CGO")
        for widget, value in (
            (self._extent_x, 1.0),
            (self._extent_y, 1.0),
            (self._extent_z, 1.0),
        ):
            if widget is not None:
                widget.set_value(value)
        if self._wireframe is not None:
            self._wireframe.setChecked(False)
        if self._table is not None:
            self._sync_table()
        self._preview.cleanup()

    def load_object(self, obj):
        from .load_visual import box_options, points_from_mesh
        from ..catalog import display_name

        self._editing_id = str(obj.id)
        self._loaded_name = display_name(obj) or "pmv_boxes"
        self._points = points_from_mesh(obj)
        opts = box_options(obj)
        extent = opts["extent"]
        self._deferred.cancel()
        self._suspend_preview = True
        try:
            if self._object_name is not None:
                self._object_name.setText(display_name(obj) or "pmv_boxes")
            if self._create_btn is not None:
                self._create_btn.setText("Update CGO")
            if self._extent_x is not None:
                self._extent_x.set_value(extent[0])
            if self._extent_y is not None:
                self._extent_y.set_value(extent[1])
            if self._extent_z is not None:
                self._extent_z.set_value(extent[2])
            if self._wireframe is not None:
                self._wireframe.setChecked(opts["wireframe"])
            if self._table is not None:
                self._sync_table()
        finally:
            self._suspend_preview = False
        self._preview.adopt(obj)

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
        title = QtWidgets.QLabel("Boxes")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        header.addWidget(back)
        header.addWidget(title)
        header.addStretch(1)
        root.addLayout(header)

        opts = QtWidgets.QGroupBox("Options")
        opts_layout = QtWidgets.QFormLayout(opts)
        self._extent_x = LogSegmentRadiusWidget(initial=1.0)
        self._extent_y = LogSegmentRadiusWidget(initial=1.0)
        self._extent_z = LogSegmentRadiusWidget(initial=1.0)
        for widget in (self._extent_x, self._extent_y, self._extent_z):
            widget.connect_changed(self._schedule_preview)
        self._wireframe = QtWidgets.QCheckBox("Wireframe")
        self._wireframe.toggled.connect(lambda *_: self._schedule_preview())
        opts_layout.addRow("Extent X", self._extent_x.widget)
        opts_layout.addRow("Extent Y", self._extent_y.widget)
        opts_layout.addRow("Extent Z", self._extent_z.widget)
        opts_layout.addRow("", self._wireframe)
        root.addWidget(opts)

        pts_box = QtWidgets.QGroupBox("Points")
        pts_layout = QtWidgets.QVBoxLayout(pts_box)
        self._snap_atom = QtWidgets.QCheckBox("Snap to atom")
        self._snap_atom.setChecked(True)
        self._hook_selection = QtWidgets.QCheckBox("Anchor new points")
        self._hook_selection.setChecked(True)
        self._zoom_selection = QtWidgets.QCheckBox("Zoom to selection")
        export_sel = QtWidgets.QPushButton("Export to selection")
        export_sel.clicked.connect(self._export_selection)
        flags = QtWidgets.QHBoxLayout()
        flags.addWidget(self._snap_atom)
        flags.addWidget(self._hook_selection)
        flags.addWidget(self._zoom_selection)
        flags.addStretch(1)
        flags.addWidget(export_sel)
        pts_layout.addLayout(flags)

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
        self._sticky_add = StickyAddOverlay(
            pts_box,
            self._table,
            text="+ Add point",
            tooltip=(
                "Add selected atoms, or the camera-center marker if nothing is selected. "
                "With Snap to atom, uses a visible atom within 2 Å of that marker."
            ),
            on_click=self._add_point,
            count=lambda: len(self._points),
            context="BoxBuilderPage",
        )
        self._sticky_add.attach()

        root.addWidget(pts_box, stretch=1)

        actions = QtWidgets.QHBoxLayout()
        self._object_name = QtWidgets.QLineEdit()
        self._object_name.setPlaceholderText("Object name")
        self._object_name.setText(unused_object_name("pmv_boxes", self.cmd))
        self._create_btn = QtWidgets.QPushButton("Create CGO")
        self._create_btn.clicked.connect(self._create_cgo)
        export_btn = QtWidgets.QPushButton("Export CGO")
        export_btn.clicked.connect(self._export_cgo)
        actions.addWidget(self._object_name, stretch=2)
        actions.addWidget(self._create_btn)
        actions.addWidget(export_btn)
        root.addLayout(actions)

        apply_required_tooltips(
            [
                (back, "Return to the mesh type list."),
                (self._extent_x, "Full box width along X in Ångströms (centered on the point)."),
                (self._extent_y, "Full box width along Y in Ångströms (centered on the point)."),
                (self._extent_z, "Full box width along Z in Ångströms (centered on the point)."),
                (self._wireframe, "Draw each box as an edge cage instead of filled faces."),
                (self._snap_atom, SNAP_TO_ATOM_TIP),
                (self._hook_selection, HOOK_TO_SELECTION_TIP),
                (self._zoom_selection, ZOOM_TO_SELECTION_TIP),
                (
                    export_sel,
                    "Create a PyMOL selection covering the table points as pseudoatoms.",
                ),
                (self._object_name, "Name of the PyMOL CGO object created or exported."),
                (self._create_btn, "Commit the boxes to the session as a named CGO object."),
                (export_btn, "Write a Python script that rebuilds this CGO."),
            ],
            context="BoxBuilderPage",
        )

        self._page = page
        warn_missing_setting_tooltips(page, context="BoxBuilderPage")

    def _extent(self) -> Tuple[float, float, float]:
        return (
            self._extent_x.value(),
            self._extent_y.value(),
            self._extent_z.value(),
        )

    def _selected_rows(self) -> List[int]:
        return sorted({i.row() for i in self._table.selectedIndexes()})

    def _pick_box_color(self):
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
        color_act.triggered.connect(self._pick_box_color)
        cam_act = menu.addAction("Update to camera center")
        cam_act.setToolTip(UPDATE_TO_CAMERA_TIP)
        cam_act.setEnabled(bool(rows))
        cam_act.triggered.connect(self._update_selected_to_camera)
        sel_act = menu.addAction("Update to selection")
        sel_act.setToolTip(UPDATE_TO_SELECTION_TIP)
        sel_act.setEnabled(bool(rows))
        sel_act.triggered.connect(self._update_selected_to_selection)
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
        if self._suspend_preview:
            return
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
                self._extent(),
                self._wireframe.isChecked(),
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
        self._points[row] = pt.with_anchor_intent(checked)

    def _sync_table(self, preview=True):
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
        if getattr(self, "_sticky_add", None) is not None:
            self._sticky_add.sync()
        if preview:
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

    def _warn_empty_pymol_selection(self, title):
        _, _, QtWidgets = qt_modules()
        if QtWidgets is not None:
            QtWidgets.QMessageBox.information(
                self._page, title, EMPTY_PYMOL_SELECTION_MSG,
            )

    def _update_selected_to_camera(self):
        rows = self._selected_rows()
        if not rows:
            return
        self._points = update_points_from_camera(
            self.cmd,
            self._points,
            rows,
            self._snap_atom.isChecked(),
            hook_to_selection=self._hook_selection.isChecked(),
        )
        self._sync_table()

    def _update_selected_to_selection(self):
        rows = self._selected_rows()
        if not rows:
            return
        updated = update_points_from_selection(
            self.cmd,
            self._points,
            rows,
            hook_to_selection=self._hook_selection.isChecked(),
        )
        if updated is None:
            self._warn_empty_pymol_selection("Update to selection")
            return
        self._points = updated
        self._sync_table()

    def _add_point(self):
        new_pts = selection_points(
            self.cmd, self._points,
            hook_to_selection=self._hook_selection.isChecked(),
        )
        if new_pts:
            new_pts = self._stamp_new_points(new_pts)
            self._points.extend(new_pts)
            self._add_preview_points(new_pts)
            return
        self._add_camera_center()

    def _add_camera_center(self):
        pt = camera_center_point(
            self.cmd,
            self._snap_atom.isChecked(),
            self._points,
            hook_to_selection=self._hook_selection.isChecked(),
        )
        new_pts = self._stamp_new_points([pt])
        self._points.extend(new_pts)
        self._add_preview_points(new_pts)

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
                    EMPTY_PYMOL_SELECTION_MSG,
                )
            return
        new_pts = self._stamp_new_points(new_pts)
        self._points.extend(new_pts)
        self._add_preview_points(new_pts)

    def _add_preview_points(self, new_pts):
        added = self._preview.add_points(
            new_pts, self._extent(), self._wireframe.isChecked(),
        )
        self._sync_table(preview=not added)

    def _delete_selected(self):
        rows = sorted({i.row() for i in self._table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        self._preview.remove_rows(rows)
        for row in rows:
            if 0 <= row < len(self._points):
                del self._points[row]
        self._sync_table(preview=False)

    def _export_selection(self):
        export_points_to_selection(self.cmd, self._points)

    def _collection(self, name: str):
        return build_box_cgo_collection(
            commit_point_anchors(self._points),
            self._extent(),
            self._wireframe.isChecked(),
            name=name,
        )

    def _create_cgo(self):
        if not self._points:
            return
        typed = self._object_name.text().strip() or "pmv_boxes"
        name = unused_object_name(typed, self.cmd, keep=self._loaded_name)
        persist_live_preview(
            self.cmd,
            self._preview,
            name,
            obj_id=self._editing_id,
            retarget=lambda coll: retarget_point_collection(
                coll, commit_point_anchors(self._points),
            ),
            fallback=lambda: self._collection(name),
        )
        if self._on_create is not None:
            self._on_create()

    def _export_cgo(self):
        if not self._points:
            return
        QtCore, _, QtWidgets = qt_modules()
        name = self._object_name.text().strip() or "pmv_boxes"
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
