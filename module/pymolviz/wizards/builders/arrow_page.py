"""Arrow mesh builder: directed pairs with dash / margin / end options."""

from __future__ import annotations

from typing import Callable, List, Optional

from ..pick import DeferredCallback, qt_modules, qt_widget_alive
from ..tooltips import (
    HOOK_TO_SELECTION_TIP,
    SNAP_TO_ATOM_TIP,
    apply_required_tooltips,
    warn_missing_setting_tooltips,
)
from .colors import colors_for_new_points, pick_rgb, readable_text_color
from .line_style import ARROW_QUALITY_SEGMENTS, LineOptionsWidget
from .pairs import VisualPair, flatten_pair_points, take_single_selection_point
from .points import (
    VisualPoint,
    camera_center_point,
    export_points_to_selection,
)
from .preview import ArrowPreview, build_arrow_collection, persist_collection

COLS = ("Start", "Start src", "End", "End src", "X0", "Y0", "Z0", "X1", "Y1", "Z1")
QUALITY_HINTS = {
    0: "2D lines",
    1: "cylinder / cone · 6 sides",
    2: "cylinder / cone · 8 sides",
    3: "cylinder / cone · 10 sides",
    4: "cylinder / cone · 14 sides",
    5: "cylinder / cone · 18 sides",
}


class ArrowBuilderPage:
    """Editor for multi-pair arrow CGOs."""

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
        self._pairs: List[VisualPair] = []
        self._preview = ArrowPreview(cmd_)
        self._phase = None
        self._pending_first = None
        self._ignore_xyz = None
        self._poll_timer = None
        self._deferred = DeferredCallback()
        self._page = None
        self._table = None
        self._quality = None
        self._quality_hint = None
        self._line_options = None
        self._snap_atom = None
        self._hook_selection = None
        self._status = None
        self._select_pair_btn = None
        self._abort_btn = None
        self._add_cam_btn = None
        self._object_name = None
        self._table_filter = None
        self._build(parent)

    @property
    def widget(self):
        return self._page

    def cleanup_preview(self):
        self._deferred.cancel()
        self._stop_poll_timer()
        self._phase = None
        self._pending_first = None
        self._ignore_xyz = None
        self._preview.cleanup()

    def _existing_points(self) -> List[VisualPoint]:
        pts = flatten_pair_points(self._pairs)
        if self._pending_first is not None:
            pts.append(self._pending_first)
        return pts

    def _build(self, parent):
        QtCore, _, QtWidgets = qt_modules()
        if QtWidgets is None:
            raise RuntimeError("PyMOL Qt UI required")

        page = QtWidgets.QWidget(parent)
        root = QtWidgets.QVBoxLayout(page)

        header = QtWidgets.QHBoxLayout()
        back = QtWidgets.QPushButton("← Back")
        back.setFlat(True)
        back.clicked.connect(self._go_back)
        title = QtWidgets.QLabel("Arrows")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        header.addWidget(back)
        header.addWidget(title)
        header.addStretch(1)
        root.addLayout(header)

        opts = QtWidgets.QGroupBox("Options")
        opts_layout = QtWidgets.QFormLayout(opts)
        self._quality = QtWidgets.QSpinBox()
        self._quality.setRange(0, 5)
        self._quality.setValue(3)
        self._quality.setToolTip("0 = 2D lines; 1–5 = cylinder / cone meshes with more vertices")
        self._quality.valueChanged.connect(self._on_quality_changed)
        self._quality_hint = QtWidgets.QLabel(QUALITY_HINTS[3])
        self._quality_hint.setStyleSheet("color: gray;")
        quality_row = QtWidgets.QHBoxLayout()
        quality_row.addWidget(self._quality)
        quality_row.addWidget(self._quality_hint, stretch=1)
        opts_layout.addRow("Quality", quality_row)
        root.addWidget(opts)

        self._line_options = LineOptionsWidget(page, on_change=self._schedule_preview)
        root.addWidget(self._line_options.widget)

        pts_box = QtWidgets.QGroupBox("Pairs")
        pts_layout = QtWidgets.QVBoxLayout(pts_box)
        self._snap_atom = QtWidgets.QCheckBox("Snap to atom")
        self._hook_selection = QtWidgets.QCheckBox("Hook to selection")
        self._hook_selection.setChecked(True)
        btn_row = QtWidgets.QHBoxLayout()
        self._select_pair_btn = QtWidgets.QPushButton("Select Pair")
        self._select_pair_btn.clicked.connect(self._on_select_pair)
        mcs_btn = QtWidgets.QPushButton("Pair MCS")
        mcs_btn.clicked.connect(self._on_pair_mcs)
        self._add_cam_btn = QtWidgets.QPushButton("Add camera")
        self._add_cam_btn.clicked.connect(self._on_add_camera)
        self._abort_btn = QtWidgets.QPushButton("Abort")
        self._abort_btn.clicked.connect(lambda: self._abort_pair(silent=False))
        self._abort_btn.setVisible(False)
        export_sel = QtWidgets.QPushButton("Export points to selection")
        export_sel.clicked.connect(self._export_selection)
        btn_row.addWidget(self._select_pair_btn)
        btn_row.addWidget(mcs_btn)
        btn_row.addWidget(self._add_cam_btn)
        btn_row.addWidget(self._abort_btn)
        btn_row.addWidget(export_sel)
        self._status = QtWidgets.QLabel("Add a pair with Select Pair or Add camera.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: gray;")
        flags = QtWidgets.QHBoxLayout()
        flags.addWidget(self._snap_atom)
        flags.addWidget(self._hook_selection)
        flags.addStretch(1)
        pts_layout.addLayout(flags)
        pts_layout.addLayout(btn_row)
        pts_layout.addWidget(self._status)

        self._table = QtWidgets.QTableWidget(0, len(COLS))
        self._table.setHorizontalHeaderLabels(list(COLS))
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.cellChanged.connect(self._on_cell_changed)
        self._table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._show_pairs_context_menu)

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
        self._object_name.setText("pmv_arrows")
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
                (
                    self._quality,
                    "0 = 2D lines; 1–5 = cylinder / cone meshes with more vertices",
                    "Quality",
                ),
                (self._snap_atom, SNAP_TO_ATOM_TIP),
                (self._hook_selection, HOOK_TO_SELECTION_TIP),
                (
                    self._select_pair_btn,
                    "Pick start and end from the current PyMOL selection (one atom at a time).",
                ),
                (mcs_btn, "Maximum common substructure pairing (not implemented)."),
                (
                    self._add_cam_btn,
                    "Use the camera/screen center as the next pair endpoint. "
                    "With Snap to atom, uses the nearest atom within 1 Å.",
                ),
                (self._abort_btn, "Cancel the in-progress pair and keep existing rows."),
                (
                    export_sel,
                    "Create a PyMOL selection covering the pair endpoints as pseudoatoms.",
                ),
                (self._object_name, "Name of the PyMOL CGO object created or exported."),
                (create_btn, "Commit the arrows to the session as a named CGO object."),
                (export_btn, "Write a Python script that rebuilds this CGO."),
            ],
            context="ArrowBuilderPage",
        )

        self._poll_timer = QtCore.QTimer(page)
        self._poll_timer.setInterval(250)
        self._poll_timer.timeout.connect(self._poll_selection)
        self._page = page
        warn_missing_setting_tooltips(page, context="ArrowBuilderPage")

    def _go_back(self):
        self._deferred.cancel()
        self._stop_poll_timer()
        self._abort_pair(silent=True)
        self._preview.cleanup()
        self._on_back()

    def _on_quality_changed(self, *_args):
        quality = int(self._quality.value())
        self._quality_hint.setText(QUALITY_HINTS.get(quality, ""))
        self._quality.setToolTip(
            "0 = 2D lines; 1–5 = cylinder / cone (%d sides)."
            % ARROW_QUALITY_SEGMENTS.get(quality, 0)
            if quality
            else "2D CGO lines. Dash / margin / ends still apply."
        )
        self._schedule_preview()

    def _style(self):
        return self._line_options.style()

    def _not_implemented(self, title, text):
        _, _, QtWidgets = qt_modules()
        if QtWidgets is not None:
            QtWidgets.QMessageBox.information(self._page, title, text)
        else:
            raise NotImplementedError(text)

    def _on_pair_mcs(self):
        self._not_implemented("Pair MCS", "Pair MCS is not implemented yet.")

    def _same_as_ignored(self, point: VisualPoint) -> bool:
        if self._ignore_xyz is None:
            return False
        dx = point.x - self._ignore_xyz[0]
        dy = point.y - self._ignore_xyz[1]
        dz = point.z - self._ignore_xyz[2]
        return (dx * dx + dy * dy + dz * dz) < 1e-6

    def _set_phase(self, phase, first=None, hint=None, schedule_preview=True):
        self._phase = phase
        self._pending_first = first
        if phase is None:
            self._ignore_xyz = None
        if qt_widget_alive(self._page):
            waiting = phase in ("first", "second")
            try:
                self._abort_btn.setVisible(waiting)
                self._select_pair_btn.setText("Select Pair" if not waiting else "Use selection")
                if hint is not None:
                    self._status.setText(hint)
            except RuntimeError:
                pass
        waiting = phase in ("first", "second")
        if waiting:
            self._start_poll_timer()
        else:
            self._stop_poll_timer()
        if schedule_preview:
            self._schedule_preview()

    def _abort_pair(self, silent=False):
        self._set_phase(
            None,
            None,
            None if silent else "Pair selection aborted.",
            schedule_preview=not silent,
        )
        if not silent and qt_widget_alive(self._status):
            try:
                self._status.setText("Add a pair with Select Pair or Add camera.")
            except RuntimeError:
                pass

    def _selection_point(self, interactive_only=False):
        return take_single_selection_point(
            self.cmd,
            self._existing_points(),
            interactive_only=interactive_only,
            hook_to_selection=self._hook_selection.isChecked(),
        )

    def _on_select_pair(self):
        if self._phase in ("first", "second"):
            self._use_current_selection()
            return
        point, status = self._selection_point()
        if status == "multiple":
            self._not_implemented(
                "Not Implemented",
                "Selecting multiple atoms as a pair is not implemented yet.",
            )
            return
        if status == "one":
            self._accept_first(point)
            return
        self._set_phase(
            "first",
            None,
            "Select the first point: pick one atom, click Use selection, or Add camera. Abort to cancel.",
        )

    def _use_current_selection(self):
        if self._phase not in ("first", "second"):
            return
        point, status = self._selection_point()
        if status == "multiple":
            self._not_implemented(
                "Not Implemented",
                "Selecting multiple atoms as a pair is not implemented yet.",
            )
            return
        if status == "empty":
            self._status.setText("Nothing selected. Pick one atom or use Add camera.")
            return
        if self._same_as_ignored(point):
            self._status.setText("That is still the first point. Pick a different atom or Add camera.")
            return
        self._accept_point(point)

    def _start_poll_timer(self):
        if not qt_widget_alive(self._poll_timer):
            return
        try:
            self._poll_timer.start()
        except RuntimeError:
            pass

    def _stop_poll_timer(self):
        if not qt_widget_alive(self._poll_timer):
            return
        try:
            self._poll_timer.stop()
        except RuntimeError:
            pass

    def _poll_selection(self):
        if self._phase not in ("first", "second"):
            return
        if not qt_widget_alive(self._page):
            self._stop_poll_timer()
            return
        point, status = self._selection_point(interactive_only=True)
        if status == "multiple":
            self._stop_poll_timer()
            self._not_implemented(
                "Not Implemented",
                "Selecting multiple atoms as a pair is not implemented yet.",
            )
            self._start_poll_timer()
            return
        if status == "one" and not self._same_as_ignored(point):
            self._accept_point(point)

    def _clear_pymol_selection(self):
        try:
            self.cmd.select("sele", "none")
        except Exception:
            pass
        try:
            self.cmd.unpick()
        except Exception:
            pass

    def _accept_point(self, point: VisualPoint):
        if self._phase == "first":
            self._accept_first(point)
        elif self._phase == "second":
            self._accept_second(point)

    def _accept_first(self, point: VisualPoint):
        self._clear_pymol_selection()
        self._ignore_xyz = point.xyz()
        self._set_phase(
            "second",
            point,
            "First point: %s. Select the second point, Add camera, or Abort." % point.name,
        )

    def _stamp_pair(self, start: VisualPoint, end: VisualPoint) -> VisualPair:
        palette = colors_for_new_points(1, start_index=len(self._pairs))
        color = palette[0]
        return VisualPair(start.with_color(color), end.with_color(color))

    def _accept_second(self, point: VisualPoint):
        first = self._pending_first
        if first is None:
            self._accept_first(point)
            return
        self._clear_pymol_selection()
        self._pairs.append(self._stamp_pair(first, point))
        self._set_phase(None, None, "Pair added: %s → %s" % (first.name, point.name))
        self._sync_table()

    def _on_add_camera(self):
        pt = camera_center_point(
            self.cmd,
            self._snap_atom.isChecked(),
            self._existing_points(),
            hook_to_selection=self._hook_selection.isChecked(),
        )
        if self._phase is None:
            self._accept_first(pt)
            return
        self._accept_point(pt)

    def _selected_rows(self) -> List[int]:
        return sorted({i.row() for i in self._table.selectedIndexes()})

    def _pick_pair_color(self):
        rows = self._selected_rows()
        if not rows:
            return
        initial = self._pairs[rows[0]].rgba()
        original = {row: self._pairs[row].rgba() for row in rows}

        def on_preview(rgba):
            for row in rows:
                if 0 <= row < len(self._pairs):
                    self._pairs[row] = self._pairs[row].with_color(rgba)
            self._refresh_preview()
            for row in rows:
                item = self._table.item(row, 0)
                if item is not None:
                    self._style_name_cell(item, self._pairs[row])

        def on_done(rgba):
            if rgba is None:
                for row, color in original.items():
                    self._pairs[row] = self._pairs[row].with_color(color)
            self._sync_table()

        pick_rgb(self._page, initial, on_change=on_preview, on_done=on_done)

    def _show_pairs_context_menu(self, pos):
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
        color_act.triggered.connect(self._pick_pair_color)
        del_act = menu.addAction("Delete selected")
        del_act.setEnabled(bool(rows))
        del_act.triggered.connect(self._delete_selected)
        menu.exec_(self._table.viewport().mapToGlobal(pos))

    def _style_name_cell(self, item, pair):
        _, QtGui, _ = qt_modules()
        text_rgb = readable_text_color(pair.color)
        item.setBackground(QtGui.QColor(
            int(pair.color[0] * 255), int(pair.color[1] * 255), int(pair.color[2] * 255),
        ))
        item.setForeground(QtGui.QColor(
            int(text_rgb[0] * 255), int(text_rgb[1] * 255), int(text_rgb[2] * 255),
        ))

    def _schedule_preview(self):
        self._deferred.schedule(self._refresh_preview, page=self._page)

    def _refresh_preview(self):
        if not qt_widget_alive(self._page):
            return
        try:
            self._preview.update(
                self._pairs,
                self._quality.value(),
                self._style(),
                pending=self._pending_first,
            )
        except RuntimeError:
            pass

    def _sync_table(self):
        _, QtGui, QtWidgets = qt_modules()
        self._table.blockSignals(True)
        try:
            self._table.setRowCount(len(self._pairs))
            for row, pair in enumerate(self._pairs):
                values = (
                    pair.start.name,
                    pair.start.source,
                    pair.end.name,
                    pair.end.source,
                    "%.3f" % pair.start.x, "%.3f" % pair.start.y, "%.3f" % pair.start.z,
                    "%.3f" % pair.end.x, "%.3f" % pair.end.y, "%.3f" % pair.end.z,
                )
                for col, text in enumerate(values):
                    item = self._table.item(row, col)
                    if item is None:
                        item = QtWidgets.QTableWidgetItem()
                        self._table.setItem(row, col, item)
                    item.setText(text)
                    if col in (0, 2):
                        self._style_name_cell(item, pair)
                    else:
                        item.setBackground(QtGui.QBrush())
                        item.setForeground(QtGui.QBrush())
        finally:
            self._table.blockSignals(False)
        self._schedule_preview()

    def _on_cell_changed(self, row, col):
        if row < 0 or row >= len(self._pairs):
            return
        item = self._table.item(row, col)
        if item is None:
            return
        text = item.text()
        pair = self._pairs[row]
        start, end = pair.start, pair.end
        try:
            if col == 0:
                start = start.with_name(text)
            elif col == 1:
                start = start.with_source(text)
            elif col == 2:
                end = end.with_name(text)
            elif col == 3:
                end = end.with_source(text)
            elif col == 4:
                start = start.with_xyz((float(text), start.y, start.z))
            elif col == 5:
                start = start.with_xyz((start.x, float(text), start.z))
            elif col == 6:
                start = start.with_xyz((start.x, start.y, float(text)))
            elif col == 7:
                end = end.with_xyz((float(text), end.y, end.z))
            elif col == 8:
                end = end.with_xyz((end.x, float(text), end.z))
            elif col == 9:
                end = end.with_xyz((end.x, end.y, float(text)))
            else:
                return
            self._pairs[row] = VisualPair(start, end)
        except ValueError:
            self._sync_table()
            return
        self._schedule_preview()

    def _delete_selected(self):
        rows = sorted(self._selected_rows(), reverse=True)
        for row in rows:
            if 0 <= row < len(self._pairs):
                del self._pairs[row]
        self._sync_table()

    def _export_selection(self):
        export_points_to_selection(self.cmd, flatten_pair_points(self._pairs))

    def _create_cgo(self):
        if not self._pairs:
            return
        name = self._object_name.text().strip() or "pmv_arrows"
        persist_collection(
            self.cmd,
            build_arrow_collection(
                self._pairs, int(self._quality.value()), self._style(), name,
            ),
        )
        self._preview.cleanup()
        if self._on_create is not None:
            self._on_create()

    def _export_cgo(self):
        if not self._pairs:
            return
        _, _, QtWidgets = qt_modules()
        name = self._object_name.text().strip() or "pmv_arrows"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self._page, "Export CGO script", "%s.py" % name, "Python (*.py)",
        )
        if not path:
            return
        build_arrow_collection(
            self._pairs, int(self._quality.value()), self._style(), name,
        ).write(path)
