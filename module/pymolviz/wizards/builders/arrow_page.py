"""Arrow mesh builder: a compact pair editor with two-step picking."""

from __future__ import annotations

from typing import Callable, List, Optional

from ...util.line_style import (
    ARROW_QUALITY_SEGMENTS,
    LineStyle,
    MAX_ARROW_MARGIN,
    max_margin_for_length,
)
from ..pick import DeferredCallback, qt_modules, qt_widget_alive
from ..tooltips import (
    HOOK_TO_SELECTION_TIP,
    SNAP_TO_ATOM_TIP,
    apply_required_tooltips,
    warn_missing_setting_tooltips,
)
from .arrow_list import ArrowPairEditor
from .arrow_type import ArrowTypeControl
from .colors import colors_for_new_points, pick_rgb
from .object_names import unused_object_name
from .pairs import (
    VisualPair,
    commit_pair_anchors,
    complete_pairs,
    endpoint_label,
    pair_index,
    take_selection_endpoints,
    take_single_selection_point,
)
from .points import VisualPoint, camera_center_point
from .preview import (
    ArrowPreview,
    build_arrow_collection,
    persist_live_preview,
    retarget_arrow_collection,
)
from .zoom_selection import ZOOM_TO_SELECTION_TIP, focus_visual_point, zoom_to_visual_points

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
        self._pick_role = None
        self._pick_pair_id = None
        self._selected_id = None
        self._ignore_xyz = None
        self._poll_timer = None
        self._deferred = DeferredCallback()
        self._page = None
        self._list = None
        self._quality = None
        self._quality_hint = None
        self._snap_atom = None
        self._hook_selection = None
        self._zoom_selection = None
        self._status = None
        self._object_name = None
        self._create_btn = None
        self._key_filter = None
        self._editing_id = None
        self._loaded_name = None
        self._line_style = LineStyle()
        self._margin = 0.0
        self._style_control = None
        self._suspend_preview = False
        self._build(parent)

    @property
    def widget(self):
        return self._page

    def cleanup_preview(self):
        self._deferred.cancel()
        self._stop_poll_timer()
        self._pick_role = None
        self._pick_pair_id = None
        self._ignore_xyz = None
        self._preview.cleanup()

    def reset_for_create(self):
        self._editing_id = None
        self._loaded_name = None
        self._pairs = []
        self._selected_id = None
        self._clear_pick()
        if self._object_name is not None:
            self._object_name.setText(unused_object_name("pmv_arrows", self.cmd))
        if self._create_btn is not None:
            self._create_btn.setText("Create CGO")
        if self._quality is not None:
            self._quality.setValue(3)
        self._line_style = LineStyle()
        self._margin = 0.0
        if self._style_control is not None:
            self._style_control.set_max_margin(MAX_ARROW_MARGIN)
            self._style_control.set_style(self._line_style)
        self._sync_list(preview=False)
        self._preview.cleanup()

    def load_object(self, obj):
        from .load_visual import arrow_options, pairs_from_mesh
        from ..catalog import display_name

        self._editing_id = str(obj.id)
        self._loaded_name = display_name(obj) or "pmv_arrows"
        self._pairs = pairs_from_mesh(obj)
        self._selected_id = self._pairs[0].pair_id if self._pairs else None
        opts = arrow_options(obj)
        self._deferred.cancel()
        self._clear_pick()
        self._suspend_preview = True
        try:
            if self._object_name is not None:
                self._object_name.setText(display_name(obj) or "pmv_arrows")
            if self._create_btn is not None:
                self._create_btn.setText("Update CGO")
            if self._quality is not None:
                self._quality.setValue(int(opts["quality"]))
            if opts.get("line_style") is not None:
                self._line_style = opts["line_style"].copy() if hasattr(opts["line_style"], "copy") else opts["line_style"]
                self._margin = float(getattr(self._line_style, "margin", 0.0) or 0.0)
                if self._style_control is not None:
                    self._style_control.set_style(self._line_style)
            self._sync_list(preview=False)
        finally:
            self._suspend_preview = False
        self._preview.adopt(obj)

    def _existing_points(self) -> List[VisualPoint]:
        pts = []
        for pair in self._pairs:
            pts.append(pair.start)
            if pair.end is not None:
                pts.append(pair.end)
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

        flags = QtWidgets.QHBoxLayout()
        self._snap_atom = QtWidgets.QCheckBox("Snap to atom")
        self._snap_atom.setChecked(True)
        self._hook_selection = QtWidgets.QCheckBox("Anchor new points")
        self._hook_selection.setChecked(True)
        self._zoom_selection = QtWidgets.QCheckBox("Zoom to selection")
        flags.addWidget(self._snap_atom)
        flags.addWidget(self._hook_selection)
        flags.addWidget(self._zoom_selection)
        flags.addStretch(1)
        root.addLayout(flags)

        quality_row = QtWidgets.QHBoxLayout()
        quality_row.addWidget(QtWidgets.QLabel("Quality"))
        self._quality = QtWidgets.QSpinBox()
        self._quality.setRange(0, 5)
        self._quality.setValue(3)
        self._quality.valueChanged.connect(self._on_quality_changed)
        self._quality_hint = QtWidgets.QLabel(QUALITY_HINTS[3])
        self._quality_hint.setStyleSheet("color: gray;")
        quality_row.addWidget(self._quality)
        quality_row.addWidget(self._quality_hint, stretch=1)
        root.addLayout(quality_row)

        style_row = QtWidgets.QHBoxLayout()
        style_row.addWidget(QtWidgets.QLabel("Style"))
        self._style_control = ArrowTypeControl(
            page,
            style=self._line_style,
            max_margin=MAX_ARROW_MARGIN,
            on_change=self.set_line_style,
        )
        self._style_control.widget.setMinimumWidth(160)
        style_row.addWidget(self._style_control.widget, stretch=1)
        root.addLayout(style_row)

        self._list = ArrowPairEditor(page, self)
        root.addWidget(self._list.widget, stretch=1)

        self._status = QtWidgets.QLabel("Add an arrow, or select two atoms first.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: gray;")
        root.addWidget(self._status)

        actions = QtWidgets.QHBoxLayout()
        self._object_name = QtWidgets.QLineEdit()
        self._object_name.setPlaceholderText("Object name")
        self._object_name.setText(unused_object_name("pmv_arrows", self.cmd))
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
                (
                    self._quality,
                    "0 = 2D lines; 1–5 = cylinder / cone meshes with more vertices",
                    "Quality",
                ),
                (self._snap_atom, SNAP_TO_ATOM_TIP),
                (self._hook_selection, HOOK_TO_SELECTION_TIP),
                (self._zoom_selection, ZOOM_TO_SELECTION_TIP),
                (self._object_name, "Name of the PyMOL CGO object created or exported."),
                (self._create_btn, "Commit the arrows to the session as a named CGO object."),
                (export_btn, "Write a Python script that rebuilds this CGO."),
            ],
            context="ArrowBuilderPage",
        )

        self._poll_timer = QtCore.QTimer(page)
        self._poll_timer.setInterval(250)
        self._poll_timer.timeout.connect(self._poll_selection)

        class _KeyFilter(QtCore.QObject):
            def __init__(self, owner):
                QtCore.QObject.__init__(self)
                self._owner = owner

            def eventFilter(self, obj, event):
                if event.type() != QtCore.QEvent.KeyPress:
                    return False
                focus = QtWidgets.QApplication.focusWidget()
                typing = isinstance(
                    focus, (QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox)
                )
                if event.key() == QtCore.Qt.Key_Escape:
                    self._owner._abort_pick()
                    return True
                if event.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
                    if typing:
                        return False
                    self._owner._delete_selected()
                    return True
                return False

        self._key_filter = _KeyFilter(self)
        page.installEventFilter(self._key_filter)
        self._page = page
        warn_missing_setting_tooltips(page, context="ArrowBuilderPage")

    def _go_back(self):
        self._deferred.cancel()
        self._stop_poll_timer()
        self._abort_pick(silent=True)
        self._deferred.cancel()
        self._preview.cleanup()
        self._on_back()

    def _on_quality_changed(self, *_args):
        quality = int(self._quality.value())
        self._quality_hint.setText(QUALITY_HINTS.get(quality, ""))
        self._quality.setToolTip(
            "0 = 2D lines; 1–5 = cylinder / cone (%d sides)."
            % ARROW_QUALITY_SEGMENTS.get(quality, 0)
            if quality
            else "2D CGO lines."
        )
        self._schedule_preview()

    def _style(self):
        style = self._line_style.copy()
        style.margin = float(self._margin)
        return style

    def set_line_style(self, style: LineStyle):
        if style is None:
            return
        self._line_style = style.copy()
        self._margin = float(self._line_style.margin)
        self._schedule_preview()

    def _fit_max_margin(self) -> float:
        cap = MAX_ARROW_MARGIN
        found = 0
        n_heads = self._line_style.n_arrow_heads() if hasattr(self._line_style, "n_arrow_heads") else 1
        double = n_heads >= 2
        use_head = n_heads > 0
        for pair in complete_pairs(self._pairs):
            try:
                start = pair.start.xyz()
                end = pair.end.xyz()
            except Exception:
                continue
            dx = start[0] - end[0]
            dy = start[1] - end[1]
            dz = start[2] - end[2]
            length = (dx * dx + dy * dy + dz * dz) ** 0.5
            head = float(pair.head) if use_head else 0.0
            cap = min(cap, max_margin_for_length(length, head, double))
            found += 1
        return cap if found else MAX_ARROW_MARGIN

    def _refresh_margin_limit(self):
        cap = self._fit_max_margin()
        if self._margin > cap:
            self._margin = cap
            self._line_style = self._line_style.updated(margin=self._margin)
        if self._style_control is not None:
            self._style_control.set_max_margin(cap)
            self._style_control.set_margin(self._margin)

    def _pairs_for_draw(self):
        style = self._style()
        return [pair.with_style(style.copy()) for pair in self._pairs]

    def _resolve_context(self):
        try:
            from ...runtime.context import ResolveContext

            return ResolveContext(self.cmd)
        except Exception:
            return None

    def _set_status(self, text):
        if qt_widget_alive(self._status):
            try:
                self._status.setText(text)
            except RuntimeError:
                pass

    def _same_as_ignored(self, point: VisualPoint) -> bool:
        if self._ignore_xyz is None or point is None:
            return False
        dx = point.x - self._ignore_xyz[0]
        dy = point.y - self._ignore_xyz[1]
        dz = point.z - self._ignore_xyz[2]
        return (dx * dx + dy * dy + dz * dz) < 1e-6

    def _clear_pick(self):
        self._pick_role = None
        self._pick_pair_id = None
        self._ignore_xyz = None
        self._stop_poll_timer()

    def _set_pick(self, role, pair_id, hint):
        self._pick_role = role
        self._pick_pair_id = pair_id
        if hint is not None:
            self._set_status(hint)
        if role is None:
            self._stop_poll_timer()
        else:
            self._start_poll_timer()

    def _abort_pick(self, silent=False):
        if self._pick_role is None:
            return
        pid = self._pick_pair_id
        if pid:
            idx = pair_index(self._pairs, pid)
            if idx >= 0 and not self._pairs[idx].is_complete():
                del self._pairs[idx]
                if self._selected_id == pid:
                    self._selected_id = None
        self._clear_pick()
        if silent:
            return
        if self._pairs:
            self._set_status("Pick cancelled.")
        else:
            self._set_status("Add an arrow, or select two atoms first.")
        self._sync_list()

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

    def _hook(self) -> bool:
        return bool(self._hook_selection.isChecked())

    def _clear_pymol_selection(self):
        try:
            self.cmd.select("sele", "none")
        except Exception:
            pass
        try:
            self.cmd.unpick()
        except Exception:
            pass

    def _stamp_pair(self, start: VisualPoint, end: Optional[VisualPoint] = None) -> VisualPair:
        palette = colors_for_new_points(1, start_index=len(complete_pairs(self._pairs)))
        color = palette[0]
        start = start.with_color(color)
        if end is not None:
            end = end.with_color(color)
        return VisualPair(start, end, style=self._line_style.copy())

    def add_arrow(self):
        if self._pick_role is not None:
            self._abort_pick()
            return
        start, end, status = take_selection_endpoints(
            self.cmd,
            self._existing_points(),
            hook_to_selection=self._hook(),
        )
        if status == "pair":
            pair = self._stamp_pair(start, end)
            self._pairs.append(pair)
            self._selected_id = pair.pair_id
            self._clear_pymol_selection()
            self._set_status("Arrow added.")
            self._sync_list()
            return
        if status == "one":
            self._begin_incomplete(start)
            return
        if status == "multiple":
            self._set_status("Select two atoms to create an arrow, or pick them one at a time.")
        self._set_pick("start", None, "Pick start")
        self._sync_list()

    def _begin_incomplete(self, start: VisualPoint):
        pair = self._stamp_pair(start, None)
        self._pairs.append(pair)
        self._selected_id = pair.pair_id
        self._clear_pymol_selection()
        self._ignore_xyz = start.xyz()
        self._set_pick("end", pair.pair_id, "%s  →  [pick end…]" % endpoint_label(pair.start))
        self._sync_list()

    def camera_pick(self):
        pt = camera_center_point(
            self.cmd,
            self._snap_atom.isChecked(),
            self._existing_points(),
            hook_to_selection=self._hook(),
        )
        if self._pick_role is None:
            return
        self._accept_point(pt)

    def _poll_selection(self):
        if self._pick_role is None:
            return
        if not qt_widget_alive(self._page):
            self._stop_poll_timer()
            return
        if self._pick_role == "start" and self._pick_pair_id is None:
            start, end, status = take_selection_endpoints(
                self.cmd,
                self._existing_points(),
                interactive_only=True,
                hook_to_selection=self._hook(),
            )
            if status == "pair" and start is not None and end is not None:
                if not self._same_as_ignored(start):
                    self._finish_new_pair(start, end)
                return
            if status == "one" and start is not None and not self._same_as_ignored(start):
                self._accept_point(start)
            return
        point, status = take_single_selection_point(
            self.cmd,
            self._existing_points(),
            interactive_only=True,
            hook_to_selection=self._hook(),
        )
        if status == "one" and not self._same_as_ignored(point):
            self._accept_point(point)

    def _finish_new_pair(self, start: VisualPoint, end: VisualPoint):
        pair = self._stamp_pair(start, end)
        self._pairs.append(pair)
        self._selected_id = pair.pair_id
        self._clear_pymol_selection()
        self._clear_pick()
        self._set_status("Arrow added.")
        self._sync_list()

    def _accept_point(self, point: VisualPoint):
        if self._pick_role is None:
            return
        if self._pick_role == "start" and self._pick_pair_id is None:
            self._begin_incomplete(point)
            return
        idx = pair_index(self._pairs, self._pick_pair_id)
        if idx < 0:
            self._clear_pick()
            self._sync_list()
            return
        pair = self._pairs[idx]
        colored = point.with_color(pair.color)
        if self._pick_role == "start":
            self._pairs[idx] = pair.with_start(colored)
        else:
            self._pairs[idx] = pair.with_end(colored)
        self._clear_pymol_selection()
        self._clear_pick()
        self._set_status("Arrow updated.")
        self._sync_list()

    def select_arrow(self, pair_id, zoom=True):
        self._selected_id = pair_id
        self._sync_list()
        if not zoom or not self._zoom_selection.isChecked():
            return
        idx = pair_index(self._pairs, pair_id)
        if idx < 0:
            return
        pair = self._pairs[idx]
        pts = [pair.start]
        if pair.end is not None:
            pts.append(pair.end)
        zoom_to_visual_points(self.cmd, pts)

    def delete_arrow(self, pair_id):
        idx = pair_index(self._pairs, pair_id)
        if idx < 0:
            return
        if self._pick_pair_id == pair_id:
            self._clear_pick()
        del self._pairs[idx]
        if self._selected_id == pair_id:
            if self._pairs:
                self._selected_id = self._pairs[min(idx, len(self._pairs) - 1)].pair_id
            else:
                self._selected_id = None
        self._sync_list()

    def _delete_selected(self):
        if self._selected_id is None:
            return
        self.delete_arrow(self._selected_id)

    def pick_endpoint(self, pair_id, role):
        idx = pair_index(self._pairs, pair_id)
        if idx < 0:
            return
        pair = self._pairs[idx]
        current = pair.start if role == "start" else pair.end
        self._selected_id = pair_id
        self._clear_pymol_selection()
        self._ignore_xyz = current.xyz() if current is not None else None
        if role == "start":
            hint = "Replace start endpoint…"
        elif current is None:
            hint = "Pick end"
        else:
            hint = "Replace end endpoint…"
        self._set_pick(role, pair_id, hint)
        self._sync_list()

    def focus_endpoint(self, pair_id, role):
        idx = pair_index(self._pairs, pair_id)
        if idx < 0:
            return
        pair = self._pairs[idx]
        pt = pair.start if role == "start" else pair.end
        self.select_arrow(pair_id, zoom=False)
        if pt is not None:
            focus_visual_point(self.cmd, pt)

    def camera_endpoint(self, pair_id, role):
        idx = pair_index(self._pairs, pair_id)
        if idx < 0:
            return
        pair = self._pairs[idx]
        pt = camera_center_point(
            self.cmd,
            self._snap_atom.isChecked(),
            self._existing_points(),
            hook_to_selection=self._hook(),
        ).with_color(pair.color)
        if role == "start":
            self._pairs[idx] = pair.with_start(pt)
        else:
            self._pairs[idx] = pair.with_end(pt)
        if self._pick_pair_id == pair_id and self._pick_role == role:
            self._clear_pick()
        self._selected_id = pair_id
        self._sync_list()

    def swap_arrow(self, pair_id):
        idx = pair_index(self._pairs, pair_id)
        if idx < 0:
            return
        self._pairs[idx] = self._pairs[idx].swapped()
        self._selected_id = pair_id
        self._sync_list()

    def edit_color(self, pair_id):
        idx = pair_index(self._pairs, pair_id)
        if idx < 0:
            return
        original = self._pairs[idx].rgba()

        def on_preview(rgba):
            i = pair_index(self._pairs, pair_id)
            if i >= 0:
                self._pairs[i] = self._pairs[i].with_color(rgba)
                self._schedule_preview()

        def on_done(rgba):
            i = pair_index(self._pairs, pair_id)
            if i < 0:
                return
            if rgba is None:
                self._pairs[i] = self._pairs[i].with_color(original)
            else:
                self._pairs[i] = self._pairs[i].with_color(rgba)
            self._sync_list()

        pick_rgb(self._page, original, on_change=on_preview, on_done=on_done)

    def set_arrow_width(self, pair_id, value):
        idx = pair_index(self._pairs, pair_id)
        if idx < 0:
            return
        self._pairs[idx] = self._pairs[idx].with_width(value)
        self._schedule_preview()

    def set_arrow_head(self, pair_id, value):
        idx = pair_index(self._pairs, pair_id)
        if idx < 0:
            return
        self._pairs[idx] = self._pairs[idx].with_head(value)
        self._schedule_preview()

    def set_arrow_title(self, pair_id, text):
        idx = pair_index(self._pairs, pair_id)
        if idx < 0:
            return
        self._pairs[idx] = self._pairs[idx].with_title(text)

    def set_endpoint_anchor(self, pair_id, role, checked):
        idx = pair_index(self._pairs, pair_id)
        if idx < 0:
            return
        pair = self._pairs[idx]
        if role == "start":
            if pair.start.can_anchor():
                self._pairs[idx] = pair.with_start(pair.start.with_anchor_intent(checked))
            return
        if pair.end is not None and pair.end.can_anchor():
            self._pairs[idx] = pair.with_end(pair.end.with_anchor_intent(checked))

    def _pending_point(self):
        if self._pick_role != "end" or not self._pick_pair_id:
            return None
        idx = pair_index(self._pairs, self._pick_pair_id)
        if idx < 0 or self._pairs[idx].is_complete():
            return None
        return self._pairs[idx].start

    def _sync_list(self, preview=True):
        self._refresh_margin_limit()
        if self._list is not None:
            self._list.rebuild(
                self._pairs,
                self._selected_id,
                self._pick_pair_id,
                self._pick_role,
                context=self._resolve_context(),
            )
        if preview:
            self._schedule_preview()

    def _schedule_preview(self):
        if self._suspend_preview:
            return
        self._refresh_margin_limit()
        self._deferred.schedule(self._refresh_preview, page=self._page)

    def _refresh_preview(self):
        if not qt_widget_alive(self._page):
            return
        try:
            self._preview.update(
                self._pairs_for_draw(),
                self._quality.value(),
                self._style(),
                pending=self._pending_point(),
                highlight_id=self._selected_id,
            )
        except RuntimeError:
            pass

    def _create_cgo(self):
        ready = commit_pair_anchors(self._pairs_for_draw())
        if not ready:
            return
        typed = self._object_name.text().strip() or "pmv_arrows"
        name = unused_object_name(typed, self.cmd, keep=self._loaded_name)
        self._deferred.cancel()
        self._preview.update(
            ready,
            int(self._quality.value()),
            self._style(),
            pending=None,
            highlight_id=None,
        )
        persist_live_preview(
            self.cmd,
            self._preview,
            name,
            obj_id=self._editing_id,
            retarget=lambda coll: retarget_arrow_collection(coll, ready),
            fallback=lambda: build_arrow_collection(
                ready,
                int(self._quality.value()),
                self._style(),
                name,
            ),
        )
        if self._on_create is not None:
            self._on_create()

    def _export_cgo(self):
        ready = commit_pair_anchors(self._pairs_for_draw())
        if not ready:
            return
        _, _, QtWidgets = qt_modules()
        name = self._object_name.text().strip() or "pmv_arrows"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self._page, "Export CGO script", "%s.py" % name, "Python (*.py)",
        )
        if not path:
            return
        build_arrow_collection(
            ready,
            int(self._quality.value()),
            self._style(),
            name,
        ).write(path)
