"""Arrow mesh builder: a compact pair editor with two-step picking."""

from __future__ import annotations

from typing import List, Optional

from ...meshes.Arrows import HEAD_WIDTH, default_head_radius, head_radius_follows_shaft
from ...util.line_style import (
    LineStyle,
    MAX_ARROW_MARGIN,
    default_head_length,
    max_margin_for_length,
    style_margins,
)
from ..pick import qt_modules, qt_widget_alive
from ..widgets.breadcrumb import CRUMB_ARROWS
from ..widgets.ascii_locale import apply_ascii_float_locale
from ..widgets.log_slider import LogSegmentRadiusWidget
from ..widgets.section import make_section
from ..widgets.theme import style_info_banner
from .appearance_section import AppearanceSection
from .arrow_list import ArrowPairEditor
from .arrow_type import ArrowTypeControl
from .base import BuilderPage
from .colors import bind_color_pick_result, colors_for_new_points, pick_rgb
from .pairs import (
    DEFAULT_ARROW_WIDTH,
    MULTI_CENTER,
    MULTI_CLICKED,
    VisualPair,
    commit_pair_anchors,
    complete_pairs,
    endpoint_label,
    pair_index,
    take_selection_endpoints,
    take_single_selection_point,
)
from .points import VisualPoint, camera_center_point
from .point_insertion import INSERT_SOURCE_CAMERA, INSERT_SOURCE_FRESH
from .preview import (
    ArrowPreview,
    build_arrow_collection,
    retarget_arrow_collection,
)
from .preview_mode import DEFAULT_PREVIEW_MODE, preview_arrow_quality, preview_is_on, read_preview_mode
from .modifiers_section import ModifiersSection
from .zoom_selection import focus_visual_point, zoom_to_visual_points

CLICKED_ATOM_LABEL = "Clicked atom"
CLICKED_ATOM_TIP = (
    "When several atoms are selected (residue, chain, or object), use the "
    "atom that was clicked. Two selected atoms still create a full arrow."
)
SELECTION_AVERAGE_LABEL = "Average of selection"
SELECTION_CENTER_TIP = (
    "When several atoms are selected, use the average position of the "
    "selection as one endpoint."
)


class ArrowBuilderPage(BuilderPage):
    """Editor for multi-pair arrow CGOs."""

    DEFAULT_NAME = "pmv_arrows"
    CRUMB_LEAF = CRUMB_ARROWS
    CONTEXT = "ArrowBuilderPage"

    def _init_editor(self):
        self._pairs: List[VisualPair] = []
        self._preview = ArrowPreview(self.cmd)
        self._pick_role = None
        self._pick_pair_id = None
        self._selected_id = None
        self._ignore_xyz = None
        self._poll_timer = None
        self._list = None
        self._appearance = None
        self._appearance_pts: List[VisualPoint] = []
        self._snap_atom = True
        self._hook_selection = True
        self._zoom_selection = None
        self._status = None
        self._key_filter = None
        self._line_style = LineStyle()
        self._modifiers = None
        self._style_control = None
        self._arrows_section = None
        self._shaft_widget = None
        self._head_length = None
        self._head_radius = None
        self._geom_form = None
        self._head_length_row = None
        self._head_radius_row = None
        self._head_follows_shaft = True
        self._clicked_atom = True
        self._clicked_atom_box = None
        self._center_radio = None

    def _cleanup_ephemeral(self):
        self._stop_poll_timer()
        if self._modifiers is not None:
            self._modifiers.cleanup()
        self._pick_role = None
        self._pick_pair_id = None
        self._ignore_xyz = None

    def reset_for_create(self):
        self._apply_create_chrome()
        self._pairs = []
        self._selected_id = None
        self._clear_pick()
        if self._appearance is not None:
            self._appearance.set_quality(3)
            self._appearance.set_preview_mode(DEFAULT_PREVIEW_MODE)
        self._line_style = LineStyle()
        if self._style_control is not None:
            self._style_control.set_max_margin(MAX_ARROW_MARGIN)
            self._style_control.set_style(self._line_style)
        if self._shaft_widget is not None:
            self._shaft_widget.set_value(DEFAULT_ARROW_WIDTH)
        if self._head_length is not None:
            self._head_length.setValue(default_head_length(DEFAULT_ARROW_WIDTH))
        self._head_follows_shaft = True
        if self._head_radius is not None:
            self._head_radius.blockSignals(True)
            self._head_radius.setValue(default_head_radius(DEFAULT_ARROW_WIDTH))
            self._head_radius.blockSignals(False)
        if self._modifiers is not None:
            self._modifiers.clip.reset()
            self._modifiers.refresh_summary()
        self._sync_list(preview=False)
        self._preview.cleanup()
        self._sync_head_geometry()

    def load_object(self, obj):
        from .load_visual import arrow_options, pairs_from_mesh
        from ..catalog import display_name

        self._editing_id = str(obj.id)
        name = display_name(obj) or self.DEFAULT_NAME
        self._pairs = pairs_from_mesh(obj)
        self._selected_id = self._pairs[0].pair_id if self._pairs else None
        opts = arrow_options(obj)
        clip = opts.get("clip_planes") or []
        self._deferred.cancel()
        self._clear_pick()
        self._suspend_preview = True
        try:
            self._apply_edit_chrome(name)
            if self._appearance is not None:
                self._appearance.set_quality(int(opts["quality"]))
                self._appearance.set_preview_mode(read_preview_mode(obj))
            if self._shaft_widget is not None:
                self._shaft_widget.set_value(float(opts.get("shaft_radius") or DEFAULT_ARROW_WIDTH))
            if self._head_length is not None:
                self._head_length.setValue(float(opts.get("head_length") or default_head_length(DEFAULT_ARROW_WIDTH)))
            shaft = float(opts.get("shaft_radius") or DEFAULT_ARROW_WIDTH)
            hr = opts.get("head_radius")
            self._head_follows_shaft = head_radius_follows_shaft(shaft, hr)
            if self._head_radius is not None:
                if hr is None:
                    hr = default_head_radius(shaft)
                self._head_radius.blockSignals(True)
                self._head_radius.setValue(float(hr))
                self._head_radius.blockSignals(False)
            if opts.get("line_style") is not None:
                self._line_style = opts["line_style"].copy() if hasattr(opts["line_style"], "copy") else opts["line_style"]
                if self._style_control is not None:
                    self._style_control.set_style(self._line_style)
            self._sync_head_geometry()
            clip = opts.get("clip_planes") or []
            self._sync_list(preview=False)
        finally:
            self._suspend_preview = False
        self._preview.adopt(obj)
        if self._modifiers is not None:
            self._modifiers.clip.set_planes(clip)
            self._modifiers.refresh_summary()
        self._schedule_preview()

    def _existing_points(self) -> List[VisualPoint]:
        pts = []
        for pair in self._pairs:
            if not pair.enabled:
                continue
            pts.append(pair.start)
            if pair.end is not None:
                pts.append(pair.end)
        return pts

    def _build(self, parent):
        QtCore, _, QtWidgets = self._require_qt()
        page, root, back = self._mount_shell(parent, QtWidgets)
        left, right = self._mount_editor_columns(root, QtWidgets)

        geom = make_section("Geometry", form=True)
        geom_layout = geom.layout
        self._geom_form = geom_layout
        self._shaft_widget = LogSegmentRadiusWidget(initial=DEFAULT_ARROW_WIDTH)
        self._shaft_widget.connect_changed(self._on_object_shaft_changed)
        self._head_length = QtWidgets.QDoubleSpinBox()
        self._head_length.setRange(0.01, 10.0)
        self._head_length.setValue(default_head_length(DEFAULT_ARROW_WIDTH))
        apply_ascii_float_locale(self._head_length, QtCore)
        self._head_length.valueChanged.connect(lambda *_: self._on_object_head_changed())
        self._head_radius = QtWidgets.QDoubleSpinBox()
        self._head_radius.setRange(0.01, 10.0)
        self._head_radius.setDecimals(3)
        self._head_radius.setValue(default_head_radius(DEFAULT_ARROW_WIDTH))
        apply_ascii_float_locale(self._head_radius, QtCore)
        self._head_radius.valueChanged.connect(self._on_object_head_radius_changed)
        geom_layout.addRow("Shaft radius (Å)", self._shaft_widget.widget)
        self._head_length_row = geom_layout.rowCount()
        geom_layout.addRow("Head length (Å)", self._head_length)
        self._head_radius_row = geom_layout.rowCount()
        geom_layout.addRow("Head radius (Å)", self._head_radius)
        right.addWidget(geom.widget)

        opts = make_section("Options")
        self._style_control = ArrowTypeControl(
            page,
            style=self._line_style,
            max_margin=MAX_ARROW_MARGIN,
            on_change=self.set_line_style,
        )
        self._style_control.widget.setMinimumWidth(160)
        opts.layout.addWidget(self._style_control.widget)
        right.addWidget(opts.widget)

        self._appearance = AppearanceSection(
            page,
            self.cmd,
            self.CONTEXT,
            show_wireframe=False,
            show_quality=True,
            quality_range=(0, 5),
            quality_tooltip=(
                "0 = 2D lines; 1–5 = native shafts (start→end gradient) with mesh heads"
            ),
            on_changed=self._on_appearance_changed,
            on_preview=self._on_appearance_preview,
        )
        self._appearance.set_quality(3)
        self._appearance.set_selected_rows_provider(self._selected_appearance_rows)
        right.addWidget(self._appearance.widget)

        arrows = make_section("Arrows", expanding=True)
        self._arrows_section = arrows
        self._status = QtWidgets.QLabel("Add an arrow, or select two atoms first.")
        self._status.setWordWrap(True)
        style_info_banner(self._status)
        arrows.layout.addWidget(self._status)
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.setSpacing(8)
        mode_label = QtWidgets.QLabel("When several atoms:")
        mode_row.addWidget(mode_label)
        self._clicked_atom_box = QtWidgets.QRadioButton(CLICKED_ATOM_LABEL)
        self._clicked_atom_box.setChecked(True)
        self._clicked_atom_box.setToolTip(CLICKED_ATOM_TIP)
        self._center_radio = QtWidgets.QRadioButton(SELECTION_AVERAGE_LABEL)
        self._center_radio.setToolTip(SELECTION_CENTER_TIP)
        mode_group = QtWidgets.QButtonGroup(page)
        mode_group.setExclusive(True)
        mode_group.addButton(self._clicked_atom_box)
        mode_group.addButton(self._center_radio)
        self._clicked_atom_box.toggled.connect(self._on_clicked_atom_toggled)
        mode_row.addWidget(self._clicked_atom_box)
        mode_row.addWidget(self._center_radio)
        mode_row.addStretch(1)
        arrows.layout.addLayout(mode_row)
        self._list = ArrowPairEditor(arrows.body, self)
        self._list.attach_add_to_section(arrows)
        arrows.layout.addWidget(self._list.widget, stretch=1)
        left.addWidget(arrows.widget, stretch=1)

        self._modifiers = ModifiersSection(
            page,
            self.cmd,
            self.CONTEXT,
            self._preview,
            get_span_points=self._existing_points,
            on_changed=self._on_modifiers_changed,
            page=page,
        )
        right.addWidget(self._modifiers.widget)
        right.addStretch(1)

        self._mount_action_bar(page, root)

        self._sync_head_geometry()

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
        self._finish_build(
            page,
            back,
            [
                (self._shaft_widget, "Object-level shaft radius for new pairs and when the slider is moved."),
                (self._head_length, "Object-level head length for new pairs and when the spinner is moved."),
                (self._head_radius, "Cone radius at the arrow head (defaults to shaft × %.1f)." % HEAD_WIDTH),
                (self._clicked_atom_box, CLICKED_ATOM_TIP, CLICKED_ATOM_LABEL),
                (self._center_radio, SELECTION_CENTER_TIP, SELECTION_AVERAGE_LABEL),
            ] + list(self._appearance.tooltips()) + list(self._modifiers.tooltips()),
        )

    def _after_build(self):
        self._sync_list(preview=False)

    def _go_back(self):
        self._abort_pick(silent=True)
        super()._go_back()

    def _quality_value(self) -> int:
        if self._appearance is None:
            return 3
        return int(self._appearance.quality())

    def _head_radius_value(self):
        if self._head_radius is None:
            return default_head_radius(DEFAULT_ARROW_WIDTH)
        return float(self._head_radius.value())

    def _head_radius_for_mesh(self):
        """None while the cone tracks shaft × HEAD_WIDTH (per-pair)."""
        if getattr(self, "_head_follows_shaft", True):
            return None
        return self._head_radius_value()

    def _clip_planes(self):
        if self._modifiers is None:
            return []
        return self._modifiers.clip.active_planes()

    def _on_modifiers_changed(self):
        if self._modifiers is not None:
            self._modifiers.refresh_summary()
            self._modifiers.clip.refresh_gizmos()
        self._schedule_preview()

    def _on_object_shaft_changed(self):
        if self._shaft_widget is None:
            return
        width = float(self._shaft_widget.value())
        self._pairs = [pair.with_width(width) for pair in self._pairs]
        if self._head_length is not None:
            self._head_length.blockSignals(True)
            self._head_length.setValue(default_head_length(width))
            self._head_length.blockSignals(False)
        if self._head_follows_shaft and self._head_radius is not None:
            self._head_radius.blockSignals(True)
            self._head_radius.setValue(default_head_radius(width))
            self._head_radius.blockSignals(False)
        self._schedule_preview()

    def _on_object_head_radius_changed(self, *_):
        self._head_follows_shaft = False
        self._schedule_preview()

    def _on_object_head_changed(self):
        if self._head_length is None:
            return
        head = float(self._head_length.value())
        self._pairs = [pair.with_head(head) for pair in self._pairs]
        self._schedule_preview()

    def _sync_appearance_points(self):
        self._appearance_pts = [pair.start for pair in self._pairs]
        if self._appearance is not None:
            self._appearance.bind_points(self._appearance_pts)

    def _write_appearance_to_pairs(self):
        for i, pair in enumerate(self._pairs):
            if i >= len(self._appearance_pts):
                break
            start = self._appearance_pts[i]
            end = pair.end
            self._pairs[i] = pair.with_start(start).with_end(end)

    def _on_appearance_preview(self):
        self._write_appearance_to_pairs()
        self._schedule_preview()

    def _on_appearance_changed(self):
        self._write_appearance_to_pairs()
        self._sync_list()

    def _selected_appearance_rows(self):
        if self._selected_id is None:
            return []
        return [i for i, pair in enumerate(self._pairs) if pair.pair_id == self._selected_id]

    def _style(self):
        return self._line_style.copy()

    def set_line_style(self, style: LineStyle):
        if style is None:
            return
        self._line_style = style.copy()
        self._sync_head_geometry()
        self._schedule_preview()

    def _set_form_row_visible(self, layout, row, visible):
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

    def _shows_arrow_heads(self) -> bool:
        style = self._line_style
        if style is None or not hasattr(style, "n_arrow_heads"):
            return True
        return int(style.n_arrow_heads()) > 0

    def _sync_head_geometry(self):
        show = self._shows_arrow_heads()
        self._set_form_row_visible(self._geom_form, self._head_length_row, show)
        self._set_form_row_visible(self._geom_form, self._head_radius_row, show)

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
        start, end = style_margins(self._line_style)
        start = min(start, cap)
        end = min(end, cap)
        if start != self._line_style.start_margin or end != self._line_style.end_margin:
            self._line_style = self._line_style.updated(start_margin=start, end_margin=end)
        if self._style_control is not None:
            self._style_control.set_max_margin(cap)
            self._style_control.set_style(self._line_style)

    def _pairs_for_draw(self):
        style = self._style()
        return [pair.with_style(style.copy()) for pair in self._pairs if pair.enabled]

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
        return bool(self._hook_selection)

    def _snap(self) -> bool:
        return bool(self._snap_atom)

    def _multi_atom(self) -> str:
        if self._clicked_atom_box is not None:
            return MULTI_CLICKED if self._clicked_atom_box.isChecked() else MULTI_CENTER
        return MULTI_CLICKED if self._clicked_atom else MULTI_CENTER

    def _on_clicked_atom_toggled(self, checked):
        self._clicked_atom = bool(checked)

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
        width = self._shaft_widget.value() if self._shaft_widget is not None else DEFAULT_ARROW_WIDTH
        head = (
            float(self._head_length.value())
            if self._head_length is not None
            else default_head_length(width)
        )
        palette = colors_for_new_points(1, start_index=len(complete_pairs(self._pairs)))
        color = palette[0]
        start = start.with_color(color)
        if end is not None:
            end = end.with_color(color)
        return VisualPair(
            start, end, style=self._line_style.copy(),
            width=float(width), head=float(head),
        )

    def add_arrow(self):
        source = self._list.add_source() if self._list is not None else INSERT_SOURCE_SELECTION
        if source == INSERT_SOURCE_CAMERA:
            pt = camera_center_point(
                self.cmd,
                self._snap(),
                self._existing_points(),
                hook_to_selection=self._hook(),
            )
            if self._pick_role is not None:
                self._accept_point(pt)
                return
            self._begin_incomplete(pt)
            return
        if self._pick_role is not None:
            self._abort_pick()
            return
        if source == INSERT_SOURCE_FRESH:
            self._clear_pymol_selection()
            self._set_pick("start", None, "Select start in PyMOL.")
            self._sync_list()
            return
        start, end, status = take_selection_endpoints(
            self.cmd,
            self._existing_points(),
            hook_to_selection=self._hook(),
            multi_atom=self._multi_atom(),
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
            self._set_status("Select atoms to place an endpoint, or pick them one at a time.")
            return
        self._set_status("Nothing selected. Select atoms, or use Fresh selection.")

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
            self._snap(),
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
                multi_atom=self._multi_atom(),
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
            multi_atom=self._multi_atom(),
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
        current = pair.start if self._pick_role == "start" else pair.end
        if current is not None:
            colored = point.with_color_choice(current.color_choice())
        else:
            colored = point.with_color_choice(pair.start.color_choice())
        if self._pick_role == "start":
            self._pairs[idx] = pair.with_start(colored)
        else:
            self._pairs[idx] = pair.with_end(colored)
        self._clear_pymol_selection()
        self._clear_pick()
        self._set_status("Arrow updated.")
        self._sync_list()

    def select_arrow(self, pair_id, zoom=True):
        if zoom and self._selected_id == pair_id:
            self._selected_id = None
            self._sync_list()
            return
        self._selected_id = pair_id
        self._sync_list()
        if not zoom:
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
        current = pair.start if role == "start" else pair.end
        keep = current.color_choice() if current is not None else pair.start.color_choice()
        pt = camera_center_point(
            self.cmd,
            self._snap(),
            self._existing_points(),
            hook_to_selection=self._hook(),
        ).with_color_choice(keep)
        if role == "start":
            self._pairs[idx] = pair.with_start(pt)
        else:
            self._pairs[idx] = pair.with_end(pt)
        if self._pick_pair_id == pair_id and self._pick_role == role:
            self._clear_pick()
        self._selected_id = pair_id
        self._sync_list()

    def set_endpoint_xyz(self, pair_id, role, xyz):
        idx = pair_index(self._pairs, pair_id)
        if idx < 0:
            return
        pair = self._pairs[idx]
        if role == "start":
            self._pairs[idx] = pair.with_start(pair.start.with_xyz(xyz))
        elif pair.end is not None:
            self._pairs[idx] = pair.with_end(pair.end.with_xyz(xyz))
        self._schedule_preview()

    def set_pair_enabled(self, pair_id, enabled):
        idx = pair_index(self._pairs, pair_id)
        if idx < 0:
            return
        self._pairs[idx] = self._pairs[idx].with_enabled(enabled)
        self._schedule_preview()

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
        original = self._pairs[idx].color_choice()

        def on_preview(choice):
            i = pair_index(self._pairs, pair_id)
            if i >= 0 and choice is not None:
                self._pairs[i] = self._pairs[i].with_color_choice(choice)
                self._schedule_preview()

        def apply_choice(choice):
            i = pair_index(self._pairs, pair_id)
            if i >= 0:
                self._pairs[i] = self._pairs[i].with_color_choice(choice)
                self._sync_list()

        def restore_originals():
            i = pair_index(self._pairs, pair_id)
            if i >= 0:
                self._pairs[i] = self._pairs[i].with_color_choice(original)
                self._sync_list()

        pick_rgb(
            self._page,
            original,
            on_change=on_preview,
            on_done=bind_color_pick_result(apply_choice, restore_originals),
            cmd=self.cmd,
        )

    def edit_endpoint_color(self, pair_id, role):
        idx = pair_index(self._pairs, pair_id)
        if idx < 0:
            return
        pair = self._pairs[idx]
        pt = pair.start if role == "start" else pair.end
        if pt is None:
            return
        original = pt.color_choice()

        def apply_role(choice, target):
            if role == "start":
                return target.with_start(target.start.with_color_choice(choice))
            if target.end is None:
                return target
            return target.with_end(target.end.with_color_choice(choice))

        def on_preview(choice):
            i = pair_index(self._pairs, pair_id)
            if i >= 0 and choice is not None:
                self._pairs[i] = apply_role(choice, self._pairs[i])
                self._schedule_preview()

        def apply_choice(choice):
            i = pair_index(self._pairs, pair_id)
            if i >= 0:
                self._pairs[i] = apply_role(choice, self._pairs[i])
                self._sync_list()

        def restore_originals():
            i = pair_index(self._pairs, pair_id)
            if i >= 0:
                self._pairs[i] = apply_role(original, self._pairs[i])
                self._sync_list()

        pick_rgb(
            self._page,
            original,
            on_change=on_preview,
            on_done=bind_color_pick_result(apply_choice, restore_originals),
            cmd=self.cmd,
        )

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
        self._sync_appearance_points()
        if self._appearance is not None:
            self._appearance.set_color_selection_enabled(self._selected_id is not None)
        if self._list is not None:
            self._list.rebuild(
                self._pairs,
                self._selected_id,
                self._pick_pair_id,
                self._pick_role,
                context=self._resolve_context(),
            )
        if self._arrows_section is not None:
            n = len(self._pairs)
            self._arrows_section.set_title("Arrows (%d)" % n if n else "Arrows")
        self._sync_commit_enabled()
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
            mode = self._current_preview_mode()
            if not preview_is_on(mode):
                self._preview.clear_meshes()
                if self._modifiers is not None:
                    self._modifiers.clip.refresh_gizmos()
                return
            self._preview.update(
                self._pairs_for_draw(),
                preview_arrow_quality(self._quality_value(), mode),
                self._style(),
                pending=self._pending_point(),
                highlight_id=self._selected_id,
                clip_planes=self._clip_planes(),
                head_radius=self._head_radius_for_mesh(),
            )
            if self._modifiers is not None:
                self._modifiers.clip.refresh_gizmos()
        except RuntimeError:
            pass

    def _can_commit(self) -> bool:
        return bool(self._committed_pairs())

    def _committed_pairs(self):
        return commit_pair_anchors(self._pairs_for_draw())

    def _prepare_persist(self, name: str):
        ready = self._committed_pairs()
        self._deferred.cancel()
        self._preview.update(
            ready,
            int(self._quality_value()),
            self._style(),
            pending=None,
            highlight_id=None,
            clip_planes=self._clip_planes(),
            head_radius=self._head_radius_for_mesh(),
        )
        super()._prepare_persist(name)

    def _collection(self, name: str):
        coll = build_arrow_collection(
            self._committed_pairs(),
            int(self._quality_value()),
            self._style(),
            name,
            clip_planes=self._clip_planes(),
            head_radius=self._head_radius_for_mesh(),
        )
        if coll is not None:
            coll.specular = True
        return coll

    def _retarget(self, collection):
        return retarget_arrow_collection(
            collection,
            self._committed_pairs(),
            clip_planes=self._clip_planes(),
            head_radius=self._head_radius_for_mesh(),
        )
