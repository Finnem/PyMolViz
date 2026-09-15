"""Compact expandable list of arrow pairs (not a property table)."""

from __future__ import annotations

from .colors import rgb_to_css
from .pairs import (
    PENDING_END,
    endpoint_label,
    endpoint_xyz_text,
    free_point_display_names,
    pair_status,
    pair_status_glyph,
)
from ..pick import qt_modules
from ..tooltips import apply_required_tooltips, warn_missing_setting_tooltips
from ..widgets.ascii_locale import apply_ascii_float_locale
from ..widgets.scrolling import apply_expanding_list_policy
from ..widgets.sticky_add import StickyAddOverlay
from ..widgets.theme import INK, SELECTED, rgb_css, swatch_button_css

ADD_ARROW_TIP = (
    "Add an arrow. If two atoms are already selected, uses them immediately; "
    "if one is selected, that becomes the start; otherwise pick start then end."
)
CANCEL_PICK_TIP = "Cancel the current start/end pick and drop any incomplete arrow."
DELETE_ARROW_TIP = "Remove this arrow from the list."
PICK_START_TIP = "Pick a new start atom or point in the viewer."
PICK_END_TIP = "Pick a new end atom or point in the viewer."
FOCUS_ENDPOINT_TIP = "Select and center this atom (or frame this point) in PyMOL."
CAMERA_ENDPOINT_TIP = (
    "Set this endpoint to the camera-center marker. "
    "With Snap new points to atoms, uses a visible atom within 2 Å of that marker."
)
SWAP_ARROW_TIP = "Swap the start and end of this arrow."
COLOR_ARROW_TIP = "Choose the color (and opacity) of this arrow."
WIDTH_ARROW_TIP = "Shaft radius in Ångströms. Arrowhead length follows width."
NAME_ARROW_TIP = "Optional label shown in the list instead of the endpoints."
ANCHOR_START_TIP = "When checked, the start stays attached to this atom and follows if it moves."
ANCHOR_END_TIP = "When checked, the end stays attached to this atom and follows if it moves."


class ArrowPairEditor:
    """Scrollable pair rows with inline details for the selected arrow."""

    def __init__(self, parent, host):
        QtCore, _, QtWidgets = qt_modules()
        self._host = host
        self._box = QtWidgets.QWidget(parent)
        layout = QtWidgets.QVBoxLayout(self._box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._inner = QtWidgets.QWidget()
        self._rows = QtWidgets.QVBoxLayout(self._inner)
        self._rows.setContentsMargins(0, 0, 0, 0)
        self._rows.setSpacing(0)
        self._rows.addStretch(1)
        self._scroll.setWidget(self._inner)
        apply_expanding_list_policy(self._box, QtWidgets)
        apply_expanding_list_policy(self._scroll, QtWidgets)
        layout.addWidget(self._scroll, stretch=1)

        self._camera_btn = QtWidgets.QPushButton("Camera")
        self._camera_btn.setToolTip(CAMERA_ENDPOINT_TIP)
        self._camera_btn.clicked.connect(self._host.camera_pick)
        self._sticky_add = StickyAddOverlay(
            self._box,
            self._scroll,
            text="+ Add arrow",
            tooltip=ADD_ARROW_TIP,
            on_click=self._host.add_arrow,
            count=self._item_count,
            row_height=self._row_height,
            extra=self._camera_btn,
            context="ArrowPairEditor",
            empty_hint="No arrows yet.",
        )
        self._sticky_add.attach()
        apply_required_tooltips(
            [(self._camera_btn, CAMERA_ENDPOINT_TIP, "Camera pick")],
            context="ArrowPairEditor",
        )

    @property
    def widget(self):
        return self._box

    def _item_count(self) -> int:
        return max(int(self._rows.count()) - 1, 0)

    def _row_height(self) -> int:
        if self._rows.count() > 1:
            item = self._rows.itemAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                height = int(widget.sizeHint().height())
                if height > 0:
                    return height
        return 28

    def set_picking(self, picking: bool) -> None:
        if picking:
            self._sticky_add.set_text("Cancel pick")
            self._sticky_add.set_tooltip(CANCEL_PICK_TIP)
        else:
            self._sticky_add.set_text("+ Add arrow")
            self._sticky_add.set_tooltip(ADD_ARROW_TIP)
        self._sticky_add.set_extra_visible(bool(picking))
        self._sticky_add.sync()

    def rebuild(self, pairs, selected_id, pick_pair_id, pick_role, context=None):
        QtCore, QtGui, QtWidgets = qt_modules()
        while self._rows.count() > 1:
            item = self._rows.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        free_names = free_point_display_names(pairs)
        picking = pick_role is not None
        self.set_picking(picking)
        for pair in pairs:
            selected = pair.pair_id == selected_id
            self._rows.insertWidget(
                self._rows.count() - 1,
                self._make_block(
                    QtCore, QtGui, QtWidgets, pair, selected,
                    pick_pair_id, pick_role, free_names, context,
                ),
            )
        if self._sticky_add is not None:
            self._sticky_add.sync()
        warn_missing_setting_tooltips(self._box, context="ArrowPairEditor")

    def _make_block(
        self, QtCore, QtGui, QtWidgets, pair, selected,
        pick_pair_id, pick_role, free_names, context,
    ):
        block = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(block)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._summary_row(
            QtCore, QtGui, QtWidgets, pair, selected, free_names, context,
        ))
        if selected:
            layout.addWidget(self._details(
                QtCore, QtGui, QtWidgets, pair, pick_pair_id, pick_role, free_names,
            ))
        return block

    def _summary_row(self, QtCore, QtGui, QtWidgets, pair, selected, free_names, context):
        host = self._host
        pid = pair.pair_id
        row = QtWidgets.QFrame()
        row.setFrameShape(QtWidgets.QFrame.NoFrame)
        row.setCursor(QtCore.Qt.PointingHandCursor)
        if selected:
            row.setStyleSheet("QFrame { background: %s; }" % rgb_css(SELECTED))
        else:
            row.setStyleSheet("QFrame { background: transparent; }")

        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(4, 3, 2, 3)
        layout.setSpacing(4)

        status = pair_status(pair, context)
        glyph = QtWidgets.QLabel(pair_status_glyph(status))
        glyph.setFixedWidth(14)
        glyph.setToolTip({
            "ok": "Arrow is complete.",
            "picking": "Waiting for the other endpoint.",
            "missing": "An anchored atom could not be resolved.",
        }.get(status, ""))
        layout.addWidget(glyph)

        title = (pair.title or "").strip()
        if title:
            name = QtWidgets.QLabel(title)
            name.setStyleSheet("font-weight: 600; color: %s;" % rgb_css(INK))
            layout.addWidget(name, stretch=1)
        else:
            start = _flat_label_button(
                QtWidgets, QtCore, row,
                endpoint_label(pair.start, pending="[pick start…]", free_names=free_names),
                FOCUS_ENDPOINT_TIP if pair.start is not None else PICK_START_TIP,
            )
            start.clicked.connect(lambda _=False, i=pid: host.focus_endpoint(i, "start"))
            end = _flat_label_button(
                QtWidgets, QtCore, row,
                endpoint_label(pair.end, pending=PENDING_END, free_names=free_names),
                FOCUS_ENDPOINT_TIP if pair.end is not None else PICK_END_TIP,
            )
            if pair.end is None:
                end.clicked.connect(lambda _=False, i=pid: host.pick_endpoint(i, "end"))
            else:
                end.clicked.connect(lambda _=False, i=pid: host.focus_endpoint(i, "end"))
            layout.addWidget(start)
            layout.addWidget(QtWidgets.QLabel("→"))
            layout.addWidget(end, stretch=1)

        delete = QtWidgets.QPushButton("×")
        delete.setFlat(True)
        delete.setFixedWidth(22)
        delete.clicked.connect(lambda _=False, i=pid: host.delete_arrow(i))
        layout.addWidget(delete)

        def on_press(event, i=pid):
            if event.button() == QtCore.Qt.LeftButton:
                host.select_arrow(i)
            QtWidgets.QFrame.mousePressEvent(row, event)

        row.mousePressEvent = on_press
        apply_required_tooltips(
            [(delete, DELETE_ARROW_TIP, "Delete arrow")],
            context="ArrowPairEditor",
        )
        return row

    def _details(self, QtCore, QtGui, QtWidgets, pair, pick_pair_id, pick_role, free_names):
        host = self._host
        pid = pair.pair_id
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(8, 2, 6, 6)
        layout.setSpacing(3)

        ends = QtWidgets.QHBoxLayout()
        ends.setSpacing(2)
        ends.addWidget(self._endpoint_controls(
            QtCore, QtWidgets, pair, "start", pick_pair_id, pick_role, free_names,
        ), stretch=1)
        swap = QtWidgets.QPushButton("↔")
        swap.setFlat(True)
        swap.setFixedWidth(22)
        swap.setEnabled(pair.is_complete())
        swap.clicked.connect(lambda _=False, i=pid: host.swap_arrow(i))
        ends.addWidget(swap)
        ends.addWidget(self._endpoint_controls(
            QtCore, QtWidgets, pair, "end", pick_pair_id, pick_role, free_names,
        ), stretch=1)
        color_btn = QtWidgets.QPushButton()
        color_btn.setFixedSize(18, 16)
        color_btn.setFlat(True)
        color_btn.setStyleSheet(swatch_button_css(rgb_to_css(pair.color)))
        color_btn.clicked.connect(lambda _=False, i=pid: host.edit_color(i))
        ends.addWidget(color_btn)
        layout.addLayout(ends)

        style_row = QtWidgets.QHBoxLayout()
        style_row.setSpacing(4)
        width = QtWidgets.QDoubleSpinBox()
        width.setRange(0.01, 2.0)
        width.setSingleStep(0.01)
        width.setDecimals(3)
        width.setSuffix(" Å")
        apply_ascii_float_locale(width, QtCore)
        width.setValue(float(pair.width))
        width.setMaximumWidth(72)
        width.valueChanged.connect(lambda value, i=pid: host.set_arrow_width(i, value))
        style_row.addWidget(width)
        name = QtWidgets.QLineEdit()
        name.setPlaceholderText("Name")
        name.setText(pair.title)
        name.setMaximumWidth(88)
        name.editingFinished.connect(
            lambda i=pid, edit=name: host.set_arrow_title(i, edit.text())
        )
        style_row.addWidget(name)
        layout.addLayout(style_row)

        apply_required_tooltips(
            [
                (color_btn, COLOR_ARROW_TIP, "Arrow color"),
                (width, WIDTH_ARROW_TIP),
                (name, NAME_ARROW_TIP),
                (swap, SWAP_ARROW_TIP),
            ],
            context="ArrowPairEditor",
        )
        return box

    def _endpoint_controls(
        self, QtCore, QtWidgets, pair, role, pick_pair_id, pick_role, free_names,
    ):
        host = self._host
        pid = pair.pair_id
        pt = pair.start if role == "start" else pair.end
        picking = pick_pair_id == pid and pick_role == role
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        label = _flat_label_button(
            QtWidgets, QtCore, row,
            endpoint_label(
                pt,
                pending=PENDING_END if role == "end" else "[pick start…]",
                free_names=free_names,
            ),
            FOCUS_ENDPOINT_TIP if pt is not None else (
                PICK_END_TIP if role == "end" else PICK_START_TIP
            ),
        )
        if pt is not None and not pt.can_anchor():
            xyz = endpoint_xyz_text(pt)
            if xyz:
                label.setToolTip("%s\nxyz %s" % (label.toolTip(), xyz))
        if pt is None:
            label.clicked.connect(lambda _=False, i=pid, r=role: host.pick_endpoint(i, r))
        else:
            label.clicked.connect(lambda _=False, i=pid, r=role: host.focus_endpoint(i, r))
        pick = QtWidgets.QPushButton("Picking…" if picking else "Pick")
        pick.setFlat(True)
        pick.setCheckable(True)
        pick.setChecked(picking)
        pick.setMaximumWidth(58)
        pick.clicked.connect(lambda _=False, i=pid, r=role: host.pick_endpoint(i, r))
        camera = QtWidgets.QPushButton("Cam")
        camera.setFlat(True)
        camera.setMaximumWidth(32)
        camera.clicked.connect(lambda _=False, i=pid, r=role: host.camera_endpoint(i, r))
        layout.addWidget(label, stretch=1)
        layout.addWidget(pick)
        layout.addWidget(camera)
        if pt is not None and pt.can_anchor():
            pin = QtWidgets.QCheckBox()
            pin.setChecked(pt.wants_anchor())
            pin.toggled.connect(
                lambda checked, i=pid, r=role: host.set_endpoint_anchor(i, r, checked)
            )
            apply_required_tooltips(
                [(pin, ANCHOR_START_TIP if role == "start" else ANCHOR_END_TIP)],
                context="ArrowPairEditor",
            )
            layout.addWidget(pin)
        apply_required_tooltips(
            [
                (pick, PICK_END_TIP if role == "end" else PICK_START_TIP),
                (camera, CAMERA_ENDPOINT_TIP),
            ],
            context="ArrowPairEditor",
        )
        return row


def _flat_label_button(QtWidgets, QtCore, parent, text, tooltip):
    btn = QtWidgets.QPushButton(text, parent)
    btn.setFlat(True)
    btn.setCursor(QtCore.Qt.PointingHandCursor)
    btn.setStyleSheet("QPushButton { text-align: left; padding: 0 4px; }")
    btn.setToolTip(tooltip)
    return btn
