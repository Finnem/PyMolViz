"""Shared expandable spatial-item list (Points and Arrows)."""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

from .colors import rgb_to_css
from ..pick import qt_modules
from ..tooltips import apply_required_tooltips
from ..widgets.scrolling import apply_expanding_list_policy
from ..widgets.switch import make_switch
from ..widgets.theme import (
    BORDER,
    DETAIL,
    HEADER,
    HEADER_INK,
    INK,
    MUTED,
    SELECTED,
    WHITE,
    muted_label_css,
    primary_button_css,
    qcolor,
    rgb_css,
    swatch_button_css,
)
from ..widgets.type_icons import apply_source_icon

ENABLE_ITEM_TIP = "Include this item in the preview and saved object."
EXPAND_ITEM_TIP = "Show or hide details for this item."
DELETE_ITEM_TIP = "Remove this item from the list."
OVERFLOW_TIP = "More actions for this item."
COLOR_ITEM_TIP = "Choose the color (and opacity) of this item."

CHEVRON_CLOSED = "▸"
CHEVRON_OPEN = "▾"

# (title, width) — width 0 means stretch. First column is the expand chevron.
POINT_LIST_COLUMNS: Tuple[Tuple[str, int], ...] = (
    ("", 18),
    ("#", 28),
    ("Enabled", 58),
    ("Label", 0),
    ("Source / Status", 0),
    ("X (Å)", 64),
    ("Y (Å)", 64),
    ("Z (Å)", 64),
    ("Color", 52),
    ("…", 28),
    ("Delete", 48),
)
ARROW_LIST_COLUMNS: Tuple[Tuple[str, int], ...] = (
    ("", 18),
    ("#", 28),
    ("Enabled", 58),
    ("Start point", 0),
    ("End point", 0),
    ("Source / Status", 0),
    ("Color", 52),
    ("…", 28),
    ("Delete", 48),
)


class SpatialItemList:
    """Column header + expandable rows, with a prominent Add control."""

    def __init__(
        self,
        parent,
        *,
        add_text: str,
        add_tip: str,
        on_add: Callable,
        columns: Sequence[Tuple[str, int]],
        empty_hint: str = "No items yet.",
        context: str = "SpatialItemList",
    ):
        QtCore, _, QtWidgets = qt_modules()
        self._columns = tuple(columns)
        self._box = QtWidgets.QWidget(parent)
        layout = QtWidgets.QVBoxLayout(self._box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        add_wrap = QtWidgets.QWidget()
        add_row = QtWidgets.QHBoxLayout(add_wrap)
        add_row.setContentsMargins(0, 0, 0, 0)
        add_row.addStretch(1)
        self._add_btn = QtWidgets.QPushButton(add_text)
        self._add_btn.setToolTip(add_tip)
        self._add_btn.setAutoDefault(False)
        self._add_btn.setAutoFillBackground(True)
        self._add_btn.setStyleSheet(primary_button_css(None))
        expanding = getattr(QtWidgets.QSizePolicy, "Expanding", None)
        fixed = getattr(QtWidgets.QSizePolicy, "Fixed", None)
        if expanding is not None and fixed is not None:
            self._add_btn.setSizePolicy(expanding, fixed)
        hand = getattr(QtCore.Qt, "PointingHandCursor", None)
        if hand is not None:
            self._add_btn.setCursor(hand)
        self._add_btn.clicked.connect(lambda *_: on_add())
        add_row.addWidget(self._add_btn)
        self._add_wrap = add_wrap
        layout.addWidget(add_wrap)

        layout.addWidget(make_spatial_header(self._columns))

        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._inner = QtWidgets.QWidget()
        self._inner.setMinimumWidth(0)
        self._rows = QtWidgets.QVBoxLayout(self._inner)
        self._rows.setContentsMargins(0, 0, 0, 0)
        self._rows.setSpacing(0)
        self._rows.addStretch(1)
        self._scroll.setWidget(self._inner)
        apply_expanding_list_policy(self._box, QtWidgets)
        apply_expanding_list_policy(self._scroll, QtWidgets)
        layout.addWidget(self._scroll, stretch=1)

        self._empty_hint = QtWidgets.QLabel(empty_hint)
        align = getattr(QtCore.Qt, "AlignCenter", None)
        if align is not None:
            self._empty_hint.setAlignment(align)
        self._empty_hint.setStyleSheet("color: %s;" % rgb_css(MUTED))
        layout.addWidget(self._empty_hint)

        apply_required_tooltips(
            [(self._add_btn, add_tip, add_text.strip() or "Add")],
            context=context,
        )
        self._default_add_text = add_text
        self._default_add_tip = add_tip

    @property
    def widget(self):
        return self._box

    @property
    def add_button(self):
        return self._add_btn

    @property
    def columns(self):
        return self._columns

    def set_add_enabled(self, enabled: bool) -> None:
        self._add_btn.setEnabled(bool(enabled))

    def set_add_source_icon(self, source: str) -> None:
        QtCore, QtGui, QtWidgets = qt_modules()
        apply_source_icon(
            self._add_btn, source, QtGui, QtCore, QtWidgets, color=WHITE,
        )

    def set_add_chrome(self, text: str, tip: str) -> None:
        self._add_btn.setText(text)
        self._add_btn.setToolTip(tip)

    def reset_add_chrome(self) -> None:
        self.set_add_chrome(self._default_add_text, self._default_add_tip)

    def attach_add_to_section(self, section) -> None:
        """Move + Add onto the section header bar (mockup placement)."""
        if section is None or not hasattr(section, "add_header_widget"):
            return
        section.add_header_widget(self._add_btn)
        if self._add_wrap is not None:
            self._add_wrap.hide()
        _, _, QtWidgets = qt_modules()
        preferred = getattr(QtWidgets.QSizePolicy, "Preferred", None)
        fixed = getattr(QtWidgets.QSizePolicy, "Fixed", None)
        if preferred is not None and fixed is not None:
            self._add_btn.setSizePolicy(preferred, fixed)

    def clear(self) -> None:
        while self._rows.count() > 1:
            item = self._rows.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def add_block(self, widget) -> None:
        self._rows.insertWidget(self._rows.count() - 1, widget)

    def set_empty(self, empty: bool) -> None:
        self._empty_hint.setVisible(bool(empty))


def make_spatial_header(columns: Sequence[Tuple[str, int]]):
    QtCore, _, QtWidgets = qt_modules()
    row = QtWidgets.QFrame()
    row.setObjectName("pmvSpatialHeader")
    row.setStyleSheet(
        "QFrame#pmvSpatialHeader { background: %s; border-radius: 4px; }"
        % rgb_css(HEADER)
    )
    layout = QtWidgets.QHBoxLayout(row)
    layout.setContentsMargins(6, 4, 4, 4)
    layout.setSpacing(6)
    for title, width in columns:
        label = QtWidgets.QLabel(title)
        label.setStyleSheet(
            "font-weight: 600; color: %s;" % rgb_css(HEADER_INK)
        )
        _place_column(layout, label, width)
    return row


def make_spatial_item_row(
    *,
    columns: Sequence[Tuple[str, int]],
    selected: bool,
    enabled: bool,
    index_text: str,
    values: Sequence[str],
    color,
    on_enabled: Callable,
    on_select: Callable,
    on_delete: Callable,
    on_color: Optional[Callable] = None,
    end_color=None,
    color_tip: Optional[str] = None,
    overflow_actions: Sequence = (),
    context: str = "SpatialItemRow",
):
    """Collapsed row aligned to ``columns``: #, enabled, labels, color, overflow, delete."""
    QtCore, _, QtWidgets = qt_modules()
    row = QtWidgets.QFrame()
    row.setFrameShape(QtWidgets.QFrame.NoFrame)
    row.setCursor(QtCore.Qt.PointingHandCursor)
    if selected:
        row.setStyleSheet("QFrame { background: %s; }" % rgb_css(SELECTED))
    else:
        row.setStyleSheet("QFrame { background: transparent; }")

    layout = QtWidgets.QHBoxLayout(row)
    layout.setContentsMargins(6, 4, 4, 4)
    layout.setSpacing(6)

    chevron = QtWidgets.QLabel(CHEVRON_OPEN if selected else CHEVRON_CLOSED)
    chevron.setStyleSheet("color: %s; font-weight: 600;" % rgb_css(INK if selected else MUTED))
    _place_column(layout, chevron, columns[0][1])

    idx = QtWidgets.QLabel(str(index_text))
    idx.setStyleSheet("color: %s;" % rgb_css(MUTED))
    _place_column(layout, idx, columns[1][1])

    toggle = make_switch("", compact=True)
    toggle.setChecked(bool(enabled))
    toggle.toggled.connect(lambda checked: on_enabled(bool(checked)))
    _place_column(layout, toggle, columns[2][1])

    data_cols = columns[3:-3]
    for (title, width), text in zip(data_cols, values):
        cell = QtWidgets.QLabel(str(text))
        if title in ("Label", "Start point", "End point"):
            cell.setStyleSheet("font-weight: 600; color: %s;" % rgb_css(INK))
        else:
            cell.setStyleSheet(muted_label_css())
        _place_column(layout, cell, width)

    color_btn = QtWidgets.QPushButton()
    color_btn.setFixedSize(28, 18)
    color_btn.setFlat(True)
    color_btn.setAutoDefault(False)
    color_btn.setDefault(False)
    _apply_row_swatch(color_btn, color, end_color)
    if on_color is not None:
        color_btn.clicked.connect(lambda *_: on_color())
    _place_column(layout, color_btn, columns[-3][1])

    overflow = QtWidgets.QToolButton()
    overflow.setText("…")
    overflow.setAutoRaise(True)
    overflow.setToolTip(OVERFLOW_TIP)
    if overflow_actions:
        menu = QtWidgets.QMenu(overflow)
        for spec in overflow_actions:
            label, callback = spec[0], spec[1]
            enabled_act = spec[2] if len(spec) > 2 else True
            act = menu.addAction(label)
            act.setEnabled(bool(enabled_act))
            if callback is not None:
                act.triggered.connect(callback)
        overflow.setMenu(menu)
        popup = getattr(QtWidgets.QToolButton, "InstantPopup", None)
        if popup is not None:
            overflow.setPopupMode(popup)
    _place_column(layout, overflow, columns[-2][1])

    delete = QtWidgets.QPushButton("×")
    delete.setFlat(True)
    delete.setFixedWidth(22)
    delete.clicked.connect(lambda *_: on_delete())
    _place_column(layout, delete, columns[-1][1])

    def on_press(event):
        if event.button() == QtCore.Qt.LeftButton:
            on_select()
        QtWidgets.QFrame.mousePressEvent(row, event)

    row.mousePressEvent = on_press
    apply_required_tooltips(
        [
            (toggle, ENABLE_ITEM_TIP, "Enable"),
            (chevron, EXPAND_ITEM_TIP, "Expand"),
            (idx, EXPAND_ITEM_TIP, "Expand"),
            (color_btn, color_tip or COLOR_ITEM_TIP, "Color"),
            (overflow, OVERFLOW_TIP, "More"),
            (delete, DELETE_ITEM_TIP, "Delete"),
        ],
        context=context,
    )
    return row


def _place_column(layout, widget, width: int) -> None:
    if width:
        widget.setFixedWidth(int(width))
        layout.addWidget(widget)
    else:
        layout.addWidget(widget, stretch=1)


def _apply_row_swatch(button, color, end_color=None) -> None:
    QtCore, QtGui, _QtWidgets = qt_modules()
    w, h = 28, 18
    inset = 1
    if end_color is None or QtGui is None or QtCore is None:
        button.setStyleSheet(
            swatch_button_css(rgb_to_css(color), extra=" min-width: 28px; max-width: 28px; min-height: 18px; max-height: 18px;")
        )
        return
    inner_w, inner_h = w - 2 * inset, h - 2 * inset
    pix = QtGui.QPixmap(inner_w, inner_h)
    pix.fill(QtGui.QColor(0, 0, 0, 0))
    painter = QtGui.QPainter(pix)
    try:
        aa = getattr(QtGui.QPainter, "Antialiasing", None)
        if aa is not None:
            painter.setRenderHint(aa, True)
        start_c = qcolor(QtGui, color)
        end_c = qcolor(QtGui, end_color)
        no_pen = getattr(QtCore.Qt, "NoPen", None)
        if no_pen is not None:
            painter.setPen(no_pen)
        painter.setBrush(QtGui.QBrush(start_c))
        painter.drawPolygon(_swatch_poly(QtGui, QtCore, ((0, 0), (inner_w, 0), (0, inner_h))))
        painter.setBrush(QtGui.QBrush(end_c))
        painter.drawPolygon(_swatch_poly(QtGui, QtCore, ((inner_w, 0), (inner_w, inner_h), (0, inner_h))))
    finally:
        painter.end()
    button.setIcon(QtGui.QIcon(pix))
    button.setIconSize(QtCore.QSize(inner_w, inner_h))
    button.setStyleSheet(
        "QPushButton { border: 1px solid %s; border-radius: 3px; padding: 0px; margin: 0px;"
        " min-width: 28px; max-width: 28px; min-height: 18px; max-height: 18px; }"
        "QPushButton:hover { border: 1px solid %s; }"
        % (rgb_css(BORDER), rgb_css(INK))
    )


def _swatch_poly(QtGui, QtCore, points):
    pts = [QtCore.QPointF(float(x), float(y)) for x, y in points]
    try:
        return QtGui.QPolygonF(pts)
    except TypeError:
        poly = QtGui.QPolygonF()
        for pt in pts:
            poly.append(pt)
        return poly


def make_spatial_detail(QtWidgets):
    """Full-width inset under an expanded row, aligned to the table."""
    box = QtWidgets.QFrame()
    box.setObjectName("pmvSpatialDetail")
    box.setStyleSheet(
        "QFrame#pmvSpatialDetail { background: %s; border: none;"
        " border-top: 1px solid %s; }" % (rgb_css(DETAIL), rgb_css(HEADER))
    )
    preferred = getattr(QtWidgets.QSizePolicy, "Preferred", None)
    ignored = getattr(QtWidgets.QSizePolicy, "Ignored", None)
    if preferred is not None and ignored is not None:
        box.setSizePolicy(ignored, preferred)
    box.setMinimumWidth(0)
    layout = QtWidgets.QVBoxLayout(box)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(6)
    return box, layout
