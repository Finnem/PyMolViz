"""Sticky in-list Add control (same behavior as the Objects table)."""

from __future__ import annotations

from typing import Callable, Optional

from ..pick import qt_modules
from ..tooltips import apply_required_tooltips

ADD_BUTTON_STYLE = (
    "QPushButton { text-align: center; padding: 5px 12px; font-weight: 600;"
    " border: 1px solid palette(dark); border-radius: 4px;"
    " background: palette(button); }"
    "QPushButton:hover { background: palette(midlight);"
    " border: 1px solid palette(highlight); }"
    "QPushButton:pressed { background: palette(mid); padding-top: 6px;"
    " padding-bottom: 4px; }"
)


def list_needs_sticky_add(n_items, viewport_height, row_height, add_height=None):
    """True when items plus an Add control do not fit in the viewport."""
    if viewport_height <= 0 or row_height <= 0:
        return False
    extra = int(row_height if add_height is None else add_height)
    if extra <= 0:
        extra = int(row_height)
    return int(n_items) * int(row_height) + extra > int(viewport_height)


def sticky_add_overlay_rect(
    table_width,
    table_height,
    viewport_x,
    viewport_y,
    viewport_height,
    n_items,
    row_height,
    add_height,
    header_height=0,
    h_scrollbar_height=0,
    inset=6,
):
    """Add-button rect in list coordinates, always inside the list widget.

    Returns ``(x, y, w, h, bottom_margin)``. ``bottom_margin`` is space to
    reserve under the viewport so rows do not sit under the button — 0 when
    the list is too short to spare that space.
    """
    table_w = max(int(table_width), 0)
    table_h = max(int(table_height), 0)
    inset = max(int(inset), 0)
    add_h = max(int(add_height), 1)
    sb_h = max(int(h_scrollbar_height), 0)
    header_h = max(int(header_height), 0)
    n_items = max(int(n_items), 0)
    row_h = max(int(row_height), 0)
    usable_h = max(table_h - sb_h, 0)
    if table_w <= 0 or usable_h <= 0:
        return (0, 0, max(table_w, 1), 1, 0)

    h = min(add_h, usable_h)
    x = inset if table_w > 2 * inset else 0
    w = max(table_w - x - min(inset, table_w - x), 1)
    max_y = max(usable_h - h, 0)
    pin = list_needs_sticky_add(
        n_items, int(viewport_height), row_h, add_height=add_h,
    )
    inline_y = int(viewport_y) + n_items * row_h
    if pin or inline_y > max_y:
        y = max_y
        max_margin = max(usable_h - header_h - 1, 0)
        bottom_margin = min(h, max_margin)
    else:
        y = inline_y
        bottom_margin = 0
    y = max(0, min(int(y), max_y))
    return (int(x), y, int(w), int(h), int(bottom_margin))


class StickyAddOverlay:
    """``+ Add …`` bar that sits in a list and pins to the bottom when it overflows."""

    def __init__(
        self,
        parent,
        target,
        text: str,
        tooltip: str,
        on_click: Callable[[], None],
        count: Callable[[], int],
        row_height: Optional[Callable[[], int]] = None,
        add_height: Optional[Callable[[], int]] = None,
        context: str = "StickyAddOverlay",
        extra=None,
    ):
        QtCore, _, QtWidgets = qt_modules()
        self._target = target
        self._count = count
        self._row_height_fn = row_height
        self._add_height_fn = add_height
        self._syncing = False
        self._filter = None

        bar = QtWidgets.QWidget(parent)
        layout = QtWidgets.QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        btn = QtWidgets.QPushButton(text)
        btn.setToolTip(tooltip)
        btn.setAutoDefault(False)
        btn.setDefault(False)
        btn.setAutoFillBackground(True)
        no_focus = getattr(QtCore.Qt, "NoFocus", None)
        if no_focus is not None:
            btn.setFocusPolicy(no_focus)
            bar.setFocusPolicy(no_focus)
        hand = getattr(QtCore.Qt, "PointingHandCursor", None)
        if hand is not None:
            btn.setCursor(hand)
        btn.setStyleSheet(ADD_BUTTON_STYLE)
        btn.clicked.connect(lambda *_args: on_click())
        layout.addWidget(btn, stretch=1)
        if extra is not None:
            extra.setParent(bar)
            extra.setVisible(False)
            layout.addWidget(extra)
        bar.hide()
        self._bar = bar
        self._button = btn
        self._extra = extra
        apply_required_tooltips(
            [(btn, tooltip, text.strip() or "Add")],
            context=context,
        )

    @property
    def widget(self):
        return self._bar

    @property
    def button(self):
        return self._button

    def set_text(self, text: str):
        self._button.setText(text)

    def set_tooltip(self, text: str):
        self._button.setToolTip(text)

    def set_extra_visible(self, visible: bool):
        if self._extra is not None:
            self._extra.setVisible(bool(visible))

    def attach(self):
        QtCore, _, _ = qt_modules()
        overlay = self

        class _ResizeFilter(QtCore.QObject):
            def eventFilter(inner, watched, event):
                try:
                    etype = event.type()
                    resize = getattr(QtCore.QEvent, "Resize", None)
                    layout_request = getattr(QtCore.QEvent, "LayoutRequest", None)
                    if resize is None:
                        resize = getattr(getattr(QtCore.QEvent, "Type", None), "Resize", None)
                    if layout_request is None:
                        layout_request = getattr(
                            getattr(QtCore.QEvent, "Type", None), "LayoutRequest", None,
                        )
                    if etype in (resize, layout_request):
                        overlay.sync()
                except Exception:
                    pass
                return False

        parent = self._bar.parentWidget()
        filt = _ResizeFilter(parent if parent is not None else self._target)
        self._filter = filt
        if parent is not None:
            parent.installEventFilter(filt)
        self._target.installEventFilter(filt)
        viewport = getattr(self._target, "viewport", None)
        if callable(viewport):
            vp = viewport()
            if vp is not None:
                vp.installEventFilter(filt)
        try:
            bar = self._target.verticalScrollBar()
            if bar is not None:
                bar.rangeChanged.connect(lambda *_args: overlay.sync())
        except Exception:
            pass
        self.sync()

    def sync(self):
        if self._syncing:
            return
        target = self._target
        bar = self._bar
        if target is None or bar is None:
            return
        self._syncing = True
        try:
            self._place()
        except Exception:
            pass
        finally:
            self._syncing = False

    def _row_height(self) -> int:
        if self._row_height_fn is not None:
            try:
                height = int(self._row_height_fn())
                if height > 0:
                    return height
            except Exception:
                pass
        target = self._target
        try:
            if int(target.rowCount()) > 0:
                height = int(target.rowHeight(0))
                if height > 0:
                    return height
        except Exception:
            pass
        try:
            return max(int(target.verticalHeader().defaultSectionSize()), 1)
        except Exception:
            return 28

    def _add_height(self) -> int:
        if self._add_height_fn is not None:
            try:
                height = int(self._add_height_fn())
                if height > 0:
                    return height
            except Exception:
                pass
        return max(self._row_height() + 8, 30)

    def _header_height(self) -> int:
        try:
            header = self._target.horizontalHeader()
            if header is not None and header.isVisible():
                return int(header.height())
        except Exception:
            pass
        return 0

    def _h_scrollbar_height(self) -> int:
        try:
            sb = self._target.horizontalScrollBar()
            if sb is not None and sb.isVisible() and int(sb.maximum()) > 0:
                return max(int(sb.height()), 0)
        except Exception:
            pass
        return 0

    def _viewport_height(self) -> int:
        height = int(self._target.height())
        height -= self._header_height()
        height -= self._h_scrollbar_height()
        return max(height, 0)

    def _place(self):
        QtCore, _, _ = qt_modules()
        target = self._target
        overlay = self._bar
        viewport = target.viewport() if callable(getattr(target, "viewport", None)) else target
        geo = viewport.geometry()
        n_items = max(int(self._count()), 0)
        x, y, w, h, bottom = sticky_add_overlay_rect(
            target.width(),
            target.height(),
            int(geo.x()),
            int(geo.y()),
            self._viewport_height(),
            n_items,
            self._row_height(),
            self._add_height(),
            header_height=self._header_height(),
            h_scrollbar_height=self._h_scrollbar_height(),
        )
        try:
            target.setViewportMargins(0, 0, 0, bottom)
        except Exception:
            pass
        ox, oy = 0, 0
        parent = overlay.parentWidget()
        if parent is not None and parent is not target and QtCore is not None:
            try:
                origin = target.mapTo(parent, QtCore.QPoint(0, 0))
                ox, oy = int(origin.x()), int(origin.y())
            except Exception:
                ox, oy = 0, 0
        gx, gy = ox + x, oy + y
        if parent is not None and parent is not target:
            try:
                crect = parent.contentsRect()
                max_h = max(int(crect.height()), 1)
                h = min(h, max_h)
                gy = max(int(crect.y()), min(gy, int(crect.y() + crect.height() - h)))
                max_w = max(int(crect.width()), 1)
                w = min(w, max_w)
                gx = max(int(crect.x()), min(gx, int(crect.x() + crect.width() - w)))
            except Exception:
                pass
        overlay.setGeometry(gx, gy, w, h)
        overlay.show()
        overlay.raise_()
