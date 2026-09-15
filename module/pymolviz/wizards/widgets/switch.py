"""On/off switch look for boolean toggles (QCheckBox API)."""

from __future__ import annotations

from ..pick import qt_modules
from .theme import (
    DASH,
    INK,
    MUTED,
    PAGE,
    PRIMARY,
    PRIMARY_HOVER,
    WHITE,
    mix_rgb,
    qcolor,
)
from .type_icons import action_icon_pixmap

TRACK_W = 36
TRACK_H = 20
KNOB = 16
GAP = 8
COMPACT_TRACK_W = 28
COMPACT_TRACK_H = 16
COMPACT_KNOB = 12
ICON_SIZE = 16

_INDICATOR_OFF_CSS = (
    "QCheckBox { spacing: 0px; background: transparent; border: none; }"
    "QCheckBox::indicator { width: 0px; height: 0px; border: none; }"
)


def track_metrics(compact=False):
    """``(width, height, knob)`` in logical pixels."""
    if compact:
        return COMPACT_TRACK_W, COMPACT_TRACK_H, COMPACT_KNOB
    return TRACK_W, TRACK_H, KNOB


def make_switch(text="", parent=None, *, compact=False, icon=None):
    """QCheckBox painted as a sliding switch (same ``isChecked`` / ``toggled`` API)."""
    QtCore, _QtGui, QtWidgets = qt_modules()
    if QtWidgets is None or not hasattr(QtWidgets, "QCheckBox"):
        raise RuntimeError("PyMOL Qt UI required")
    box = QtWidgets.QCheckBox(text or "", parent)
    box._pmv_icon = str(icon) if icon else None
    _apply_switch_look(box, compact=compact)
    return box


def _apply_switch_look(box, compact=False):
    QtCore, _QtGui, QtWidgets = qt_modules()
    box._compact = bool(compact)
    box.setStyleSheet(_INDICATOR_OFF_CSS)
    hand = _qt_enum(QtCore, "PointingHandCursor")
    if hand is not None:
        box.setCursor(hand)
    hover = _qt_enum(QtCore, "WA_Hover")
    if hover is not None:
        box.setAttribute(hover, True)
    policy = getattr(QtWidgets, "QSizePolicy", None)
    if policy is not None:
        h_pol = getattr(policy, "Fixed" if compact else "Minimum", None)
        v_pol = getattr(policy, "Fixed", None)
        if h_pol is not None and v_pol is not None:
            box.setSizePolicy(h_pol, v_pol)
    _update_switch_size(box)
    box.toggled.connect(lambda *_: box.update())

    class _SwitchFilter(QtCore.QObject):
        def eventFilter(self, obj, event):
            qevent = getattr(QtCore, "QEvent", None)
            if obj is None or event is None or qevent is None:
                return False
            et = event.type()
            paint = getattr(qevent, "Paint", None)
            if paint is not None and et == paint:
                _paint_switch_widget(obj)
                return True
            press = getattr(qevent, "MouseButtonPress", None)
            left = _qt_enum(QtCore, "LeftButton")
            if press is not None and et == press:
                button = getattr(event, "button", None)
                if callable(button):
                    button = button()
                if left is not None and button not in (None, left):
                    return False
                if obj.isEnabled():
                    obj.toggle()
                return True
            dbl = getattr(qevent, "MouseButtonDblClick", None)
            if dbl is not None and et == dbl:
                return True
            return False

    filt = _SwitchFilter(box)
    box.installEventFilter(filt)
    box._pmv_switch_filter = filt


def _update_switch_size(box):
    QtCore, QtGui, _QtWidgets = qt_modules()
    tw, th, _knob = track_metrics(getattr(box, "_compact", False))
    text = box.text() if callable(getattr(box, "text", None)) else ""
    extra_w = 0
    extra_h = 0
    if getattr(box, "_pmv_icon", None):
        extra_w += GAP + ICON_SIZE
        extra_h = max(extra_h, ICON_SIZE)
    if text:
        fm = box.fontMetrics()
        extra_w += GAP + _text_width(fm, text)
        extra_h = max(extra_h, int(fm.height()))
    w = tw + extra_w + 2
    h = max(th, extra_h) + 4
    setter = getattr(box, "setMinimumSize", None)
    if callable(setter):
        setter(w, h)
    if getattr(box, "_compact", False) and not text:
        fixed = getattr(box, "setFixedSize", None)
        if callable(fixed):
            fixed(w, h)


def _paint_switch_widget(widget):
    QtCore, QtGui, _QtWidgets = qt_modules()
    if widget is None or QtGui is None or not hasattr(QtGui, "QPainter"):
        return
    try:
        painter = QtGui.QPainter(widget)
    except Exception:
        return
    if not painter.isActive():
        return
    try:
        aa = getattr(QtGui.QPainter, "Antialiasing", None)
        if aa is None:
            aa = getattr(
                getattr(QtGui.QPainter, "RenderHint", None),
                "Antialiasing",
                None,
            )
        if aa is not None:
            painter.setRenderHint(aa, True)
        _paint_switch(widget, painter, QtCore, QtGui)
    finally:
        if painter.isActive():
            painter.end()


def _qt_enum(QtCore, name):
    qt = getattr(QtCore, "Qt", None)
    if qt is None:
        return None
    return getattr(qt, name, None)


def _text_width(fm, text) -> int:
    if hasattr(fm, "horizontalAdvance"):
        return int(fm.horizontalAdvance(text))
    return int(fm.width(text))


def _paint_switch(widget, painter, QtCore, QtGui):
    tw, th, knob = track_metrics(getattr(widget, "_compact", False))
    rect = widget.contentsRect()
    enabled = bool(widget.isEnabled())
    checked = bool(widget.isChecked())
    hovered = bool(enabled and widget.underMouse())
    cy = rect.y() + rect.height() / 2.0
    track = _qrectf(QtCore, QtGui, rect.x() + 1, cy - th / 2.0, tw, th)
    if track is None:
        return
    track_rgb = _track_rgb(checked, enabled, hovered)
    painter.setPen(_qt_nopen(QtCore, QtGui))
    painter.setBrush(qcolor(QtGui, track_rgb))
    painter.drawRoundedRect(track, th / 2.0, th / 2.0)
    pad = max(1.0, (th - knob) / 2.0)
    if checked:
        kx = track.right() - pad - knob
    else:
        kx = track.left() + pad
    ky = track.top() + pad
    knob_rect = _qrectf(QtCore, QtGui, kx, ky, knob, knob)
    knob_rgb = WHITE if enabled else mix_rgb(WHITE, MUTED, 0.35)
    painter.setBrush(qcolor(QtGui, knob_rgb))
    if knob_rect is not None:
        painter.drawEllipse(knob_rect)
    text = widget.text()
    tx = int(round(track.right() + GAP))
    icon_kind = getattr(widget, "_pmv_icon", None)
    if icon_kind:
        icon_rgb = INK if enabled else MUTED
        pix = action_icon_pixmap(
            icon_kind, QtGui, QtCore, None, size=ICON_SIZE, color=icon_rgb,
        )
        if pix is not None:
            iy = int(round(cy - ICON_SIZE / 2.0))
            painter.drawPixmap(tx, iy, pix)
            tx += ICON_SIZE + GAP
    if not text:
        return
    text_rect = _qrectf(
        QtCore, QtGui, tx, rect.y(), max(0, rect.right() - tx), rect.height(),
    )
    if text_rect is None:
        return
    painter.setPen(qcolor(QtGui, INK if enabled else MUTED))
    align = _qt_enum(QtCore, "AlignVCenter")
    left = _qt_enum(QtCore, "AlignLeft")
    if align is not None and left is not None:
        painter.drawText(text_rect, int(left) | int(align), text)
        return
    painter.drawText(text_rect, text)


def _track_rgb(checked, enabled, hovered):
    if not enabled:
        if checked:
            return mix_rgb(PRIMARY, PAGE, 0.55)
        return mix_rgb(DASH, PAGE, 0.35)
    if checked:
        return PRIMARY_HOVER if hovered else PRIMARY
    if hovered:
        return mix_rgb(DASH, MUTED, 0.45)
    return DASH


def _qrectf(QtCore, QtGui, x, y, w, h):
    cls = getattr(QtCore, "QRectF", None) or getattr(QtGui, "QRectF", None)
    if cls is None:
        return None
    return cls(float(x), float(y), float(w), float(h))


def _qt_nopen(QtCore, QtGui):
    pen_cls = getattr(QtGui, "QPen", None)
    no_pen = _qt_enum(QtCore, "NoPen")
    if pen_cls is None:
        return no_pen
    if no_pen is None:
        try:
            return pen_cls()
        except Exception:
            return None
    try:
        return pen_cls(no_pen)
    except Exception:
        return no_pen
