"""Stylized global ``o---->`` control: ticks (margin), shaft (dash), two heads."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

from ...util.line_style import DASH_PRESETS, HEAD_STYLES, LineStyle, MAX_ARROW_MARGIN
from ..pick import qt_modules
from ..tooltips import apply_required_tooltips

ARROW_TYPE_TIP = (
    "Collection style. Drag the ticks to inset both ends, click the shaft to "
    "choose a dash, click a head to choose that end's cap."
)
CIRCLE_TIP = "Start point. The ticks slide away from the endpoints as margin grows."
SHAFT_TIP = "Click to choose a solid or dashed shaft."
HEAD_START_TIP = "Click to choose the start cap: none, arrow, or circle."
HEAD_END_TIP = "Click to choose the end cap: none, arrow, or circle."
TICK_TIP = "Drag to inset the arrow from both endpoints (margin)."

_PAD = 6.0
_CIRCLE_R = 5.5
_HEAD_LEN = 14.0
_HEAD_HALF = 6.0
_TICK_SLOP_X = 5.0
_END_GAP = 3.0
_KNOB = 4.5
_MAX_INSET_FRAC = 0.38
_HEAD_CIRCLE = 4.0
_MIN_SHAFT = 4.0


def arrow_type_geometry(width, height, margin, max_margin=MAX_ARROW_MARGIN):
    """Pixel layout for the stylized ``o---->`` control.

    Circles at the ends are the *points* and stay put. Both head slots are
    always reserved so each side is clickable. Ticks at the head tips slide
    inward as margin grows.
    """
    width = max(float(width), 1.0)
    height = max(float(height), 1.0)
    y = max(height - _PAD - _CIRCLE_R, _PAD + 18.0)
    circle_x = _PAD + _CIRCLE_R
    end_x = width - _PAD - _CIRCLE_R
    inner = (
        end_x - circle_x
        - 2.0 * _CIRCLE_R
        - 2.0 * _END_GAP
        - 2.0 * _HEAD_LEN
        - _MIN_SHAFT
    )
    usable = max(inner, 1.0)
    cap = max(float(max_margin), 1e-9)
    t = min(max(float(margin) / cap, 0.0), 1.0)
    inset = t * usable * _MAX_INSET_FRAC
    start_tip = circle_x + _CIRCLE_R + _END_GAP + inset
    end_tip = end_x - _CIRCLE_R - _END_GAP - inset
    start_base = start_tip + _HEAD_LEN
    end_base = end_tip - _HEAD_LEN
    if end_base < start_base + _MIN_SHAFT:
        mid = 0.5 * (start_tip + end_tip)
        start_base = mid - _MIN_SHAFT * 0.5
        end_base = mid + _MIN_SHAFT * 0.5
        start_tip = start_base - _HEAD_LEN
        end_tip = end_base + _HEAD_LEN
    shaft0 = start_base + 1.0
    shaft1 = max(shaft0, end_base - 1.0)
    tick_top = _PAD
    tick_bottom = y + 5.0
    return {
        "y": y,
        "circle_x": circle_x,
        "end_circle_x": end_x,
        "circle_r": _CIRCLE_R,
        "shaft0": shaft0,
        "shaft1": shaft1,
        "start_tip": start_tip,
        "start_base": start_base,
        "end_base": end_base,
        "end_tip": end_tip,
        "head_base": end_base,
        "head_tip": end_tip,
        "head_half": _HEAD_HALF,
        "tick0": start_tip,
        "tick1": end_tip,
        "tick_top": tick_top,
        "tick_bottom": tick_bottom,
        "knob": _KNOB,
        "end_x": end_x,
    }


def _near_tick(x, y, tx, tick_top, tick_bottom):
    return (
        abs(float(x) - tx) <= _TICK_SLOP_X
        and tick_top - 2.0 <= float(y) <= tick_bottom + 2.0
    )


def hit_arrow_type_part(
    x, y, width, height, margin, max_margin=MAX_ARROW_MARGIN,
) -> str:
    """Return ``tick0``, ``tick1``, ``circle``, ``shaft``, ``head_start``, ``head_end``, or empty."""
    g = arrow_type_geometry(width, height, margin, max_margin)
    y0 = g["y"]
    r_hit = g["circle_r"] + 1.5
    dx = float(x) - g["circle_x"]
    dy = float(y) - y0
    if dx * dx + dy * dy <= r_hit * r_hit:
        return "circle"
    edx = float(x) - g["end_circle_x"]
    if edx * edx + dy * dy <= r_hit * r_hit:
        return "circle"
    if _near_tick(x, y, g["tick0"], g["tick_top"], g["tick_bottom"]):
        return "tick0"
    if _near_tick(x, y, g["tick1"], g["tick_top"], g["tick_bottom"]):
        return "tick1"
    px = float(x)
    if g["start_tip"] - 2.0 <= px <= g["start_base"] + 2.0:
        return "head_start"
    if g["end_base"] - 2.0 <= px <= g["end_tip"] + 2.0:
        return "head_end"
    if g["shaft0"] - 2.0 <= px <= g["shaft1"] + 2.0:
        return "shaft"
    return ""


def part_highlight_rect(g, part):
    """Axis-aligned ``(x, y, w, h)`` for hover/section chrome, or None."""
    y = g["y"]
    if part == "shaft":
        width = max(g["shaft1"] - g["shaft0"], 1.0)
        return (g["shaft0"], y - 9.0, width, 18.0)
    if part == "head_start":
        return (
            g["start_tip"] - 2.0,
            y - g["head_half"] - 5.0,
            g["start_base"] - g["start_tip"] + 6.0,
            g["head_half"] * 2.0 + 10.0,
        )
    if part in ("head_end", "head"):
        return (
            g["end_base"] - 2.0,
            y - g["head_half"] - 5.0,
            g["end_tip"] - g["end_base"] + 6.0,
            g["head_half"] * 2.0 + 10.0,
        )
    if part in ("tick0", "tick1"):
        tx = g[part]
        return (tx - 6.0, g["tick_top"] - 1.0, 12.0, g["tick_bottom"] - g["tick_top"] + 2.0)
    return None


def _event_xy(event):
    if hasattr(event, "position"):
        pos = event.position()
        return float(pos.x()), float(pos.y())
    return float(event.x()), float(event.y())


def margin_from_tick0(x, width, height, max_margin=MAX_ARROW_MARGIN) -> float:
    """Map the left tick x position back to a margin in Å."""
    g0 = arrow_type_geometry(width, height, 0.0, max_margin)
    g1 = arrow_type_geometry(width, height, max_margin, max_margin)
    span = g1["tick0"] - g0["tick0"]
    if span <= 1e-9:
        return 0.0
    t = (float(x) - g0["tick0"]) / span
    return min(max(t, 0.0), 1.0) * float(max_margin)


def margin_from_tick1(x, width, height, max_margin=MAX_ARROW_MARGIN) -> float:
    """Map the right tick x position back to a margin in Å."""
    g0 = arrow_type_geometry(width, height, 0.0, max_margin)
    g1 = arrow_type_geometry(width, height, max_margin, max_margin)
    span = g0["tick1"] - g1["tick1"]
    if span <= 1e-9:
        return 0.0
    t = (g0["tick1"] - float(x)) / span
    return min(max(t, 0.0), 1.0) * float(max_margin)


class ArrowTypeControl:
    """Collection-level ``o---->`` editor: ticks, dash, and independent end caps."""

    def __init__(
        self,
        parent,
        style: Optional[LineStyle] = None,
        width: float = 0.045,
        color: Optional[Tuple[float, float, float]] = None,
        on_change: Optional[Callable[[LineStyle], None]] = None,
        max_margin: float = MAX_ARROW_MARGIN,
    ):
        QtCore, _, QtWidgets = qt_modules()
        self._style = (style or LineStyle()).copy()
        self._width = float(width)
        self._color = color
        self._on_change = on_change
        self._max_margin = max(float(max_margin), 1e-9)
        self._drag = None
        self._drag_offset = 0.0
        self._hover = ""
        self._widget = QtWidgets.QWidget(parent)
        self._widget.setMinimumHeight(44)
        self._widget.setMaximumHeight(50)
        self._widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self._widget.setMouseTracking(True)
        self._widget.paintEvent = self._paint
        self._widget.mousePressEvent = self._mouse_press
        self._widget.mouseMoveEvent = self._mouse_move
        self._widget.mouseReleaseEvent = self._mouse_release
        self._widget.leaveEvent = self._leave
        apply_required_tooltips(
            [(self._widget, ARROW_TYPE_TIP, "Arrow style")],
            context="ArrowTypeControl",
        )

    @property
    def widget(self):
        return self._widget

    def style(self) -> LineStyle:
        return self._style.copy()

    def set_style(self, style: LineStyle):
        if style is None:
            return
        self._style = style.copy()
        self._widget.update()

    def set_margin(self, margin: float):
        self._style = self._style.updated(
            margin=min(max(float(margin), 0.0), self._max_margin),
        )
        self._widget.update()

    def set_max_margin(self, value: float):
        self._max_margin = max(float(value), 1e-9)
        if self._style.margin > self._max_margin:
            self._style = self._style.updated(margin=self._max_margin)
        self._widget.update()

    def _emit(self):
        self._widget.update()
        if self._on_change is not None:
            self._on_change(self._style.copy())

    def _geometry(self):
        rect = self._widget.rect()
        return arrow_type_geometry(
            rect.width(), rect.height(), self._style.margin, self._max_margin,
        )

    def _paint(self, event):
        QtCore, QtGui, _ = qt_modules()
        painter = QtGui.QPainter(self._widget)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.fillRect(self._widget.rect(), self._widget.palette().base())
        g = self._geometry()
        y = g["y"]
        highlight = self._widget.palette().color(self._widget.palette().Highlight)
        mid = self._widget.palette().color(self._widget.palette().Midlight)
        hover = self._hover
        for part, alpha in (("shaft", 28), ("head_start", 28), ("head_end", 28)):
            if hover == part:
                continue
            self._fill_rect(painter, QtCore, QtGui, part_highlight_rect(g, part), mid, alpha)
        if hover in ("shaft", "head_start", "head_end", "tick0", "tick1"):
            self._fill_rect(painter, QtCore, QtGui, part_highlight_rect(g, hover), highlight, 70)
        if self._color is not None:
            r, gch, b = self._color
            ink = QtGui.QColor(int(r * 255), int(gch * 255), int(b * 255))
        else:
            ink = self._widget.palette().color(self._widget.palette().WindowText)
        pen_w = max(1.6, min(4.0, self._width * 40.0))
        pen = QtGui.QPen(ink, pen_w)
        pen.setCapStyle(QtCore.Qt.FlatCap)
        pen.setJoinStyle(QtCore.Qt.RoundJoin)
        pattern = self._style.pattern()
        if len(pattern) < 2 or pattern[1] <= 1e-6:
            pen.setStyle(QtCore.Qt.SolidLine)
        else:
            scale = max(self._style.dash_scale, 0.15) * 18.0
            pen.setStyle(QtCore.Qt.CustomDashLine)
            pen.setDashPattern([max(float(v) * scale, 0.4) for v in pattern])
        painter.setPen(pen)
        shaft0, shaft1 = g["shaft0"], g["shaft1"]
        if shaft1 > shaft0:
            painter.drawLine(int(shaft0), int(y), int(shaft1), int(y))

        self._draw_cap(
            painter, QtGui, QtCore, ink,
            self._style.start_head, g["start_tip"], y, -1, g["head_half"],
        )
        self._draw_cap(
            painter, QtGui, QtCore, ink,
            self._style.end_head, g["end_tip"], y, 1, g["head_half"],
        )

        tick_pen = QtGui.QPen(ink, 1.8 if hover in ("tick0", "tick1") else 1.5)
        tick_pen.setStyle(QtCore.Qt.SolidLine)
        tick_pen.setCapStyle(QtCore.Qt.FlatCap)
        painter.setPen(tick_pen)
        knob = g["knob"]
        for name in ("tick0", "tick1"):
            tx = g[name]
            painter.drawLine(int(tx), int(g["tick_top"] + knob), int(tx), int(min(g["y"], g["tick_bottom"])))
            painter.setBrush(ink if hover == name else self._widget.palette().base())
            painter.drawEllipse(QtCore.QPointF(tx, g["tick_top"] + knob), knob, knob)

        head_pen = QtGui.QPen(ink, 1.4)
        head_pen.setStyle(QtCore.Qt.SolidLine)
        painter.setPen(head_pen)
        painter.setBrush(ink)
        painter.drawEllipse(QtCore.QPointF(g["circle_x"], y), g["circle_r"], g["circle_r"])
        painter.setBrush(self._widget.palette().base())
        painter.drawEllipse(QtCore.QPointF(g["end_circle_x"], y), g["circle_r"], g["circle_r"])
        painter.end()

    def _fill_rect(self, painter, QtCore, QtGui, rect, color, alpha):
        if rect is None:
            return
        x, y, w, h = rect
        if w <= 0 or h <= 0:
            return
        fill = QtGui.QColor(color)
        fill.setAlpha(int(alpha))
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(fill)
        painter.drawRoundedRect(QtCore.QRectF(x, y, w, h), 4.0, 4.0)

    def _draw_head(self, painter, QtGui, QtCore, tip_x, y, direction, half):
        painter.drawPolygon(QtGui.QPolygonF([
            QtCore.QPointF(tip_x, y),
            QtCore.QPointF(tip_x - _HEAD_LEN * direction, y - half),
            QtCore.QPointF(tip_x - _HEAD_LEN * direction, y + half),
        ]))

    def _draw_cap(self, painter, QtGui, QtCore, ink, kind, tip_x, y, direction, half):
        head_pen = QtGui.QPen(ink, 1.4)
        head_pen.setStyle(QtCore.Qt.SolidLine)
        head_pen.setCapStyle(QtCore.Qt.RoundCap)
        if kind == "Arrow":
            painter.setPen(head_pen)
            painter.setBrush(ink)
            self._draw_head(painter, QtGui, QtCore, tip_x, y, direction, half)
            return
        if kind == "Circles":
            painter.setPen(head_pen)
            painter.setBrush(ink)
            painter.drawEllipse(QtCore.QPointF(tip_x, y), _HEAD_CIRCLE, _HEAD_CIRCLE)
            return
        ghost = QtGui.QColor(ink)
        ghost.setAlpha(110)
        painter.setPen(QtGui.QPen(ghost, 1.2))
        painter.setBrush(QtCore.Qt.NoBrush)
        self._draw_head(painter, QtGui, QtCore, tip_x, y, direction, half)

    def _mouse_press(self, event):
        QtCore, _, QtWidgets = qt_modules()
        if event.button() != QtCore.Qt.LeftButton:
            return
        x, y = _event_xy(event)
        part = hit_arrow_type_part(
            x, y,
            self._widget.width(), self._widget.height(),
            self._style.margin, self._max_margin,
        )
        if part in ("tick0", "tick1"):
            g = self._geometry()
            self._drag = part
            self._set_hover(part)
            self._drag_offset = x - g[part]
            self._apply_tick(part, x)
            return
        if part == "shaft":
            self._popup(event, DASH_PRESETS, self._pick_dash, SHAFT_TIP)
            return
        items = [(name, name) for name in HEAD_STYLES]
        if part == "head_start":
            self._popup(event, items, self._pick_start_head, HEAD_START_TIP)
            return
        if part in ("head_end", "head"):
            self._popup(event, items, self._pick_end_head, HEAD_END_TIP)

    def _mouse_move(self, event):
        QtCore, _, QtWidgets = qt_modules()
        if self._drag:
            self._apply_tick(self._drag, _event_xy(event)[0])
            return
        x, y = _event_xy(event)
        part = hit_arrow_type_part(
            x, y,
            self._widget.width(), self._widget.height(),
            self._style.margin, self._max_margin,
        )
        tips = {
            "tick0": TICK_TIP,
            "tick1": TICK_TIP,
            "circle": CIRCLE_TIP,
            "shaft": SHAFT_TIP,
            "head_start": HEAD_START_TIP,
            "head_end": HEAD_END_TIP,
            "head": HEAD_END_TIP,
        }
        self._widget.setToolTip(tips.get(part, ARROW_TYPE_TIP))
        self._set_hover(part)
        if part in ("tick0", "tick1"):
            self._widget.setCursor(QtCore.Qt.SizeHorCursor)
        elif part in ("shaft", "head_start", "head_end", "head"):
            self._widget.setCursor(QtCore.Qt.PointingHandCursor)
        else:
            self._widget.setCursor(QtCore.Qt.ArrowCursor)

    def _mouse_release(self, event):
        self._drag = None

    def _leave(self, event):
        self._set_hover("")

    def _set_hover(self, part):
        part = part or ""
        if part == self._hover:
            return
        self._hover = part
        self._widget.update()

    def _apply_tick(self, which, x):
        mapped = float(x) - self._drag_offset
        args = (mapped, self._widget.width(), self._widget.height(), self._max_margin)
        if which == "tick1":
            margin = margin_from_tick1(*args)
        else:
            margin = margin_from_tick0(*args)
        if abs(margin - self._style.margin) < 1e-4:
            return
        self._style = self._style.updated(margin=margin)
        self._emit()

    def _popup(self, event, items, on_pick, tooltip):
        _, _, QtWidgets = qt_modules()
        menu = QtWidgets.QMenu(self._widget)
        menu.setToolTip(tooltip)
        for name, _payload in items:
            action = menu.addAction(name)
            action.triggered.connect(lambda _checked=False, n=name: on_pick(n))
        pos = event.globalPos() if hasattr(event, "globalPos") else self._widget.mapToGlobal(event.pos())
        menu.exec_(pos)

    def _pick_dash(self, name):
        self._style = self._style.updated(dash=name)
        self._emit()

    def _pick_start_head(self, name):
        self._style = self._style.updated(start_head=name)
        self._emit()

    def _pick_end_head(self, name):
        self._style = self._style.updated(end_head=name)
        self._emit()
