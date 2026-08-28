"""Dash / margin / end-cap options plus a 2D line-style preview."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

from ..pick import qt_modules

DASH_PRESETS = (
    ("Solid", (1.0, 0.0)),
    ("Dashed", (0.45, 0.28)),
    ("Dotted", (0.08, 0.18)),
    ("Dash-dot", (0.45, 0.16, 0.08, 0.16)),
    ("Dash-dot-dot", (0.45, 0.14, 0.08, 0.14, 0.08, 0.14)),
)

END_STYLES = ("None", "Arrow", "Double arrow", "Circles")

ARROW_QUALITY_SEGMENTS = {0: 0, 1: 6, 2: 8, 3: 10, 4: 14, 5: 18}


@dataclass
class LineStyle:
    dash: str = "Solid"
    dash_scale: float = 1.0
    margin: float = 0.0
    ends: str = "Arrow"

    def pattern(self) -> Tuple[float, ...]:
        for name, pattern in DASH_PRESETS:
            if name == self.dash:
                return pattern
        return (1.0, 0.0)


def dash_on_segments(
    p0: Sequence[float],
    p1: Sequence[float],
    pattern: Sequence[float],
    scale: float,
) -> list:
    """Split p0→p1 into ON segments for a repeating on/off pattern."""
    x0, y0, z0 = (float(p0[0]), float(p0[1]), float(p0[2]))
    x1, y1, z1 = (float(p1[0]), float(p1[1]), float(p1[2]))
    dx, dy, dz = (x1 - x0, y1 - y0, z1 - z0)
    length = (dx * dx + dy * dy + dz * dz) ** 0.5
    if length < 1e-8:
        return []
    inv = 1.0 / length
    nx, ny, nz = (dx * inv, dy * inv, dz * inv)
    raw = [max(float(v), 0.0) for v in pattern]
    if not raw:
        raw = [1.0, 0.0]
    if len(raw) == 1:
        raw = [raw[0], 0.0]
    scale = max(float(scale), 1e-6)
    units = [v * scale for v in raw]
    off_spans = units[1::2]
    if not off_spans or max(off_spans) <= 1e-8:
        return [((x0, y0, z0), (x1, y1, z1))]
    period = sum(units)
    if period < 1e-8:
        return [((x0, y0, z0), (x1, y1, z1))]

    segments = []
    travelled = 0.0
    cycle = 0
    while travelled < length - 1e-8:
        span = units[cycle % len(units)]
        on = (cycle % 2) == 0
        if span <= 1e-8:
            cycle += 1
            continue
        start_t = travelled
        end_t = min(length, travelled + span)
        if on and end_t > start_t + 1e-8:
            segments.append((
                (x0 + nx * start_t, y0 + ny * start_t, z0 + nz * start_t),
                (x0 + nx * end_t, y0 + ny * end_t, z0 + nz * end_t),
            ))
        travelled = end_t
        cycle += 1
    return segments


def apply_margin(p0, p1, margin: float):
    x0, y0, z0 = (float(p0[0]), float(p0[1]), float(p0[2]))
    x1, y1, z1 = (float(p1[0]), float(p1[1]), float(p1[2]))
    dx, dy, dz = (x1 - x0, y1 - y0, z1 - z0)
    length = (dx * dx + dy * dy + dz * dz) ** 0.5
    pad = max(float(margin), 0.0)
    if length < 1e-8 or pad <= 0.0:
        return (x0, y0, z0), (x1, y1, z1)
    if pad * 2.0 >= length:
        pad = length * 0.25
    inv = 1.0 / length
    return (
        (x0 + dx * inv * pad, y0 + dy * inv * pad, z0 + dz * inv * pad),
        (x1 - dx * inv * pad, y1 - dy * inv * pad, z1 - dz * inv * pad),
    )


class LineStylePreview:
    """Small 2D sketch of dash / margin / end caps."""

    def __init__(self, parent=None):
        QtCore, QtGui, QtWidgets = qt_modules()
        self._style = LineStyle()
        self._widget = QtWidgets.QWidget(parent)
        self._widget.setMinimumHeight(64)
        self._widget.setMaximumHeight(72)
        self._widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self._widget.paintEvent = self._paint

    @property
    def widget(self):
        return self._widget

    def set_style(self, style: LineStyle):
        self._style = style
        self._widget.update()

    def _paint(self, event):
        QtCore, QtGui, _ = qt_modules()
        painter = QtGui.QPainter(self._widget)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        rect = self._widget.rect().adjusted(10, 8, -10, -8)
        painter.fillRect(self._widget.rect(), self._widget.palette().base())
        y = rect.center().y()
        x0 = float(rect.left())
        x1 = float(rect.right())
        width = max(x1 - x0, 1.0)
        margin_px = min(self._style.margin * 18.0, width * 0.2)
        a = x0 + margin_px
        b = x1 - margin_px
        if b <= a:
            a, b = x0 + 8.0, x1 - 8.0

        pen = QtGui.QPen(QtGui.QColor(40, 40, 40), 2.2)
        pattern = self._style.pattern()
        if len(pattern) < 2 or pattern[1] <= 1e-6:
            pen.setStyle(QtCore.Qt.SolidLine)
        else:
            scale = max(self._style.dash_scale, 0.15) * 22.0
            pen.setDashPattern([max(float(v) * scale, 0.01) for v in pattern])
        painter.setPen(pen)
        painter.drawLine(int(a), int(y), int(b), int(y))

        painter.setPen(QtGui.QPen(QtGui.QColor(40, 40, 40), 1.6))
        painter.setBrush(QtGui.QColor(40, 40, 40))
        ends = self._style.ends
        if ends in ("Arrow", "Double arrow"):
            self._draw_arrow(painter, b, y, 1)
        if ends == "Double arrow":
            self._draw_arrow(painter, a, y, -1)
        if ends == "Circles":
            painter.drawEllipse(QtCore.QPointF(a, y), 4, 4)
            painter.drawEllipse(QtCore.QPointF(b, y), 4, 4)
        painter.end()

    def _draw_arrow(self, painter, x, y, direction):
        QtCore, QtGui, _ = qt_modules()
        tip = QtGui.QPolygonF([
            QtCore.QPointF(x + 9 * direction, y),
            QtCore.QPointF(x - 2 * direction, y - 5),
            QtCore.QPointF(x - 2 * direction, y + 5),
        ])
        painter.drawPolygon(tip)


class LineOptionsWidget:
    """Dash / margin / ends controls plus the 2D preview."""

    def __init__(self, parent=None, on_change: Callable[[], None] = None):
        QtCore, _, QtWidgets = qt_modules()
        self._on_change = on_change
        box = QtWidgets.QGroupBox("Line options", parent)
        layout = QtWidgets.QFormLayout(box)
        self._dash = QtWidgets.QComboBox()
        for name, _pattern in DASH_PRESETS:
            self._dash.addItem(name)
        self._scale = QtWidgets.QDoubleSpinBox()
        self._scale.setRange(0.1, 20.0)
        self._scale.setSingleStep(0.1)
        self._scale.setValue(1.0)
        self._scale.setSuffix(" Å")
        self._margin = QtWidgets.QDoubleSpinBox()
        self._margin.setRange(0.0, 20.0)
        self._margin.setSingleStep(0.1)
        self._margin.setValue(0.0)
        self._margin.setSuffix(" Å")
        self._ends = QtWidgets.QComboBox()
        for name in END_STYLES:
            self._ends.addItem(name)
        self._ends.setCurrentText("Arrow")
        self._preview = LineStylePreview(box)
        for widget in (self._dash, self._ends):
            widget.currentIndexChanged.connect(self._emit)
        for widget in (self._scale, self._margin):
            widget.valueChanged.connect(self._emit)
        layout.addRow("Dash", self._dash)
        layout.addRow("Dash scale", self._scale)
        layout.addRow("Margin", self._margin)
        layout.addRow("Ends", self._ends)
        layout.addRow("Preview", self._preview.widget)
        self._box = box
        self._preview.set_style(self.style())

    @property
    def widget(self):
        return self._box

    def style(self) -> LineStyle:
        return LineStyle(
            dash=self._dash.currentText(),
            dash_scale=float(self._scale.value()),
            margin=float(self._margin.value()),
            ends=self._ends.currentText(),
        )

    def _emit(self, *_args):
        self._preview.set_style(self.style())
        if self._on_change is not None:
            self._on_change()
