"""Dash / margin / end-cap options plus a 2D line-style preview."""

from __future__ import annotations

from typing import Callable

from ...util.line_style import (
    ARROW_QUALITY_SEGMENTS,
    DASH_PRESETS,
    HEAD_STYLES,
    END_STYLES,
    LineStyle,
    apply_margin,
    dash_on_segments,
)
from ..pick import qt_modules
from ..tooltips import apply_required_tooltips
from ..widgets.section import make_section
from ..widgets.theme import INK, ROW, qcolor

__all__ = [
    "DASH_PRESETS",
    "HEAD_STYLES",
    "END_STYLES",
    "ARROW_QUALITY_SEGMENTS",
    "LineStyle",
    "dash_on_segments",
    "apply_margin",
    "LineStylePreview",
    "LineOptionsWidget",
]


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
        painter.fillRect(self._widget.rect(), qcolor(QtGui, ROW))
        y = rect.center().y()
        x0 = float(rect.left())
        x1 = float(rect.right())
        width = max(x1 - x0, 1.0)
        margin0 = min(self._style.start_margin * 18.0, width * 0.2)
        margin1 = min(self._style.end_margin * 18.0, width * 0.2)
        a = x0 + margin0
        b = x1 - margin1
        if b <= a:
            a, b = x0 + 8.0, x1 - 8.0

        pen = QtGui.QPen(qcolor(QtGui, INK), 2.2)
        pattern = self._style.pattern()
        if len(pattern) < 2 or pattern[1] <= 1e-6:
            pen.setStyle(QtCore.Qt.SolidLine)
        else:
            scale = max(self._style.dash_scale, 0.15) * 22.0
            pen.setDashPattern([max(float(v) * scale, 0.01) for v in pattern])
        painter.setPen(pen)
        painter.drawLine(int(a), int(y), int(b), int(y))

        painter.setPen(QtGui.QPen(qcolor(QtGui, INK), 1.6))
        painter.setBrush(qcolor(QtGui, INK))
        self._draw_cap(painter, self._style.start_head, a, y, -1)
        self._draw_cap(painter, self._style.end_head, b, y, 1)
        painter.end()

    def _draw_cap(self, painter, kind, x, y, direction):
        QtCore, QtGui, _ = qt_modules()
        if kind == "Circles":
            painter.drawEllipse(QtCore.QPointF(x, y), 4, 4)
            return
        if kind == "Arrow":
            self._draw_arrow(painter, x, y, direction)
            return
        ghost = qcolor(QtGui, INK, alpha=110)
        painter.setPen(QtGui.QPen(ghost, 1.2))
        painter.setBrush(QtCore.Qt.NoBrush)
        self._draw_arrow(painter, x, y, direction)
        painter.setPen(QtGui.QPen(qcolor(QtGui, INK), 1.6))
        painter.setBrush(qcolor(QtGui, INK))

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
        section = make_section("Line options", parent, form=True)
        layout = section.layout
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
        self._start_head = QtWidgets.QComboBox()
        self._end_head = QtWidgets.QComboBox()
        for name in HEAD_STYLES:
            self._start_head.addItem(name)
            self._end_head.addItem(name)
        self._start_head.setCurrentText("None")
        self._end_head.setCurrentText("Arrow")
        self._preview = LineStylePreview(section.body)
        for widget in (self._dash, self._start_head, self._end_head):
            widget.currentIndexChanged.connect(self._emit)
        for widget in (self._scale, self._margin):
            widget.valueChanged.connect(self._emit)
        layout.addRow("Dash", self._dash)
        layout.addRow("Dash scale", self._scale)
        layout.addRow("Margin", self._margin)
        layout.addRow("Start cap", self._start_head)
        layout.addRow("End cap", self._end_head)
        layout.addRow("Preview", self._preview.widget)
        self._box = section.widget
        self._preview.set_style(self.style())
        apply_required_tooltips(
            [
                (self._dash, "Line pattern: solid or a dashed preset."),
                (self._scale, "Stretch the dash pattern along the line (Ångström-scaled)."),
                (self._margin, "Shorten both ends by this many Ångströms before drawing."),
                (self._start_head, "Start cap: none, arrow, or circle."),
                (self._end_head, "End cap: none, arrow, or circle."),
                (self._preview.widget, "Sketch of the current dash, margin, and end style."),
            ],
            context="LineOptionsWidget",
        )

    @property
    def widget(self):
        return self._box

    def style(self) -> LineStyle:
        return LineStyle(
            dash=self._dash.currentText(),
            dash_scale=float(self._scale.value()),
            margin=float(self._margin.value()),
            start_head=self._start_head.currentText(),
            end_head=self._end_head.currentText(),
        )

    def set_style(self, style: LineStyle):
        if style is None:
            return
        self._dash.blockSignals(True)
        self._scale.blockSignals(True)
        self._margin.blockSignals(True)
        self._start_head.blockSignals(True)
        self._end_head.blockSignals(True)
        try:
            dash = getattr(style, "dash", None)
            if dash:
                index = self._dash.findText(str(dash))
                if index >= 0:
                    self._dash.setCurrentIndex(index)
            self._scale.setValue(float(getattr(style, "dash_scale", 1.0)))
            self._margin.setValue(float(getattr(style, "margin", 0.0)))
            start = getattr(style, "start_head", None)
            if start:
                index = self._start_head.findText(str(start))
                if index >= 0:
                    self._start_head.setCurrentIndex(index)
            end = getattr(style, "end_head", None)
            if end:
                index = self._end_head.findText(str(end))
                if index >= 0:
                    self._end_head.setCurrentIndex(index)
        finally:
            self._dash.blockSignals(False)
            self._scale.blockSignals(False)
            self._margin.blockSignals(False)
            self._start_head.blockSignals(False)
            self._end_head.blockSignals(False)
        self._preview.set_style(self.style())

    def _emit(self, *_args):
        self._preview.set_style(self.style())
        if self._on_change is not None:
            self._on_change()
