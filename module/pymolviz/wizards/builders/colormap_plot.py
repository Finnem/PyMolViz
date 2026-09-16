"""Colormap histogram, colorbar preview, stop-drag widgets, and export dialog."""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ...util.colormap_spec import (
    ColorStop,
    ColorbarExportSettings,
    ColormapDefinition,
    FieldColorMapping,
    Normalization,
    alpha_to_screen_y,
    axis_to_unit,
    clamp_range,
    colorbar_caption,
    colorbar_export_display,
    colorbar_ramp_rgba,
    colorbar_tick_values,
    data_to_unit,
    field_histogram,
    histogram_axis_to_value,
    HISTOGRAM_VIEW_FULL,
    HISTOGRAM_VIEW_LOG,
    HISTOGRAM_VIEW_PERCENTILE,
    value_to_histogram_axis,
    field_title,
    field_units,
    field_values_for_stats,
    format_number,
    format_numbers_for_ticks,
    definition_from_preset,
    tick_label_indices_without_overlap,
    tick_label_span,
    ramp_rgba,
    sample_unit,
    screen_y_to_alpha,
    unit_to_axis,
    unit_to_data,
)
from ...util.colorbar_export import export_colorbar
from ...util.field_sample import DEFAULT_SURFACE_COLORMAP
from ..pick import (
    bind_tool_window,
    overlay_exec,
    overlay_get_save_file_name,
    overlay_window,
    qt_modules,
    qt_widget_alive,
)
from ..widgets.ascii_locale import apply_ascii_float_locale
from ..widgets.section import make_section
from ..widgets.spin_step import apply_magnitude_step
from ..widgets.theme import apply_wizard_page_style, mark_primary_button


EXPORT_TITLE = "Export Colorbar"
SCALE_FULL_LABEL = "Absolute (full range)"
SCALE_LOG_LABEL = "Log scale"
SCALE_PCT_LABEL = "Percentile (1–99%)"


def _configure_editor_window(widget):
    QtCore, _, QtWidgets = qt_modules()
    if QtCore is None:
        return
    # Do not Qt-parent this to Fields or setTransientParent(PyMOL). Clicking
    # the viewer used to hide the Fields window and delete this editor.
    widget._pmv_window_anchor = None
    widget._pmv_no_transient = True
    widget._pmv_raise_last = True
    widget.setWindowFlags(widget.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)


def _font_text_width(fm, text: str) -> float:
    if hasattr(fm, "horizontalAdvance"):
        return float(fm.horizontalAdvance(text))
    return float(fm.width(text))


def _tick_labels_for_paint(
    painter,
    ticks: Sequence[float],
    xs: Sequence[float],
    settings: Optional[ColorbarExportSettings] = None,
) -> Tuple[List[float], List[str], List[int]]:
    """Pick compact labels and drop indices that would overlap on screen."""
    settings = settings or ColorbarExportSettings()
    fmt = settings.number_format
    decimals = settings.decimals
    sig = int(settings.sig_digits or 3)
    fm = painter.fontMetrics()
    for sig_try in (sig, max(2, sig - 1), 2):
        labels = format_numbers_for_ticks(ticks, fmt, decimals, sig_try)
        widths = [_font_text_width(fm, t) for t in labels]
        if len(widths) >= 2:
            w0 = tick_label_span(widths, 0, xs[0])
            w1 = tick_label_span(widths, len(widths) - 1, xs[-1])
            if w0[1] + 4.0 > w1[0]:
                continue
        keep = tick_label_indices_without_overlap(widths, xs)
        if keep:
            return list(ticks), labels, keep
    labels = format_numbers_for_ticks(ticks, fmt, decimals, 2)
    widths = [_font_text_width(fm, t) for t in labels]
    keep = tick_label_indices_without_overlap(widths, xs)
    return list(ticks), labels, keep


def _draw_horizontal_tick_labels(painter, QtCore, labels, xs, keep, y_line: int, y_text: int):
    n = len(labels)
    last = max(n - 1, 1)
    align_top = getattr(QtCore.Qt, "AlignTop", 0x20)
    align_left = getattr(QtCore.Qt, "AlignLeft", 0x1) | align_top
    align_right = getattr(QtCore.Qt, "AlignRight", 0x2) | align_top
    align_center = getattr(QtCore.Qt, "AlignHCenter", 0x4) | align_top
    fm = painter.fontMetrics()
    for i in keep:
        x = int(round(xs[i]))
        text = labels[i]
        painter.drawLine(x, y_line, x, y_line + 4)
        box = max(48, int(_font_text_width(fm, text)) + 8)
        if i == 0:
            painter.drawText(x, y_text, box, 16, align_left, text)
        elif i == last:
            painter.drawText(x - box, y_text, box, 16, align_right, text)
        else:
            painter.drawText(x - box // 2, y_text, box, 16, align_center, text)


from ..widgets.theme import BORDER, INK, MUTED, PRIMARY, ROW, apply_secondary_button_style, swatch_button_css
from .colors import ColorChoice, bind_color_pick_result, pick_rgb, rgba_to_css

def _qcolor(QtGui, rgba):
    r, g, b, a = [max(0, min(255, int(round(float(c) * 255.0)))) for c in _as_rgba(rgba)]
    return QtGui.QColor(r, g, b, a)


def _as_rgba(value):
    if value is None:
        seq = []
    else:
        try:
            seq = np.asarray(value, dtype=float).reshape(-1).tolist()
        except (TypeError, ValueError):
            seq = list(value)
    r = float(seq[0] if len(seq) > 0 else 0.0)
    g = float(seq[1] if len(seq) > 1 else 0.0)
    b = float(seq[2] if len(seq) > 2 else 0.0)
    a = float(seq[3] if len(seq) > 3 else 1.0)
    return (r, g, b, a)


def _pixel(QtGui, rgba):
    r, g, b, a = [max(0, min(255, int(round(float(c) * 255.0)))) for c in _as_rgba(rgba)]
    qrgba = getattr(QtGui, "qRgba", None)
    if qrgba is not None:
        return qrgba(r, g, b, a)
    return (a << 24) | (r << 16) | (g << 8) | b


def _image_format(QtGui, with_alpha=False):
    image_cls = getattr(QtGui, "QImage", None)
    if image_cls is None:
        return None
    names = ("Format_ARGB32", "Format_RGB32") if with_alpha else ("Format_RGB32", "Format_ARGB32")
    for name in names:
        fmt = getattr(image_cls, name, None)
        if fmt is not None:
            return fmt
    nested = getattr(image_cls, "Format", None)
    for name in names:
        fmt = getattr(nested, name, None) if nested is not None else None
        if fmt is not None:
            return fmt
    return None


def _ramp_image(QtGui, defn, width, height, vertical=False, *, map_limits=None, display_limits=None, axis=HISTOGRAM_VIEW_FULL):
    width = max(2, int(width))
    height = max(2, int(height))
    n = height if vertical else width
    fmt = _image_format(QtGui, with_alpha=True)
    empty = QtGui.QPixmap(width, height)
    empty.fill(QtGui.QColor(0, 0, 0, 0))
    if fmt is None:
        return empty
    try:
        if display_limits is not None and map_limits is not None:
            colors = colorbar_ramp_rgba(
                defn, n,
                float(map_limits[0]), float(map_limits[1]),
                float(display_limits[0]), float(display_limits[1]),
                axis,
            )
        else:
            colors = ramp_rgba(defn, n=n)
    except Exception:
        return empty
    try:
        if vertical:
            image = QtGui.QImage(1, n, fmt)
        else:
            image = QtGui.QImage(n, 1, fmt)
        if image.isNull():
            return empty
        for i, row in enumerate(colors):
            pixel = _pixel(QtGui, row)
            if vertical:
                image.setPixel(0, n - 1 - i, pixel)
            else:
                image.setPixel(i, 0, pixel)
        pix = QtGui.QPixmap.fromImage(image)
        if pix.isNull():
            return empty
        return pix.scaled(width, height)
    except Exception:
        return empty


def _event_xy(event):
    if event is None:
        return 0.0, 0.0
    if hasattr(event, "position"):
        pos = event.position()
        return float(pos.x()), float(pos.y())
    if hasattr(event, "localPos"):
        pos = event.localPos()
        return float(pos.x()), float(pos.y())
    return float(event.x()), float(event.y())


def _realized_wh(widget, fallback_w, fallback_h):
    """QWidget defaults to 640x480 before it is shown; do not bake that into pixmaps."""
    if widget is None:
        return int(fallback_w), int(fallback_h)
    try:
        visible = bool(widget.isVisible())
        w = int(widget.width() or 0)
        h = int(widget.height() or 0)
    except RuntimeError:
        return int(fallback_w), int(fallback_h)
    if not visible or w <= 1 or h <= 1:
        return int(fallback_w), int(fallback_h)
    return max(2, w), max(2, h)


def _begin_widget_paint(QtGui, widget):
    if widget is None or QtGui is None or not qt_widget_alive(widget):
        return None
    try:
        painter = QtGui.QPainter(widget)
    except Exception:
        return None
    if not painter.isActive():
        return None
    return painter


def _end_widget_paint(painter):
    if painter is not None and painter.isActive():
        painter.end()


def _draw_stop_circle(painter, QtGui, x, y, rgba, selected, radius=8):
    r = int(radius)
    cx, cy = int(round(x)), int(round(y))
    painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 3 if selected else 2))
    painter.setBrush(_qcolor(QtGui, rgba))
    painter.drawEllipse(cx - r, cy - r, r * 2, r * 2)
    painter.setPen(QtGui.QPen(QtGui.QColor(*(PRIMARY if selected else INK)), 1))
    painter.setBrush(QtGui.QColor(0, 0, 0, 0))
    painter.drawEllipse(cx - r, cy - r, r * 2, r * 2)


def _histogram_tick_values(dmin, dmax, axis, count: int = 5):
    """Tick positions along the displayed histogram span (log or linear)."""
    return colorbar_tick_values(dmin, dmax, axis, count)


def _data_span(hist, limits):
    if hist and hist.get("display_min") is not None and hist.get("display_max") is not None:
        return clamp_range(float(hist["display_min"]), float(hist["display_max"]))
    hmin = None if not hist else hist.get("vmin")
    hmax = None if not hist else hist.get("vmax")
    if limits:
        lo, hi = float(limits[0]), float(limits[1])
        if hmin is None:
            hmin = lo
        if hmax is None:
            hmax = hi
        hmin = min(float(hmin), lo)
        hmax = max(float(hmax), hi)
    if hmin is None or hmax is None:
        return 0.0, 1.0
    return clamp_range(hmin, hmax)


def _hist_axis(hist) -> str:
    return str((hist or {}).get("axis") or HISTOGRAM_VIEW_FULL)


def _x_for_value(rect, value, dmin, dmax, axis=HISTOGRAM_VIEW_FULL) -> float:
    if value is None:
        value = dmin
    unit = value_to_histogram_axis(value, dmin, dmax, axis)
    return unit_to_axis(unit, rect.left(), rect.right())


def _x_for_stop_position(rect, position, vmin, vmax, dmin, dmax, axis=HISTOGRAM_VIEW_FULL) -> float:
    """Pixel X for a unit-interval color stop on the current min/max range."""
    value = unit_to_data(float(position), vmin, vmax)
    return _x_for_value(rect, value, dmin, dmax, axis)


def _value_for_x(rect, x, dmin, dmax, axis=HISTOGRAM_VIEW_FULL) -> float:
    unit = axis_to_unit(x, rect.left(), rect.right())
    return histogram_axis_to_value(unit, dmin, dmax, axis)


def _stop_neighbor_limits(index: int, stops) -> Tuple[float, float]:
    """Unit-interval bounds for stop ``index``. End stops stay at 0 and 1."""
    n = len(stops or ())
    if n <= 0:
        return 0.0, 1.0
    index = max(0, min(int(index), n - 1))
    if _is_end_stop(index, n):
        pinned = 0.0 if index <= 0 else 1.0
        return pinned, pinned
    lo = float(stops[index - 1].position) + 0.004
    hi = float(stops[index + 1].position) - 0.004
    if hi < lo:
        mid = 0.5 * (lo + hi)
        return mid, mid
    return lo, hi


def _is_end_stop(index: int, stops_or_n) -> bool:
    if isinstance(stops_or_n, int):
        n = stops_or_n
    else:
        n = len(stops_or_n or ())
    if n <= 0:
        return False
    index = int(index)
    return index <= 0 or index >= n - 1


def _clamp_stop_position(index: int, stops, position: float) -> float:
    lo, hi = _stop_neighbor_limits(index, stops)
    return max(lo, min(hi, float(position)))


def _unit_position_for_histogram_x(hist, x, vmin, vmax, dmin, dmax, axis=HISTOGRAM_VIEW_FULL) -> float:
    """Invert ``_x_for_stop_position`` so a dragged stop stays under the cursor."""
    value = _value_for_x(hist, x, dmin, dmax, axis)
    return data_to_unit(value, vmin, vmax)


def _ask_float(parent, title, label, value, lo=-1e8, hi=1e8, decimals=4):
    QtCore, _, QtWidgets = qt_modules()
    if QtWidgets is None or not hasattr(QtWidgets, "QDialog"):
        return None
    overlay = overlay_window(parent)
    dialog = QtWidgets.QDialog(overlay if overlay is not None else parent)
    dialog.setWindowTitle(str(title or "Value"))
    dialog.setModal(True)
    form = QtWidgets.QFormLayout(dialog)
    spin = QtWidgets.QDoubleSpinBox()
    spin.setDecimals(int(decimals))
    spin.setRange(float(lo), float(hi))
    apply_ascii_float_locale(spin, QtCore)
    spin.setValue(float(value))
    apply_magnitude_step(spin, min_decimals=int(decimals))
    form.addRow(str(label or "Value"), spin)
    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    form.addRow(buttons)
    accepted = overlay_exec(dialog, parent)
    ok = getattr(QtWidgets.QDialog, "Accepted", 1)
    if int(accepted) != int(ok):
        return None
    return float(spin.value())


class _DistributionEditor:
    """Histogram + transfer-function plot: range handles, 2D stops (X=value, Y=alpha)."""

    _STOP_R = 9

    def __init__(self, parent, on_stop, on_select, on_range, on_ask, on_add=None, on_edit_stop=None, on_drag_end=None):
        QtCore, QtGui, QtWidgets = qt_modules()
        self._on_stop = on_stop
        self._on_select = on_select
        self._on_range = on_range
        self._on_ask = on_ask
        self._on_add = on_add
        self._on_edit_stop = on_edit_stop
        self._on_drag_end = on_drag_end
        self._hist = None
        self._limits = (0.0, 1.0)
        self._center = None
        self._defn = definition_from_preset(DEFAULT_SURFACE_COLORMAP)
        self._selected = 0
        self._drag = None
        self._drag_offset = (0.0, 0.0)
        self._drag_moved = False
        self._press_was_already_selected = False
        view = QtWidgets.QWidget(parent)
        view.setObjectName("pmvColormapDistribution")
        view.setMinimumHeight(220)
        view.setMaximumHeight(280)
        expanding = getattr(QtWidgets.QSizePolicy, "Expanding", None)
        fixed = getattr(QtWidgets.QSizePolicy, "Fixed", None)
        if expanding is not None and fixed is not None:
            view.setSizePolicy(expanding, fixed)
        hint = getattr(QtCore, "QSize", None)
        if hint is not None:
            view.sizeHint = lambda: hint(400, 220)
            view.minimumSizeHint = lambda: hint(240, 200)
        view.setMouseTracking(True)
        view.setToolTip(
            "Drag the blue triangles above the plot to set min/max. Drag interior "
            "color stops left/right (value) and up/down for opacity on the histogram. "
            "The top of the plot is opaque, the bottom is transparent. End colors stay "
            "on min/max and only move up/down. Left-click empty space to add a stop. "
            "Left-click a selected stop again to edit it; right-click a range handle "
            "to type a value."
        )
        view.paintEvent = lambda _event: self._paint(view, QtCore, QtGui)
        view.mousePressEvent = lambda event: self._press(view, event, QtCore)
        view.mouseMoveEvent = lambda event: self._move(view, event)
        view.mouseReleaseEvent = lambda event: self._release(view, event)
        view.contextMenuEvent = lambda event: self._context(view, event)

        def _resize(event):
            QtWidgets.QWidget.resizeEvent(view, event)
            view.update()

        view.resizeEvent = _resize
        self.widget = view

    def set_data(self, hist, limits=None, center=None, defn=None, selected=0):
        self._hist = hist
        self._limits = limits or (0.0, 1.0)
        self._center = center
        if defn is not None:
            self._defn = defn
            self._selected = max(0, min(int(selected), len(defn.stops) - 1))
        self.widget.update()

    def _plot(self, widget):
        # Top pad: range-handle triangles. Bottom pad: value ticks.
        return widget.rect().adjusted(12, 22, -12, -22)

    def _layout(self, widget):
        """Shared plot: histogram bars and stop alpha use the same vertical span."""
        plot = self._plot(widget)
        return {"plot": plot, "bars": plot, "alpha": plot}

    def _span(self):
        return _data_span(self._hist, self._limits)

    def _axis(self) -> str:
        return _hist_axis(self._hist)

    def _stop_xy(self, layout, stop, dmin, dmax):
        plot = layout["plot"]
        alpha = layout["alpha"]
        vmin, vmax = self._limits
        x = _x_for_stop_position(
            plot, stop.position, vmin, vmax, dmin, dmax, self._axis(),
        )
        inset = 6
        y = alpha_to_screen_y(stop.rgba[3], alpha.top() + inset, alpha.bottom() - inset)
        return x, y

    def _paint(self, widget, QtCore, QtGui):
        painter = _begin_widget_paint(QtGui, widget)
        if painter is None:
            return
        try:
            aa = getattr(QtGui.QPainter, "Antialiasing", None)
            if aa is not None:
                painter.setRenderHint(aa, True)
            painter.fillRect(widget.rect(), QtGui.QColor(*ROW))
            layout = self._layout(widget)
            plot = layout["plot"]
            dmin, dmax = self._span()
            vmin, vmax = self._limits
            self._paint_histogram(painter, QtCore, QtGui, layout, dmin, dmax, vmin, vmax)
            try:
                painter.setPen(QtGui.QPen(QtGui.QColor(*BORDER), 1))
                painter.setBrush(QtGui.QColor(0, 0, 0, 0))
                painter.drawRect(plot.adjusted(0, 0, -1, -1))
            except Exception:
                pass
            self._paint_axis_ticks(painter, QtCore, QtGui, layout, dmin, dmax)
            self._paint_handles(painter, QtCore, QtGui, layout, dmin, dmax, vmin, vmax)
        finally:
            _end_widget_paint(painter)

    def _paint_histogram(self, painter, QtCore, QtGui, layout, dmin, dmax, vmin, vmax):
        counts = list((self._hist or {}).get("counts") or ())
        edges = list((self._hist or {}).get("edges") or ())
        bars = layout["bars"]
        plot = layout["plot"]
        if not counts:
            try:
                painter.setPen(QtGui.QColor(*MUTED))
                painter.drawText(
                    bars.adjusted(8, 8, -8, -8),
                    getattr(QtCore.Qt, "AlignTop", 0x20),
                    "No field values",
                )
            except Exception:
                pass
            return
        axis = self._axis()
        dmin, dmax = self._span()
        bar_h = max(8, int(bars.height()))
        peak = math.sqrt(float(max(max(counts), 1)))
        fallback = QtGui.QColor(*PRIMARY)
        fallback.setAlpha(180)
        muted = QtGui.QColor(*MUTED)
        muted.setAlpha(110)
        no_pen = getattr(getattr(QtCore, "Qt", None), "NoPen", None)
        if no_pen is not None:
            painter.setPen(no_pen)
        else:
            painter.setPen(QtGui.QPen(fallback, 0))
        tops = []
        for i, count in enumerate(counts):
            if len(edges) < i + 2:
                break
            x0 = _x_for_value(plot, float(edges[i]), dmin, dmax, axis)
            x1 = _x_for_value(plot, float(edges[i + 1]), dmin, dmax, axis)
            left = int(round(min(x0, x1)))
            right = int(round(max(x0, x1)))
            w = max(2, right - left)
            bh = int(round((math.sqrt(float(max(count, 0))) / peak) * bar_h))
            if count > 0:
                bh = max(3, bh)
            e0, e1 = float(edges[i]), float(edges[i + 1])
            if axis == "log" and e0 > 0.0 and e1 > 0.0:
                mid = math.sqrt(e0 * e1)
            else:
                mid = 0.5 * (e0 + e1)
            fill = fallback
            if mid < vmin or mid > vmax:
                fill = muted
            else:
                try:
                    rgba = sample_unit(self._defn, data_to_unit(mid, vmin, vmax))
                    fill = _qcolor(QtGui, rgba)
                    fill.setAlpha(200)
                except Exception:
                    fill = fallback
            painter.setBrush(fill)
            painter.drawRect(left, int(bars.bottom()) - bh, w, bh)
            if count > 0:
                tops.append((left + w * 0.5, int(bars.bottom()) - bh))
        if len(tops) >= 2:
            painter.setPen(QtGui.QPen(QtGui.QColor(*INK), 1.25))
            for (x0, y0), (x1, y1) in zip(tops, tops[1:]):
                painter.drawLine(int(x0), int(y0), int(x1), int(y1))

    def _paint_axis_ticks(self, painter, QtCore, QtGui, layout, dmin, dmax):
        plot = layout["plot"]
        axis = self._axis()
        ticks = _histogram_tick_values(dmin, dmax, axis, 5)
        if not ticks:
            return
        xs = [_x_for_value(plot, value, dmin, dmax, axis) for value in ticks]
        _, labels, keep = _tick_labels_for_paint(painter, ticks, xs)
        painter.setPen(QtGui.QColor(*INK))
        y0 = int(plot.bottom())
        _draw_horizontal_tick_labels(painter, QtCore, labels, xs, keep, y0, y0 + 4)

    def _paint_handles(self, painter, QtCore, QtGui, layout, dmin, dmax, vmin, vmax):
        dash = getattr(QtCore.Qt, "DashLine", 2)
        axis = self._axis()
        plot = layout["plot"]
        bars = layout["bars"]
        alpha = layout["alpha"]
        try:
            painter.setPen(QtGui.QPen(QtGui.QColor(*INK), 1.5))
            prev = None
            for stop in self._defn.stops:
                x, y = self._stop_xy(layout, stop, dmin, dmax)
                if prev is not None:
                    painter.drawLine(int(prev[0]), int(prev[1]), int(x), int(y))
                prev = (x, y)
            for i, stop in enumerate(self._defn.stops):
                x, y = self._stop_xy(layout, stop, dmin, dmax)
                painter.setPen(QtGui.QPen(QtGui.QColor(*MUTED), 1, dash))
                painter.drawLine(int(x), int(alpha.top()), int(x), int(bars.bottom()))
                _draw_stop_circle(painter, QtGui, x, y, stop.rgba, i == self._selected, self._STOP_R)
            for which, value in (("vmin", vmin), ("vmax", vmax)):
                x = _x_for_value(plot, value, dmin, dmax, axis)
                painter.setPen(QtGui.QPen(QtGui.QColor(*PRIMARY), 2))
                painter.drawLine(int(x), int(plot.top()), int(x), int(plot.bottom()))
                self._draw_range_handle(painter, QtCore, QtGui, x, plot.top(), which == "vmin")
            if self._center is not None:
                painter.setPen(QtGui.QPen(QtGui.QColor(*MUTED), 1, dash))
                cx = _x_for_value(plot, float(self._center), dmin, dmax, axis)
                painter.drawLine(int(cx), int(plot.top()), int(cx), int(plot.bottom()))
        except Exception:
            pass

    def _draw_range_handle(self, painter, QtCore, QtGui, x, top, left_side):
        point = getattr(QtCore, "QPoint", None)
        poly_cls = getattr(QtGui, "QPolygon", None)
        y = int(top)
        painter.setBrush(QtGui.QColor(*PRIMARY))
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
        if poly_cls is not None and point is not None:
            if left_side:
                pts = [point(int(x), y), point(int(x - 11), y - 16), point(int(x + 5), y - 16)]
            else:
                pts = [point(int(x), y), point(int(x - 5), y - 16), point(int(x + 11), y - 16)]
            painter.drawPolygon(poly_cls(pts))
        else:
            painter.drawRect(int(x) - 7, y - 16, 14, 16)

    def _hit(self, widget, x, y):
        layout = self._layout(widget)
        plot = layout["plot"]
        bars = layout["bars"]
        alpha = layout["alpha"]
        dmin, dmax = self._span()
        axis = self._axis()
        vmin, vmax = self._limits
        handle_top = int(plot.top()) - 18
        handle_bottom = int(plot.top()) + 2
        if handle_top <= y <= handle_bottom:
            best = None
            best_dist = 1e9
            for which, value in (("vmin", vmin), ("vmax", vmax)):
                hx = _x_for_value(plot, value, dmin, dmax, axis)
                dist = abs(x - hx)
                if dist <= 16 and dist < best_dist:
                    best_dist = dist
                    best = ("range", which)
            if best is not None:
                return best
        stop_hit = None
        stop_dist = 1e9
        radius = self._STOP_R + 5
        for i, stop in enumerate(self._defn.stops):
            sx, sy = self._stop_xy(layout, stop, dmin, dmax)
            dist = ((x - sx) ** 2 + (y - sy) ** 2) ** 0.5
            if dist <= radius and dist < stop_dist:
                stop_dist = dist
                stop_hit = ("stop", i)
        if stop_hit is not None:
            return stop_hit
        line_hit = None
        line_dist = 1e9
        for i, stop in enumerate(self._defn.stops):
            sx, _sy = self._stop_xy(layout, stop, dmin, dmax)
            dist = abs(x - sx)
            if dist <= radius and alpha.top() <= y <= bars.bottom() and dist < line_dist:
                line_dist = dist
                line_hit = ("stop", i)
        if line_hit is not None:
            return line_hit
        return None

    def _press(self, widget, event, QtCore):
        button = getattr(event, "button", lambda: None)()
        right = getattr(QtCore.Qt, "RightButton", None)
        if right is not None and button == right:
            return
        x, y = _event_xy(event)
        hit = self._hit(widget, x, y)
        if hit is None:
            self._add_at(widget, x, y)
            return
        kind, payload = hit
        self._drag_moved = False
        if kind == "stop":
            index = int(payload)
            self._press_was_already_selected = index == self._selected
        else:
            self._press_was_already_selected = False
        self._drag = hit
        self._drag_offset = (0.0, 0.0)
        grab = getattr(widget, "grabMouse", None)
        if callable(grab):
            try:
                grab()
            except Exception:
                pass
        if kind == "stop":
            self._selected = int(payload)
            layout = self._layout(widget)
            dmin, dmax = self._span()
            sx, sy = self._stop_xy(layout, self._defn.stops[int(payload)], dmin, dmax)
            self._drag_offset = (x - sx, y - sy)
            if self._on_select is not None:
                self._on_select(int(payload))
        widget.update()

    def _release(self, widget, _event):
        if self._drag is None:
            return
        kind, payload = self._drag
        moved = self._drag_moved
        already = self._press_was_already_selected
        self._drag = None
        self._drag_offset = (0.0, 0.0)
        self._drag_moved = False
        release = getattr(widget, "releaseMouse", None)
        if callable(release):
            try:
                release()
            except Exception:
                pass
        if (
            kind == "stop"
            and not moved
            and already
            and self._on_edit_stop is not None
        ):
            self._on_edit_stop(int(payload))
        if self._on_drag_end is not None:
            self._on_drag_end()
        widget.update()

    def _add_at(self, widget, x, y):
        if self._on_add is None:
            return
        layout = self._layout(widget)
        plot = layout["plot"]
        alpha = layout["alpha"]
        if x < plot.left() or x > plot.right() or y < plot.top() or y > plot.bottom():
            return
        dmin, dmax = self._span()
        axis = self._axis()
        vmin, vmax = self._limits
        pos = data_to_unit(_value_for_x(plot, x, dmin, dmax, axis), vmin, vmax)
        inset = 6
        alpha_val = screen_y_to_alpha(y, alpha.top() + inset, alpha.bottom() - inset)
        self._on_add(pos, alpha_val)

    def _move(self, widget, event):
        if self._drag is None:
            return
        self._drag_moved = True
        kind, payload = self._drag
        x, y = _event_xy(event)
        layout = self._layout(widget)
        plot = layout["plot"]
        alpha = layout["alpha"]
        dmin, dmax = self._span()
        axis = self._axis()
        if kind == "range":
            value = _value_for_x(plot, x, dmin, dmax, axis)
            vmin, vmax = self._limits
            if payload == "vmin":
                vmin, vmax = clamp_range(value, vmax)
            else:
                vmin, vmax = clamp_range(vmin, value)
            if self._on_range is not None:
                self._on_range(vmin, vmax)
            return
        index = int(payload)
        stops = self._defn.stops
        vmin, vmax = self._limits
        ox, oy = self._drag_offset
        x = x - ox
        y = y - oy
        if _is_end_stop(index, stops):
            pos = 0.0 if index <= 0 else 1.0
        else:
            pos = _unit_position_for_histogram_x(plot, x, vmin, vmax, dmin, dmax, axis)
            pos = _clamp_stop_position(index, stops, pos)
        inset = 6
        alpha = screen_y_to_alpha(y, alpha.top() + inset, alpha.bottom() - inset)
        if self._on_stop is not None:
            self._on_stop(index, pos, alpha)

    def _context(self, widget, event):
        x, y = _event_xy(event)
        hit = self._hit(widget, x, y)
        if hit is None:
            return
        kind, payload = hit
        if kind == "range":
            if self._on_ask is None:
                return
            vmin, vmax = self._limits
            current = vmin if payload == "vmin" else vmax
            self._on_ask("range", payload, current)
            return
        return

class _ColorbarPreview:
    """Live colorbar with ticks and values — this is the editor preview."""

    _TICK_PAD = 42

    def __init__(
        self,
        parent,
        vertical=False,
        compact=False,
        show_stops=False,
        on_select=None,
        on_add=None,
        on_edit_stop=None,
        on_stop=None,
        on_drag_end=None,
    ):
        QtCore, QtGui, QtWidgets = qt_modules()
        self._vertical = bool(vertical)
        self._compact = bool(compact)
        self._show_stops = bool(show_stops)
        self._on_select = on_select
        self._on_add = on_add
        self._on_edit_stop = on_edit_stop
        self._on_stop = on_stop
        self._on_drag_end = on_drag_end
        self._defn = definition_from_preset(DEFAULT_SURFACE_COLORMAP)
        self._limits = (0.0, 1.0)
        self._map_limits = (0.0, 1.0)
        self._axis = HISTOGRAM_VIEW_FULL
        self._settings = ColorbarExportSettings()
        self._selected = 0
        self._drag = None
        self._drag_offset = (0.0, 0.0)
        self._drag_moved = False
        self._press_was_already_selected = False
        bar = QtWidgets.QWidget(parent)
        bar.setObjectName("pmvColormapColorbar")
        expanding = getattr(QtWidgets.QSizePolicy, "Expanding", None)
        preferred = getattr(QtWidgets.QSizePolicy, "Preferred", None)
        fixed = getattr(QtWidgets.QSizePolicy, "Fixed", None)
        if vertical:
            bar.setMinimumWidth(72)
            bar.setMaximumWidth(110)
            bar.setMinimumHeight(90)
        elif self._compact:
            bar.setMinimumHeight(60)
            bar.setMaximumHeight(68)
            if expanding is not None and fixed is not None:
                bar.setSizePolicy(expanding, fixed)
        else:
            bar.setMinimumHeight(64)
            bar.setMaximumHeight(80)
            if expanding is not None and preferred is not None:
                bar.setSizePolicy(expanding, preferred)
        bar.setToolTip(
            "Colormap preview with tick values. Drag interior stops along the bar; "
            "end colors stay at min/max. Left-click a selected stop again to edit it."
        )
        bar.paintEvent = lambda _event: self._paint(bar, QtCore, QtGui)
        bar.setMouseTracking(True)
        bar.mousePressEvent = lambda event: self._press(bar, event, QtCore)
        bar.mouseMoveEvent = lambda event: self._move(bar, event)
        bar.mouseReleaseEvent = lambda event: self._release(bar, event)
        bar.contextMenuEvent = lambda event: self._context(bar, event)

        def _resize(event):
            QtWidgets.QWidget.resizeEvent(bar, event)
            bar.update()

        bar.resizeEvent = _resize
        self.widget = bar

    def tick_pad(self) -> int:
        return int(self._TICK_PAD)

    def set_preview(self, defn, limits, settings: ColorbarExportSettings, selected=0, axis=HISTOGRAM_VIEW_FULL, map_limits=None):
        self._defn = defn
        self._limits = limits or (0.0, 1.0)
        self._map_limits = tuple(map_limits) if map_limits is not None else self._limits
        self._axis = axis or HISTOGRAM_VIEW_FULL
        self._settings = settings
        self._selected = int(selected or 0)
        self.widget.update()

    def _paint(self, widget, QtCore, QtGui):
        painter = _begin_widget_paint(QtGui, widget)
        if painter is None:
            return
        try:
            aa = getattr(QtGui.QPainter, "Antialiasing", None)
            if aa is not None:
                painter.setRenderHint(aa, True)
            painter.fillRect(widget.rect(), QtGui.QColor(*ROW))
            caption = "" if self._compact else colorbar_caption(self._settings.title, self._settings.units)
            lo, hi = self._limits
            if self._vertical:
                bar = widget.rect().adjusted(28, 10, -10, -10)
            else:
                pad = self._TICK_PAD
                top = 4 if self._compact or not caption else 14
                bottom = -20 if self._compact else -18
                if caption:
                    painter.setPen(QtGui.QColor(*INK))
                    painter.drawText(pad, 12, caption)
                bar = widget.rect().adjusted(pad, top, -pad, bottom)
            try:
                self._draw_checker(painter, QtGui, bar)
                grad = _ramp_image(
                    QtGui, self._defn,
                    max(2, bar.width()), max(8, bar.height()),
                    vertical=self._vertical,
                    map_limits=self._map_limits,
                    display_limits=self._limits,
                    axis=self._axis,
                )
                painter.drawPixmap(bar.topLeft(), grad)
                painter.setPen(QtGui.QPen(QtGui.QColor(*BORDER), 1))
                painter.drawRect(bar.adjusted(0, 0, -1, -1))
            except Exception:
                pass
            self._paint_ticks(painter, QtCore, QtGui, widget, bar, caption, lo, hi)
            if self._show_stops:
                self._paint_stops(painter, QtGui, bar)
        finally:
            _end_widget_paint(painter)

    def _paint_ticks(self, painter, QtCore, QtGui, widget, bar, caption, lo, hi):
        try:
            ticks = colorbar_tick_values(lo, hi, self._axis, self._settings.tick_count)
            painter.setPen(QtGui.QColor(*INK))
            if self._vertical:
                labels = format_numbers_for_ticks(
                    ticks,
                    self._settings.number_format,
                    self._settings.decimals,
                    self._settings.sig_digits,
                )
                span = max(1, bar.height() - 1)
                for i, text in enumerate(labels):
                    t = value_to_histogram_axis(ticks[i], lo, hi, self._axis)
                    y = bar.bottom() - t * span
                    painter.drawLine(bar.left() - 4, int(y), bar.left(), int(y))
                    painter.drawText(4, int(y) + 4, text)
                if caption:
                    painter.save()
                    painter.translate(12, widget.height() / 2)
                    painter.rotate(-90)
                    painter.drawText(-40, 0, caption)
                    painter.restore()
                return
            span = max(1, bar.width() - 1)
            xs = [
                bar.left() + value_to_histogram_axis(tick, lo, hi, self._axis) * span
                for tick in ticks
            ]
            _, labels, keep = _tick_labels_for_paint(painter, ticks, xs, self._settings)
            _draw_horizontal_tick_labels(
                painter,
                QtCore,
                labels,
                xs,
                keep,
                int(bar.bottom()),
                int(bar.bottom()) + 4,
            )
        except Exception:
            return

    def _bar_rect(self, widget, caption=""):
        if self._vertical:
            return widget.rect().adjusted(28, 10, -10, -10)
        pad = self._TICK_PAD
        top = 4 if self._compact or not caption else 14
        bottom = -20 if self._compact else -18
        return widget.rect().adjusted(pad, top, -pad, bottom)

    def _paint_stops(self, painter, QtGui, bar):
        try:
            stops = getattr(self._defn, "stops", ()) or ()
            selected = max(0, min(int(self._selected), max(0, len(stops) - 1)))
            y = bar.center().y()
            for i, stop in enumerate(stops):
                x = unit_to_axis(stop.position, bar.left(), bar.right())
                _draw_stop_circle(painter, QtGui, x, y, stop.rgba, i == selected, 7)
        except Exception:
            return

    def _hit_stop(self, widget, x, y):
        bar = self._bar_rect(widget)
        if y < bar.top() - 6 or y > bar.bottom() + 10:
            return None
        best = None
        for i, stop in enumerate(getattr(self._defn, "stops", ()) or ()):
            sx = unit_to_axis(stop.position, bar.left(), bar.right())
            dist = abs(x - sx)
            if dist <= 12 and (best is None or dist < best[0]):
                best = (dist, i)
        if best is None:
            return None
        return best[1]

    def _press(self, widget, event, QtCore):
        button = getattr(event, "button", lambda: None)()
        right = getattr(QtCore.Qt, "RightButton", None)
        if right is not None and button == right:
            return
        x, y = _event_xy(event)
        index = self._hit_stop(widget, x, y)
        if index is not None:
            was_selected = int(index) == self._selected
            self._selected = int(index)
            if self._on_select is not None:
                self._on_select(int(index))
            if self._on_stop is not None:
                self._drag_moved = False
                self._press_was_already_selected = was_selected
                self._drag = ("stop", int(index))
                bar = self._bar_rect(widget)
                sx = unit_to_axis(
                    self._defn.stops[int(index)].position, bar.left(), bar.right(),
                )
                self._drag_offset = (x - sx, 0.0)
                grab = getattr(widget, "grabMouse", None)
                if callable(grab):
                    try:
                        grab()
                    except Exception:
                        pass
            elif was_selected and self._on_edit_stop is not None:
                self._on_edit_stop(int(index))
            widget.update()
            return
        if self._on_add is None:
            return
        bar = self._bar_rect(widget)
        if x < bar.left() or x > bar.right() or y < bar.top() - 4 or y > bar.bottom() + 8:
            return
        pos = axis_to_unit(x, bar.left(), bar.right())
        self._on_add(pos, None)

    def _release(self, widget, _event):
        if self._drag is None:
            return
        kind, payload = self._drag
        moved = self._drag_moved
        already = self._press_was_already_selected
        self._drag = None
        self._drag_offset = (0.0, 0.0)
        self._drag_moved = False
        release = getattr(widget, "releaseMouse", None)
        if callable(release):
            try:
                release()
            except Exception:
                pass
        if (
            kind == "stop"
            and not moved
            and already
            and self._on_edit_stop is not None
        ):
            self._on_edit_stop(int(payload))
        if self._on_drag_end is not None:
            self._on_drag_end()
        widget.update()

    def _move(self, widget, event):
        if self._drag is None or self._on_stop is None:
            return
        kind, payload = self._drag
        if kind != "stop":
            return
        index = int(payload)
        stops = getattr(self._defn, "stops", ()) or ()
        if _is_end_stop(index, stops):
            return
        self._drag_moved = True
        x, _y = _event_xy(event)
        x = x - self._drag_offset[0]
        bar = self._bar_rect(widget)
        pos = axis_to_unit(x, bar.left(), bar.right())
        pos = _clamp_stop_position(index, stops, pos)
        stop = stops[index]
        rgba = _as_rgba(stop.rgba)
        self._on_stop(index, pos, float(rgba[3]))

    def _context(self, widget, event):
        return

    def _draw_checker(self, painter, QtGui, rect):
        if rect.width() <= 1 or rect.height() <= 1:
            return
        size = 6
        light = QtGui.QColor(230, 230, 230)
        dark = QtGui.QColor(200, 200, 200)
        left, top = int(rect.left()), int(rect.top())
        right, bottom = int(rect.right()), int(rect.bottom())
        y = top
        while y < bottom:
            x = left
            row = (y - top) // size
            while x < right:
                painter.fillRect(
                    x, y,
                    min(size, right - x),
                    min(size, bottom - y),
                    light if ((x - left) // size + row) % 2 == 0 else dark,
                )
                x += size
            y += size


def _style_swatch(button, rgba):
    button.setFixedSize(28, 20)
    button.setStyleSheet(swatch_button_css(rgba_to_css(_as_rgba(rgba)), " min-width: 28px; min-height: 18px;"))


class ExportColorbarDialog:
    def __init__(self, parent, mapping: FieldColorMapping, values=None, cmd=None):
        QtCore, _, QtWidgets = qt_modules()
        overlay = overlay_window(parent)
        dialog = QtWidgets.QDialog(overlay if overlay is not None else parent)
        dialog.setWindowTitle(EXPORT_TITLE)
        dialog.setModal(False)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        _configure_editor_window(dialog)
        apply_wizard_page_style(dialog)
        self._dialog = dialog
        self._mapping = mapping
        self._values = values
        self._cmd = cmd
        self._settings = ColorbarExportSettings(
            title=mapping.title, units=mapping.units,
        )
        root = QtWidgets.QVBoxLayout(dialog)
        form = QtWidgets.QFormLayout()
        self._orient_h = QtWidgets.QRadioButton("Horizontal")
        self._orient_v = QtWidgets.QRadioButton("Vertical")
        self._orient_h.setChecked(True)
        orient = QtWidgets.QHBoxLayout()
        orient.addWidget(self._orient_h)
        orient.addWidget(self._orient_v)
        orient.addStretch(1)
        form.addRow("Orientation", orient)
        self._scale = QtWidgets.QComboBox()
        self._scale.addItem(SCALE_FULL_LABEL, HISTOGRAM_VIEW_FULL)
        self._scale.addItem(SCALE_LOG_LABEL, HISTOGRAM_VIEW_LOG)
        self._scale.addItem(SCALE_PCT_LABEL, HISTOGRAM_VIEW_PERCENTILE)
        form.addRow("Scale", self._scale)
        self._width = QtWidgets.QSpinBox()
        self._height = QtWidgets.QSpinBox()
        self._dpi = QtWidgets.QSpinBox()
        self._width.setRange(32, 4000)
        self._height.setRange(32, 4000)
        self._dpi.setRange(36, 1200)
        self._width.setValue(600)
        self._height.setValue(100)
        self._dpi.setValue(300)
        self._width.setSuffix(" px")
        self._height.setSuffix(" px")
        form.addRow("Width", self._width)
        form.addRow("Height", self._height)
        form.addRow("DPI", self._dpi)
        self._title = QtWidgets.QLineEdit(mapping.title)
        self._units = QtWidgets.QLineEdit(mapping.units)
        self._ticks = QtWidgets.QSpinBox()
        self._ticks.setRange(2, 20)
        self._ticks.setValue(5)
        self._format = QtWidgets.QComboBox()
        self._format.addItem("Auto", "auto")
        self._format.addItem("Fixed decimal", "fixed")
        self._format.addItem("Scientific notation", "scientific")
        self._decimals = QtWidgets.QSpinBox()
        self._decimals.setRange(0, 8)
        self._decimals.setValue(2)
        form.addRow("Title", self._title)
        form.addRow("Units", self._units)
        form.addRow("Tick count", self._ticks)
        form.addRow("Number format", self._format)
        form.addRow("Decimals", self._decimals)
        self._bg = QtWidgets.QComboBox()
        self._bg.addItem("Transparent", "transparent")
        self._bg.addItem("White", "white")
        self._bg.addItem("Black", "black")
        form.addRow("Background", self._bg)
        fmt_row = QtWidgets.QHBoxLayout()
        self._fmt_png = QtWidgets.QRadioButton("PNG")
        self._fmt_svg = QtWidgets.QRadioButton("SVG")
        self._fmt_png.setChecked(True)
        fmt_row.addWidget(self._fmt_png)
        fmt_row.addWidget(self._fmt_svg)
        fmt_row.addStretch(1)
        form.addRow("Format", fmt_row)
        root.addLayout(form)
        preview_box = make_section("Preview")
        self._preview = _ColorbarPreview(dialog, vertical=False)
        preview_box.layout.addWidget(self._preview.widget)
        root.addWidget(preview_box.widget)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        cancel = QtWidgets.QPushButton("Cancel")
        export = QtWidgets.QPushButton("Export")
        apply_secondary_button_style(cancel)
        mark_primary_button(export)
        cancel.clicked.connect(dialog.reject)
        export.clicked.connect(self._export)
        buttons.addWidget(cancel)
        buttons.addWidget(export)
        root.addLayout(buttons)
        for widget in (
            self._orient_h, self._orient_v, self._scale, self._width, self._height, self._dpi,
            self._title, self._units, self._ticks, self._format, self._decimals,
            self._bg, self._fmt_png, self._fmt_svg,
        ):
            signal = getattr(widget, "valueChanged", None) or getattr(widget, "toggled", None) or getattr(widget, "currentIndexChanged", None) or getattr(widget, "textChanged", None)
            if signal is not None:
                signal.connect(lambda *_: self._refresh_preview())
        self._refresh_preview()

    def _current_settings(self) -> ColorbarExportSettings:
        return ColorbarExportSettings(
            orientation="vertical" if self._orient_v.isChecked() else "horizontal",
            width=int(self._width.value()),
            height=int(self._height.value()),
            dpi=int(self._dpi.value()),
            title=self._title.text(),
            units=self._units.text(),
            tick_count=int(self._ticks.value()),
            number_format=str(self._format.currentData() or "auto"),
            decimals=int(self._decimals.value()),
            background=str(self._bg.currentData() or "transparent"),
            fmt="svg" if self._fmt_svg.isChecked() else "png",
            scale=str(self._scale.currentData() or HISTOGRAM_VIEW_FULL),
        )

    def _refresh_preview(self):
        settings = self._current_settings()
        axis, lo, hi, map_lo, map_hi = colorbar_export_display(
            self._mapping.normalization, self._values, settings.scale,
        )
        self._preview.set_preview(
            self._mapping.colormap, (lo, hi), settings,
            axis=axis, map_limits=(map_lo, map_hi),
        )

    def _export(self):
        settings = self._current_settings()
        suffix = ".svg" if settings.fmt == "svg" else ".png"
        path, _ = overlay_get_save_file_name(
            self._dialog,
            "Export colorbar",
            "colorbar%s" % suffix,
            "SVG (*.svg);;PNG (*.png)" if settings.fmt == "svg" else "PNG (*.png);;SVG (*.svg)",
        )
        if not path:
            return
        if not path.lower().endswith((".png", ".svg")):
            path = path + suffix
        export_colorbar(path, self._mapping, settings, values=self._values)
        self._dialog.accept()

    def show(self):
        self._dialog.show()
        bind_tool_window(self._dialog)
        self._dialog.raise_()
        self._dialog.activateWindow()
