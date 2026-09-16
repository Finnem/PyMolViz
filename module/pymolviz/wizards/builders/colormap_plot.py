"""Colormap histogram, colorbar preview, stop-drag widgets, and export dialog."""

from __future__ import annotations

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
    data_to_unit,
    field_histogram,
    field_title,
    field_units,
    field_values_for_stats,
    format_number,
    definition_from_preset,
    ramp_rgba,
    resolve_limits,
    sample_unit,
    screen_y_to_alpha,
    tick_values,
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
from ..widgets.theme import INK, PRIMARY, apply_secondary_button_style, swatch_button_css
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


def _ramp_image(QtGui, defn, width, height, vertical=False):
    width = max(2, int(width))
    height = max(2, int(height))
    n = height if vertical else width
    fmt = _image_format(QtGui, with_alpha=True)
    empty = QtGui.QPixmap(width, height)
    empty.fill(QtGui.QColor(0, 0, 0, 0))
    if fmt is None:
        return empty
    try:
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


def _plot_rect(widget, pad_l=10, pad_r=10, pad_t=8, pad_b=22):
    return widget.rect().adjusted(pad_l, pad_t, -pad_r, -pad_b)


def _data_span(hist, limits):
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


def _x_for_value(rect, value, dmin, dmax) -> float:
    if value is None:
        value = dmin
    return unit_to_axis(data_to_unit(value, dmin, dmax), rect.left(), rect.right())


def _value_for_x(rect, x, dmin, dmax) -> float:
    return unit_to_data(data_to_unit(x, rect.left(), rect.right()), dmin, dmax)


def _stop_neighbor_limits(index: int, stops) -> Tuple[float, float]:
    """Unit-interval bounds for stop ``index`` (endpoints may move inward from 0/1)."""
    n = len(stops or ())
    if n <= 0:
        return 0.0, 1.0
    index = max(0, min(int(index), n - 1))
    lo = 0.0 if index <= 0 else float(stops[index - 1].position) + 0.004
    hi = 1.0 if index >= n - 1 else float(stops[index + 1].position) - 0.004
    if hi < lo:
        mid = 0.5 * (lo + hi)
        return mid, mid
    return lo, hi


def _clamp_stop_position(index: int, stops, position: float) -> float:
    lo, hi = _stop_neighbor_limits(index, stops)
    return max(lo, min(hi, float(position)))


def _unit_position_for_histogram_x(hist, x, vmin, vmax, dmin, dmax) -> float:
    """Map a histogram x pixel to colormap unit position along ``vmin``…``vmax``."""
    x_lo = _x_for_value(hist, vmin, dmin, dmax)
    x_hi = _x_for_value(hist, vmax, dmin, dmax)
    return axis_to_unit(x, x_lo, x_hi)


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
    spin.setSingleStep(0.1)
    apply_ascii_float_locale(spin, QtCore)
    spin.setValue(float(value))
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
        self._drag_moved = False
        self._press_was_already_selected = False
        view = QtWidgets.QWidget(parent)
        view.setObjectName("pmvColormapDistribution")
        view.setMinimumHeight(168)
        view.setMaximumHeight(220)
        expanding = getattr(QtWidgets.QSizePolicy, "Expanding", None)
        fixed = getattr(QtWidgets.QSizePolicy, "Fixed", None)
        if expanding is not None and fixed is not None:
            view.setSizePolicy(expanding, fixed)
        hint = getattr(QtCore, "QSize", None)
        if hint is not None:
            view.sizeHint = lambda: hint(400, 180)
            view.minimumSizeHint = lambda: hint(240, 168)
        view.setMouseTracking(True)
        view.setToolTip(
            "Drag min/max handles to set the colormap range. Drag color stops "
            "left/right (value) and up/down (opacity). Left-click empty space to add a "
            "stop. Left-click a selected stop again to edit it; right-click a range handle "
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
        return widget.rect().adjusted(12, 16, -12, -8)

    def _span(self):
        return _data_span(self._hist, self._limits)

    def _stop_xy(self, hist, stop, dmin, dmax):
        vmin, vmax = self._limits
        value = unit_to_data(stop.position, vmin, vmax)
        x = _x_for_value(hist, value, dmin, dmax)
        inset = 10
        y = alpha_to_screen_y(stop.rgba[3], hist.top() + inset, hist.bottom() - inset)
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
            hist = self._plot(widget)
            dmin, dmax = self._span()
            vmin, vmax = self._limits
            self._paint_histogram(painter, QtCore, QtGui, hist, dmin, dmax, vmin, vmax)
            try:
                painter.setPen(QtGui.QPen(QtGui.QColor(*BORDER), 1))
                painter.setBrush(QtGui.QColor(0, 0, 0, 0))
                painter.drawRect(hist.adjusted(0, 0, -1, -1))
            except Exception:
                pass
            self._paint_handles(painter, QtCore, QtGui, hist, dmin, dmax, vmin, vmax)
        finally:
            _end_widget_paint(painter)

    def _paint_histogram(self, painter, QtCore, QtGui, hist, dmin, dmax, vmin, vmax):
        counts = list((self._hist or {}).get("counts") or ())
        edges = list((self._hist or {}).get("edges") or ())
        if not counts:
            try:
                painter.setPen(QtGui.QColor(*MUTED))
                painter.drawText(
                    hist.adjusted(8, 8, -8, -8),
                    getattr(QtCore.Qt, "AlignTop", 0x20),
                    "No field values",
                )
            except Exception:
                pass
            return
        try:
            peak = max(counts) or 1
            bar_w = max(1.0, hist.width() / float(len(counts)))
            painter.setPen(getattr(QtCore.Qt, "NoPen", 0))
            fallback = QtGui.QColor(*PRIMARY)
            fallback.setAlpha(140)
            muted = QtGui.QColor(*MUTED)
            muted.setAlpha(90)
            for i, count in enumerate(counts):
                bh = int(round((count / float(peak)) * hist.height()))
                x = hist.left() + i * bar_w
                fill = fallback
                if len(edges) >= i + 2:
                    mid = 0.5 * (float(edges[i]) + float(edges[i + 1]))
                    if mid < vmin or mid > vmax:
                        fill = muted
                    else:
                        try:
                            rgba = sample_unit(self._defn, data_to_unit(mid, vmin, vmax))
                            fill = _qcolor(QtGui, rgba)
                            fill.setAlpha(160)
                        except Exception:
                            fill = fallback
                painter.setBrush(fill)
                painter.drawRect(int(x), hist.bottom() - bh, max(1, int(bar_w) - 1), bh)
        except Exception:
            return

    def _paint_handles(self, painter, QtCore, QtGui, hist, dmin, dmax, vmin, vmax):
        dash = getattr(QtCore.Qt, "DashLine", 2)
        try:
            painter.setPen(QtGui.QPen(QtGui.QColor(*INK), 1.5))
            prev = None
            for stop in self._defn.stops:
                x, y = self._stop_xy(hist, stop, dmin, dmax)
                if prev is not None:
                    painter.drawLine(int(prev[0]), int(prev[1]), int(x), int(y))
                prev = (x, y)
            for i, stop in enumerate(self._defn.stops):
                x, y = self._stop_xy(hist, stop, dmin, dmax)
                painter.setPen(QtGui.QPen(QtGui.QColor(*MUTED), 1, dash))
                painter.drawLine(int(x), hist.top(), int(x), hist.bottom())
                _draw_stop_circle(painter, QtGui, x, y, stop.rgba, i == self._selected, self._STOP_R)
            for which, value in (("vmin", vmin), ("vmax", vmax)):
                x = _x_for_value(hist, value, dmin, dmax)
                painter.setPen(QtGui.QPen(QtGui.QColor(*PRIMARY), 2))
                painter.drawLine(int(x), hist.top(), int(x), hist.bottom())
                self._draw_range_handle(painter, QtCore, QtGui, x, hist.top() - 2, which == "vmin")
            if self._center is not None:
                painter.setPen(QtGui.QPen(QtGui.QColor(*MUTED), 1, dash))
                cx = _x_for_value(hist, float(self._center), dmin, dmax)
                painter.drawLine(int(cx), hist.top(), int(cx), hist.bottom())
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
                pts = [point(int(x), y), point(int(x - 9), y - 12), point(int(x + 3), y - 12)]
            else:
                pts = [point(int(x), y), point(int(x - 3), y - 12), point(int(x + 9), y - 12)]
            painter.drawPolygon(poly_cls(pts))
        else:
            painter.drawRect(int(x) - 5, y - 12, 10, 12)

    def _hit(self, widget, x, y):
        hist = self._plot(widget)
        dmin, dmax = self._span()
        vmin, vmax = self._limits
        stop_hit = None
        stop_dist = 1e9
        radius = self._STOP_R + 5
        for i, stop in enumerate(self._defn.stops):
            sx, sy = self._stop_xy(hist, stop, dmin, dmax)
            dist = ((x - sx) ** 2 + (y - sy) ** 2) ** 0.5
            if dist <= radius and dist < stop_dist:
                stop_dist = dist
                stop_hit = ("stop", i)
        if stop_hit is not None:
            return stop_hit
        # The endpoint stops sit on the vmin/vmax lines. Prefer those over
        # "add a stop" when the click is on a stop's value line.
        line_hit = None
        line_dist = 1e9
        for i, stop in enumerate(self._defn.stops):
            sx, _sy = self._stop_xy(hist, stop, dmin, dmax)
            dist = abs(x - sx)
            if dist <= radius and hist.top() <= y <= hist.bottom() and dist < line_dist:
                line_dist = dist
                line_hit = ("stop", i)
        if line_hit is not None:
            return line_hit
        # Range triangles live *above* the plot so they do not steal endpoint stops.
        for which, value in (("vmin", vmin), ("vmax", vmax)):
            hx = _x_for_value(hist, value, dmin, dmax)
            if abs(x - hx) <= 12 and (hist.top() - 16) <= y <= hist.top() + 2:
                return ("range", which)
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
        grab = getattr(widget, "grabMouse", None)
        if callable(grab):
            try:
                grab()
            except Exception:
                pass
        if kind == "stop":
            self._selected = int(payload)
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
        hist = self._plot(widget)
        if x < hist.left() or x > hist.right() or y < hist.top() or y > hist.bottom():
            return
        dmin, dmax = self._span()
        vmin, vmax = self._limits
        pos = data_to_unit(_value_for_x(hist, x, dmin, dmax), vmin, vmax)
        inset = 10
        alpha = screen_y_to_alpha(y, hist.top() + inset, hist.bottom() - inset)
        self._on_add(pos, alpha)

    def _move(self, widget, event):
        if self._drag is None:
            return
        self._drag_moved = True
        kind, payload = self._drag
        x, y = _event_xy(event)
        hist = self._plot(widget)
        dmin, dmax = self._span()
        if kind == "range":
            value = _value_for_x(hist, x, dmin, dmax)
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
        pos = _unit_position_for_histogram_x(hist, x, self._limits[0], self._limits[1], dmin, dmax)
        pos = _clamp_stop_position(index, stops, pos)
        inset = 10
        alpha = screen_y_to_alpha(y, hist.top() + inset, hist.bottom() - inset)
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
        self._settings = ColorbarExportSettings()
        self._selected = 0
        self._drag = None
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
            "Colormap preview with tick values. Left-click a selected stop again to edit it."
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

    def set_preview(self, defn, limits, settings: ColorbarExportSettings, selected=0):
        self._defn = defn
        self._limits = limits or (0.0, 1.0)
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
            ticks = tick_values(lo, hi, self._settings.tick_count)
            labels = [
                format_number(v, self._settings.number_format, self._settings.decimals, self._settings.sig_digits)
                for v in ticks
            ]
            painter.setPen(QtGui.QColor(*INK))
            if self._vertical:
                span = max(1, bar.height() - 1)
                for i, text in enumerate(labels):
                    t = i / float(max(len(labels) - 1, 1))
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
            last = max(len(labels) - 1, 1)
            align_top = getattr(QtCore.Qt, "AlignTop", 0x20)
            align_left = getattr(QtCore.Qt, "AlignLeft", 0x1) | align_top
            align_right = getattr(QtCore.Qt, "AlignRight", 0x2) | align_top
            align_center = getattr(QtCore.Qt, "AlignHCenter", 0x4) | align_top
            for i, text in enumerate(labels):
                t = i / float(last)
                x = bar.left() + t * span
                painter.drawLine(int(x), bar.bottom(), int(x), bar.bottom() + 4)
                if i == 0:
                    painter.drawText(int(x), bar.bottom() + 4, 72, 16, align_left, text)
                elif i == len(labels) - 1:
                    painter.drawText(int(x) - 72, bar.bottom() + 4, 72, 16, align_right, text)
                else:
                    painter.drawText(int(x) - 36, bar.bottom() + 4, 72, 16, align_center, text)
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
        self._drag_moved = True
        index = int(payload)
        x, _y = _event_xy(event)
        bar = self._bar_rect(widget)
        stops = getattr(self._defn, "stops", ()) or ()
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
            self._orient_h, self._orient_v, self._width, self._height, self._dpi,
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
        )

    def _refresh_preview(self):
        settings = self._current_settings()
        limits = resolve_limits(self._mapping.normalization, self._values) or (0.0, 1.0)
        self._preview.set_preview(self._mapping.colormap, limits, settings)

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
