"""Full Colormap Editor and Export Colorbar dialogs (cheap previews only)."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ...util.colormap_spec import (
    INTERP_HSV,
    INTERP_RGB,
    MAP_CONTINUOUS,
    MAP_DISCRETE,
    OOR_CLAMP,
    OOR_CUSTOM,
    OOR_TRANSPARENT,
    RANGE_MODE_AUTO,
    RANGE_MODE_CUSTOM,
    RANGE_MODE_PERCENTILE,
    RANGE_MODE_SYMMETRIC,
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
    definition_from_preset,
    field_histogram,
    field_title,
    field_units,
    field_values_for_stats,
    format_number,
    listed_preset_names,
    load_custom_presets,
    ramp_rgba,
    resolve_limits,
    reverse_definition,
    sample_unit,
    screen_y_to_alpha,
    tick_values,
    unit_to_axis,
    unit_to_data,
    unused_custom_preset_name,
)
from ...util.colorbar_export import export_colorbar
from ...util.field_sample import DEFAULT_SURFACE_COLORMAP
from ..pick import (
    bind_tool_window,
    overlay_exec,
    overlay_get_save_file_name,
    overlay_information,
    overlay_window,
    qt_modules,
    qt_widget_alive,
)
from ..widgets.ascii_locale import apply_ascii_float_locale
from ..widgets.section import make_section
from ..widgets.switch import make_switch
from ..widgets.theme import (
    BORDER,
    INK,
    MUTED,
    PRIMARY,
    ROW,
    apply_secondary_button_style,
    apply_wizard_page_style,
    catalog_table_css,
    compact_primary_button_css,
    mark_primary_button,
    rgb_css,
    swatch_button_css,
)
from .colormap_editor import (
    ADD_CUSTOM_COLORMAP_TIP,
    ColormapPresetPicker,
)
from .colors import (
    ColorChoice,
    bind_color_pick_result,
    pick_rgb,
    rgba_to_css,
)

EDITOR_TITLE = "Edit Colormap"
EXPORT_TITLE = "Export Colorbar"
EDIT_LABEL = "Edit..."
EXPORT_COLORBAR_LABEL = "Export Colorbar..."
SAVE_AS_LABEL = "Save As..."
RESET_LABEL = "Reset"
REVERSE_LABEL = "Reverse"
ADD_STOP_LABEL = "Add stop"
DELETE_STOP_LABEL = "Delete stop"
HELP_TITLE = "Colormap editor"
HELP_TEXT = (
    "Drag the distribution handles to set min/max. Right-click a range handle to "
    "type an explicit value. Color stops move left/right along the mapped range and "
    "up/down for opacity. Left-click empty space to add a stop. Left-click a "
    "selected stop again (without dragging) to edit its color, position, and opacity. "
    "The colorbar below is the live preview."
)

RANGE_AUTO_LABEL = "Auto"
RANGE_CUSTOM_LABEL = "Custom"
RANGE_SYMMETRIC_LABEL = "Symmetric around zero"
RANGE_PERCENTILE_LABEL = "Percentile"

INTERP_RGB_LABEL = "Linear RGB"
INTERP_HSV_LABEL = "Linear HSV"
MAP_CONTINUOUS_LABEL = "Continuous"
MAP_DISCRETE_LABEL = "Discrete"
OOR_CLAMP_LABEL = "Clamp to end colors"
OOR_TRANSPARENT_LABEL = "Transparent"
OOR_CUSTOM_LABEL = "Custom below / above"

_LIVE_EDITOR = []


def colormap_editor_title() -> str:
    return EDITOR_TITLE


def colormap_editor_action_labels() -> Tuple[str, str, str, str]:
    return (ADD_CUSTOM_COLORMAP_TIP, SAVE_AS_LABEL, EDIT_LABEL, EXPORT_COLORBAR_LABEL)


def colormap_full_range_labels() -> Tuple[str, str, str, str]:
    return (RANGE_AUTO_LABEL, RANGE_CUSTOM_LABEL, RANGE_SYMMETRIC_LABEL, RANGE_PERCENTILE_LABEL)


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


class ColormapEditorDialog:
    def __init__(
        self,
        parent,
        mapping: FieldColorMapping,
        *,
        values=None,
        cmd=None,
        on_change=None,
        on_apply=None,
        on_done=None,
    ):
        QtCore, _, QtWidgets = qt_modules()

        class _Window(QtWidgets.QDialog):
            def closeEvent(inner, event):
                if self._user_cancel or self._closing:
                    event.accept()
                    return
                self._close_stop_editor()
                if not self._store_custom_colormap():
                    event.ignore()
                    return
                self._applied = True
                if self._on_apply is not None:
                    self._on_apply(self._mapping)
                if self._on_done is not None:
                    self._on_done(self._mapping)
                self._closing = True
                _forget_editor(self)
                event.accept()

        dialog = _Window()
        dialog.setWindowTitle(EDITOR_TITLE)
        dialog.setModal(False)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        _configure_editor_window(dialog)
        apply_wizard_page_style(dialog)
        dialog.resize(720, 760)
        self._dialog = dialog
        self._cmd = cmd
        self._on_change = on_change
        self._on_apply = on_apply
        self._on_done = on_done
        self._original = mapping
        self._mapping = mapping
        self._values = values
        self._selected = 0
        self._syncing = False
        self._applied = False
        self._stored_key = None
        self._user_cancel = False
        self._closing = False
        self._debounce = QtCore.QTimer(dialog)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(250)
        self._debounce.timeout.connect(self._emit_live)

        root = QtWidgets.QVBoxLayout(dialog)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        body = QtWidgets.QWidget()
        apply_wizard_page_style(body)
        col = QtWidgets.QVBoxLayout(body)
        scroll.setWidget(body)
        root.addWidget(scroll, stretch=1)

        preset = make_section("Preset and actions")
        row = QtWidgets.QHBoxLayout()
        self._picker = ColormapPresetPicker(
            dialog,
            on_selected=lambda name: self._on_preset(),
            on_add=self._add_custom_preset,
            on_edit=self._edit_named_preset,
            default_name=DEFAULT_SURFACE_COLORMAP,
        )
        self._preset = self._picker.combo
        self._save = QtWidgets.QPushButton(SAVE_AS_LABEL)
        self._reset = QtWidgets.QPushButton(RESET_LABEL)
        self._reverse = QtWidgets.QPushButton(REVERSE_LABEL)
        self._more = QtWidgets.QToolButton()
        self._more.setText("More")
        self._more.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(self._more)
        menu.addAction(EXPORT_COLORBAR_LABEL, self._export_colorbar)
        self._more.setMenu(menu)
        apply_secondary_button_style(self._save)
        apply_secondary_button_style(self._reset)
        apply_secondary_button_style(self._reverse)
        row.addWidget(QtWidgets.QLabel("Preset"))
        row.addWidget(self._picker.widget, stretch=1)
        row.addWidget(self._save)
        row.addWidget(self._reset)
        row.addWidget(self._reverse)
        row.addWidget(self._more)
        preset.layout.addLayout(row)
        col.addWidget(preset.widget)

        hist = make_section("Value distribution")
        hist_row = QtWidgets.QHBoxLayout()
        self._histogram = _DistributionEditor(
            body,
            on_stop=self._on_stop_dragged,
            on_select=self._on_handle_selected,
            on_range=self._on_range_dragged,
            on_ask=self._on_handle_asked,
            on_add=self._add_color_stop_at,
            on_edit_stop=self._open_stop_editor,
            on_drag_end=self._on_handle_drag_finished,
        )
        self._hist_stats = QtWidgets.QLabel("")
        self._hist_stats.setWordWrap(True)
        hist_row.addWidget(self._histogram.widget, stretch=1)
        hist_row.addWidget(self._hist_stats)
        hist.layout.addLayout(hist_row)
        self._colorbar = _ColorbarPreview(
            body, compact=True, show_stops=True,
            on_select=self._on_handle_selected,
            on_add=self._add_color_stop_at,
            on_edit_stop=self._open_stop_editor,
            on_stop=self._on_stop_dragged,
            on_drag_end=self._on_handle_drag_finished,
        )
        hist.layout.addWidget(self._colorbar.widget)
        self._hist_section = hist
        col.addWidget(hist.widget)

        self._pos = None
        self._swatch = None
        self._opacity = None
        self._delete = None
        self._stop_popup = None

        table_sec = make_section("Color stops")
        self._table = QtWidgets.QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["#", "Position", "Color", "Opacity", "Label"])
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self._table.verticalHeader().setVisible(False)
        self._table.setStyleSheet(catalog_table_css("pmvColormapStops", "pmvColormapStopsHeader"))
        self._table.setObjectName("pmvColormapStops")
        self._add_stop = QtWidgets.QPushButton("+ " + ADD_STOP_LABEL)
        apply_secondary_button_style(self._add_stop)
        table_sec.layout.addWidget(self._table)
        table_sec.layout.addWidget(self._add_stop)
        col.addWidget(table_sec.widget)

        rng = make_section("Range & scaling", form=True)
        self._range = QtWidgets.QComboBox()
        self._range.addItem(RANGE_AUTO_LABEL, RANGE_MODE_AUTO)
        self._range.addItem(RANGE_CUSTOM_LABEL, RANGE_MODE_CUSTOM)
        self._range.addItem(RANGE_SYMMETRIC_LABEL, RANGE_MODE_SYMMETRIC)
        self._range.addItem(RANGE_PERCENTILE_LABEL, RANGE_MODE_PERCENTILE)
        self._vmin = QtWidgets.QDoubleSpinBox()
        self._vmax = QtWidgets.QDoubleSpinBox()
        self._center = QtWidgets.QDoubleSpinBox()
        for spin in (self._vmin, self._vmax, self._center):
            spin.setDecimals(4)
            spin.setRange(-1e8, 1e8)
            spin.setSingleStep(0.1)
            apply_ascii_float_locale(spin, QtCore)
        self._link_zero = make_switch("Link center to 0")
        self._link_zero.setChecked(True)
        self._pct_lo = QtWidgets.QDoubleSpinBox()
        self._pct_hi = QtWidgets.QDoubleSpinBox()
        self._pct_lo.setRange(0.0, 50.0)
        self._pct_hi.setRange(50.0, 100.0)
        self._pct_lo.setSuffix(" %")
        self._pct_hi.setSuffix(" %")
        self._pct_lo.setValue(2.0)
        self._pct_hi.setValue(98.0)
        rng.layout.addRow("Range mode", self._range)
        rng.layout.addRow("Min", self._vmin)
        rng.layout.addRow("Center", self._center)
        rng.layout.addRow("Max", self._vmax)
        rng.layout.addRow("", self._link_zero)
        pct = QtWidgets.QHBoxLayout()
        pct.addWidget(self._pct_lo)
        pct.addWidget(self._pct_hi)
        rng.layout.addRow("Percentile", pct)
        col.addWidget(rng.widget)

        mapping_sec = make_section("Mapping options", form=True)
        self._interp = QtWidgets.QComboBox()
        self._interp.addItem(INTERP_RGB_LABEL, INTERP_RGB)
        self._interp.addItem(INTERP_HSV_LABEL, INTERP_HSV)
        type_row = QtWidgets.QHBoxLayout()
        self._type_cont = QtWidgets.QRadioButton(MAP_CONTINUOUS_LABEL)
        self._type_disc = QtWidgets.QRadioButton(MAP_DISCRETE_LABEL)
        self._type_cont.setChecked(True)
        type_row.addWidget(self._type_cont)
        type_row.addWidget(self._type_disc)
        type_row.addStretch(1)
        self._levels = QtWidgets.QSpinBox()
        self._levels.setRange(2, 32)
        self._levels.setValue(5)
        self._nan_swatch = QtWidgets.QPushButton()
        self._nan_transparent = make_switch("Transparent")
        self._oor = QtWidgets.QComboBox()
        self._oor.addItem(OOR_CLAMP_LABEL, OOR_CLAMP)
        self._oor.addItem(OOR_TRANSPARENT_LABEL, OOR_TRANSPARENT)
        self._oor.addItem(OOR_CUSTOM_LABEL, OOR_CUSTOM)
        self._below = QtWidgets.QPushButton()
        self._above = QtWidgets.QPushButton()
        mapping_sec.layout.addRow("Interpolation", self._interp)
        mapping_sec.layout.addRow("Map type", type_row)
        mapping_sec.layout.addRow("Levels", self._levels)
        nan_row = QtWidgets.QHBoxLayout()
        nan_row.addWidget(self._nan_swatch)
        nan_row.addWidget(self._nan_transparent)
        nan_row.addStretch(1)
        mapping_sec.layout.addRow("NaN / missing", nan_row)
        mapping_sec.layout.addRow("Out of range", self._oor)
        mapping_sec.layout.addRow("Below min", self._below)
        mapping_sec.layout.addRow("Above max", self._above)
        self._export_btn = QtWidgets.QPushButton(EXPORT_COLORBAR_LABEL)
        apply_secondary_button_style(self._export_btn)
        mapping_sec.layout.addRow("", self._export_btn)
        col.addWidget(mapping_sec.widget)
        col.addStretch(1)

        footer = QtWidgets.QHBoxLayout()
        help_btn = QtWidgets.QPushButton("Help")
        apply_secondary_button_style(help_btn)
        help_btn.clicked.connect(lambda *_: overlay_information(dialog, HELP_TITLE, HELP_TEXT))
        cancel = QtWidgets.QPushButton("Cancel")
        apply_btn = QtWidgets.QPushButton("Apply")
        ok = QtWidgets.QPushButton("OK")
        apply_secondary_button_style(cancel)
        apply_secondary_button_style(apply_btn)
        mark_primary_button(ok)
        cancel.clicked.connect(self._cancel)
        apply_btn.clicked.connect(self._apply)
        ok.clicked.connect(self._ok)
        footer.addWidget(help_btn)
        footer.addStretch(1)
        footer.addWidget(cancel)
        footer.addWidget(apply_btn)
        footer.addWidget(ok)
        root.addLayout(footer)

        self._save.clicked.connect(self._save_as)
        self._reset.clicked.connect(self._reset_preset)
        self._reverse.clicked.connect(self._reverse_cmap)
        self._add_stop.clicked.connect(self._add_color_stop)
        self._table.itemSelectionChanged.connect(self._on_table_select)
        self._range.currentIndexChanged.connect(lambda *_: self._edit_range())
        for spin in (self._vmin, self._vmax, self._center, self._pct_lo, self._pct_hi):
            spin.valueChanged.connect(lambda *_: self._edit_range())
        self._link_zero.toggled.connect(lambda *_: self._edit_range())
        self._interp.currentIndexChanged.connect(lambda *_: self._edit_mapping())
        self._type_cont.toggled.connect(lambda *_: self._edit_mapping())
        self._type_disc.toggled.connect(lambda *_: self._edit_mapping())
        self._levels.valueChanged.connect(lambda *_: self._edit_mapping())
        self._nan_transparent.toggled.connect(lambda *_: self._edit_mapping())
        self._oor.currentIndexChanged.connect(lambda *_: self._edit_mapping())
        self._nan_swatch.clicked.connect(lambda *_: self._pick_special("nan"))
        self._below.clicked.connect(lambda *_: self._pick_special("below"))
        self._above.clicked.connect(lambda *_: self._pick_special("above"))
        self._export_btn.clicked.connect(self._export_colorbar)
        dialog.destroyed.connect(lambda *_: (self._close_stop_editor(), _forget_editor(self)))
        self._load_mapping(mapping)

    @property
    def widget(self):
        return self._dialog

    def mapping(self) -> FieldColorMapping:
        return self._mapping

    def _fill_presets(self):
        if getattr(self, "_picker", None) is not None:
            self._picker.reload(select=self._preset.currentText() if self._preset is not None else None)
            return
        self._preset.blockSignals(True)
        self._preset.clear()
        for name in listed_preset_names():
            self._preset.addItem(name, name)
        self._preset.blockSignals(False)

    def _edit_named_preset(self, name: str):
        if self._syncing:
            return
        if self._picker is not None:
            self._picker.set_current(name)
        self._on_preset()

    def _load_mapping(self, mapping: FieldColorMapping):
        if not qt_widget_alive(self._dialog):
            return
        self._syncing = True
        try:
            self._mapping = mapping
            name = mapping.colormap.preset or DEFAULT_SURFACE_COLORMAP
            if self._picker is not None:
                self._picker.reload(select=name)
            else:
                idx = self._preset.findData(name)
                if idx < 0:
                    idx = self._preset.findText(name)
                if idx < 0:
                    self._preset.addItem(name, name)
                    idx = self._preset.findData(name)
                if idx >= 0:
                    self._preset.setCurrentIndex(idx)
            stops = mapping.colormap.stops
            self._selected = max(0, min(self._selected, len(stops) - 1))
            self._fill_stop_editor()
            self._rebuild_table()
            norm = mapping.normalization
            ridx = self._range.findData(norm.mode)
            if ridx >= 0:
                self._range.setCurrentIndex(ridx)
            limits = resolve_limits(norm, self._values)
            if norm.mode in (RANGE_MODE_CUSTOM, RANGE_MODE_SYMMETRIC):
                if norm.vmin is not None:
                    self._vmin.setValue(float(norm.vmin))
                elif limits is not None:
                    self._vmin.setValue(float(limits[0]))
                if norm.vmax is not None:
                    self._vmax.setValue(float(norm.vmax))
                elif limits is not None:
                    self._vmax.setValue(float(limits[1]))
            elif limits is not None:
                self._vmin.setValue(float(limits[0]))
                self._vmax.setValue(float(limits[1]))
            if norm.center is not None:
                self._center.setValue(float(norm.center))
            self._link_zero.setChecked(bool(norm.link_center_zero))
            self._pct_lo.setValue(float(norm.percentile_low))
            self._pct_hi.setValue(float(norm.percentile_high))
            iidx = self._interp.findData(mapping.colormap.interpolation)
            if iidx >= 0:
                self._interp.setCurrentIndex(iidx)
            self._type_disc.setChecked(mapping.colormap.map_type == MAP_DISCRETE)
            self._type_cont.setChecked(mapping.colormap.map_type != MAP_DISCRETE)
            self._levels.setValue(int(mapping.colormap.levels))
            _style_swatch(self._nan_swatch, mapping.colormap.nan_rgba)
            self._nan_transparent.setChecked(bool(mapping.colormap.nan_transparent))
            oidx = self._oor.findData(mapping.colormap.out_of_range)
            if oidx >= 0:
                self._oor.setCurrentIndex(oidx)
            _style_swatch(self._below, mapping.colormap.below_rgba or mapping.colormap.stops[0].rgba)
            _style_swatch(self._above, mapping.colormap.above_rgba or mapping.colormap.stops[-1].rgba)
            self._sync_range_enabled()
            self._levels.setEnabled(mapping.colormap.map_type == MAP_DISCRETE)
            custom_oor = mapping.colormap.out_of_range == OOR_CUSTOM
            self._below.setEnabled(custom_oor)
            self._above.setEnabled(custom_oor)
        finally:
            self._syncing = False
        self._refresh_histogram()
        self._refresh_previews()

    def _rebuild_table(self):
        QtCore, QtGui, QtWidgets = qt_modules()
        stops = self._mapping.colormap.stops
        self._table.blockSignals(True)
        self._table.setRowCount(len(stops))
        for i, stop in enumerate(stops):
            self._table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i + 1)))
            self._table.setItem(i, 1, QtWidgets.QTableWidgetItem("%.3f" % stop.position))
            swatch = QtWidgets.QLabel()
            pix = QtGui.QPixmap(22, 12)
            pix.fill(_qcolor(QtGui, stop.rgba))
            swatch.setPixmap(pix)
            self._table.setCellWidget(i, 2, swatch)
            self._table.setItem(i, 3, QtWidgets.QTableWidgetItem("%d%%" % int(round(stop.rgba[3] * 100))))
            self._table.setItem(i, 4, QtWidgets.QTableWidgetItem(stop.label))
        self._table.selectRow(self._selected)
        self._table.blockSignals(False)
        self._table.resizeColumnsToContents()

    def _sync_range_enabled(self):
        mode = str(self._range.currentData() or RANGE_MODE_AUTO)
        custom = mode == RANGE_MODE_CUSTOM
        symmetric = mode == RANGE_MODE_SYMMETRIC
        pct = mode == RANGE_MODE_PERCENTILE
        self._vmin.setEnabled(custom or symmetric)
        self._vmax.setEnabled(custom or symmetric)
        self._center.setEnabled(symmetric and not self._link_zero.isChecked())
        self._link_zero.setEnabled(symmetric)
        self._pct_lo.setEnabled(pct)
        self._pct_hi.setEnabled(pct)

    def _current_normalization(self) -> Normalization:
        mode = str(self._range.currentData() or RANGE_MODE_AUTO)
        manual = mode in (RANGE_MODE_CUSTOM, RANGE_MODE_SYMMETRIC)
        return Normalization(
            mode=mode,
            vmin=float(self._vmin.value()) if manual else None,
            vmax=float(self._vmax.value()) if manual else None,
            center=float(self._center.value()),
            percentile_low=float(self._pct_lo.value()),
            percentile_high=float(self._pct_hi.value()),
            link_center_zero=bool(self._link_zero.isChecked()),
        )

    def _replace_colormap(self, **kwargs):
        cmap = self._mapping.colormap
        self._mapping = FieldColorMapping(
            field_id=self._mapping.field_id,
            colormap=ColormapDefinition(
                preset=kwargs.get("preset", cmap.preset),
                stops=kwargs.get("stops", cmap.stops),
                interpolation=kwargs.get("interpolation", cmap.interpolation),
                map_type=kwargs.get("map_type", cmap.map_type),
                levels=kwargs.get("levels", cmap.levels),
                nan_rgba=kwargs.get("nan_rgba", cmap.nan_rgba),
                nan_transparent=kwargs.get("nan_transparent", cmap.nan_transparent),
                out_of_range=kwargs.get("out_of_range", cmap.out_of_range),
                below_rgba=kwargs.get("below_rgba", cmap.below_rgba),
                above_rgba=kwargs.get("above_rgba", cmap.above_rgba),
                customized=kwargs.get("customized", cmap.customized),
            ),
            normalization=kwargs.get("normalization", self._mapping.normalization),
            title=self._mapping.title,
            units=self._mapping.units,
        )

    def _commit_ui(self, live=True, immediate=False):
        if not qt_widget_alive(self._dialog):
            return
        self._load_mapping(self._mapping)
        if not live:
            return
        if immediate:
            self._debounce.stop()
            self._emit_live()
            return
        self._debounce.start()

    def _emit_live(self):
        if self._on_change is not None:
            self._on_change(self._mapping)

    def _on_preset(self):
        if self._syncing:
            return
        name = str(self._preset.currentData() or self._preset.currentText() or DEFAULT_SURFACE_COLORMAP)
        for item in load_custom_presets():
            if item.get("name") == name:
                defn = ColormapDefinition.from_dict(item.get("definition"))
                self._replace_colormap(
                    preset=defn.preset, stops=defn.stops, interpolation=defn.interpolation,
                    map_type=defn.map_type, levels=defn.levels, nan_rgba=defn.nan_rgba,
                    nan_transparent=defn.nan_transparent, out_of_range=defn.out_of_range,
                    below_rgba=defn.below_rgba, above_rgba=defn.above_rgba, customized=True,
                )
                self._commit_ui()
                return
        defn = definition_from_preset(name)
        self._replace_colormap(
            preset=defn.preset, stops=defn.stops, customized=False,
            interpolation=INTERP_RGB, map_type=MAP_CONTINUOUS,
        )
        self._commit_ui()

    def _reset_preset(self):
        name = str(self._preset.currentData() or self._original.colormap.preset or DEFAULT_SURFACE_COLORMAP)
        defn = definition_from_preset(name)
        self._replace_colormap(
            preset=defn.preset, stops=defn.stops, customized=False,
            interpolation=INTERP_RGB, map_type=MAP_CONTINUOUS,
            nan_rgba=defn.nan_rgba, nan_transparent=False, out_of_range=OOR_CLAMP,
            below_rgba=None, above_rgba=None, levels=5,
        )
        self._commit_ui()

    def _reverse_cmap(self):
        self._mapping = FieldColorMapping(
            field_id=self._mapping.field_id,
            colormap=reverse_definition(self._mapping.colormap),
            normalization=self._mapping.normalization,
            title=self._mapping.title,
            units=self._mapping.units,
        )
        self._commit_ui()

    def _save_as(self):
        from .colormap_similar import persist_custom_colormap
        from ...util.colormap_spec import custom_preset_definition

        QtCore, _, QtWidgets = qt_modules()
        name, ok = QtWidgets.QInputDialog.getText(self._dialog, "Save colormap", "Preset name")
        if not ok or not str(name).strip():
            return
        name = str(name).strip()
        kept = persist_custom_colormap(self._dialog, self._mapping.colormap, name)
        if kept is None:
            return
        if kept != name:
            existing = custom_preset_definition(kept)
            if existing is not None:
                self._replace_colormap(preset=kept, stops=existing.stops, customized=True)
        else:
            self._replace_colormap(preset=name, customized=True)
        self._stored_key = repr(self._mapping.colormap.to_dict())
        self._commit_ui()

    def _add_custom_preset(self):
        name = unused_custom_preset_name()
        self._replace_colormap(preset=name, customized=True)
        self._commit_ui()

    def _fill_stop_editor(self):
        pos = getattr(self, "_pos", None)
        if pos is None or not qt_widget_alive(pos):
            return
        stops = self._mapping.colormap.stops
        self._selected = max(0, min(self._selected, len(stops) - 1))
        stop = stops[self._selected]
        lo, hi = _stop_neighbor_limits(self._selected, stops)
        was = self._syncing
        self._syncing = True
        try:
            self._pos.setRange(lo, hi)
            self._pos.setValue(_clamp_stop_position(self._selected, stops, stop.position))
            if self._opacity is not None and qt_widget_alive(self._opacity):
                self._opacity.setValue(int(round(stop.rgba[3] * 100)))
            if self._swatch is not None and qt_widget_alive(self._swatch):
                _style_swatch(self._swatch, stop.rgba)
            if self._delete is not None and qt_widget_alive(self._delete):
                self._delete.setEnabled(len(stops) > 2)
        finally:
            self._syncing = was

    def _close_stop_editor(self):
        popup = getattr(self, "_stop_popup", None)
        if popup is not None and qt_widget_alive(popup):
            try:
                popup.close()
            except Exception:
                pass
        self._pos = None
        self._swatch = None
        self._opacity = None
        self._delete = None
        self._stop_popup = None

    def _open_stop_editor(self, index: int):
        QtCore, _, QtWidgets = qt_modules()
        if QtWidgets is None or not hasattr(QtWidgets, "QDialog"):
            return
        self._selected = int(index)
        self._on_handle_selected(self._selected)
        if self._stop_popup is not None and qt_widget_alive(self._stop_popup):
            self._fill_stop_editor()
            try:
                self._stop_popup.show()
                self._stop_popup.raise_()
                self._stop_popup.activateWindow()
            except Exception:
                pass
            return
        dialog = QtWidgets.QDialog(self._dialog)
        dialog.setObjectName("pmvColormapStopEditor")
        dialog.setWindowTitle("Selected stop")
        dialog.setModal(False)
        non_modal = getattr(QtCore.Qt, "NonModal", None)
        if non_modal is not None:
            dialog.setWindowModality(non_modal)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        _configure_editor_window(dialog)
        apply_wizard_page_style(dialog)
        form = QtWidgets.QFormLayout(dialog)
        self._pos = QtWidgets.QDoubleSpinBox()
        self._pos.setDecimals(3)
        self._pos.setRange(0.0, 1.0)
        self._pos.setSingleStep(0.01)
        apply_ascii_float_locale(self._pos, QtCore)
        self._swatch = QtWidgets.QPushButton()
        self._opacity = QtWidgets.QSpinBox()
        self._opacity.setRange(0, 100)
        self._opacity.setSuffix(" %")
        self._delete = QtWidgets.QPushButton(DELETE_STOP_LABEL)
        apply_secondary_button_style(self._delete)
        done = QtWidgets.QPushButton("Done")
        done.setObjectName("pmvStopEditorDone")
        done.setStyleSheet(compact_primary_button_css("pmvStopEditorDone"))
        form.addRow("Position", self._pos)
        form.addRow("Color", self._swatch)
        form.addRow("Opacity", self._opacity)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self._delete)
        buttons.addStretch(1)
        buttons.addWidget(done)
        form.addRow(buttons)
        self._pos.valueChanged.connect(lambda *_: self._edit_selected())
        self._opacity.valueChanged.connect(lambda *_: self._edit_selected())
        self._swatch.clicked.connect(self._pick_stop_color)
        self._delete.clicked.connect(self._delete_stop)
        done.clicked.connect(dialog.close)

        def _clear(*_):
            self._pos = None
            self._swatch = None
            self._opacity = None
            self._delete = None
            self._stop_popup = None

        dialog.finished.connect(_clear)
        dialog.destroyed.connect(_clear)
        self._stop_popup = dialog
        self._fill_stop_editor()
        dialog.show()
        bind_tool_window(dialog)
        dialog.raise_()
        dialog.activateWindow()

    def _on_handle_selected(self, index: int):
        self._selected = int(index)
        if self._syncing:
            return
        self._syncing = True
        try:
            self._fill_stop_editor()
            self._table.selectRow(self._selected)
            self._refresh_histogram()
            self._refresh_previews()
        finally:
            self._syncing = False

    def _sync_plot(self):
        hist = getattr(self, "_histogram", None)
        if hist is None:
            return
        hist.set_data(
            hist._hist,
            self._limits_for_plot(),
            self._center_for_plot(),
            defn=self._mapping.colormap,
            selected=self._selected,
        )
        self._refresh_previews()

    def _limits_for_plot(self):
        limits = resolve_limits(self._mapping.normalization, self._values)
        if limits is None:
            hist = getattr(self._histogram, "_hist", None)
            if hist and hist.get("vmin") is not None:
                return (float(hist["vmin"]), float(hist["vmax"]))
            return (0.0, 1.0)
        return limits

    def _center_for_plot(self):
        if self._mapping.normalization.mode == RANGE_MODE_SYMMETRIC:
            return 0.0 if self._mapping.normalization.link_center_zero else self._mapping.normalization.center
        return None

    def _on_stop_dragged(self, index: int, position: float, alpha: float):
        stops = list(self._mapping.colormap.stops)
        index = max(0, min(int(index), len(stops) - 1))
        stop = stops[index]
        rgba = _as_rgba(stop.rgba)
        position = _clamp_stop_position(index, stops, position)
        rgba = (rgba[0], rgba[1], rgba[2], float(alpha))
        stops[index] = ColorStop(position, rgba, stop.label)
        self._replace_colormap(stops=tuple(stops), customized=True)
        new_stops = self._mapping.colormap.stops
        selected = index
        for i, item in enumerate(new_stops):
            if abs(float(item.position) - float(position)) > 1e-6:
                continue
            if abs(float(item.rgba[3]) - float(rgba[3])) > 1e-5:
                continue
            selected = i
            break
        self._selected = selected
        self._sync_plot()

    def _on_range_dragged(self, vmin: float, vmax: float):
        if self._syncing:
            return
        vmin, vmax = clamp_range(vmin, vmax)
        mode = RANGE_MODE_CUSTOM
        current = self._mapping.normalization
        if current.mode == RANGE_MODE_SYMMETRIC:
            center = 0.0 if current.link_center_zero else (current.center if current.center is not None else 0.0)
            mag = max(abs(vmin - center), abs(vmax - center), 1e-12)
            vmin, vmax = center - mag, center + mag
            mode = RANGE_MODE_SYMMETRIC
        self._mapping = FieldColorMapping(
            field_id=self._mapping.field_id,
            colormap=self._mapping.colormap,
            normalization=Normalization(
                mode=mode,
                vmin=vmin,
                vmax=vmax,
                center=current.center,
                percentile_low=current.percentile_low,
                percentile_high=current.percentile_high,
                link_center_zero=current.link_center_zero,
            ),
            title=self._mapping.title,
            units=self._mapping.units,
        )
        self._sync_plot()

    def _on_handle_drag_finished(self):
        self._commit_ui(live=True, immediate=True)

    def _on_handle_asked(self, kind: str, payload, current: float):
        if kind == "range":
            value = _ask_float(self._dialog, "Colormap range", "Value", current)
            if value is None:
                return
            vmin, vmax = self._mapping.normalization.vmin, self._mapping.normalization.vmax
            limits = resolve_limits(self._mapping.normalization, self._values) or (0.0, 1.0)
            if vmin is None or vmax is None:
                vmin, vmax = limits
            if payload == "vmin":
                vmin = value
            else:
                vmax = value
            self._on_range_dragged(vmin, vmax)
            return
        value = _ask_float(self._dialog, "Color stop", "Value", current)
        if value is None:
            return
        limits = resolve_limits(self._mapping.normalization, self._values) or (0.0, 1.0)
        pos = data_to_unit(value, limits[0], limits[1])
        stops = list(self._mapping.colormap.stops)
        index = int(payload)
        stop = stops[index]
        pos = _clamp_stop_position(index, stops, pos)
        stops[index] = ColorStop(pos, stop.rgba, stop.label)
        self._replace_colormap(stops=tuple(stops), customized=True)
        self._selected = index
        self._commit_ui()

    def _on_table_select(self):
        if self._syncing:
            return
        rows = self._table.selectionModel().selectedRows() if self._table.selectionModel() else []
        if not rows:
            return
        self._on_handle_selected(rows[0].row())

    def _edit_selected(self):
        if self._syncing:
            return
        pos = getattr(self, "_pos", None)
        opacity = getattr(self, "_opacity", None)
        if pos is None or opacity is None or not qt_widget_alive(pos) or not qt_widget_alive(opacity):
            return
        stops = list(self._mapping.colormap.stops)
        stop = stops[self._selected]
        rgba = _as_rgba(stop.rgba)
        rgba = (rgba[0], rgba[1], rgba[2], opacity.value() / 100.0)
        position = _clamp_stop_position(
            self._selected, stops, float(pos.value()),
        )
        stops[self._selected] = ColorStop(position, rgba, stop.label)
        self._replace_colormap(stops=tuple(stops), customized=True)
        self._commit_ui()

    def _pick_stop_color(self):
        stop = self._mapping.colormap.stops[self._selected]
        initial = ColorChoice(rgba=_as_rgba(stop.rgba))

        def apply_choice(choice):
            if choice is None or not qt_widget_alive(self._dialog):
                return
            stops = list(self._mapping.colormap.stops)
            current = stops[self._selected]
            rgba = (choice.rgb[0], choice.rgb[1], choice.rgb[2], current.rgba[3])
            stops[self._selected] = ColorStop(current.position, rgba, current.label)
            self._replace_colormap(stops=tuple(stops), customized=True)
            self._commit_ui()

        # Parent the picker to the colormap editor, not the stop popup, so a
        # StayOnTop color dialog is not trapped behind a modal event loop.
        pick_rgb(
            self._dialog,
            initial,
            on_done=bind_color_pick_result(apply_choice, lambda: None),
            cmd=self._cmd,
            allow_field=False,
        )

    def _pick_special(self, which: str):
        cmap = self._mapping.colormap
        current = cmap.nan_rgba if which == "nan" else (cmap.below_rgba or cmap.stops[0].rgba if which == "below" else cmap.above_rgba or cmap.stops[-1].rgba)

        def apply_choice(choice):
            if choice is None or not qt_widget_alive(self._dialog):
                return
            rgba = (choice.rgb[0], choice.rgb[1], choice.rgb[2], 1.0)
            if which == "nan":
                self._replace_colormap(nan_rgba=rgba)
            elif which == "below":
                self._replace_colormap(below_rgba=rgba)
            else:
                self._replace_colormap(above_rgba=rgba)
            self._commit_ui()

        pick_rgb(
            self._dialog,
            ColorChoice(rgba=_as_rgba(current)),
            on_done=bind_color_pick_result(apply_choice, lambda: None),
            cmd=self._cmd,
            allow_field=False,
        )

    def _add_color_stop_at(self, position, alpha=None):
        stops = list(self._mapping.colormap.stops)
        pos = max(0.004, min(0.996, float(position)))
        for i, stop in enumerate(stops):
            if abs(stop.position - pos) < 0.012:
                self._on_handle_selected(i)
                return
        if len(stops) >= 16:
            return
        try:
            rgba = _as_rgba(sample_unit(self._mapping.colormap, pos))
        except Exception:
            rgba = (0.5, 0.5, 0.5, 1.0)
        if alpha is not None:
            rgba = (rgba[0], rgba[1], rgba[2], float(alpha))
        stops.append(ColorStop(pos, rgba))
        self._replace_colormap(stops=tuple(stops), customized=True)
        nearest = min(
            range(len(self._mapping.colormap.stops)),
            key=lambda i: abs(self._mapping.colormap.stops[i].position - pos),
        )
        self._selected = nearest
        self._commit_ui()

    def _add_color_stop(self):
        stops = list(self._mapping.colormap.stops)
        if len(stops) >= 16:
            return
        left = stops[self._selected]
        right = stops[min(self._selected + 1, len(stops) - 1)]
        pos = 0.5 * (left.position + right.position)
        if abs(pos - left.position) < 1e-4:
            pos = min(1.0, left.position + 0.05)
        rgba = tuple(0.5 * (left.rgba[i] + right.rgba[i]) for i in range(4))
        stops.append(ColorStop(pos, rgba))
        self._replace_colormap(stops=tuple(stops), customized=True)
        nearest = min(
            range(len(self._mapping.colormap.stops)),
            key=lambda i: abs(self._mapping.colormap.stops[i].position - pos),
        )
        self._selected = nearest
        self._commit_ui()

    def _delete_stop(self):
        stops = list(self._mapping.colormap.stops)
        if len(stops) <= 2:
            return
        del stops[self._selected]
        self._selected = max(0, min(self._selected, len(stops) - 1))
        self._replace_colormap(stops=tuple(stops), customized=True)
        self._commit_ui()

    def _edit_range(self):
        if self._syncing:
            return
        self._mapping = FieldColorMapping(
            field_id=self._mapping.field_id,
            colormap=self._mapping.colormap,
            normalization=self._current_normalization(),
            title=self._mapping.title,
            units=self._mapping.units,
        )
        self._sync_range_enabled()
        self._refresh_histogram()
        self._refresh_previews()
        self._debounce.start()

    def _edit_mapping(self):
        if self._syncing:
            return
        self._replace_colormap(
            interpolation=str(self._interp.currentData() or INTERP_RGB),
            map_type=MAP_DISCRETE if self._type_disc.isChecked() else MAP_CONTINUOUS,
            levels=int(self._levels.value()),
            nan_transparent=bool(self._nan_transparent.isChecked()),
            out_of_range=str(self._oor.currentData() or OOR_CLAMP),
            customized=True,
        )
        self._commit_ui()

    def _refresh_histogram(self):
        hist = field_histogram(self._mapping.field_id) if self._mapping.field_id else None
        if hist is None and self._values is not None:
            from ...util.colormap_spec import histogram_from_values
            hist = histogram_from_values(self._values)
        limits = resolve_limits(self._mapping.normalization, self._values)
        if limits is None and hist and hist.get("vmin") is not None:
            limits = (float(hist["vmin"]), float(hist["vmax"]))
        if limits is None:
            limits = (0.0, 1.0)
        center = None
        if self._mapping.normalization.mode == RANGE_MODE_SYMMETRIC:
            center = 0.0 if self._mapping.normalization.link_center_zero else self._mapping.normalization.center
        self._histogram.set_data(
            hist, limits, center, defn=self._mapping.colormap, selected=self._selected,
        )
        if not hist or not hist.get("n"):
            self._hist_stats.setText("No field")
            return
        lines = [
            "Min: %s" % format_number(hist["vmin"]),
            "Max: %s" % format_number(hist["vmax"]),
            "Mean: %s" % format_number(hist["mean"]),
            "Std: %s" % format_number(hist["std"]),
            "Samples: %s" % hist["n"],
        ]
        self._hist_stats.setText("\n".join(lines))

    def _refresh_previews(self):
        bar = getattr(self, "_colorbar", None)
        if bar is None:
            return
        limits = resolve_limits(self._mapping.normalization, self._values)
        if limits is None:
            hist = field_histogram(self._mapping.field_id) if self._mapping.field_id else None
            if hist is None and self._values is not None:
                from ...util.colormap_spec import histogram_from_values
                hist = histogram_from_values(self._values)
            if hist and hist.get("vmin") is not None:
                limits = (float(hist["vmin"]), float(hist["vmax"]))
        if limits is None:
            limits = (0.0, 1.0)
        settings = ColorbarExportSettings(
            title="",
            units="",
            tick_count=5,
        )
        bar.set_preview(self._mapping.colormap, limits, settings, selected=self._selected)

    def _export_colorbar(self):
        ExportColorbarDialog(self._dialog, self._mapping, values=self._values, cmd=self._cmd).show()

    def _store_custom_colormap(self) -> bool:
        from .colormap_similar import persist_custom_colormap
        from ...util.colormap_spec import builtin_preset_names, custom_preset_definition

        cmap = self._mapping.colormap
        name = cmap.preset
        if not cmap.customized or not name:
            return True
        if name in builtin_preset_names() and custom_preset_definition(name) is None:
            return True
        key = repr(cmap.to_dict())
        if key == getattr(self, "_stored_key", None):
            return True
        kept = persist_custom_colormap(self._dialog, cmap, name)
        if kept is None:
            return False
        if kept != name:
            existing = custom_preset_definition(kept)
            if existing is not None:
                self._replace_colormap(
                    preset=kept, stops=existing.stops, interpolation=existing.interpolation,
                    map_type=existing.map_type, levels=existing.levels,
                    nan_rgba=existing.nan_rgba, nan_transparent=existing.nan_transparent,
                    out_of_range=existing.out_of_range, below_rgba=existing.below_rgba,
                    above_rgba=existing.above_rgba, customized=True,
                )
                self._commit_ui()
        self._stored_key = repr(self._mapping.colormap.to_dict())
        return True

    def _apply(self):
        if not self._store_custom_colormap():
            return False
        self._applied = True
        if self._on_apply is not None:
            self._on_apply(self._mapping)
        elif self._on_change is not None:
            self._on_change(self._mapping)
        return True

    def _ok(self):
        if not self._apply():
            return
        if self._on_done is not None:
            self._on_done(self._mapping)
        self._close_stop_editor()
        self._closing = True
        self._dialog.accept()

    def _cancel(self):
        self._user_cancel = True
        self._closing = True
        if self._on_done is not None:
            self._on_done(None if not self._applied else self._mapping)
        self._close_stop_editor()
        self._dialog.reject()

    def show(self):
        self._dialog.show()
        bind_tool_window(self._dialog)
        self._dialog.raise_()
        self._dialog.activateWindow()


def _forget_editor(editor):
    try:
        _LIVE_EDITOR.remove(editor)
    except ValueError:
        pass


def open_colormap_editor(
    parent,
    mapping: FieldColorMapping,
    *,
    values=None,
    cmd=None,
    on_change=None,
    on_apply=None,
    on_done=None,
) -> Optional[ColormapEditorDialog]:
    QtCore, _, QtWidgets = qt_modules()
    if QtWidgets is None:
        if on_done is not None:
            on_done(None)
        return None
    if values is None and mapping.field_id:
        values = field_values_for_stats(mapping.field_id)
    if not mapping.title and mapping.field_id:
        mapping = FieldColorMapping(
            field_id=mapping.field_id,
            colormap=mapping.colormap,
            normalization=mapping.normalization,
            title=field_title(mapping.field_id),
            units=mapping.units or field_units(mapping.field_id),
        )
    existing = _LIVE_EDITOR[-1] if _LIVE_EDITOR else None
    if (
        existing is not None
        and not getattr(existing, "_closing", False)
        and qt_widget_alive(existing.widget)
    ):
        try:
            existing._on_change = on_change
            existing._on_apply = on_apply
            existing._on_done = on_done
            existing._cmd = cmd
            existing._values = values
            existing._original = mapping
            existing._applied = False
            existing._stored_key = None
            existing._user_cancel = False
            existing._closing = False
            existing._load_mapping(mapping)
            existing.widget.show()
            existing.widget.raise_()
            existing.widget.activateWindow()
            return existing
        except Exception:
            _forget_editor(existing)
    try:
        editor = ColormapEditorDialog(
            parent, mapping, values=values, cmd=cmd,
            on_change=on_change, on_apply=on_apply, on_done=on_done,
        )
    except Exception:
        if on_done is not None:
            on_done(None)
        return None
    _LIVE_EDITOR.append(editor)
    editor.show()
    return editor
