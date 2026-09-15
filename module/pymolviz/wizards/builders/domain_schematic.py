"""Orthogonal 2D sketch of a Field Domain AABB (paint data is Qt-free)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ...fields.domain import (
    BOUNDS_AROUND_SELECTION,
    Domain,
    aabb_from_points,
    domain_preview_aabbs,
    normalize_aabb,
)
from ..pick import qt_modules
from ..widgets.theme import PRIMARY, ROW, SELECTED, TREE_LINE, qcolor

AXIS_LABELS = ("X", "Y", "Z")
SCHEMATIC_WIDTH = 160
SCHEMATIC_MIN_HEIGHT = 140
SCHEMATIC_MAX_HEIGHT = 180
_MAX_GRID_TICKS = 8
_MIN_WORLD_SPAN = 1.0
SCHEMATIC_TIP = (
    "Orthogonal sketch of the voxel domain (outer box) versus the current "
    "selection. Padding is the gap; faint ticks hint grid spacing."
)


@dataclass(frozen=True)
class SchematicScene:
    """World-space 2D projection inputs. Safe to unit-test without Qt."""

    axes: tuple = (0, 1)
    domain_aabb: Optional[list] = None
    selection_aabb: Optional[list] = None
    points: tuple = ()
    spacing: float = 0.5
    padding: float = 0.0
    bounds_mode: str = BOUNDS_AROUND_SELECTION


@dataclass(frozen=True)
class SchematicLayout:
    """Pixel-space drawing model. ``empty`` when there is nothing to sketch."""

    empty: bool = True
    width: float = 0.0
    height: float = 0.0
    domain_rect: Optional[tuple] = None
    selection_rect: Optional[tuple] = None
    points: tuple = ()
    grid_lines: tuple = ()
    axes: tuple = (0, 1)
    axis_labels: tuple = ("X", "Y")


def best_projection_axes(aabb) -> tuple:
    """Two axis indices with the largest AABB spans (XY when tied)."""
    box = normalize_aabb(aabb)
    if box is None:
        return (0, 1)
    span = np.abs(np.asarray(box[1], dtype=float) - np.asarray(box[0], dtype=float))
    order = np.argsort(-span, kind="stable")
    i, j = int(order[0]), int(order[1])
    if i > j:
        i, j = j, i
    return (i, j)


def domain_schematic_scene(domain, centers=None, spacing=None) -> SchematicScene:
    """Build a schematic from a Domain and optional selection centers."""
    if domain is None:
        domain = Domain()
    elif not isinstance(domain, Domain):
        domain = Domain.from_dict(domain)
    pts = _center_tuples(centers)
    outer, inner = domain_preview_aabbs(domain, pts)
    ref = outer or inner or aabb_from_points(pts, 0.0)
    return SchematicScene(
        axes=best_projection_axes(ref),
        domain_aabb=outer,
        selection_aabb=inner,
        points=pts,
        spacing=float(spacing if spacing is not None else domain.spacing),
        padding=float(domain.padding),
        bounds_mode=str(domain.bounds_mode),
    )


def layout_schematic(scene: SchematicScene, width, height, margin=12) -> SchematicLayout:
    """Map a world-space scene into pixel rects (Y flipped for screen space)."""
    width = max(float(width), 1.0)
    height = max(float(height), 1.0)
    axes = tuple(scene.axes) if scene is not None else (0, 1)
    labels = (AXIS_LABELS[axes[0]], AXIS_LABELS[axes[1]])
    empty = SchematicLayout(
        empty=True, width=width, height=height, axes=axes, axis_labels=labels,
    )
    if scene is None:
        return empty
    world = _world_bounds(scene)
    if world is None:
        return empty
    x0, y0, x1, y1 = world
    span_x = max(x1 - x0, _MIN_WORLD_SPAN)
    span_y = max(y1 - y0, _MIN_WORLD_SPAN)
    if x1 - x0 < _MIN_WORLD_SPAN:
        mid = 0.5 * (x0 + x1)
        x0, x1 = mid - 0.5 * span_x, mid + 0.5 * span_x
    if y1 - y0 < _MIN_WORLD_SPAN:
        mid = 0.5 * (y0 + y1)
        y0, y1 = mid - 0.5 * span_y, mid + 0.5 * span_y
        span_y = y1 - y0
    span_x = x1 - x0
    inner_w = max(width - 2.0 * margin, 1.0)
    inner_h = max(height - 2.0 * margin, 1.0)
    scale = min(inner_w / span_x, inner_h / span_y)
    ox = margin + 0.5 * (inner_w - span_x * scale)
    oy = margin + 0.5 * (inner_h - span_y * scale)

    def to_px(x, y):
        px = ox + (float(x) - x0) * scale
        py = height - (oy + (float(y) - y0) * scale)
        return (px, py)

    def rect_px(aabb):
        box = normalize_aabb(aabb)
        if box is None:
            return None
        rx0, ry0, rx1, ry1 = _project_aabb(box, axes)
        px0, py0 = to_px(rx0, ry0)
        px1, py1 = to_px(rx1, ry1)
        left, top = min(px0, px1), min(py0, py1)
        return (left, top, abs(px1 - px0), abs(py1 - py0))

    points_px = tuple(to_px(*_project_point(pt, axes)) for pt in scene.points)
    grid = _grid_lines_px(scene, axes, to_px)
    return SchematicLayout(
        empty=False,
        width=width,
        height=height,
        domain_rect=rect_px(scene.domain_aabb),
        selection_rect=rect_px(scene.selection_aabb),
        points=tuple(points_px),
        grid_lines=grid,
        axes=axes,
        axis_labels=labels,
    )


def wrap_domain_knobs(QtWidgets, schematic, knobs_widget):
    """Horizontal Domain body: schematic on the left, knobs on the right."""
    row = QtWidgets.QWidget()
    row.setObjectName("pmvDomainBody")
    layout = QtWidgets.QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(12)
    schematic_widget = schematic.widget if hasattr(schematic, "widget") else schematic
    layout.addWidget(schematic_widget, 0)
    layout.addWidget(knobs_widget, 1)
    return row


class DomainSchematicWidget:
    """Small orthogonal Domain diagram painted from a ``SchematicScene``."""

    def __init__(self, parent=None):
        QtCore, _, QtWidgets = qt_modules()
        self._scene = SchematicScene()
        self._widget = None
        if QtWidgets is None:
            return
        widget = QtWidgets.QWidget(parent)
        widget.setObjectName("pmvDomainSchematic")
        widget.setFixedWidth(SCHEMATIC_WIDTH)
        widget.setMinimumHeight(SCHEMATIC_MIN_HEIGHT)
        widget.setMaximumHeight(SCHEMATIC_MAX_HEIGHT)
        expanding = getattr(QtWidgets.QSizePolicy, "Expanding", None)
        fixed = getattr(QtWidgets.QSizePolicy, "Fixed", None)
        if expanding is not None and fixed is not None:
            widget.setSizePolicy(fixed, expanding)
        widget.setToolTip(SCHEMATIC_TIP)
        widget.paintEvent = self._paint
        self._widget = widget

    @property
    def widget(self):
        return self._widget

    def set_scene(self, scene: Optional[SchematicScene]) -> None:
        self._scene = scene if isinstance(scene, SchematicScene) else SchematicScene()
        if self._widget is not None:
            self._widget.update()

    def _paint(self, event):
        QtCore, QtGui, _ = qt_modules()
        widget = self._widget
        if widget is None or QtGui is None:
            return
        painter = QtGui.QPainter(widget)
        try:
            aa = getattr(QtGui.QPainter, "Antialiasing", None)
            if aa is not None:
                painter.setRenderHint(aa, True)
            rect = widget.rect()
            painter.fillRect(rect, qcolor(QtGui, ROW) or widget.palette().base())
            layout = layout_schematic(self._scene, rect.width(), rect.height())
            _paint_layout(painter, QtCore, QtGui, layout)
        finally:
            painter.end()


def _center_tuples(centers) -> tuple:
    if centers is None:
        return ()
    try:
        pts = np.asarray(centers, dtype=float)
    except (TypeError, ValueError):
        return ()
    if pts.size == 0:
        return ()
    try:
        pts = pts.reshape(-1, 3)
    except ValueError:
        return ()
    out = []
    for row in pts:
        if not np.all(np.isfinite(row)):
            continue
        out.append((float(row[0]), float(row[1]), float(row[2])))
    return tuple(out)


def _project_point(xyz, axes):
    i, j = axes
    return float(xyz[i]), float(xyz[j])


def _project_aabb(aabb, axes):
    lo, hi = aabb
    i, j = axes
    return float(lo[i]), float(lo[j]), float(hi[i]), float(hi[j])


def _world_bounds(scene: SchematicScene):
    rects = []
    if scene.domain_aabb is not None:
        rects.append(_project_aabb(scene.domain_aabb, scene.axes))
    if scene.selection_aabb is not None:
        rects.append(_project_aabb(scene.selection_aabb, scene.axes))
    for pt in scene.points:
        x, y = _project_point(pt, scene.axes)
        rects.append((x, y, x, y))
    if not rects:
        return None
    return (
        min(row[0] for row in rects),
        min(row[1] for row in rects),
        max(row[2] for row in rects),
        max(row[3] for row in rects),
    )


def _grid_ticks(lo, hi, spacing, max_count=_MAX_GRID_TICKS):
    span = float(hi) - float(lo)
    step = float(spacing or 0.0)
    if step <= 1e-12 or span <= 1e-12:
        return ()
    n = int(np.floor(span / step))
    if n <= 1:
        return ()
    stride = max(1, int(np.ceil(n / float(max_count))))
    ticks = []
    i = stride
    while i < n:
        ticks.append(float(lo) + i * step)
        i += stride
    return tuple(ticks)


def _grid_lines_px(scene: SchematicScene, axes, to_px):
    box = normalize_aabb(scene.domain_aabb)
    if box is None:
        return ()
    rx0, ry0, rx1, ry1 = _project_aabb(box, axes)
    lines = []
    for x in _grid_ticks(rx0, rx1, scene.spacing):
        a = to_px(x, ry0)
        b = to_px(x, ry1)
        lines.append((a[0], a[1], b[0], b[1]))
    for y in _grid_ticks(ry0, ry1, scene.spacing):
        a = to_px(rx0, y)
        b = to_px(rx1, y)
        lines.append((a[0], a[1], b[0], b[1]))
    return tuple(lines)


def _paint_layout(painter, QtCore, QtGui, layout: SchematicLayout) -> None:
    if layout.empty:
        _paint_empty(painter, QtCore, QtGui, layout)
        return
    if layout.grid_lines:
        grid = qcolor(QtGui, TREE_LINE, alpha=90)
        if grid is not None:
            pen = QtGui.QPen(grid, 1.0)
            painter.setPen(pen)
            painter.setBrush(getattr(QtCore.Qt, "NoBrush", QtCore.Qt.NoBrush))
            for x0, y0, x1, y1 in layout.grid_lines:
                painter.drawLine(int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))
    domain = layout.domain_rect
    if domain is not None:
        fill = qcolor(QtGui, SELECTED, alpha=70)
        edge = qcolor(QtGui, PRIMARY)
        if fill is not None:
            painter.setBrush(fill)
        if edge is not None:
            painter.setPen(QtGui.QPen(edge, 1.8))
        painter.drawRect(_qrect(QtCore, domain))
    selection = layout.selection_rect
    if selection is not None and not _rect_near(selection, domain):
        edge = qcolor(QtGui, TREE_LINE)
        if edge is not None:
            pen = QtGui.QPen(edge, 1.2)
            dash = getattr(QtCore.Qt, "DashLine", None)
            if dash is not None:
                pen.setStyle(dash)
            painter.setPen(pen)
        painter.setBrush(getattr(QtCore.Qt, "NoBrush", QtCore.Qt.NoBrush))
        painter.drawRect(_qrect(QtCore, selection))
    fill = qcolor(QtGui, PRIMARY)
    if fill is not None:
        painter.setPen(QtGui.QPen(fill, 1.0))
        painter.setBrush(fill)
        for x, y in layout.points:
            painter.drawEllipse(int(round(x)) - 2, int(round(y)) - 2, 4, 4)
    _paint_axis_caption(painter, QtGui, layout)


def _paint_empty(painter, QtCore, QtGui, layout: SchematicLayout) -> None:
    edge = qcolor(QtGui, TREE_LINE)
    if edge is None:
        return
    pen = QtGui.QPen(edge, 1.0)
    dash = getattr(QtCore.Qt, "DashLine", None)
    if dash is not None:
        pen.setStyle(dash)
    painter.setPen(pen)
    painter.setBrush(getattr(QtCore.Qt, "NoBrush", QtCore.Qt.NoBrush))
    inset = 18
    painter.drawRect(
        _qrect(QtCore, (inset, inset, layout.width - 2 * inset, layout.height - 2 * inset))
    )


def _paint_axis_caption(painter, QtGui, layout: SchematicLayout) -> None:
    ink = qcolor(QtGui, TREE_LINE)
    if ink is None:
        return
    painter.setPen(ink)
    text = "%s–%s" % layout.axis_labels
    painter.drawText(8, int(layout.height) - 6, text)


def _qrect(QtCore, rect):
    x, y, w, h = rect
    if hasattr(QtCore, "QRectF"):
        return QtCore.QRectF(float(x), float(y), float(w), float(h))
    return QtCore.QRect(int(x), int(y), int(w), int(h))


def _rect_near(a, other, eps=1.5) -> bool:
    if a is None or other is None:
        return False
    return all(abs(float(x) - float(y)) < eps for x, y in zip(a, other))
