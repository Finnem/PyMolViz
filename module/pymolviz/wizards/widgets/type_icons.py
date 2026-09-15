"""Simple monochrome glyphs for visual type cards."""

from __future__ import annotations

from .theme import INK, MUTED, PRIMARY, mix_rgb, qcolor

ICON_SIZE = 32


def type_icon_pixmap(kind, QtGui, QtCore, QtWidgets=None, size=ICON_SIZE, color=None):
    """Paint a 32×32-style glyph. ``kind`` is sphere, cube, surface, arrow, volume, or isomesh."""
    if QtGui is None or QtCore is None:
        return None
    pix = QtGui.QPixmap(int(size), int(size))
    pix.fill(QtGui.QColor(0, 0, 0, 0))
    painter = QtGui.QPainter(pix)
    try:
        aa = getattr(QtGui.QPainter, "Antialiasing", None)
        if aa is None:
            aa = getattr(
                getattr(QtGui.QPainter, "RenderHint", None), "Antialiasing", None
            )
        if aa is not None:
            painter.setRenderHint(aa, True)
        paint = _explicit_color(QtGui, color) or _icon_color(QtGui, QtWidgets)
        drawer = _DRAWERS.get(str(kind))
        if drawer is not None:
            drawer(painter, QtGui, QtCore, float(size), paint)
    finally:
        painter.end()
    return pix


def action_icon_pixmap(kind, QtGui, QtCore, QtWidgets=None, size=16, color=None):
    """Small glyph for catalog row actions (trash)."""
    if QtGui is None or QtCore is None:
        return None
    pix = QtGui.QPixmap(int(size), int(size))
    pix.fill(QtGui.QColor(0, 0, 0, 0))
    painter = QtGui.QPainter(pix)
    try:
        aa = getattr(QtGui.QPainter, "Antialiasing", None)
        if aa is None:
            aa = getattr(
                getattr(QtGui.QPainter, "RenderHint", None), "Antialiasing", None
            )
        if aa is not None:
            painter.setRenderHint(aa, True)
        paint = _explicit_color(QtGui, color) or qcolor(QtGui, MUTED)
        drawer = _ACTION_DRAWERS.get(str(kind))
        if drawer is not None:
            drawer(painter, QtGui, QtCore, float(size), paint)
    finally:
        painter.end()
    return pix


def _explicit_color(QtGui, color):
    if color is None or QtGui is None:
        return None
    try:
        if hasattr(color, "red"):
            return color
        vals = [float(x) for x in list(color)[:3]]
        if vals and max(vals) <= 1.0:
            vals = [v * 255.0 for v in vals]
        return QtGui.QColor(
            *[max(0, min(255, int(round(v)))) for v in vals]
        )
    except (TypeError, ValueError):
        return None


def _icon_color(QtGui, QtWidgets):
    return qcolor(QtGui, INK)


def _scheme_icon_color(QtGui, QtWidgets):
    """Ink mixed with primary so glyphs follow the Fields catalog chrome."""
    rgb = mix_rgb(INK, PRIMARY, 0.48)
    return qcolor(QtGui, rgb)


def _pen(QtGui, QtCore, color, size, width=1.6):
    pen = QtGui.QPen(color)
    pen.setWidthF(max(1.2, width * size / 32.0))
    cap = getattr(QtCore.Qt, "RoundCap", None)
    join = getattr(QtCore.Qt, "RoundJoin", None)
    if cap is not None:
        pen.setCapStyle(cap)
    if join is not None:
        pen.setJoinStyle(join)
    return pen


def _pt(QtCore, size, x, y):
    s = size / 32.0
    return QtCore.QPointF(x * s, y * s)


def _poly(QtGui, QtCore, size, points):
    pts = [_pt(QtCore, size, x, y) for x, y in points]
    try:
        return QtGui.QPolygonF(pts)
    except TypeError:
        poly = QtGui.QPolygonF()
        for pt in pts:
            poly.append(pt)
        return poly


def _fill(color, QtGui, alpha):
    return QtGui.QBrush(QtGui.QColor(color.red(), color.green(), color.blue(), alpha))


def _draw_sphere(painter, QtGui, QtCore, size, color):
    painter.setPen(_pen(QtGui, QtCore, color, size, 1.7))
    painter.setBrush(_fill(color, QtGui, 45))
    s = size / 32.0
    painter.drawEllipse(QtCore.QRectF(6 * s, 6 * s, 20 * s, 20 * s))
    no_pen = getattr(QtCore.Qt, "NoPen", None)
    if no_pen is not None:
        painter.setPen(no_pen)
    painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 90)))
    painter.drawEllipse(QtCore.QRectF(10 * s, 8.5 * s, 7 * s, 5 * s))


def _draw_cube(painter, QtGui, QtCore, size, color):
    painter.setPen(_pen(QtGui, QtCore, color, size, 1.5))
    faces = (
        ((16, 6), (26, 11.5), (16, 17), (6, 11.5), 35),
        ((6, 11.5), (16, 17), (16, 26), (6, 20.5), 55),
        ((26, 11.5), (16, 17), (16, 26), (26, 20.5), 25),
    )
    for *pts, alpha in faces:
        painter.setBrush(_fill(color, QtGui, alpha))
        painter.drawPolygon(_poly(QtGui, QtCore, size, pts))


def _draw_surface(painter, QtGui, QtCore, size, color):
    painter.setPen(_pen(QtGui, QtCore, color, size, 1.4))
    painter.setBrush(_fill(color, QtGui, 30))
    rows = (
        ((7, 13), (13, 10), (20, 11), (26, 9)),
        ((6, 19), (13, 17), (20, 18), (27, 16)),
        ((7, 25), (14, 24), (21, 23), (26, 25)),
    )
    painter.drawPolygon(
        _poly(
            QtGui,
            QtCore,
            size,
            list(rows[0]) + list(reversed(rows[-1])),
        )
    )
    for row in rows:
        for a, b in zip(row, row[1:]):
            painter.drawLine(_pt(QtCore, size, *a), _pt(QtCore, size, *b))
    for col in range(4):
        painter.drawLine(
            _pt(QtCore, size, *rows[0][col]),
            _pt(QtCore, size, *rows[-1][col]),
        )


def _draw_dashed_line(painter, QtCore, size, x0, y0, x1, y1, dashes=3):
    dx = (x1 - x0) / (dashes * 2 - 1)
    dy = (y1 - y0) / (dashes * 2 - 1)
    for i in range(dashes):
        t0 = i * 2
        painter.drawLine(
            _pt(QtCore, size, x0 + dx * t0, y0 + dy * t0),
            _pt(QtCore, size, x0 + dx * (t0 + 1), y0 + dy * (t0 + 1)),
        )


def _unit(dx, dy):
    length = (dx * dx + dy * dy) ** 0.5
    if length <= 1e-6:
        return 1.0, 0.0
    return dx / length, dy / length


def _draw_arrow_mark(painter, QtGui, QtCore, size, x0, y0, x1, y1, head_len=7.0, head_half=3.2):
    ux, uy = _unit(x1 - x0, y1 - y0)
    px, py = -uy, ux
    base_x = x1 - ux * head_len
    base_y = y1 - uy * head_len
    painter.drawLine(
        _pt(QtCore, size, x0, y0),
        _pt(QtCore, size, base_x, base_y),
    )
    painter.drawPolygon(
        _poly(
            QtGui,
            QtCore,
            size,
            (
                (x1, y1),
                (base_x + px * head_half, base_y + py * head_half),
                (base_x - px * head_half, base_y - py * head_half),
            ),
        )
    )


def _draw_arrow(painter, QtGui, QtCore, size, color):
    painter.setPen(_pen(QtGui, QtCore, color, size, 1.8))
    painter.setBrush(_fill(color, QtGui, 80))
    x0, y0, x1, y1 = 8.0, 16.0, 26.0, 5.0
    ux, uy = _unit(x1 - x0, y1 - y0)
    px, py = -uy, ux
    if py < 0:
        px, py = -px, -py
    gap = 9.5
    _draw_dashed_line(
        painter,
        QtCore,
        size,
        x0 + px * gap,
        y0 + py * gap,
        x0 + ux * 16.0 + px * gap,
        y0 + uy * 16.0 + py * gap,
        dashes=3,
    )
    _draw_arrow_mark(painter, QtGui, QtCore, size, x0, y0, x1, y1)


def _draw_volume(painter, QtGui, QtCore, size, color):
    painter.setPen(_pen(QtGui, QtCore, color, size, 1.4))
    s = size / 32.0
    layers = (
        (7, 18, 18, 10, 35),
        (5, 12, 22, 12, 50),
        (8, 6, 16, 10, 28),
    )
    for x, y, w, h, alpha in layers:
        painter.setBrush(_fill(color, QtGui, alpha))
        painter.drawEllipse(QtCore.QRectF(x * s, y * s, w * s, h * s))


def _draw_isomesh(painter, QtGui, QtCore, size, color):
    painter.setPen(_pen(QtGui, QtCore, color, size, 1.35))
    painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 0)))
    rows = (
        ((7, 12), (13, 9), (20, 11), (26, 8)),
        ((6, 18), (13, 16), (20, 17), (27, 15)),
        ((7, 24), (14, 23), (21, 22), (26, 24)),
    )
    for row in rows:
        for a, b in zip(row, row[1:]):
            painter.drawLine(_pt(QtCore, size, *a), _pt(QtCore, size, *b))
    for col in range(4):
        painter.drawLine(
            _pt(QtCore, size, *rows[0][col]),
            _pt(QtCore, size, *rows[-1][col]),
        )


def _draw_trash(painter, QtGui, QtCore, size, color):
    painter.setPen(_pen(QtGui, QtCore, color, size, 1.45))
    painter.setBrush(_fill(color, QtGui, 28))
    s = size / 32.0
    painter.drawRoundedRect(QtCore.QRectF(9 * s, 12 * s, 14 * s, 15 * s), 1.6 * s, 1.6 * s)
    painter.drawLine(_pt(QtCore, size, 8, 12), _pt(QtCore, size, 24, 12))
    painter.drawLine(_pt(QtCore, size, 13, 8.5), _pt(QtCore, size, 19, 8.5))
    painter.drawLine(_pt(QtCore, size, 13.2, 15), _pt(QtCore, size, 13.2, 23))
    painter.drawLine(_pt(QtCore, size, 16, 15), _pt(QtCore, size, 16, 23))
    painter.drawLine(_pt(QtCore, size, 18.8, 15), _pt(QtCore, size, 18.8, 23))


def _draw_lattice(painter, QtGui, QtCore, size, color):
    """Three offset unit cells — crystal repeat, not a free translate."""
    painter.setPen(_pen(QtGui, QtCore, color, size, 1.35))
    painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 0)))
    s = size / 32.0
    for dx, dy in ((5.5, 8.5), (10.0, 11.5), (14.5, 14.5)):
        painter.drawRect(QtCore.QRectF(dx * s, dy * s, 11.5 * s, 11.5 * s))


def _draw_plus(painter, QtGui, QtCore, size, color):
    painter.setPen(_pen(QtGui, QtCore, color, size, 2.2))
    painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 0)))
    painter.drawLine(_pt(QtCore, size, 16, 7), _pt(QtCore, size, 16, 25))
    painter.drawLine(_pt(QtCore, size, 7, 16), _pt(QtCore, size, 25, 16))


def _draw_gear(painter, QtGui, QtCore, size, color):
    import math

    painter.setPen(_pen(QtGui, QtCore, color, size, 1.5))
    painter.setBrush(_fill(color, QtGui, 40))
    s = size / 32.0
    cx, cy, r = 16.0, 16.0, 7.2
    painter.drawEllipse(QtCore.QRectF((cx - r) * s, (cy - r) * s, 2 * r * s, 2 * r * s))
    no_pen = getattr(QtCore.Qt, "NoPen", None)
    if no_pen is not None:
        painter.setPen(no_pen)
    painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 0)))
    painter.setPen(_pen(QtGui, QtCore, color, size, 1.5))
    painter.drawEllipse(QtCore.QRectF(12.2 * s, 12.2 * s, 7.6 * s, 7.6 * s))
    painter.setPen(_pen(QtGui, QtCore, color, size, 2.1))
    for i in range(6):
        ang = math.radians(i * 60.0)
        painter.drawLine(
            _pt(QtCore, size, cx + 8.0 * math.cos(ang), cy + 8.0 * math.sin(ang)),
            _pt(QtCore, size, cx + 13.2 * math.cos(ang), cy + 13.2 * math.sin(ang)),
        )


def source_icon_kind(source: str) -> str:
    token = str(source or "").strip().lower()
    if token in ("camera", "manual"):
        return "camera"
    return "selection"


SOURCE_ICON_SIZE = 16


def source_icon_pixmap(source, QtGui, QtCore, QtWidgets=None, size=SOURCE_ICON_SIZE):
    """16px glyph for Add-from mode: selection (atom pick) or camera."""
    if QtGui is None or QtCore is None:
        return None
    kind = source_icon_kind(source)
    pix = QtGui.QPixmap(int(size), int(size))
    pix.fill(QtGui.QColor(0, 0, 0, 0))
    painter = QtGui.QPainter(pix)
    try:
        aa = getattr(QtGui.QPainter, "Antialiasing", None)
        if aa is None:
            aa = getattr(
                getattr(QtGui.QPainter, "RenderHint", None), "Antialiasing", None
            )
        if aa is not None:
            painter.setRenderHint(aa, True)
        color = _scheme_icon_color(QtGui, QtWidgets)
        drawer = _SOURCE_DRAWERS.get(kind)
        if drawer is not None:
            drawer(painter, QtGui, QtCore, float(size), color)
    finally:
        painter.end()
    return pix


def _draw_selection_source(painter, QtGui, QtCore, size, color):
    """Dashed pick box around a central atom."""
    pen = _pen(QtGui, QtCore, color, size, 1.45)
    dash = getattr(QtCore.Qt, "DashLine", None)
    if dash is not None:
        pen.setStyle(dash)
    painter.setPen(pen)
    painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 0)))
    s = size / 32.0
    radius = 3.2 * s
    painter.drawRoundedRect(QtCore.QRectF(4.5 * s, 4.5 * s, 23 * s, 23 * s), radius, radius)
    painter.setPen(_pen(QtGui, QtCore, color, size, 1.35))
    painter.setBrush(_fill(color, QtGui, 90))
    painter.drawEllipse(QtCore.QRectF(12 * s, 12 * s, 8 * s, 8 * s))


def _draw_camera_source(painter, QtGui, QtCore, size, color):
    """Camera body, lens, and viewfinder in the Visuals glyph language."""
    painter.setPen(_pen(QtGui, QtCore, color, size, 1.45))
    painter.setBrush(_fill(color, QtGui, 40))
    s = size / 32.0
    painter.drawRoundedRect(QtCore.QRectF(3.5 * s, 11 * s, 20 * s, 14 * s), 2.8 * s, 2.8 * s)
    painter.drawRoundedRect(QtCore.QRectF(13.5 * s, 6.5 * s, 7.5 * s, 5 * s), 1.4 * s, 1.4 * s)
    painter.setBrush(_fill(color, QtGui, 28))
    painter.drawEllipse(QtCore.QRectF(8.2 * s, 13.2 * s, 9.5 * s, 9.5 * s))
    painter.setBrush(_fill(color, QtGui, 100))
    painter.drawEllipse(QtCore.QRectF(10.4 * s, 15.4 * s, 5.2 * s, 5.2 * s))


_SOURCE_DRAWERS = {
    "selection": _draw_selection_source,
    "camera": _draw_camera_source,
}


_DRAWERS = {
    "sphere": _draw_sphere,
    "cube": _draw_cube,
    "surface": _draw_surface,
    "arrow": _draw_arrow,
    "volume": _draw_volume,
    "isomesh": _draw_isomesh,
}

_ACTION_DRAWERS = {
    "trash": _draw_trash,
    "plus": _draw_plus,
    "gear": _draw_gear,
    "lattice": _draw_lattice,
}
