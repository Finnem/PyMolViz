"""Screen-space atom picking from a cached get_view() + coordinates."""

import math

from ..util.view import model_to_camera

PICK_SELE = "visible and enabled and not hydro"


def qt_modules():
    try:
        from pymol.Qt import QtCore, QtGui, QtWidgets
        return QtCore, QtGui, QtWidgets
    except Exception:
        return None, None, None


def find_viewer_widget(QtWidgets):
    app = QtWidgets.QApplication.instance()
    if app is None:
        return None
    ranked = []
    for widget in app.allWidgets():
        if not widget.isVisible() or widget.width() < 80 or widget.height() < 80:
            continue
        score = 0
        if hasattr(widget, "pymol") and hasattr(widget, "fb_scale"):
            score += 6
        elif hasattr(widget, "pymol"):
            score += 4
        if widget.inherits("QOpenGLWidget") or widget.inherits("QGLWidget"):
            score += 3
        cls = widget.metaObject().className()
        if any(token in cls for token in ("GLWidget", "OpenGL", "PyMOLGL", "CMol")):
            score += 2
        if score:
            ranked.append((score, widget.width() * widget.height(), widget))
    if not ranked:
        return None
    ranked.sort()
    return ranked[-1][-1]


def widget_fb_scale(widget):
    scale = getattr(widget, "fb_scale", None)
    if scale:
        return float(scale)
    if hasattr(widget, "devicePixelRatioF"):
        return float(widget.devicePixelRatioF())
    if hasattr(widget, "devicePixelRatio"):
        return float(widget.devicePixelRatio())
    return 1.0


def qt_to_pymol_xy(widget, x, y):
    """Match PyMOLGLWidget._event_x_y_mod: framebuffer pixels, Y from the bottom."""
    scale = widget_fb_scale(widget)
    return int(scale * x), int(scale * (widget.height() - y))


def atom_sele(ids, index):
    if not ids or index < 0 or index >= len(ids):
        return None
    model, atm = ids[index]
    return "(%s)`%d" % (model, atm)


def pick_atom(view, coords, widget, x, y, viewport, fov, ortho, ids=None, max_px=24.0):
    """Nearest atom under the cursor using cached view/coords. No cmd calls."""
    click_x, click_y = qt_to_pymol_xy(widget, x, y)
    try:
        width, height = float(viewport[0]), float(viewport[1])
    except Exception:
        width, height = 0.0, 0.0
    if width < 1 or height < 1:
        scale = widget_fb_scale(widget)
        width, height = float(widget.width()) * scale, float(widget.height()) * scale
    if width < 1 or height < 1 or coords is None or len(coords) == 0:
        return None
    rect_bottom = max(widget_fb_scale(widget) * widget.height() - height, 0.0)
    sx = float(click_x)
    sy = float(click_y) - rect_bottom
    fov_width = 2.0 * math.tan(math.radians(max(abs(float(fov)), 1.0)) / 2.0)
    origin_depth = max(abs(float(view[11])), 1e-6)
    best = None
    best_index = None
    best_key = None
    front = float(view[15])
    back = float(view[16])
    for index, pos in enumerate(coords):
        point = (float(pos[0]), float(pos[1]), float(pos[2]))
        cx, cy, cz = model_to_camera(view, point)
        depth = origin_depth if ortho else -cz
        if depth < 1e-4:
            continue
        if front > 0.0 and back > front and (depth < front * 0.5 or depth > back * 1.5):
            continue
        angstrom_per_px = depth * fov_width / height
        if angstrom_per_px < 1e-8:
            continue
        dx = (width * 0.5 + cx / angstrom_per_px) - sx
        dy = (height * 0.5 + cy / angstrom_per_px) - sy
        dist2 = dx * dx + dy * dy
        if dist2 > max_px * max_px:
            continue
        key = (dist2, depth)
        if best_key is None or key < best_key:
            best_key = key
            best = point
            best_index = index
    if best is None:
        return None
    return best, atom_sele(ids, best_index)
