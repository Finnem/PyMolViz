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


_OPEN_TOOL_WINDOWS = []
_RAISE_FILTER = None


def find_pymol_window(QtWidgets):
    """Top-level PyMOL window that owns the 3D viewer."""
    viewer = find_viewer_widget(QtWidgets)
    if viewer is not None:
        window = viewer.window()
        if window is not None:
            return window
    app = QtWidgets.QApplication.instance()
    if app is None:
        return None
    for widget in app.topLevelWidgets():
        if not widget.isVisible():
            continue
        cls = widget.metaObject().className()
        if "PyMOL" in cls or "pymol" in cls.lower():
            return widget
    return None


def _qt_platform_name():
    try:
        from pymol.Qt import QtGui

        app = QtGui.QGuiApplication.instance()
        if app is not None and hasattr(app, "platformName"):
            return str(app.platformName()).lower()
    except Exception:
        pass
    return ""


def _native_wid(widget):
    try:
        return int(widget.winId())
    except Exception:
        return None


def _x11_pin_to_parent_desktop(child_wid, parent_wid):
    """Copy PyMOL's EWMH desktop onto the popup and clear sticky/all-desktops.

    Qt.Tool + WindowStaysOnTopHint typically set _NET_WM_STATE_STICKY and/or
    _NET_WM_DESKTOP=0xFFFFFFFF, which WMs treat as "show on every workspace".
    See EWMH _NET_WM_DESKTOP / _NET_WM_STATE_STICKY.
    """
    import ctypes
    import ctypes.util

    libname = ctypes.util.find_library("X11")
    if not libname:
        return
    x11 = ctypes.CDLL(libname)
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XInternAtom.restype = ctypes.c_ulong
    x11.XDefaultRootWindow.restype = ctypes.c_ulong
    x11.XGetWindowProperty.restype = ctypes.c_int

    dpy = x11.XOpenDisplay(None)
    if not dpy:
        return

    class XClientMessageEvent(ctypes.Structure):
        _fields_ = [
            ("type", ctypes.c_int),
            ("serial", ctypes.c_ulong),
            ("send_event", ctypes.c_int),
            ("display", ctypes.c_void_p),
            ("window", ctypes.c_ulong),
            ("message_type", ctypes.c_ulong),
            ("format", ctypes.c_int),
            ("data", ctypes.c_long * 5),
        ]

    ClientMessage = 33
    SubstructureNotify = 1 << 19
    SubstructureRedirect = 1 << 20
    ALL_DESKTOPS = 0xFFFFFFFF

    def atom(name):
        return x11.XInternAtom(dpy, name.encode("ascii"), 0)

    def get_cardinal(wid, name):
        actual_type = ctypes.c_ulong()
        actual_format = ctypes.c_int()
        nitems = ctypes.c_ulong()
        bytes_after = ctypes.c_ulong()
        prop = ctypes.c_void_p()
        status = x11.XGetWindowProperty(
            dpy,
            ctypes.c_ulong(wid),
            atom(name),
            ctypes.c_long(0),
            ctypes.c_long(1),
            0,
            atom("CARDINAL"),
            ctypes.byref(actual_type),
            ctypes.byref(actual_format),
            ctypes.byref(nitems),
            ctypes.byref(bytes_after),
            ctypes.byref(prop),
        )
        if status != 0 or not nitems.value or not prop.value:
            return None
        value = int(ctypes.cast(prop, ctypes.POINTER(ctypes.c_ulong))[0])
        x11.XFree(prop)

    def send(wid, message, *values):
        event = XClientMessageEvent()
        event.type = ClientMessage
        event.serial = 0
        event.send_event = 1
        event.display = dpy
        event.window = ctypes.c_ulong(wid)
        event.message_type = atom(message)
        event.format = 32
        data = [0, 0, 0, 0, 0]
        for i, value in enumerate(values[:5]):
            data[i] = int(value)
        event.data = (ctypes.c_long * 5)(*data)
        root = x11.XDefaultRootWindow(dpy)
        x11.XSendEvent(
            dpy,
            ctypes.c_ulong(root),
            0,
            SubstructureNotify | SubstructureRedirect,
            ctypes.byref(event),
        )

    try:
        desktop = get_cardinal(parent_wid, "_NET_WM_DESKTOP")
        if desktop is None or desktop == ALL_DESKTOPS:
            root = x11.XDefaultRootWindow(dpy)
            desktop = get_cardinal(root, "_NET_CURRENT_DESKTOP")
        if desktop is not None and desktop != ALL_DESKTOPS:
            send(child_wid, "_NET_WM_DESKTOP", desktop, 1)
        send(child_wid, "_NET_WM_STATE", 0, atom("_NET_WM_STATE_STICKY"), 0, 1)
        x11.XFlush(dpy)
    finally:
        x11.XCloseDisplay(dpy)


def _raise_open_tool_windows():
    alive = []
    for widget in _OPEN_TOOL_WINDOWS:
        try:
            if widget.isVisible():
                widget.raise_()
                alive.append(widget)
        except RuntimeError:
            continue
    _OPEN_TOOL_WINDOWS[:] = alive


def _install_raise_on_parent_activate(anchor):
    """Raise our dialogs when PyMOL is focused, without WS_EX_TOPMOST / ABOVE."""
    global _RAISE_FILTER
    QtCore, _, _ = qt_modules()
    if QtCore is None or anchor is None:
        return

    class _RaiseFilter(QtCore.QObject):
        def eventFilter(self, obj, event):
            etype = event.type()
            interesting = (
                QtCore.QEvent.WindowActivate,
                getattr(QtCore.QEvent, "ActivationChange", -1),
                getattr(QtCore.QEvent, "FocusIn", -1),
            )
            if etype in interesting:
                _raise_open_tool_windows()
            return False

    if _RAISE_FILTER is None:
        _RAISE_FILTER = _RaiseFilter(anchor)
        try:
            anchor.installEventFilter(_RAISE_FILTER)
        except Exception:
            _RAISE_FILTER = None


def _track_tool_window(widget):
    if widget not in _OPEN_TOOL_WINDOWS:
        _OPEN_TOOL_WINDOWS.append(widget)

    def _forget(*_args):
        try:
            _OPEN_TOOL_WINDOWS.remove(widget)
        except ValueError:
            pass

    try:
        widget.destroyed.connect(_forget)
    except Exception:
        pass


def bind_tool_window(widget):
    """After show(): own the popup as a child of PyMOL's desktop, not all desktops."""
    QtCore, _, QtWidgets = qt_modules()
    if QtCore is None:
        return
    anchor = getattr(widget, "_pmv_window_anchor", None)
    if anchor is None:
        anchor = find_pymol_window(QtWidgets)
    _track_tool_window(widget)
    if anchor is None:
        return
    _install_raise_on_parent_activate(anchor)

    def _do_bind():
        try:
            wh = widget.windowHandle()
            ah = anchor.windowHandle()
            if wh is not None and ah is not None and hasattr(wh, "setTransientParent"):
                wh.setTransientParent(ah)
        except Exception:
            pass
        if _qt_platform_name() == "xcb":
            child_wid = _native_wid(widget)
            parent_wid = _native_wid(anchor)
            if child_wid and parent_wid:
                try:
                    _x11_pin_to_parent_desktop(child_wid, parent_wid)
                except Exception:
                    pass

    _do_bind()
    QtCore.QTimer.singleShot(0, _do_bind)


def configure_tool_window(widget, anchor=None):
    """Normal dialog owned by PyMOL — not a sticky Tool / always-on-top window.

    Qt.Tool + WindowStaysOnTopHint is what leaked across virtual desktops:
    on X11 that becomes UTILITY + STICKY / _NET_WM_DESKTOP=all; on Windows /
    WSLg it becomes WS_EX_TOPMOST, which is shown on every desktop.
    """
    QtCore, _, QtWidgets = qt_modules()
    if QtCore is None:
        return
    if anchor is None:
        anchor = find_pymol_window(QtWidgets)
    widget._pmv_window_anchor = anchor
    flags = widget.windowFlags()
    flags &= ~QtCore.Qt.Tool
    flags &= ~QtCore.Qt.WindowStaysOnTopHint
    flags |= QtCore.Qt.Dialog
    widget.setWindowFlags(flags)


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
