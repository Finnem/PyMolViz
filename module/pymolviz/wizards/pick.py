"""Screen-space atom picking from a cached get_view() + coordinates."""

import math

_CACHED_VIEWER = None


def qt_modules():
    try:
        from pymol.Qt import QtCore, QtGui, QtWidgets
        return QtCore, QtGui, QtWidgets
    except Exception:
        return None, None, None


def qt_widget_alive(widget) -> bool:
    """Return False when the underlying C/C++ Qt object was deleted."""
    if widget is None:
        return False
    try:
        import shiboken6 as shiboken
        return shiboken.isValid(widget)
    except ImportError:
        try:
            import shiboken2 as shiboken
            return shiboken.isValid(widget)
        except ImportError:
            pass
    try:
        import sip
        return not sip.isdeleted(widget)
    except ImportError:
        pass
    try:
        widget.objectName()
        return True
    except RuntimeError:
        return False


class DeferredCallback:
    """Coalesce QTimer.singleShot(0, ...) callbacks and cancel on page teardown."""

    def __init__(self):
        self._generation = 0

    def cancel(self):
        self._generation += 1

    def schedule(self, callback, page=None):
        QtCore, _, _ = qt_modules()
        if QtCore is None:
            return
        if page is not None and not qt_widget_alive(page):
            return
        self._generation += 1
        generation = self._generation

        def _run():
            if generation != self._generation:
                return
            if page is not None and not qt_widget_alive(page):
                return
            try:
                callback()
            except RuntimeError:
                pass

        QtCore.QTimer.singleShot(0, _run)


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


def stacked_tool_windows(widgets):
    """Raise color pickers last so a live preview cannot bury them."""
    leading = []
    trailing = []
    for widget in widgets:
        if getattr(widget, "_pmv_raise_last", False):
            trailing.append(widget)
        else:
            leading.append(widget)
    return leading + trailing


def _raise_open_tool_windows():
    alive = []
    for widget in stacked_tool_windows(list(_OPEN_TOOL_WINDOWS)):
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
        _untrack_tool_window(widget)

    try:
        widget.destroyed.connect(_forget)
    except Exception:
        pass


def _untrack_tool_window(widget):
    try:
        _OPEN_TOOL_WINDOWS.remove(widget)
    except ValueError:
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
    if getattr(widget, "_pmv_no_transient", False):
        _install_raise_on_parent_activate(find_pymol_window(QtWidgets))
        return
    if anchor is None:
        return
    _install_raise_on_parent_activate(anchor)

    def _do_bind():
        if not qt_widget_alive(widget):
            _untrack_tool_window(widget)
            return
        try:
            was_visible = widget.isVisible()
            try:
                wh = widget.windowHandle()
                ah = None
                if qt_widget_alive(anchor):
                    ah = anchor.windowHandle()
                if wh is not None and ah is not None and hasattr(wh, "setTransientParent"):
                    wh.setTransientParent(ah)
            except Exception:
                pass
            if _qt_platform_name() == "xcb":
                # Only pin to the PyMOL desktop. Pinning a picker to another
                # StayOnTop dialog can pass a non-EWMH Qt winId into X11.
                pymol = find_pymol_window(QtWidgets)
                if pymol is not None and anchor is pymol:
                    child_wid = _native_wid(widget)
                    parent_wid = _native_wid(anchor)
                    if child_wid and parent_wid:
                        try:
                            _x11_pin_to_parent_desktop(child_wid, parent_wid)
                        except Exception:
                            pass
            if was_visible and qt_widget_alive(widget) and not widget.isVisible():
                widget.show()
                widget.raise_()
        except RuntimeError:
            _untrack_tool_window(widget)

    _do_bind()
    QtCore.QTimer.singleShot(0, _do_bind)


def configure_tool_window(widget, anchor=None):
    """Keep wizard popups visible above the PyMOL viewer.

    StayOnTop only — Qt.Tool windows hide (and WA_DeleteOnClose parents die)
    when the 3D viewer is focused. Do not Qt-parent or setTransientParent the
    Fields/Visuals window to PyMOL. bind_tool_window() still tracks the
    window so it can be raised when the viewer is activated.
    """
    QtCore, _, QtWidgets = qt_modules()
    if QtCore is None:
        return
    widget._pmv_window_anchor = None
    widget._pmv_no_transient = True
    widget.setWindowFlags(
        widget.windowFlags() | QtCore.Qt.WindowStaysOnTopHint
    )
    from .widgets.dialog_enter import install_enter_commits_editor

    install_enter_commits_editor(widget)


def overlay_window(widget):
    """Top-level window owning ``widget`` (Fields / Visuals / pose dialog)."""
    if widget is None:
        return None
    window_fn = getattr(widget, "window", None)
    if callable(window_fn):
        try:
            window = window_fn()
        except Exception:
            window = None
        if window is not None:
            return window
    return widget


def _parent_widget(widget):
    getter = getattr(widget, "parentWidget", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            return None
    return None


def _stay_on_top_hint(QtCore):
    qt = getattr(QtCore, "Qt", None)
    return getattr(qt, "WindowStaysOnTopHint", None)


def _has_stay_on_top(widget, stays):
    if widget is None or stays is None:
        return False
    try:
        return bool(widget.windowFlags() & stays)
    except Exception:
        return False


def _set_transient_parent(child, parent):
    if child is None or parent is None or child is parent:
        return
    try:
        child.winId()
    except Exception:
        pass
    try:
        parent.winId()
    except Exception:
        pass
    try:
        ch = child.windowHandle()
        ph = parent.windowHandle()
        if ch is not None and ph is not None and hasattr(ch, "setTransientParent"):
            ch.setTransientParent(ph)
    except Exception:
        pass


def _restore_dialog_parent(dialog, overlay):
    if overlay is None or dialog is overlay:
        return
    if _parent_widget(dialog) is overlay:
        return
    try:
        dialog.setParent(overlay, dialog.windowFlags())
    except TypeError:
        try:
            dialog.setParent(overlay)
        except Exception:
            pass
    except Exception:
        pass


def configure_overlay_dialog(dialog, parent=None):
    """Stack a popup above StayOnTop Tool overlays (Fields, Visuals).

    ``QMessageBox.question(parent)`` does not copy ``WindowStaysOnTopHint``,
    and raise-on-activate would otherwise lift the catalog over the confirm.
    Parent to the overlay, inherit StayOnTop, and raise after the catalog.
    """
    if dialog is None:
        return dialog
    overlay = overlay_window(parent if parent is not None else _parent_widget(dialog))
    QtCore, _, _ = qt_modules()
    stays = _stay_on_top_hint(QtCore) if QtCore is not None else None
    should_stay = _has_stay_on_top(overlay, stays) or overlay in _OPEN_TOOL_WINDOWS
    if should_stay and stays is not None:
        try:
            dialog.setWindowFlags(dialog.windowFlags() | stays)
        except Exception:
            pass
        _restore_dialog_parent(dialog, overlay)
    try:
        dialog._pmv_raise_last = True
    except Exception:
        pass
    from .widgets.dialog_enter import install_enter_commits_editor

    install_enter_commits_editor(dialog)
    _track_tool_window(dialog)
    _set_transient_parent(dialog, overlay)
    return dialog


def _exec_dialog(dialog):
    run = getattr(dialog, "exec_", None)
    if not callable(run):
        run = getattr(dialog, "exec", None)
    if not callable(run):
        return 0
    return run()


def overlay_exec(dialog, parent=None):
    """``exec()`` a dialog after stacking it above the StayOnTop overlay."""
    configure_overlay_dialog(dialog, parent)
    try:
        try:
            dialog.raise_()
            dialog.activateWindow()
        except Exception:
            pass
        return _exec_dialog(dialog)
    finally:
        _untrack_tool_window(dialog)


def overlay_message_box(
    parent,
    title,
    text,
    icon=None,
    buttons=None,
    default=None,
):
    """QMessageBox parented to the visible StayOnTop wizard window. Does not exec()."""
    _, _, QtWidgets = qt_modules()
    if QtWidgets is None or not hasattr(QtWidgets, "QMessageBox"):
        return None
    overlay = overlay_window(parent)
    box = QtWidgets.QMessageBox(overlay)
    box.setWindowTitle(str(title or ""))
    box.setText(str(text or ""))
    if icon is not None:
        box.setIcon(icon)
    if buttons is not None:
        box.setStandardButtons(buttons)
    if default is not None:
        box.setDefaultButton(default)
    configure_overlay_dialog(box, overlay)
    return box


def overlay_question(parent, title, text, buttons=None, default=None):
    _, _, QtWidgets = qt_modules()
    if QtWidgets is None or not hasattr(QtWidgets, "QMessageBox"):
        return 0
    if buttons is None:
        buttons = QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
    box = overlay_message_box(
        parent,
        title,
        text,
        icon=QtWidgets.QMessageBox.Question,
        buttons=buttons,
        default=default,
    )
    if box is None:
        return 0
    return overlay_exec(box, overlay_window(parent))


def overlay_warning(parent, title, text):
    _, _, QtWidgets = qt_modules()
    if QtWidgets is None or not hasattr(QtWidgets, "QMessageBox"):
        return 0
    box = overlay_message_box(
        parent,
        title,
        text,
        icon=QtWidgets.QMessageBox.Warning,
    )
    if box is None:
        return 0
    return overlay_exec(box, overlay_window(parent))


def overlay_information(parent, title, text):
    _, _, QtWidgets = qt_modules()
    if QtWidgets is None or not hasattr(QtWidgets, "QMessageBox"):
        return 0
    box = overlay_message_box(
        parent,
        title,
        text,
        icon=QtWidgets.QMessageBox.Information,
    )
    if box is None:
        return 0
    return overlay_exec(box, overlay_window(parent))


def _overlay_file_dialog(parent, caption, directory, name_filter, save):
    _, _, QtWidgets = qt_modules()
    if QtWidgets is None or not hasattr(QtWidgets, "QFileDialog"):
        return None
    overlay = overlay_window(parent)
    dialog = QtWidgets.QFileDialog(overlay, str(caption or ""), directory or "", name_filter or "")
    if save:
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dialog.setFileMode(QtWidgets.QFileDialog.AnyFile)
    else:
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
    configure_overlay_dialog(dialog, overlay)
    return dialog


def overlay_get_open_file_name(parent, caption, directory="", filter=""):
    dialog = _overlay_file_dialog(parent, caption, directory, filter, save=False)
    if dialog is None:
        return "", ""
    if not overlay_exec(dialog, overlay_window(parent)):
        return "", ""
    files = dialog.selectedFiles()
    chosen = files[0] if files else ""
    name_filter = ""
    getter = getattr(dialog, "selectedNameFilter", None)
    if callable(getter):
        try:
            name_filter = getter()
        except Exception:
            name_filter = ""
    return chosen, name_filter


def overlay_get_save_file_name(parent, caption, directory="", filter=""):
    dialog = _overlay_file_dialog(parent, caption, directory, filter, save=True)
    if dialog is None:
        return "", ""
    if not overlay_exec(dialog, overlay_window(parent)):
        return "", ""
    files = dialog.selectedFiles()
    chosen = files[0] if files else ""
    name_filter = ""
    getter = getattr(dialog, "selectedNameFilter", None)
    if callable(getter):
        try:
            name_filter = getter()
        except Exception:
            name_filter = ""
    return chosen, name_filter


def find_viewer_widget(QtWidgets):
    global _CACHED_VIEWER
    cached = _CACHED_VIEWER
    if cached is not None:
        try:
            if (
                qt_widget_alive(cached)
                and cached.isVisible()
                and cached.width() >= 80
                and cached.height() >= 80
            ):
                return cached
        except Exception:
            pass
        _CACHED_VIEWER = None
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
    _CACHED_VIEWER = ranked[-1][-1]
    return _CACHED_VIEWER


def _widget_is_descendant(widget, root):
    if widget is None or root is None:
        return False
    current = widget
    for _ in range(64):
        if current is root:
            return True
        try:
            current = current.parentWidget()
        except Exception:
            return False
        if current is None:
            return False
    return False


def _cursor_in_widget(QtGui, widget) -> bool:
    if widget is None:
        return False
    try:
        local = widget.mapFromGlobal(QtGui.QCursor.pos())
        return bool(widget.rect().contains(local))
    except Exception:
        return False


def idle_selection_poll_ok(page=None) -> bool:
    """True when an idle ``cmd`` poll will not cancel names-panel hover.

    ``cmd.get_names`` / ``count_atoms`` / ``iterate`` from a QTimer abort
    PyMOL object-list hover, so skip those polls unless the pointer is on the
    3D view or the wizard page that owns the timer.

    OpenGL viewers often fail ``widgetAt`` (``under is None``). Fall back to
    the viewer rectangle so Add Clicked Atoms still polls while picking.
    """
    QtCore, QtGui, QtWidgets = qt_modules()
    if QtWidgets is None or QtGui is None:
        return True
    try:
        app = QtWidgets.QApplication.instance()
        if app is None or not hasattr(app, "widgetAt"):
            return True
        under = app.widgetAt(QtGui.QCursor.pos())
    except Exception:
        return True
    viewer = find_viewer_widget(QtWidgets)
    if under is None:
        return _cursor_in_widget(QtGui, viewer)
    if viewer is not None and _widget_is_descendant(under, viewer):
        return True
    if page is not None and _widget_is_descendant(under, page):
        return True
    return False


def pointer_over_viewer():
    """True when the cursor is on the 3D view (or Qt is unavailable).

    Idle ``cmd`` polls from QTimer cancel PyMOL object-list hover; skip them
    while the pointer is on the names panel or other chrome.
    """
    QtCore, QtGui, QtWidgets = qt_modules()
    if QtWidgets is None or QtGui is None:
        return True
    try:
        app = QtWidgets.QApplication.instance()
        if app is None or not hasattr(app, "widgetAt"):
            return True
        under = app.widgetAt(QtGui.QCursor.pos())
    except Exception:
        return True
    if under is None:
        try:
            buttons = app.mouseButtons()
            if buttons and int(buttons) != 0:
                return True
        except Exception:
            pass
        return False
    viewer = find_viewer_widget(QtWidgets)
    if viewer is None:
        return True
    return _widget_is_descendant(under, viewer)


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


def viewer_click_selection_expr(widget, x, y, view, viewport, fov, pad=2.5):
    """``(visible and enabled)`` clipped to the click-ray AABB. No ``cmd`` calls."""
    from ..util.view import click_ray_selection

    click_x, click_y = qt_to_pymol_xy(widget, x, y)
    try:
        width, height = float(viewport[0]), float(viewport[1])
    except Exception:
        width, height = 0.0, 0.0
    if width < 1 or height < 1:
        scale = widget_fb_scale(widget)
        width = float(widget.width()) * scale
        height = float(widget.height()) * scale
    rect_bottom = max(widget_fb_scale(widget) * widget.height() - height, 0.0)
    sx = float(click_x)
    sy = float(click_y) - rect_bottom
    return click_ray_selection(view, sx, sy, width, height, fov, pad=pad)


def pick_atom(view, coords, widget, x, y, viewport, fov, ortho, ids=None, max_px=24.0):
    """Nearest atom under the cursor using cached view/coords. No cmd calls."""
    import numpy as np

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
    pts = np.asarray(coords, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return None
    pts = pts[:, :3]
    rect_bottom = max(widget_fb_scale(widget) * widget.height() - height, 0.0)
    sx = float(click_x)
    sy = float(click_y) - rect_bottom
    fov_width = 2.0 * math.tan(math.radians(max(abs(float(fov)), 1.0)) / 2.0)
    origin_depth = max(abs(float(view[11])), 1e-6)
    front = float(view[15])
    back = float(view[16])
    relx = pts[:, 0] - float(view[12])
    rely = pts[:, 1] - float(view[13])
    relz = pts[:, 2] - float(view[14])
    cx = float(view[0]) * relx + float(view[3]) * rely + float(view[6]) * relz + float(view[9])
    cy = float(view[1]) * relx + float(view[4]) * rely + float(view[7]) * relz + float(view[10])
    cz = float(view[2]) * relx + float(view[5]) * rely + float(view[8]) * relz + float(view[11])
    if ortho:
        depth = np.full(pts.shape[0], origin_depth, dtype=float)
    else:
        depth = -cz
    valid = depth >= 1e-4
    if front > 0.0 and back > front:
        valid &= (depth >= front * 0.5) & (depth <= back * 1.5)
    angstrom_per_px = depth * fov_width / height
    valid &= angstrom_per_px >= 1e-8
    if not np.any(valid):
        return None
    safe = np.where(valid, angstrom_per_px, 1.0)
    dx = (width * 0.5 + cx / safe) - sx
    dy = (height * 0.5 + cy / safe) - sy
    dist2 = dx * dx + dy * dy
    valid &= dist2 <= (max_px * max_px)
    if not np.any(valid):
        return None
    # Prefer nearer in screen pixels, then closer in depth.
    order = dist2 + depth * 1e-9
    order = np.where(valid, order, np.inf)
    best_index = int(np.argmin(order))
    if not np.isfinite(order[best_index]):
        return None
    best = (float(pts[best_index, 0]), float(pts[best_index, 1]), float(pts[best_index, 2]))
    return best, atom_sele(ids, best_index)


_LIVE_CLICK_SELE_MAX = 32
_CLICK_PICK_PX = 48.0


def _cmd_state(cmd_) -> int:
    try:
        state = int(cmd_.get_state())
    except Exception:
        return 1
    return state if state > 0 else 1


def _view_fov(cmd_) -> float:
    try:
        fov = abs(float(cmd_.get("field_of_view")))
    except Exception:
        fov = 0.0
    return fov if fov >= 1.0 else 20.0


def _view_ortho(cmd_, view) -> int:
    try:
        return int(float(cmd_.get("orthoscopic")))
    except Exception:
        pass
    if len(view) > 17 and float(view[17]) > 0.5:
        return 1
    return 0


def _live_sele_expr(cmd_):
    """``(sele)`` when it is enabled and small enough to iterate on a click."""
    get_names = getattr(cmd_, "get_names", None)
    if callable(get_names):
        try:
            names = get_names("selections", enabled_only=1)
        except TypeError:
            try:
                names = get_names("selections", 1)
            except Exception:
                names = None
        except Exception:
            names = None
        if names is not None:
            try:
                enabled = {str(name) for name in names}
            except TypeError:
                enabled = None
            if enabled is not None and "sele" not in enabled:
                return None
    try:
        n = int(cmd_.count_atoms("(sele)"))
    except Exception:
        return None
    if n <= 0 or n > _LIVE_CLICK_SELE_MAX:
        return None
    return "(sele)"


def _pick_atom_rows(cmd_, sele, state=None):
    """``(model, id, x, y, z)`` for ``sele``. Empty if iterate is unavailable."""
    if state is None:
        state = _cmd_state(cmd_)
    expressions = (
        "rows.append((model, ID, x, y, z))",
        "rows.append((model, index, x, y, z))",
    )
    iterate_state = getattr(cmd_, "iterate_state", None)
    iterate = getattr(cmd_, "iterate", None)
    for expr in expressions:
        rows = []
        try:
            if callable(iterate_state):
                iterate_state(state, sele, expr, space={"rows": rows})
            elif callable(iterate):
                iterate(sele, expr, space={"rows": rows})
        except Exception:
            rows = []
        if rows:
            return rows
    return []


def _atom_id_near(cmd_, pos, state):
    """``(model, id)`` of a visible atom at ``pos``. Does not create a named selection."""
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    sele = "(visible and enabled) within 0.2 of [%g,%g,%g]" % (x, y, z)
    expressions = (
        "ids.append((model, ID))",
        "ids.append((model, index))",
    )
    iterate_state = getattr(cmd_, "iterate_state", None)
    iterate = getattr(cmd_, "iterate", None)
    for expr in expressions:
        ids = []
        try:
            if callable(iterate_state):
                iterate_state(state, sele, expr, space={"ids": ids})
            elif callable(iterate):
                iterate(sele, expr, space={"ids": ids})
        except Exception:
            ids = []
        if ids:
            return (str(ids[0][0]), int(ids[0][1]))
    return None


def record_viewer_atom_click(cmd_, widget, x, y) -> None:
    """Remember the atom under a viewer click. Does not consume the click.

    Prefers PyMOL's own pick (``You clicked`` / ``pk1``). Screen-space
    fallback never iterates all visible atoms and never clears a better hit.
    """
    from .last_click import (
        last_clicked_atom,
        last_clicked_path,
        parse_atom_sele,
        record_pymol_click,
        set_last_clicked_atom,
    )

    if cmd_ is None or widget is None:
        return
    record_pymol_click(cmd_)
    if last_clicked_atom() is not None or last_clicked_path() is not None:
        return
    try:
        view = tuple(cmd_.get_view())
    except Exception:
        return
    try:
        viewport = tuple(float(v) for v in cmd_.get_viewport())
    except Exception:
        viewport = (float(widget.width()), float(widget.height()))
    fov = _view_fov(cmd_)
    ortho = _view_ortho(cmd_, view)
    state = _cmd_state(cmd_)
    live = _live_sele_expr(cmd_)
    ray = viewer_click_selection_expr(widget, x, y, view, viewport, fov)
    rows = _pick_atom_rows(cmd_, live, state) if live else []
    if not rows:
        rows = _pick_atom_rows(cmd_, ray, state)
    target = None
    parsed = None
    if rows:
        try:
            coords = [[float(row[2]), float(row[3]), float(row[4])] for row in rows]
            ids = [(row[0], int(row[1])) for row in rows]
        except Exception:
            coords = None
            ids = None
        else:
            target = pick_atom(
                view, coords, widget, x, y, viewport, fov, ortho,
                ids=ids, max_px=_CLICK_PICK_PX,
            )
            if target is None and len(rows) == 1:
                parsed = (str(rows[0][0]), int(rows[0][1]))
    if target is None and parsed is None:
        try:
            coords = cmd_.get_coords(ray, state)
        except Exception:
            coords = None
        if coords is not None and len(coords):
            target = pick_atom(
                view, coords, widget, x, y, viewport, fov, ortho,
                max_px=_CLICK_PICK_PX,
            )
            if target is not None:
                parsed = _atom_id_near(cmd_, target[0], state)
    if parsed is None and target is not None:
        parsed = parse_atom_sele(target[1])
    if parsed is None:
        return
    set_last_clicked_atom(parsed[0], parsed[1])

