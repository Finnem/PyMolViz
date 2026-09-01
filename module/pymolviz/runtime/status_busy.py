"""Show a busy bar during a blocking CGO reload.

PyMOL's ray bar lives on the External GUI dock, not QStatusBar, and an
indeterminate QProgressBar will not paint while ``load_cgo`` freezes Qt.
We fill the native ray bar (if present) and draw a solid overlay along
the bottom of the 3D view so something is visible for the whole freeze.
"""

from __future__ import annotations

from contextlib import contextmanager

_MSG = "PyMOLViz: Updating anchored CGOs"
_depth = 0
_active = None


def reset_status_busy():
    """Drop a leftover bar (tests / failed follow)."""
    global _depth, _active
    _depth = 0
    _hide(_active)
    _active = None


def _pymol_qt_window():
    try:
        from pymol.gui import get_qtwindow
        window = get_qtwindow()
        if window is not None:
            return window
    except Exception:
        pass
    try:
        import pmg_qt.pymol_qt_gui as gui
        return getattr(gui, "window", None)
    except Exception:
        return None


def _qt():
    from ..wizards.pick import qt_modules
    return qt_modules()


def _flush_paint(widgets):
    QtCore, _, QtWidgets = _qt()
    if QtWidgets is None:
        return
    try:
        app = QtWidgets.QApplication.instance()
    except Exception:
        app = None
    if app is not None:
        send = getattr(app, "sendPostedEvents", None)
        if callable(send):
            try:
                send()
            except Exception:
                pass
        try:
            app.processEvents()
        except Exception:
            pass
    for widget in widgets:
        if widget is None:
            continue
        try:
            widget.repaint()
        except Exception:
            pass


def _cmd_progress(window):
    cmd = getattr(window, "cmd", None)
    getter = getattr(cmd, "get_progress", None)
    if not callable(getter):
        return -1.0
    try:
        return float(getter())
    except Exception:
        return -1.0


def _find_native_progressbar(window):
    if window is not None:
        bar = getattr(window, "progressbar", None)
        if bar is not None:
            return bar, getattr(window, "abortbutton", None)
    QtCore, _, QtWidgets = _qt()
    if QtWidgets is None:
        return None, None
    try:
        app = QtWidgets.QApplication.instance()
    except Exception:
        app = None
    if app is None or not hasattr(app, "allWidgets"):
        return None, None
    Progress = getattr(QtWidgets, "QProgressBar", None)
    for widget in app.allWidgets():
        try:
            if Progress is not None and isinstance(widget, Progress):
                return widget, None
            cls = widget.metaObject().className()
            if cls == "QProgressBar":
                return widget, None
        except Exception:
            continue
    return None, None


def _fill_native_bar(bar, message):
    try:
        bar.setRange(0, 100)
        bar.setValue(100)
    except Exception:
        pass
    if hasattr(bar, "setTextVisible"):
        try:
            bar.setTextVisible(True)
        except Exception:
            pass
    if hasattr(bar, "setFormat"):
        try:
            bar.setFormat(str(message))
        except Exception:
            pass
    for method in ("setMinimumHeight", "setMinimumWidth"):
        fn = getattr(bar, method, None)
        if callable(fn):
            try:
                fn(18 if "Height" in method else 80)
            except Exception:
                pass
    try:
        bar.show()
        bar.raise_()
    except Exception:
        pass
    parent = getattr(bar, "parentWidget", None)
    try:
        host = parent() if callable(parent) else None
    except Exception:
        host = None
    if host is not None:
        try:
            host.show()
        except Exception:
            pass
        layout = getattr(host, "layout", None)
        try:
            layout = layout() if callable(layout) else None
        except Exception:
            layout = None
        if layout is not None and hasattr(layout, "activate"):
            try:
                layout.activate()
            except Exception:
                pass


def _overlay_on_viewer(message):
    from ..wizards.pick import find_pymol_window, find_viewer_widget

    QtCore, QtGui, QtWidgets = _qt()
    if QtWidgets is None or QtCore is None:
        return None
    Progress = getattr(QtWidgets, "QProgressBar", None)
    if Progress is None:
        return None
    viewer = find_viewer_widget(QtWidgets)
    if viewer is None:
        return None
    window = viewer.window() or find_pymol_window(QtWidgets)
    parent = window if window is not None else viewer
    try:
        bar = Progress(parent)
    except Exception:
        return None
    try:
        bar.setObjectName("pmv_cgo_progress")
        if hasattr(bar, "setTextVisible"):
            bar.setTextVisible(True)
        bar.setRange(0, 100)
        bar.setValue(100)
        if hasattr(bar, "setFormat"):
            bar.setFormat(str(message))
        if hasattr(bar, "setStyleSheet"):
            bar.setStyleSheet(
                "QProgressBar { background: #1a1a1a; color: #fff; border: 1px solid #888;"
                " text-align: center; min-height: 18px; }"
                "QProgressBar::chunk { background: #2e8b57; }"
            )
        left, top, width = 8, 8, 200
        try:
            height = max(22, min(28, int(viewer.height() * 0.04) or 22))
            width = max(180, int(viewer.width()) - 16)
            local = QtCore.QPoint(8, max(8, int(viewer.height()) - height - 8))
            mapped = viewer.mapTo(parent, local) if parent is not viewer else local
            left, top = int(mapped.x()), int(mapped.y())
        except Exception:
            height = 22
        bar.setGeometry(left, top, width, height)
        bar.show()
        bar.raise_()
        bar.repaint()
    except Exception:
        try:
            bar.hide()
            later = getattr(bar, "deleteLater", None)
            if callable(later):
                later()
        except Exception:
            pass
        return None
    return bar


def _show(message):
    window = _pymol_qt_window()
    bar, abort = _find_native_progressbar(window)
    orig_update = getattr(window, "update_progress", None) if window is not None else None
    saved_range = (0, 100)
    saved_format = None
    abort_hidden = True
    if bar is not None:
        try:
            saved_range = (int(bar.minimum()), int(bar.maximum()))
        except Exception:
            saved_range = (0, 100)
        if hasattr(bar, "format"):
            try:
                saved_format = bar.format()
            except Exception:
                saved_format = None
        if abort is not None and hasattr(abort, "isHidden"):
            try:
                abort_hidden = bool(abort.isHidden())
            except Exception:
                abort_hidden = True

    def _keep_ours():
        if _cmd_progress(window) >= 0:
            if orig_update is not None and orig_update is not _keep_ours:
                orig_update()
            return
        if bar is not None:
            _fill_native_bar(bar, message)
        if abort is not None:
            try:
                abort.hide()
            except Exception:
                pass

    patched = False
    if window is not None and orig_update is not None:
        try:
            window.update_progress = _keep_ours
            patched = True
        except Exception:
            patched = False
    if bar is not None:
        _fill_native_bar(bar, message)
        if abort is not None:
            try:
                abort.hide()
            except Exception:
                pass
    overlay = _overlay_on_viewer(message)
    _flush_paint((bar, overlay, window))
    if bar is None and overlay is None:
        if patched and orig_update is not None:
            try:
                window.update_progress = orig_update
            except Exception:
                pass
        return None
    return {
        "window": window,
        "bar": bar,
        "overlay": overlay,
        "abort": abort,
        "orig_update": orig_update,
        "saved_range": saved_range,
        "saved_format": saved_format,
        "abort_hidden": abort_hidden,
        "patched": patched,
    }


def _hide(state):
    if not state:
        return
    overlay = state.get("overlay")
    if overlay is not None:
        try:
            overlay.hide()
        except Exception:
            pass
        later = getattr(overlay, "deleteLater", None)
        if callable(later):
            try:
                later()
            except Exception:
                pass
        try:
            overlay.setParent(None)
        except Exception:
            pass
    window = state.get("window")
    bar = state.get("bar")
    abort = state.get("abort")
    orig_update = state.get("orig_update")
    if state.get("patched") and window is not None and orig_update is not None:
        try:
            window.update_progress = orig_update
        except Exception:
            pass
    if bar is not None:
        saved_range = state.get("saved_range") or (0, 100)
        try:
            bar.setRange(int(saved_range[0]), int(saved_range[1]))
        except Exception:
            pass
        saved_format = state.get("saved_format")
        if saved_format is not None and hasattr(bar, "setFormat"):
            try:
                bar.setFormat(saved_format)
            except Exception:
                pass
        try:
            bar.hide()
        except Exception:
            pass
    if abort is not None:
        try:
            if state.get("abort_hidden", True):
                abort.hide()
            else:
                abort.show()
        except Exception:
            pass
    if orig_update is not None:
        try:
            orig_update()
        except Exception:
            pass


def begin_cgo_update(message=_MSG):
    global _depth, _active
    _depth += 1
    if _depth > 1:
        return
    try:
        _active = _show(message)
    except Exception:
        _active = None


def end_cgo_update():
    global _depth, _active
    if _depth <= 0:
        return
    _depth -= 1
    if _depth:
        return
    try:
        _hide(_active)
    finally:
        _active = None


def run_after_paint(func):
    """Run ``func`` after Qt has painted. Sync when PyMOL's GUI is not live."""
    if _pymol_qt_window() is None:
        func()
        return False
    QtCore, _, QtWidgets = _qt()
    app = None
    try:
        if QtWidgets is not None:
            app = QtWidgets.QApplication.instance()
    except Exception:
        app = None
    if QtCore is None or app is None or not hasattr(QtCore, "QTimer"):
        func()
        return False
    try:
        QtCore.QTimer.singleShot(0, func)
        return True
    except Exception:
        func()
        return False


@contextmanager
def cgo_update_status(message=_MSG):
    """Show a filled progress bar around a blocking CGO reload."""
    begin_cgo_update(message)
    try:
        yield
    finally:
        end_cgo_update()
