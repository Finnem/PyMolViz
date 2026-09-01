"""Middle-click gesture ownership for click-to-center vs translate."""

from ..util.pymol_helpers import set_button_action
from .pick import find_viewer_widget, qt_modules

_ACTIVE_CLICK_FILTER = None
_ACTIVE_CLICK_WIDGET = None


def _teardown_global_click_filter():
    global _ACTIVE_CLICK_FILTER, _ACTIVE_CLICK_WIDGET
    widget = _ACTIVE_CLICK_WIDGET
    filt = _ACTIVE_CLICK_FILTER
    _ACTIVE_CLICK_FILTER = None
    _ACTIVE_CLICK_WIDGET = None
    if widget is not None and filt is not None:
        try:
            widget.removeEventFilter(filt)
        except Exception:
            pass


def take_over_center_click(cmd_):
    """Wizard owns click-to-center so PyMOL's delayed 'cent' cannot be cancelled."""
    set_button_action(cmd_, "single_middle", "none", "none")


def restore_viewing_mouse(cmd_):
    """Restore default middle-click centering when the wizard closes."""
    _teardown_global_click_filter()
    try:
        cmd_.button("all", "all", "reset")
    except Exception:
        pass
    set_button_action(cmd_, "single_middle", "none", "cent")
    try:
        cmd_.unpick()
    except Exception:
        pass
    try:
        cmd_.edit_mode(0)
    except Exception:
        pass
    for args in (
        ("3button", "all", "reset"),
        ("3button", "all", "auto"),
    ):
        try:
            cmd_.button(*args)
        except Exception:
            pass


def install_middle_click_filter(wizard):
    QtCore, QtGui, QtWidgets = qt_modules()
    if QtCore is None:
        return None, None
    widget = find_viewer_widget(QtWidgets)
    if widget is None:
        return None, None

    _teardown_global_click_filter()

    class _MiddleClickFilter(QtCore.QObject):
        _drag_px = 20

        def __init__(self, parent):
            QtCore.QObject.__init__(self, parent)
            self._press = None
            self._dragged = False

        def eventFilter(self, watched, event):
            if watched is not widget:
                return False
            if getattr(wizard, "_closed", False):
                return False
            middle = getattr(QtCore.Qt, "MiddleButton", None)
            if middle is None:
                middle = QtCore.Qt.MouseButton.MiddleButton
            etype = event.type()
            sphere = getattr(wizard, "camera_sphere", None)
            if sphere is None:
                return False

            if etype == QtCore.QEvent.MouseMove and event.buttons():
                wizard._request_sphere_sync()
            if etype == getattr(QtCore.QEvent, "Wheel", None) or etype == 31:
                wizard._request_sphere_sync()

            if etype == QtCore.QEvent.MouseButtonPress and event.button() == middle:
                local = widget.mapFromGlobal(QtGui.QCursor.pos())
                self._press = (float(local.x()), float(local.y()))
                self._dragged = False
                return False

            if etype == QtCore.QEvent.MouseMove and self._press is not None:
                if event.buttons() & middle:
                    local = widget.mapFromGlobal(QtGui.QCursor.pos())
                    dx = local.x() - self._press[0]
                    dy = local.y() - self._press[1]
                    if (dx * dx + dy * dy) > (self._drag_px * self._drag_px):
                        if not self._dragged:
                            self._dragged = True
                            sphere.follow()
                return False

            if etype == QtCore.QEvent.MouseButtonRelease and event.button() == middle:
                press = self._press
                dragged = self._dragged
                self._press = None
                self._dragged = False
                if press is not None and not dragged:
                    wizard._on_middle_click(press[0], press[1])
                return False

            return False

    filt = _MiddleClickFilter(widget)
    widget.installEventFilter(filt)
    global _ACTIVE_CLICK_FILTER, _ACTIVE_CLICK_WIDGET
    _ACTIVE_CLICK_FILTER = filt
    _ACTIVE_CLICK_WIDGET = widget
    return filt, widget
