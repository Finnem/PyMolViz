"""Middle-click gesture ownership for click-to-center vs translate."""

from ..util.pymol_helpers import set_button_action
from .pick import find_viewer_widget, qt_modules


def take_over_center_click(cmd_):
    """Wizard owns click-to-center so PyMOL's delayed 'cent' cannot be cancelled."""
    set_button_action(cmd_, "single_middle", "none", "none")


def restore_viewing_mouse(cmd_):
    """Restore default middle-click centering when the wizard closes."""
    set_button_action(cmd_, "single_middle", "none", "cent")
    try:
        cmd_.unpick()
    except Exception:
        pass


def install_middle_click_filter(wizard):
    QtCore, QtGui, QtWidgets = qt_modules()
    if QtCore is None:
        return None, None
    widget = find_viewer_widget(QtWidgets)
    if widget is None:
        return None, None

    class _MiddleClickFilter(QtCore.QObject):
        _drag_px = 20

        def __init__(self, parent):
            QtCore.QObject.__init__(self, parent)
            self._press = None
            self._dragged = False

        def eventFilter(self, watched, event):
            if watched is not widget:
                return False
            middle = getattr(QtCore.Qt, "MiddleButton", None)
            if middle is None:
                middle = QtCore.Qt.MouseButton.MiddleButton
            etype = event.type()
            sphere = getattr(wizard, "camera_sphere", None)
            if sphere is None:
                return False

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
                    # Keep tiny jitter from becoming a translate; we own click-to-center.
                    return True
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
    return filt, widget
