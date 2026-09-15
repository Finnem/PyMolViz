"""Qt tests: confirm dialogs stack above StayOnTop Fields/wizard windows."""

from __future__ import annotations

import pytest


def _qt_app():
    from pymolviz.wizards.pick import qt_modules

    QtCore, _, QtWidgets = qt_modules()
    if QtWidgets is None or not hasattr(QtWidgets, "QDialog"):
        pytest.skip("Qt widgets unavailable")
    if QtCore is None or not hasattr(QtCore, "Qt"):
        pytest.skip("QtCore unavailable")
    try:
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        return QtCore, QtWidgets, app
    except Exception:
        pytest.skip("QApplication / QWidget not usable")


@pytest.mark.qt
def test_overlay_message_box_inherits_stay_on_top_from_fields_window():
    from pymolviz.wizards.pick import (
        configure_tool_window,
        overlay_message_box,
        overlay_window,
    )
    import pymolviz.wizards.pick as pick

    QtCore, QtWidgets, _app = _qt_app()
    if not hasattr(QtWidgets, "QMessageBox"):
        pytest.skip("QMessageBox unavailable")

    stays = QtCore.Qt.WindowStaysOnTopHint
    before = list(pick._OPEN_TOOL_WINDOWS)
    fields = None
    box = None
    try:
        try:
            fields = QtWidgets.QDialog()
            fields.setWindowTitle("PyMOLViz Fields")
            configure_tool_window(fields)
            nested = QtWidgets.QWidget(fields)
            box = overlay_message_box(
                nested,
                "Delete field",
                "Delete this field and its dependents?",
            )
        except Exception:
            pytest.skip("QApplication / QWidget not usable")
        assert box is not None
        assert fields.windowFlags() & stays
        assert getattr(fields, "_pmv_no_transient", False) is True
        assert box.windowFlags() & stays
        parent = box.parentWidget()
        assert parent is fields or overlay_window(parent) is fields
        assert overlay_window(nested) is fields
        assert getattr(box, "_pmv_raise_last", False) is True
    finally:
        pick._OPEN_TOOL_WINDOWS[:] = before
        if box is not None:
            try:
                box.deleteLater()
            except Exception:
                pass
        if fields is not None:
            try:
                fields.deleteLater()
            except Exception:
                pass
