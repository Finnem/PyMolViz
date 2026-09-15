"""Qt smoke tests for the on/off switch control."""

from __future__ import annotations

import pytest


def _qt_app():
    from pymolviz.wizards.pick import qt_modules

    QtCore, _, QtWidgets = qt_modules()
    if QtWidgets is None or not hasattr(QtWidgets, "QApplication"):
        pytest.skip("Qt widgets unavailable")
    if not hasattr(QtWidgets, "QCheckBox"):
        pytest.skip("QCheckBox unavailable")
    try:
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        return QtCore, QtWidgets, app
    except Exception:
        pytest.skip("QApplication not usable")


@pytest.mark.qt
def test_switch_checkable_api_and_compact_size():
    from pymolviz.wizards.widgets.switch import make_switch

    QtCore, QtWidgets, _app = _qt_app()
    try:
        labeled = make_switch("Live preview")
        compact = make_switch(compact=True)
        with_icon = make_switch("Snap to atoms", icon="snap")
    except Exception:
        pytest.skip("QCheckBox not usable")
    assert labeled.isCheckable()
    assert not labeled.isChecked()
    labeled.setChecked(True)
    assert labeled.isChecked()
    assert labeled.text() == "Live preview"
    assert compact.minimumWidth() < labeled.minimumWidth()
    assert compact.minimumHeight() <= labeled.minimumHeight() + 4
    assert with_icon._pmv_icon == "snap"
    assert with_icon.minimumWidth() > labeled.minimumWidth()
    assert "indicator" in labeled.styleSheet()


@pytest.mark.qt
def test_iter_setting_widgets_includes_switch():
    from pymolviz.wizards.tooltips import iter_setting_widgets
    from pymolviz.wizards.widgets.switch import make_switch

    QtCore, QtWidgets, _app = _qt_app()
    try:
        host = QtWidgets.QWidget()
        sw = make_switch("Reverse", host)
        sw.setObjectName("pmvColormapReverse")
    except Exception:
        pytest.skip("QWidget not usable")
    found = iter_setting_widgets(host)
    assert sw in found
    assert type(sw).__name__ == "QCheckBox"
