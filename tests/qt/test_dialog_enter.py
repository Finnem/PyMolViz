"""Qt: Enter in a focused spin commits without accepting the dialog."""

from __future__ import annotations

import pytest


def _qt_app():
    from pymolviz.wizards.pick import qt_modules

    QtCore, _, QtWidgets = qt_modules()
    if QtWidgets is None or not hasattr(QtWidgets, "QDialog"):
        pytest.skip("Qt widgets unavailable")
    try:
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        return QtCore, QtWidgets, app
    except Exception:
        pytest.skip("QApplication not usable")


def _send_enter(QtCore, QtWidgets, widget):
    from pymolviz.wizards.pick import qt_modules

    _, QtGui, _ = qt_modules()
    if QtGui is None or not hasattr(QtGui, "QKeyEvent"):
        pytest.skip("QKeyEvent unavailable")
    app = QtWidgets.QApplication.instance()
    kinds = [QtCore.QEvent.KeyPress]
    override = getattr(QtCore.QEvent, "ShortcutOverride", None)
    if override is not None:
        kinds = [override, QtCore.QEvent.KeyPress]
    for etype in kinds:
        current = widget
        while current is not None:
            key_event = QtGui.QKeyEvent(
                etype,
                QtCore.Qt.Key_Return,
                QtCore.Qt.NoModifier,
            )
            QtWidgets.QApplication.sendEvent(current, key_event)
            if key_event.isAccepted():
                break
            parent_fn = getattr(current, "parentWidget", None)
            current = parent_fn() if callable(parent_fn) else None
    if app is not None:
        app.processEvents()


@pytest.mark.qt
def test_enter_in_spin_commits_without_accepting_dialog():
    from pymolviz.wizards.widgets.dialog_enter import install_enter_commits_editor

    QtCore, QtWidgets, _app = _qt_app()
    dialog = QtWidgets.QDialog()
    install_enter_commits_editor(dialog)
    layout = QtWidgets.QVBoxLayout(dialog)
    spin = QtWidgets.QDoubleSpinBox()
    spin.setRange(-100.0, 100.0)
    spin.setDecimals(2)
    spin.setValue(1.25)
    ok = QtWidgets.QPushButton("OK")
    ok.setDefault(True)
    ok.setAutoDefault(True)
    accepted = {"n": 0}
    ok.clicked.connect(lambda *_: accepted.__setitem__("n", accepted["n"] + 1))
    layout.addWidget(spin)
    layout.addWidget(ok)
    dialog.show()
    spin.setFocus()
    if spin.lineEdit() is not None:
        spin.lineEdit().setFocus()
        spin.lineEdit().setText("4.5")
    else:
        spin.setValue(4.5)
    target = spin.lineEdit() if spin.lineEdit() is not None else spin
    _send_enter(QtCore, QtWidgets, target)
    assert accepted["n"] == 0
    assert dialog.result() != QtWidgets.QDialog.Accepted
    focus = QtWidgets.QApplication.focusWidget()
    assert focus is not spin
    assert focus is not spin.lineEdit()
    assert spin.value() == pytest.approx(4.5)
    dialog.close()


@pytest.mark.qt
def test_enter_with_no_editor_focus_clicks_default():
    from pymolviz.wizards.widgets.dialog_enter import install_enter_commits_editor

    QtCore, QtWidgets, _app = _qt_app()
    dialog = QtWidgets.QDialog()
    install_enter_commits_editor(dialog)
    layout = QtWidgets.QVBoxLayout(dialog)
    spin = QtWidgets.QDoubleSpinBox()
    ok = QtWidgets.QPushButton("OK")
    ok.setDefault(True)
    accepted = {"n": 0}
    ok.clicked.connect(lambda *_: accepted.__setitem__("n", accepted["n"] + 1))
    layout.addWidget(spin)
    layout.addWidget(ok)
    dialog.show()
    dialog.setFocus()
    _send_enter(QtCore, QtWidgets, dialog)
    assert accepted["n"] == 1
    dialog.close()
