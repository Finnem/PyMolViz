"""Enter in a focused editor commits; it does not accept the dialog."""

from __future__ import annotations

from pymolviz.wizards.widgets.dialog_enter import (
    ACTION_ACCEPT,
    ACTION_COMMIT,
    ACTION_DEFOCUS,
    dialog_enter_action,
    handle_dialog_enter,
    value_editor_for,
)


def test_dialog_enter_action_accepts_only_without_child_focus():
    assert dialog_enter_action("none") == ACTION_ACCEPT
    assert dialog_enter_action("window") == ACTION_ACCEPT
    assert dialog_enter_action("editor") == ACTION_COMMIT
    assert dialog_enter_action("other") == ACTION_DEFOCUS


class _FakeEditor:
    def __init__(self, parent=None):
        self._parent = parent
        self.interpreted = False
        self.blurred = False

    def interpretText(self):
        self.interpreted = True

    def clearFocus(self):
        self.blurred = True

    def parentWidget(self):
        return self._parent


class _FakeOther:
    def __init__(self, parent):
        self._parent = parent
        self.blurred = False

    def clearFocus(self):
        self.blurred = True

    def parentWidget(self):
        return self._parent


def test_handle_dialog_enter_commits_spin_and_blocks_accept():
    dialog = object()
    spin = _FakeEditor(dialog)
    widgets = type("W", (), {"QAbstractSpinBox": _FakeEditor, "QLineEdit": type("L", (), {})})()
    assert handle_dialog_enter(dialog, spin, widgets) is True
    assert spin.interpreted is True
    assert spin.blurred is True


def test_handle_dialog_enter_allows_accept_when_nothing_focused():
    dialog = object()
    widgets = type("W", (), {"QAbstractSpinBox": type("S", (), {}), "QLineEdit": type("L", (), {})})()
    assert handle_dialog_enter(dialog, None, widgets) is False
    assert handle_dialog_enter(dialog, dialog, widgets) is False


def test_handle_dialog_enter_defocuses_other_child():
    dialog = object()
    child = _FakeOther(dialog)
    widgets = type("W", (), {"QAbstractSpinBox": type("S", (), {}), "QLineEdit": type("L", (), {})})()
    assert handle_dialog_enter(dialog, child, widgets) is True
    assert child.blurred is True


def test_value_editor_for_walks_to_spin_parent():
    class Line:
        pass

    class Spin:
        pass

    line = Line()
    spin = Spin()
    line.parentWidget = lambda: spin
    spin.parentWidget = lambda: None
    widgets = type("W", (), {"QAbstractSpinBox": Spin, "QLineEdit": Line, "QComboBox": type("C", (), {})})()
    assert value_editor_for(line, widgets) is spin
