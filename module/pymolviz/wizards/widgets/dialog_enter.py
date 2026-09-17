"""Enter in a focused field commits the value; it does not accept the dialog.

QLineEdit / QAbstractSpinBox ignore Return after committing so QDialog clicks
the default button (Done / OK). Swallow that when any child has focus.
"""

from __future__ import annotations

ACTION_ACCEPT = "accept"
ACTION_COMMIT = "commit"
ACTION_DEFOCUS = "defocus"

_FOCUS_NONE = "none"
_FOCUS_WINDOW = "window"
_FOCUS_EDITOR = "editor"
_FOCUS_OTHER = "other"


def dialog_enter_action(focus_role: str) -> str:
    """What Enter should do given the focused widget's role."""
    if focus_role in (_FOCUS_NONE, _FOCUS_WINDOW):
        return ACTION_ACCEPT
    if focus_role == _FOCUS_EDITOR:
        return ACTION_COMMIT
    return ACTION_DEFOCUS


def value_editor_for(widget, QtWidgets):
    """Spin box, line edit, or editable combo — or None."""
    if widget is None or QtWidgets is None:
        return None
    for cls_name in ("QTextEdit", "QPlainTextEdit"):
        cls = getattr(QtWidgets, cls_name, None)
        if cls is not None and isinstance(widget, cls):
            return None
    spin_cls = getattr(QtWidgets, "QAbstractSpinBox", None)
    line_cls = getattr(QtWidgets, "QLineEdit", None)
    combo_cls = getattr(QtWidgets, "QComboBox", None)
    current = widget
    for _ in range(8):
        if current is None:
            break
        if spin_cls is not None and isinstance(current, spin_cls):
            return current
        if combo_cls is not None and isinstance(current, combo_cls):
            is_editable = getattr(current, "isEditable", None)
            if callable(is_editable) and is_editable():
                return current
            return None
        if line_cls is not None and isinstance(current, line_cls):
            parent = _widget_parent(current)
            if spin_cls is not None and isinstance(parent, spin_cls):
                return parent
            if combo_cls is not None and isinstance(parent, combo_cls):
                return parent
            return current
        current = _widget_parent(current)
    return None


def commit_value_editor(editor) -> None:
    interpret = getattr(editor, "interpretText", None)
    if callable(interpret):
        try:
            interpret()
        except Exception:
            pass
    _clear_focus(editor)
    inner = getattr(editor, "lineEdit", None)
    if callable(inner):
        try:
            line = inner()
        except Exception:
            line = None
        if line is not None:
            _clear_focus(line)


def focus_role_for(dialog, focus, QtWidgets) -> str:
    if focus is None:
        return _FOCUS_NONE
    if focus is dialog:
        return _FOCUS_WINDOW
    if not _is_ancestor(dialog, focus):
        return _FOCUS_NONE
    if value_editor_for(focus, QtWidgets) is not None:
        return _FOCUS_EDITOR
    return _FOCUS_OTHER


def handle_dialog_enter(dialog, focus, QtWidgets) -> bool:
    """Commit/blur a focused child. True if the dialog must not accept."""
    action = dialog_enter_action(focus_role_for(dialog, focus, QtWidgets))
    if action == ACTION_ACCEPT:
        return False
    if action == ACTION_COMMIT:
        editor = value_editor_for(focus, QtWidgets)
        if editor is not None:
            commit_value_editor(editor)
            return True
    _clear_focus(focus)
    return True


def install_enter_commits_editor(dialog):
    """Filter Return/Enter so a focused child does not accept the dialog.

    QLineEdit ignores Return after commit, and QDialog then clicks the default
    button. ShortcutOverride is delivered to the focused widget (not the
    dialog), so this filter is installed on the application as well.
    """
    if dialog is None:
        return None
    existing = getattr(dialog, "_pmv_enter_filter", None)
    if existing is not None:
        return existing
    from ..pick import qt_modules

    QtCore, _, QtWidgets = qt_modules()
    if QtCore is None or QtWidgets is None:
        return None

    def _is_enter(event) -> bool:
        try:
            return event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter)
        except Exception:
            return False

    def _event_on_dialog(obj) -> bool:
        if obj is dialog:
            return True
        return _is_ancestor(dialog, obj)

    class _EnterCommitsFilter(QtCore.QObject):
        def eventFilter(inner, obj, event):
            if event is None:
                return False
            etype = event.type()
            if etype not in (QtCore.QEvent.KeyPress, QtCore.QEvent.ShortcutOverride):
                return False
            if not _is_enter(event):
                return False
            focus = obj
            app = QtWidgets.QApplication.instance()
            if app is not None:
                focused = app.focusWidget()
                if focused is not None:
                    focus = focused
            if not _event_on_dialog(focus) and not _event_on_dialog(obj):
                return False
            if etype == QtCore.QEvent.ShortcutOverride:
                role = focus_role_for(dialog, focus, QtWidgets)
                if dialog_enter_action(role) == ACTION_ACCEPT:
                    return False
                try:
                    event.accept()
                except Exception:
                    pass
                return True
            if not handle_dialog_enter(dialog, focus, QtWidgets):
                return False
            try:
                event.accept()
            except Exception:
                pass
            return True

    filt = _EnterCommitsFilter(dialog)
    dialog.installEventFilter(filt)
    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.installEventFilter(filt)

        def _remove(*_):
            try:
                if app is not None:
                    app.removeEventFilter(filt)
            except Exception:
                pass

        destroyed = getattr(dialog, "destroyed", None)
        if destroyed is not None:
            destroyed.connect(_remove)
    dialog._pmv_enter_filter = filt
    return filt


def _widget_parent(widget):
    getter = getattr(widget, "parentWidget", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            return None
    parent = getattr(widget, "parent", None)
    if callable(parent):
        try:
            return parent()
        except Exception:
            return None
    return parent


def _is_ancestor(ancestor, widget) -> bool:
    current = widget
    for _ in range(64):
        if current is None:
            return False
        if current is ancestor:
            return True
        current = _widget_parent(current)
    return False


def _clear_focus(widget) -> None:
    clear = getattr(widget, "clearFocus", None)
    if callable(clear):
        try:
            clear()
        except Exception:
            pass
