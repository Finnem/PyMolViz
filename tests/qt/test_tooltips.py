"""Optional Qt wizard tests (run with ``pytest -m qt``)."""

from __future__ import annotations

import logging
from unittest import mock

import pytest

from pymolviz.wizards.pick import DeferredCallback, qt_widget_alive
from pymolviz.wizards.tooltips import (
    apply_required_tooltips,
    require_tooltips,
)


class _FakeWidget:
    def __init__(self, class_name="QPushButton", tooltip="", *, alive=True):
        self.__class__.__name__ = class_name
        self._tooltip = tooltip
        self._alive = alive

    def setToolTip(self, text):
        self._tooltip = str(text)

    def toolTip(self):
        return self._tooltip

    def objectName(self):
        if not self._alive:
            raise RuntimeError("wrapped C/C++ object has been deleted")
        return self.__class__.__name__


def _qt_alive_via_object_name(widget) -> bool:
    """Exercise the objectName() fallback path used when shiboken/sip are absent."""
    if widget is None:
        return False
    try:
        widget.objectName()
        return True
    except RuntimeError:
        return False


@pytest.mark.qt
def test_apply_required_tooltips_leaves_no_gaps():
    widgets = [
        _FakeWidget(class_name="QPushButton", tooltip=""),
        _FakeWidget(class_name="QSpinBox", tooltip=""),
    ]
    missing = apply_required_tooltips(
        [(widgets[0], "Create CGO"), (widgets[1], "Quality level")],
        context="TestPage",
    )
    assert missing == []


@pytest.mark.qt
def test_require_tooltips_reports_gap(caplog):
    widget = _FakeWidget(class_name="QPushButton", tooltip="")
    caplog.set_level(logging.DEBUG, logger="pymolviz.wizards")
    missing = require_tooltips([widget], context="TestPage")
    assert len(missing) == 1
    assert any("missing tooltip" in rec.message.lower() for rec in caplog.records)


@pytest.mark.qt
def test_qt_widget_alive_detects_deleted_widget():
    alive = _FakeWidget()
    dead = _FakeWidget(alive=False)
    with mock.patch.dict("sys.modules", {"shiboken6": None, "shiboken2": None, "sip": None}):
        assert _qt_alive_via_object_name(alive) is True
        assert _qt_alive_via_object_name(dead) is False
        assert qt_widget_alive(None) is False


@pytest.mark.qt
def test_deferred_callback_skips_after_page_destroyed():
    calls = []
    pending = []

    class _FakeTimer:
        @staticmethod
        def singleShot(_delay, callback):
            pending.append(callback)

    class _FakeQtCore:
        QTimer = _FakeTimer

    page = _FakeWidget(alive=True)

    import pymolviz.wizards.pick as pick_mod

    original = pick_mod.qt_modules
    pick_mod.qt_modules = lambda: (_FakeQtCore, None, None)
    try:
        deferred = DeferredCallback()
        with mock.patch.object(pick_mod, "qt_widget_alive", side_effect=_qt_alive_via_object_name):
            deferred.schedule(lambda: calls.append("run"), page=page)
            pending.pop()()
            assert calls == ["run"]

            page._alive = False
            calls.clear()
            deferred.schedule(lambda: calls.append("late"), page=page)
            assert pending == []
            assert calls == []
    finally:
        pick_mod.qt_modules = original


@pytest.mark.qt
def test_deferred_callback_cancel_prevents_stale_run():
    calls = []
    pending = []

    class _FakeTimer:
        @staticmethod
        def singleShot(_delay, callback):
            pending.append(callback)

    class _FakeQtCore:
        QTimer = _FakeTimer

    import pymolviz.wizards.pick as pick_mod

    original = pick_mod.qt_modules
    pick_mod.qt_modules = lambda: (_FakeQtCore, None, None)
    try:
        deferred = DeferredCallback()
        deferred.schedule(lambda: calls.append("stale"))
        deferred.cancel()
        deferred.schedule(lambda: calls.append("fresh"))
        for callback in pending:
            callback()
        assert calls == ["fresh"]
    finally:
        pick_mod.qt_modules = original
