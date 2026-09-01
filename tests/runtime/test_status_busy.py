"""Ray-tracing progress bar reuse during blocking CGO reloads."""

from __future__ import annotations

from pymolviz.runtime import status_busy
from pymolviz.util.view import translation_ttt

from tests.runtime.test_follow import _armed, _pump


class _FakeProgress:
    def __init__(self, hidden=True):
        self.range = (0, 100)
        self.shown = False
        self.hidden = hidden
        self._format = "%p%"
        self.value = 0

    def minimum(self):
        return self.range[0]

    def maximum(self):
        return self.range[1]

    def setRange(self, lo, hi):
        self.range = (lo, hi)

    def setFormat(self, text):
        self._format = str(text)

    def format(self):
        return self._format

    def setValue(self, value):
        self.value = value

    def setTextVisible(self, _visible):
        pass

    def setMinimumHeight(self, _height):
        pass

    def setMinimumWidth(self, _width):
        pass

    def raise_(self):
        pass

    def parentWidget(self):
        return None

    def show(self):
        self.shown = True
        self.hidden = False

    def hide(self):
        self.hidden = True
        self.shown = False

    def isHidden(self):
        return self.hidden

    def repaint(self):
        pass


class _FakeWindow:
    def __init__(self):
        self.progressbar = _FakeProgress()
        self.abortbutton = _FakeProgress(hidden=True)
        self.cmd = type("cmd", (), {"get_progress": staticmethod(lambda: -1.0)})()
        self.update_calls = 0

    def update_progress(self):
        self.update_calls += 1
        if self.cmd.get_progress() < 0:
            self.progressbar.hide()
            self.abortbutton.hide()

    def repaint(self):
        pass


def test_status_busy_is_noop_without_window(monkeypatch):
    status_busy.reset_status_busy()
    monkeypatch.setattr(status_busy, "_pymol_qt_window", lambda: None)
    monkeypatch.setattr(status_busy, "_overlay_on_viewer", lambda _msg: None)
    with status_busy.cgo_update_status():
        pass
    assert status_busy._depth == 0
    assert status_busy._active is None


def test_status_busy_uses_pymol_ray_progressbar(monkeypatch):
    status_busy.reset_status_busy()
    window = _FakeWindow()
    monkeypatch.setattr(status_busy, "_pymol_qt_window", lambda: window)
    monkeypatch.setattr(status_busy, "_overlay_on_viewer", lambda _msg: None)
    with status_busy.cgo_update_status():
        assert window.progressbar.range == (0, 100)
        assert window.progressbar.value == 100
        assert window.progressbar.shown
        assert not window.progressbar.hidden
        assert window.progressbar.format() == "PyMOLViz: Updating anchored CGOs"
        assert window.abortbutton.hidden
        window.update_progress()
        assert window.progressbar.shown
    assert window.progressbar.hidden
    assert window.progressbar.range == (0, 100)
    assert window.progressbar.format() == "%p%"
    assert status_busy._depth == 0


def test_status_busy_nests_as_one_bar(monkeypatch):
    status_busy.reset_status_busy()
    window = _FakeWindow()
    monkeypatch.setattr(status_busy, "_pymol_qt_window", lambda: window)
    monkeypatch.setattr(status_busy, "_overlay_on_viewer", lambda _msg: None)
    with status_busy.cgo_update_status():
        with status_busy.cgo_update_status():
            assert window.progressbar.shown
        assert window.progressbar.shown
    assert window.progressbar.hidden


def test_follow_uses_status_bar_on_atom_move(monkeypatch, fake_cmd):
    hits = []
    monkeypatch.setattr(status_busy, "begin_cgo_update", lambda *_a, **_k: hits.append("enter"))
    monkeypatch.setattr(status_busy, "end_cgo_update", lambda *_a, **_k: hits.append("exit"))
    _armed(fake_cmd)
    fake_cmd.atoms[0].x = 5.0
    _pump(1)
    assert hits == ["enter", "exit"]


def test_follow_skips_status_bar_on_ttt_only(monkeypatch, fake_cmd):
    hits = []
    monkeypatch.setattr(status_busy, "begin_cgo_update", lambda *_a, **_k: hits.append("enter"))
    monkeypatch.setattr(status_busy, "end_cgo_update", lambda *_a, **_k: hits.append("exit"))
    runtime, _coll = _armed(fake_cmd)
    fake_cmd.set_object_ttt("prot", translation_ttt((4.0, 1.0, 0.0)))
    _pump(1)
    assert hits == []
    visual = runtime.bindings.get(_coll.id).pymol_name
    assert fake_cmd.settings[visual]["_ttt"] is not None
