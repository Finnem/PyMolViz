"""Wizard startup registers runtime integration."""

from __future__ import annotations

from pymolviz.wizard import start_wizard
from tests.fakes.cmd import FakeCmd


def test_start_wizard_calls_install(monkeypatch):
    calls = []

    def _install(cmd=None):
        calls.append(cmd)

    monkeypatch.setattr("pymolviz.runtime.integration.install", _install)
    class _StubWizard:
        def _sync_sphere(self):
            pass

    monkeypatch.setattr("pymolviz.wizard.PyMolVizWizard", _StubWizard)
    monkeypatch.setattr("pymolviz.wizard.exit_wizard", lambda *_a, **_k: None)
    monkeypatch.setattr("pymolviz.wizard.restore_view", lambda *_a, **_k: None)
    monkeypatch.setattr("pymolviz.wizard.register_wizard_commands", lambda: None)

    cmd = FakeCmd()
    cmd._wizard = None

    def set_wizard(wizard=None, replace=0):
        cmd._wizard = wizard

    cmd.set_wizard = set_wizard
    cmd.get_view = lambda: ()
    cmd.refresh_wizard = lambda: None
    monkeypatch.setattr("pymolviz.wizard.cmd", cmd)
    start_wizard()
    assert calls == [None]
