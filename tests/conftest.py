"""Pytest configuration: pymol stubs, path setup, shared fixtures.

Agent guidance: see ``.cursor/rules/pymolviz-testing.mdc`` and scoped
``testing-*.mdc`` rules. Default run: ``pytest tests/unit tests/runtime``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = ROOT / "module"
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

_CGO_TOKEN_NAMES = (
    "POINTS", "SPHERE", "COLOR", "VERTEX", "NORMAL", "CYLINDER", "CONE",
    "BEGIN", "END", "LINEWIDTH", "LINES", "TRIANGLES", "ALPHA",
)


def _install_pymol_stubs() -> None:
    if "pymol.cgo" in sys.modules:
        return

    cgo = types.ModuleType("pymol.cgo")
    for index, name in enumerate(_CGO_TOKEN_NAMES):
        setattr(cgo, name, 1000 + index)

    cmd_mod = types.ModuleType("pymol.cmd")

    pymol = types.ModuleType("pymol")
    pymol.cgo = cgo
    pymol.cmd = cmd_mod
    pymol.session = types.SimpleNamespace()

    wizard_mod = types.ModuleType("pymol.wizard")

    class Wizard:
        event_mask_pick = 1
        event_mask_select = 2

    wizard_mod.Wizard = Wizard

    qt_core = types.ModuleType("pymol.Qt.QtCore")
    qt_gui = types.ModuleType("pymol.Qt.QtGui")
    qt_widgets = types.ModuleType("pymol.Qt.QtWidgets")
    qt_pkg = types.ModuleType("pymol.Qt")
    qt_pkg.QtCore = qt_core
    qt_pkg.QtGui = qt_gui
    qt_pkg.QtWidgets = qt_widgets

    sys.modules["pymol"] = pymol
    sys.modules["pymol.cgo"] = cgo
    sys.modules["pymol.cmd"] = cmd_mod
    sys.modules["pymol.wizard"] = wizard_mod
    sys.modules["pymol.Qt"] = qt_pkg
    sys.modules["pymol.Qt.QtCore"] = qt_core
    sys.modules["pymol.Qt.QtGui"] = qt_gui
    sys.modules["pymol.Qt.QtWidgets"] = qt_widgets


def _real_pymol_loaded() -> bool:
    cmd_mod = sys.modules.get("pymol.cmd")
    return cmd_mod is not None and hasattr(cmd_mod, "load_cgo")


def _install_pymol_stubs_if_needed() -> None:
    if _real_pymol_loaded():
        return
    if "pymol.cgo" in sys.modules:
        return

    try:
        import importlib

        importlib.import_module("pymol.cgo")
        cmd_mod = importlib.import_module("pymol.cmd")
        if hasattr(cmd_mod, "load_cgo"):
            return
    except ImportError:
        pass

    _install_pymol_stubs()


_install_pymol_stubs_if_needed()


def pytest_configure(config):
    config.addinivalue_line("markers", "pymol: requires real PyMOL (opt-in)")
    config.addinivalue_line("markers", "qt: requires PyMOL Qt (opt-in)")


@pytest.fixture
def fake_cmd():
    from tests.fakes.cmd import FakeCmd

    return FakeCmd()


@pytest.fixture
def resolve_context(fake_cmd):
    from pymolviz.runtime.context import ResolveContext

    return ResolveContext(fake_cmd, state=1)


@pytest.fixture
def runtime(fake_cmd):
    from pymolviz.runtime.runtime import PyMOLRuntime, reset_runtime

    reset_runtime()
    rt = PyMOLRuntime(fake_cmd)
    yield rt
    reset_runtime()


@pytest.fixture(autouse=True)
def reset_session_store():
    from pymolviz.runtime import session as pmv_session

    pmv_session.clear()
    yield
    pmv_session.clear()
