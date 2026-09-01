"""Runtime tests for wizard exit / session reconcile."""

from __future__ import annotations

from pymolviz.wizard import PyMolVizWizard, exit_wizard, reconcile_wizard_after_session_load
from tests.fakes.cmd import FakeCmd


class _FakeWizard:
    def __init__(self):
        self.cleaned = False

    def cleanup(self):
        self.cleaned = True


def test_exit_wizard_cleans_pymolviz_wizard():
    cmd = FakeCmd()
    wizard = PyMolVizWizard()
    wizard.cmd = cmd
    cmd._stack = [wizard]

    def get_wizard():
        return cmd._stack[-1] if cmd._stack else None

    def get_wizard_stack():
        return list(cmd._stack)

    def set_wizard_stack(stack):
        cmd._stack = list(stack)

    def set_wizard(value=None):
        if value is None:
            if cmd._stack:
                cmd._stack.pop()
            return
        cmd._stack.append(value)

    cmd.get_wizard = get_wizard
    cmd.get_wizard_stack = get_wizard_stack
    cmd.set_wizard_stack = set_wizard_stack
    cmd.set_wizard = set_wizard
    cmd.refresh_wizard = lambda: None

    exit_wizard(cmd)
    assert wizard._closed is True
    assert cmd._stack == []


def test_wizard_event_mask_is_silent():
    from pymol.wizard import Wizard

    wizard = PyMolVizWizard.__new__(PyMolVizWizard)
    mask = PyMolVizWizard.get_event_mask(wizard)
    dirty = getattr(Wizard, "event_mask_dirty", 128)
    view = getattr(Wizard, "event_mask_view", 256)
    position = getattr(Wizard, "event_mask_position", 512)
    scene = getattr(Wizard, "event_mask_scene", 16)
    assert mask & dirty == 0
    assert mask & view == 0
    assert mask & position == 0
    assert mask & scene == 0
    assert mask == Wizard.event_mask_pick + Wizard.event_mask_select


def test_done_button_calls_wizard_do_done():
    wizard = PyMolVizWizard.__new__(PyMolVizWizard)
    wizard.menu_items = [("Add Visual", None)]
    panel = PyMolVizWizard.get_panel(wizard)
    done = panel[-1]
    assert done[1] == "Done"
    assert done[2] == "cmd.get_wizard().do_done()"


def test_extend_cmd_sets_module_attribute():
    from pymolviz.util.pymol_helpers import extend_cmd

    class _Cmd:
        def extend(self, name, func):
            self.extended = (name, func)

    cmd = _Cmd()
    fn = lambda: None
    extend_cmd(cmd, "pymolviz_exit_wizard", fn)
    assert cmd.extended[0] == "pymolviz_exit_wizard"
    assert cmd.pymolviz_exit_wizard is fn


def test_exit_wizard_clears_non_pymolviz_wizards_too():
    cmd = FakeCmd()
    leftover = _FakeWizard()
    wizard = PyMolVizWizard()
    wizard.cmd = cmd
    cmd._stack = [leftover, wizard]

    def get_wizard():
        return cmd._stack[-1] if cmd._stack else None

    cmd.get_wizard = get_wizard
    cmd.get_wizard_stack = lambda: list(cmd._stack)
    cmd.set_wizard_stack = lambda stack: setattr(cmd, "_stack", list(stack))
    cmd.set_wizard = lambda value=None: cmd._stack.pop() if value is None and cmd._stack else None
    cmd.refresh_wizard = lambda: None

    wizard.do_done()
    assert wizard._closed is True
    assert cmd._stack == []


def test_camera_place_skips_identical_position():
    from pymolviz.wizards.camera_center import CameraCenterSphere

    class _Cmd:
        def __init__(self):
            self.ttt = []

        def set_object_ttt(self, name, matrix):
            self.ttt.append((name, list(matrix)))

    sphere = CameraCenterSphere.__new__(CameraCenterSphere)
    sphere.cmd = _Cmd()
    sphere.name = "pmv_camera_center"
    sphere._current_pos = None
    sphere._place((1.0, 2.0, 3.0))
    sphere._place((1.0, 2.0, 3.0))
    assert len(sphere.cmd.ttt) == 1
    sphere._place((1.0, 2.0, 4.0))
    assert len(sphere.cmd.ttt) == 2


def test_exit_wizard_pops_stacked_wizards():
    cmd = FakeCmd()
    outer = PyMolVizWizard()
    inner = PyMolVizWizard()
    outer.cmd = cmd
    inner.cmd = cmd
    cmd._stack = [outer, inner]

    def get_wizard():
        return cmd._stack[-1] if cmd._stack else None

    def get_wizard_stack():
        return list(cmd._stack)

    def set_wizard_stack(stack):
        cmd._stack = list(stack)

    def set_wizard(value=None):
        if value is None:
            if cmd._stack:
                cmd._stack.pop()
            return
        cmd._stack.append(value)

    cmd.get_wizard = get_wizard
    cmd.get_wizard_stack = get_wizard_stack
    cmd.set_wizard_stack = set_wizard_stack
    cmd.set_wizard = set_wizard
    cmd.refresh_wizard = lambda: None

    exit_wizard(cmd)
    assert outer._closed is True
    assert inner._closed is True
    assert cmd._stack == []


def test_reconcile_after_session_load_exits_wizard():
    cmd = FakeCmd()
    wizard = PyMolVizWizard()
    wizard.cmd = cmd
    cmd._stack = [wizard]
    cmd.get_wizard = lambda: cmd._stack[-1] if cmd._stack else None
    cmd.get_wizard_stack = lambda: list(cmd._stack)
    cmd.set_wizard_stack = lambda stack: setattr(cmd, "_stack", list(stack))
    cmd.set_wizard = lambda value=None: cmd._stack.pop() if value is None and cmd._stack else None
    cmd.refresh_wizard = lambda: None

    reconcile_wizard_after_session_load(cmd)
    assert wizard._closed is True
    assert cmd._stack == []
