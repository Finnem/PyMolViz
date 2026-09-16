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


def test_wizard_records_you_clicked_on_pick_and_select():
    import inspect

    from pymolviz.wizard import PyMolVizWizard

    pick = inspect.getsource(PyMolVizWizard.do_pick)
    select = inspect.getsource(PyMolVizWizard.do_select)
    init = inspect.getsource(PyMolVizWizard._init_runtime)
    cleanup = inspect.getsource(PyMolVizWizard.cleanup)
    assert "record_pymol_click" in pick
    assert "record_pymol_click" in select
    assert "install_click_feedback_hook" in init
    assert "uninstall_click_feedback_hook" in cleanup
    assert "restore_atom_selection_mode" in cleanup


def test_last_click_does_not_scrape_qt_widgets():
    import inspect

    from pymolviz.wizards import last_click

    src = inspect.getsource(last_click)
    assert "allWidgets" not in src
    assert "toPlainText" not in src
    assert "cmd._get_feedback" in src


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
    wizard.menu_items = [("Open 3D Objects Menu", None), ("Open Field Visuals Menu", None)]
    panel = PyMolVizWizard.get_panel(wizard)
    labels = [row[1] for row in panel]
    assert "Open 3D Objects Menu" in labels
    assert "Open Field Visuals Menu" in labels
    assert "Item B" not in labels
    done = panel[-1]
    assert done[1] == "Done"
    assert done[2] == "cmd.get_wizard().do_done()"
    assert not any(str(label).startswith("Cam center:") for label in labels)
    wizard.prompt = ["PyMOLViz"]
    prompt = PyMolVizWizard.get_prompt(wizard)
    assert any(str(line).startswith("Cam center:") for line in prompt)


def test_wizard_menu_includes_field_visuals():
    wizard = PyMolVizWizard()
    labels = [label for label, _ in wizard.menu_items]
    assert labels[0] == "Open 3D Objects Menu"
    assert labels[1] == "Open Field Visuals Menu"
    assert labels[2] == "Open Colormap Menu"
    assert labels[3] == "Import PyMolViz File"
    assert labels[4] == "Export PyMolViz Session"
    assert labels[5] == "Create Pseudoatom at Cam Center"
    assert "Item B" not in labels
    assert "Item C" not in labels
    assert wizard.field_visuals_window is not None
    assert wizard.colormap_window is not None
    assert hasattr(wizard, "on_field_visuals")
    assert hasattr(wizard, "on_colormaps")
    assert hasattr(wizard, "on_import")
    assert hasattr(wizard, "on_export")
    panel_labels = [row[1] for row in wizard.get_panel()]
    assert "Open Colormap Menu" in panel_labels
    assert panel_labels.index("Open Colormap Menu") < panel_labels.index(
        "Import PyMolViz File"
    )
    assert panel_labels.index("Import PyMolViz File") < panel_labels.index(
        "Export PyMolViz Session"
    )
    assert panel_labels.index("Export PyMolViz Session") < panel_labels.index(
        "Create Pseudoatom at Cam Center"
    )


def test_create_pseudoatom_at_cam_center():
    cmd = FakeCmd()
    wizard = PyMolVizWizard.__new__(PyMolVizWizard)
    wizard.cmd = cmd
    wizard.prompt = ["PyMOLViz"]

    class _Sphere:
        def current_position(self):
            return (1.25, -4.5, 8.0)

    wizard.camera_sphere = _Sphere()
    PyMolVizWizard.on_create_cam_pseudoatom(wizard)
    assert "cam_center" in cmd.objects
    assert cmd.objects["cam_center"] == [1.25, -4.5, 8.0]
    assert wizard.prompt == ["Created cam_center"]
    PyMolVizWizard.on_create_cam_pseudoatom(wizard)
    assert "cam_center_1" in cmd.objects or "cam_center_2" in cmd.objects


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
    from pymolviz.util.pymol_helpers import CAMERA_CENTER_NAME
    from pymolviz.wizards.camera_center import CameraCenterSphere

    class _Cmd:
        def __init__(self):
            self.ttt = []

        def set_object_ttt(self, name, matrix):
            self.ttt.append((name, list(matrix)))

    sphere = CameraCenterSphere.__new__(CameraCenterSphere)
    sphere.cmd = _Cmd()
    sphere.name = CAMERA_CENTER_NAME
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


def test_on_import_interns_pack(tmp_path, monkeypatch):
    from pymolviz.io import save
    from pymolviz.meshes.Sphere import Sphere
    from pymolviz.runtime.session import get
    from pymolviz.wizards.session_io import import_session_path

    sphere = Sphere(
        (0.0, 0.0, 0.0), 1.0, bypass_colormap=True, obj_id="wiz1", name="wiz1",
    )
    path = tmp_path / "wiz.pmv"
    save(sphere, path)

    wizard = PyMolVizWizard.__new__(PyMolVizWizard)
    wizard.cmd = FakeCmd()
    wizard.prompt = ["PyMOLViz"]
    wizard.add_visual_window = None
    wizard.field_visuals_window = None
    wizard.colormap_window = None
    refreshed = []
    wizard._refresh_open_windows = lambda: refreshed.append(True)

    def _prompt(parent, cmd):
        return import_session_path(cmd, path)

    monkeypatch.setattr("pymolviz.wizards.session_io.prompt_import_session", _prompt)
    PyMolVizWizard.on_import(wizard)
    assert get("wiz1") is not None
    assert wizard.prompt == ["Imported 1 object"]
    assert refreshed == [True]


def test_on_export_empty_session(monkeypatch):
    wizard = PyMolVizWizard.__new__(PyMolVizWizard)
    wizard.prompt = ["PyMOLViz"]
    monkeypatch.setattr(
        "pymolviz.wizards.session_io.prompt_export_session",
        lambda parent: False,
    )
    PyMolVizWizard.on_export(wizard)
    assert wizard.prompt == ["Nothing to export"]


def test_on_export_writes_session(tmp_path, monkeypatch):
    from pymolviz.io import load, save
    from pymolviz.meshes.Sphere import Sphere
    from pymolviz.runtime.session import add
    from pymolviz.wizards.session_io import session_export_items

    sphere = Sphere(
        (4.0, 0.0, 0.0), 0.2, bypass_colormap=True, obj_id="wiz2", name="wiz2",
    )
    add(sphere)
    dest = tmp_path / "out.pmv"

    def _prompt(parent):
        save(session_export_items(), dest)
        return str(dest)

    wizard = PyMolVizWizard.__new__(PyMolVizWizard)
    wizard.prompt = ["PyMOLViz"]
    monkeypatch.setattr("pymolviz.wizards.session_io.prompt_export_session", _prompt)
    PyMolVizWizard.on_export(wizard)
    assert wizard.prompt == ["Exported %s" % dest]
    assert load(dest)[0].id == "wiz2"
