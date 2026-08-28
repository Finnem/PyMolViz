"""Interactive PyMOL wizard with a side-panel menu."""

from pymol import cmd
from pymol.wizard import Wizard

from .util.pymol_helpers import center_on, center_on_point, restore_view
from .wizards.add_visual import AddVisualWindow
from .wizards.camera_center import CameraCenterSphere
from .wizards.middle_click import (
    install_middle_click_filter,
    restore_viewing_mouse,
    take_over_center_click,
)
from .wizards.pick import qt_modules


class PyMolVizWizard(Wizard):
    """Interactive PyMOL wizard with a side-panel menu."""

    def __init__(self):
        Wizard.__init__(self)
        self.prompt = ["PyMOLViz"]
        self.menu_items = [
            ("Add Visual", self.on_add_visual),
            ("Item B", self.on_item_b),
            ("Item C", self.on_item_c),
        ]
        self.add_visual_window = AddVisualWindow(self)
        self.camera_sphere = CameraCenterSphere(self.cmd)
        self._last_center_display = None
        self._click_filter, self._click_widget = install_middle_click_filter(self)
        take_over_center_click(self.cmd)

    def get_event_mask(self):
        mask = Wizard.event_mask_pick + Wizard.event_mask_select
        mask += getattr(Wizard, "event_mask_dirty", 32)
        mask += getattr(Wizard, "event_mask_scene", 16)
        return mask

    def do_pick(self, *args, **kwargs):
        self._sync_sphere()

    def do_dirty(self):
        self._sync_sphere()

    def do_scene(self):
        self._sync_sphere()

    def _sync_sphere(self):
        sphere = getattr(self, "camera_sphere", None)
        if sphere is None:
            return
        sphere.sync()
        pos = sphere.current_position()
        display = None
        if pos is not None:
            display = (round(pos[0], 3), round(pos[1], 3), round(pos[2], 3))
        if display != self._last_center_display:
            self._last_center_display = display
            try:
                self.cmd.refresh_wizard()
            except Exception:
                pass

    def get_prompt(self):
        lines = list(self.prompt)
        sphere = getattr(self, "camera_sphere", None)
        if sphere is not None:
            pos = sphere.current_position()
            if pos is not None:
                lines.append(
                    "Center: %.3f, %.3f, %.3f" % (pos[0], pos[1], pos[2])
                )
        return lines

    def get_panel(self):
        panel = [[1, "PyMOLViz", ""]]
        for index, (label, _) in enumerate(self.menu_items):
            panel.append([2, label, f"cmd.get_wizard().select_item({index})"])
        panel.append([2, "Done", "cmd.set_wizard()"])
        return panel

    def select_item(self, index):
        _, callback = self.menu_items[index]
        callback()
        self.cmd.refresh_wizard()

    def on_add_visual(self):
        self.prompt = ["Add Visual"]
        self.add_visual_window.show()

    def on_item_b(self):
        self.prompt = ["Selected Item B"]

    def on_item_c(self):
        self.prompt = ["Selected Item C"]

    def _on_middle_click(self, x, y):
        """Snap the cage, then issue cmd.center ourselves after the mouse event."""
        sphere = getattr(self, "camera_sphere", None)
        widget = getattr(self, "_click_widget", None)
        if sphere is None or widget is None:
            return
        sphere.request_snap(widget, x, y)
        QtCore, _, _ = qt_modules()
        if QtCore is None:
            self._resubmit_center()
            return
        QtCore.QTimer.singleShot(0, self._resubmit_center)

    def _resubmit_center(self):
        sphere = getattr(self, "camera_sphere", None)
        if sphere is None:
            return
        sphere._apply_pending_snap()
        sele, pos = sphere.take_center_target()
        if sele and center_on(self.cmd, sele, animate=-1):
            return
        if pos is None:
            return
        center_on_point(self.cmd, pos, animate=-1)

    def cleanup(self):
        restore_viewing_mouse(self.cmd)
        if getattr(self, "add_visual_window", None) is not None:
            self.add_visual_window.close()
            self.add_visual_window = None
        widget = getattr(self, "_click_widget", None)
        filt = getattr(self, "_click_filter", None)
        if widget is not None and filt is not None:
            widget.removeEventFilter(filt)
        self._click_widget = None
        self._click_filter = None
        if getattr(self, "camera_sphere", None) is not None:
            self.camera_sphere.close()
            self.camera_sphere = None


def start_wizard():
    """Open the PyMOLViz wizard panel."""
    view = cmd.get_view()
    cmd.set_wizard(PyMolVizWizard())
    restore_view(cmd, view)


def reload_wizard():
    """Drop cached pymolviz modules, reimport, and start the wizard."""
    import importlib
    import sys

    cmd.set_wizard()
    for name in list(sys.modules):
        if name == "pymolviz" or name.startswith("pymolviz."):
            del sys.modules[name]
    module = importlib.import_module("pymolviz.wizard")
    module.start_wizard()


cmd.extend("pymolviz_wizard", start_wizard)
cmd.extend("pymolviz_reload_wizard", reload_wizard)
