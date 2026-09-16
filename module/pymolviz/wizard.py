"""Interactive PyMOL wizard with a side-panel menu."""

import time

from pymol import cmd
from pymol.wizard import Wizard

from .util.pymol_helpers import center_on, center_on_point, extend_cmd, restore_view
from .wizards.add_visual import AddVisualWindow
from .wizards.field_visuals import FieldVisualsWindow
from .wizards.colormaps import ColormapMenuWindow
from .wizards.camera_center import (
    CameraCenterSphere,
    create_cam_center_pseudoatom,
    format_center_xyz,
)
from .wizards.middle_click import (
    install_middle_click_filter,
    restore_viewing_mouse,
    take_over_center_click,
    _teardown_global_click_filter,
)
from .wizards.pick import pointer_over_viewer, qt_modules
from .wizards.session_io import EXPORT_LABEL, IMPORT_LABEL


def is_pymolviz_wizard(wizard) -> bool:
    if wizard is None:
        return False
    if type(wizard).__name__ == "PyMolVizWizard":
        return True
    return hasattr(wizard, "add_visual_window") and hasattr(wizard, "camera_sphere")


class PyMolVizWizard(Wizard):
    """Interactive PyMOL wizard with a side-panel menu."""

    def __init__(self):
        Wizard.__init__(self)
        self.prompt = ["PyMOLViz"]
        self._closed = False
        self._syncing = False
        self._sphere_sync_posted = False
        self._last_center_display = None
        self._last_center_refresh = 0.0
        self._init_runtime()

    def _teardown_click_filter(self):
        _teardown_global_click_filter()
        self._click_widget = None
        self._click_filter = None

    def _init_runtime(self):
        """Attach menu callbacks, Qt hooks, and the camera cage."""
        if getattr(self, "_closed", False):
            return
        self._teardown_click_filter()
        if getattr(self, "camera_sphere", None) is not None:
            try:
                self.camera_sphere.close()
            except Exception:
                pass
            self.camera_sphere = None
        self.menu_items = [
            ("Open 3D Objects Menu", self.on_add_visual),
            ("Open Field Visuals Menu", self.on_field_visuals),
            ("Open Colormap Menu", self.on_colormaps),
            (IMPORT_LABEL, self.on_import),
            (EXPORT_LABEL, self.on_export),
            ("Create Pseudoatom at Cam Center", self.on_create_cam_pseudoatom),
        ]
        self.add_visual_window = AddVisualWindow(self)
        self.field_visuals_window = FieldVisualsWindow(self)
        self.colormap_window = ColormapMenuWindow(self)
        self.camera_sphere = CameraCenterSphere(self.cmd)
        self._click_filter, self._click_widget = install_middle_click_filter(self)
        take_over_center_click(self.cmd)
        try:
            from .wizards.last_click import install_click_feedback_hook

            install_click_feedback_hook()
        except Exception:
            pass
        try:
            from .runtime.follow import ensure_follow_input_hook
            ensure_follow_input_hook()
        except Exception:
            pass

    def _ensure_runtime(self):
        """Rebuild wizard hooks if session save dropped the camera cage."""
        if getattr(self, "_closed", False):
            return
        sphere = getattr(self, "camera_sphere", None)
        if sphere is None:
            self._init_runtime()
            return
        try:
            sphere.ensure_object()
        except Exception:
            self._init_runtime()
            return
        if getattr(self, "_click_filter", None) is None:
            if getattr(self, "_closed", False):
                return
            self._click_filter, self._click_widget = install_middle_click_filter(self)
            take_over_center_click(self.cmd)

    def __getstate__(self):
        """PyMOL session save pickles the wizard stack — Qt objects cannot go in."""
        self._suspend_for_session()
        return {"prompt": list(self.prompt)}

    def __setstate__(self, state):
        Wizard.__init__(self)
        if getattr(self, "cmd", None) is None:
            self.cmd = cmd
        self.prompt = list(state.get("prompt") or ["PyMOLViz"])
        self._closed = False
        self._syncing = False
        self._sphere_sync_posted = False
        self._last_center_display = None
        self._last_center_refresh = 0.0
        self._init_runtime()
        try:
            self.cmd.refresh_wizard()
        except Exception:
            pass

    def _suspend_for_session(self):
        """Drop transient Qt state before session pickling."""
        if getattr(self, "add_visual_window", None) is not None:
            try:
                self.add_visual_window.close()
            except Exception:
                pass
            self.add_visual_window = None
        if getattr(self, "field_visuals_window", None) is not None:
            try:
                self.field_visuals_window.close()
            except Exception:
                pass
            self.field_visuals_window = None
        if getattr(self, "colormap_window", None) is not None:
            try:
                self.colormap_window.close()
            except Exception:
                pass
            self.colormap_window = None
        self._teardown_click_filter()
        if getattr(self, "camera_sphere", None) is not None:
            try:
                self.camera_sphere.close()
            except Exception:
                pass
            self.camera_sphere = None
        restore_viewing_mouse(self.cmd)
        try:
            from .wizards.last_click import restore_atom_selection_mode

            restore_atom_selection_mode(self.cmd)
        except Exception:
            pass

    def get_event_mask(self):
        # Keep the default pick/select bits so PyMOL still shows the panel.
        # Do not subscribe to dirty/view/scene/position: those re-enter from
        # set_object_ttt and flood the command queue.
        return Wizard.event_mask_pick + Wizard.event_mask_select

    def do_pick(self, *_args, **_kwargs):
        from .wizards.last_click import record_pymol_click

        record_pymol_click(getattr(self, "cmd", None))

    def do_select(self, *_args, **_kwargs):
        from .wizards.last_click import record_pymol_click

        record_pymol_click(getattr(self, "cmd", None))

    def _request_sphere_sync(self, delay_ms=0):
        """Run follow_view after PyMOL applies the current mouse/wheel event."""
        if getattr(self, "_closed", False) or getattr(self, "_sphere_sync_posted", False):
            return
        QtCore, _, _ = qt_modules()
        if QtCore is None or not hasattr(QtCore, "QTimer"):
            self._sync_sphere()
            return
        self._sphere_sync_posted = True
        try:
            QtCore.QTimer.singleShot(max(0, int(delay_ms)), self._flush_sphere_sync)
        except Exception:
            self._sphere_sync_posted = False
            self._sync_sphere()

    def _flush_sphere_sync(self):
        self._sphere_sync_posted = False
        self._sync_sphere()

    def _sync_sphere(self):
        if getattr(self, "_closed", False) or getattr(self, "_syncing", False):
            return
        self._syncing = True
        try:
            sphere = getattr(self, "camera_sphere", None)
            if sphere is None:
                return
            if not getattr(sphere, "_hold", False) and not pointer_over_viewer():
                return
            try:
                view = tuple(float(v) for v in self.cmd.get_view())
            except Exception:
                return
            if view == getattr(self, "_last_sync_view", None):
                if getattr(sphere, "_hold", False):
                    self._request_sphere_sync(50)
                return
            self._last_sync_view = view
            sphere.follow_view(view)
            self._maybe_refresh_center_display()
            if getattr(sphere, "_hold", False):
                self._request_sphere_sync(50)
        except Exception:
            pass
        finally:
            self._syncing = False

    def get_prompt(self):
        lines = [str(line) for line in (self.prompt or ["PyMOLViz"])]
        lines.append(self._center_display_text())
        return lines

    def _center_display_text(self):
        sphere = getattr(self, "camera_sphere", None)
        pos = None
        if sphere is not None:
            try:
                pos = sphere.current_position()
            except Exception:
                pos = None
        return format_center_xyz(pos)

    def _maybe_refresh_center_display(self):
        text = self._center_display_text()
        if text == getattr(self, "_last_center_display", None):
            return
        now = time.monotonic()
        last = float(getattr(self, "_last_center_refresh", 0.0) or 0.0)
        if last and (now - last) < 0.2:
            return
        self._last_center_display = text
        self._last_center_refresh = now
        try:
            self.cmd.refresh_wizard()
        except Exception:
            pass

    def get_panel(self):
        panel = [
            [1, "PyMOLViz", ""],
            [1, self._center_display_text(), ""],
        ]
        for index, (label, _) in enumerate(self.menu_items):
            panel.append([2, label, "cmd.get_wizard().select_item(%d)" % index])
        panel.append([2, "Done", "cmd.get_wizard().do_done()"])
        return panel

    def do_done(self):
        exit_wizard(self.cmd)

    def select_item(self, index):
        if index < 0 or index >= len(self.menu_items):
            return
        _, callback = self.menu_items[index]
        callback()
        try:
            self.cmd.refresh_wizard()
        except Exception:
            pass

    def on_add_visual(self):
        self.prompt = ["Open 3D Objects Menu"]
        try:
            self.add_visual_window.show()
        except Exception as exc:
            self.prompt = ["Open 3D Objects Menu failed: %s" % exc]

    def on_field_visuals(self):
        self.prompt = ["Open Field Visuals Menu"]
        try:
            self.field_visuals_window.show()
        except Exception as exc:
            self.prompt = ["Open Field Visuals Menu failed: %s" % exc]

    def on_colormaps(self):
        self.prompt = ["Open Colormap Menu"]
        try:
            self.colormap_window.show()
        except Exception as exc:
            self.prompt = ["Open Colormap Menu failed: %s" % exc]

    def _menu_parent(self):
        for attr in ("field_visuals_window", "add_visual_window", "colormap_window"):
            host = getattr(self, attr, None)
            if host is None:
                continue
            window = getattr(host, "_window", None)
            if window is not None:
                return window
        return None

    def _refresh_open_windows(self):
        pairs = (
            ("add_visual_window", "_refresh_objects_table"),
            ("field_visuals_window", "_refresh_fields_table"),
            ("colormap_window", "_refresh_table"),
        )
        for attr, method in pairs:
            host = getattr(self, attr, None)
            if host is None or getattr(host, "_window", None) is None:
                continue
            fn = getattr(host, method, None)
            if not callable(fn):
                continue
            try:
                fn()
            except Exception:
                pass

    def on_import(self):
        self.prompt = [IMPORT_LABEL]
        try:
            from .wizards.session_io import prompt_import_session

            interned = prompt_import_session(self._menu_parent(), self.cmd)
        except Exception as exc:
            self.prompt = ["Import failed: %s" % exc]
            return
        if interned is None:
            return
        self._refresh_open_windows()
        n = len(interned)
        self.prompt = ["Imported %d object%s" % (n, "" if n == 1 else "s")]

    def on_export(self):
        self.prompt = [EXPORT_LABEL]
        try:
            from .wizards.session_io import prompt_export_session

            path = prompt_export_session(self._menu_parent())
        except Exception as exc:
            self.prompt = ["Export failed: %s" % exc]
            return
        if path is False:
            self.prompt = ["Nothing to export"]
            return
        if not path:
            return
        self.prompt = ["Exported %s" % path]

    def on_create_cam_pseudoatom(self):
        sphere = getattr(self, "camera_sphere", None)
        pos = None
        if sphere is not None:
            try:
                pos = sphere.current_position()
            except Exception:
                pos = None
        if pos is None:
            self.prompt = ["Camera center is unavailable"]
            return
        try:
            name = create_cam_center_pseudoatom(self.cmd, pos)
            self.prompt = ["Created %s" % name]
        except Exception as exc:
            self.prompt = ["Create pseudoatom failed: %s" % exc]

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
            self._request_sphere_sync(50)
            return
        if pos is None:
            return
        center_on_point(self.cmd, pos, animate=-1)
        self._request_sphere_sync(50)

    def cleanup(self):
        self._closed = True
        self._syncing = False
        self._teardown_click_filter()
        if getattr(self, "add_visual_window", None) is not None:
            try:
                self.add_visual_window.close()
            except Exception:
                pass
            self.add_visual_window = None
        if getattr(self, "field_visuals_window", None) is not None:
            try:
                self.field_visuals_window.close()
            except Exception:
                pass
            self.field_visuals_window = None
        if getattr(self, "colormap_window", None) is not None:
            try:
                self.colormap_window.close()
            except Exception:
                pass
            self.colormap_window = None
        if getattr(self, "camera_sphere", None) is not None:
            try:
                self.camera_sphere.close()
            except Exception:
                pass
            self.camera_sphere = None
        restore_viewing_mouse(self.cmd)
        try:
            from .wizards.last_click import restore_atom_selection_mode, uninstall_click_feedback_hook

            restore_atom_selection_mode(self.cmd)
            uninstall_click_feedback_hook()
        except Exception:
            pass


def exit_wizard(cmd_=None):
    """Leave wizard mode by popping the stack with set_wizard() (public API)."""
    if cmd_ is None:
        cmd_ = cmd
    from .util.pymol_helpers import purge_ephemeral_wizard_objects

    # Do not use set_wizard_stack([]): on some PyMOL 3 builds it succeeds
    # but leaves the wizard system unable to show a new panel.
    for _ in range(4096):
        wizard = cmd_.get_wizard()
        if wizard is None:
            break
        if is_pymolviz_wizard(wizard):
            try:
                wizard.cleanup()
            except Exception:
                pass
        try:
            cmd_.set_wizard()
        except Exception:
            break

    purge_ephemeral_wizard_objects(cmd_)
    restore_viewing_mouse(cmd_)
    try:
        from .wizards.last_click import restore_atom_selection_mode

        restore_atom_selection_mode(cmd_)
    except Exception:
        pass
    try:
        cmd_.refresh_wizard()
    except Exception:
        pass


def reconcile_wizard_after_session_load(cmd_=None):
    """Drop saved wizard UI state after .pse load; leave a normal PyMOL session."""
    exit_wizard(cmd_)


def start_wizard():
    """Open the PyMOLViz wizard panel."""
    register_wizard_commands()
    view = cmd.get_view()
    try:
        exit_wizard(cmd)
    except Exception:
        try:
            cmd.set_wizard()
        except Exception:
            pass
    wizard = PyMolVizWizard()
    try:
        cmd.set_wizard(wizard, replace=1)
    except TypeError:
        cmd.set_wizard(wizard)
    try:
        cmd.refresh_wizard()
    except Exception:
        pass
    restore_view(cmd, view)
    wizard._sync_sphere()


def reload_wizard():
    """Uninstall hooks, stop wizard, purge pymolviz.*, reinstall, reconcile."""
    from .runtime.integration import reload_pymolviz

    reload_pymolviz(restart_wizard=True)


_WIZARD_COMMANDS_REGISTERED = False


def register_wizard_commands():
    """Register ``pmvw`` / related CLI hooks when running inside PyMOL."""
    global _WIZARD_COMMANDS_REGISTERED
    if _WIZARD_COMMANDS_REGISTERED:
        return
    extend_cmd(cmd, "pymolviz_wizard", start_wizard)
    extend_cmd(cmd, "pmvw", start_wizard)
    extend_cmd(cmd, "pymolviz_reload_wizard", reload_wizard)
    extend_cmd(cmd, "pymolviz_exit_wizard", exit_wizard)
    from .util.array_backend import announce_cuda

    announce_cuda(cmd)
    _WIZARD_COMMANDS_REGISTERED = True


try:
    register_wizard_commands()
except Exception:
    pass
