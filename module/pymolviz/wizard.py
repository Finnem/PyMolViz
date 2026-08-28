import math
import time

from pymol import cmd
from pymol.cgo import CYLINDER
from pymol.wizard import Wizard


def _point_on_sphere(center, radius, theta, phi):
    cx, cy, cz = center
    st = math.sin(theta)
    return (
        cx + radius * st * math.cos(phi),
        cy + radius * st * math.sin(phi),
        cz + radius * math.cos(theta),
    )


def _wireframe_sphere_cgo(center, radius, color, n_lon=6, n_lat=4, n_seg=10, line_r=0.012):
    red, green, blue = [float(c) for c in color]
    obj = []

    def add_edge(p0, p1):
        obj.extend([CYLINDER, *p0, *p1, line_r, red, green, blue, red, green, blue])

    for i in range(n_lon):
        phi = 2.0 * math.pi * i / n_lon
        for j in range(n_seg):
            add_edge(
                _point_on_sphere(center, radius, math.pi * j / n_seg, phi),
                _point_on_sphere(center, radius, math.pi * (j + 1) / n_seg, phi),
            )
    for i in range(1, n_lat + 1):
        theta = math.pi * i / (n_lat + 1)
        for j in range(n_seg):
            add_edge(
                _point_on_sphere(center, radius, theta, 2.0 * math.pi * j / n_seg),
                _point_on_sphere(center, radius, theta, 2.0 * math.pi * (j + 1) / n_seg),
            )
    return obj


def _restore_view(cmd_, view):
    if view is None:
        return
    try:
        cmd_.set_view(view, animate=0)
    except TypeError:
        cmd_.set_view(view)


def _purge_camera_helpers(cmd_, name):
    """Drop leftover cage / callback objects from a previous wizard run."""
    try:
        existing = list(cmd_.get_names("objects"))
    except Exception:
        existing = [name]
    for obj_name in existing:
        if obj_name == name or obj_name.startswith("_pmv_cam_cb"):
            try:
                cmd_.delete(obj_name)
            except Exception:
                pass


class _CameraCenterSphere:
    """Keeps a small wireframe sphere at the current screen / view center."""

    def __init__(self, cmd_, name="pmv_camera_center", radius=0.35, color=(1.0, 0.85, 0.15)):
        self.cmd = cmd_
        self.name = name
        self.radius = radius
        self.color = color
        self.prev_view = None
        self._closed = False
        self._hold = False
        self._hold_pos = None
        self._seen_motion = False
        self._hold_until = 0.0
        self._settle = 0
        self._pick_view = None
        self._pick_coords = None
        self._pick_ids = None
        self._pick_viewport = (1.0, 1.0)
        self._pick_fov = 20.0
        self._pick_ortho = 0
        self._pick_time = 0.0
        self._anim_duration = 0.4
        self._pending_snap = None
        self._pending_center_sele = None
        self._pending_center_pos = None
        _purge_camera_helpers(self.cmd, self.name)
        saved_view = self.cmd.get_view()
        cgo = _wireframe_sphere_cgo((0.0, 0.0, 0.0), self.radius, self.color)
        try:
            # zoom=-1 (default) calls cmd.zoom on this origin-centered CGO.
            self.cmd.load_cgo(cgo, self.name, 1, zoom=0)
        except TypeError:
            loadable = getattr(self.cmd, "loadable", None)
            if loadable is not None:
                self.cmd.load_object(loadable.cgo, cgo, self.name, zoom=0)
            else:
                self.cmd.load_cgo(cgo, self.name, 1)
        self.cmd.enable(self.name)
        _restore_view(self.cmd, saved_view)
        self.sync()
        self.refresh_pick_cache()

    def request_snap(self, widget, x, y):
        """Queue a click snap; applied on the next draw callback (no cmd on the mouse path)."""
        self._pending_snap = (widget, float(x), float(y))

    def _apply_pending_snap(self):
        pending = self._pending_snap
        if pending is None:
            return None
        self._pending_snap = None
        widget, x, y = pending
        target = self.atom_at_cursor(widget, x, y)
        if target is None:
            self.hold()
            return None
        pos, sele = target
        self.snap_to(pos)
        self._pending_center_sele = sele
        self._pending_center_pos = pos
        return sele

    def take_center_target(self):
        sele = self._pending_center_sele
        pos = self._pending_center_pos
        self._pending_center_sele = None
        self._pending_center_pos = None
        return sele, pos

    def snap_to(self, pos):
        """Jump to a camera-move target and ignore interpolated frames until it settles."""
        self._begin_hold(tuple(float(c) for c in pos))

    def atom_at_cursor(self, widget, x, y):
        """Nearest cached atom under the cursor. Does not call into cmd."""
        if self._pick_view is None or self._pick_coords is None:
            return None
        return _pick_atom(
            self._pick_view,
            self._pick_coords,
            widget,
            x,
            y,
            self._pick_viewport,
            self._pick_fov,
            self._pick_ortho,
            self._pick_ids,
        )

    def refresh_pick_cache(self):
        """Refresh atom/view data used for click snaps. Safe outside mouse events."""
        if self._closed:
            return
        view = tuple(self.cmd.get_view())
        self._pick_view = view
        try:
            self._pick_viewport = tuple(float(v) for v in self.cmd.get_viewport())
        except Exception:
            self._pick_viewport = (1.0, 1.0)
        try:
            self._pick_fov = abs(float(self.cmd.get("field_of_view")))
        except Exception:
            self._pick_fov = 20.0
        try:
            self._pick_ortho = int(float(self.cmd.get("orthoscopic")))
        except Exception:
            self._pick_ortho = 1 if float(view[17]) > 0.5 else 0
        try:
            self._anim_duration = max(float(self.cmd.get("animation_duration")), 0.3)
        except Exception:
            self._anim_duration = 0.4
        try:
            state = int(self.cmd.get_state())
        except Exception:
            state = 1
        try:
            coords = self.cmd.get_coords(_PICK_SELE, state)
        except Exception:
            coords = None
        ids = []
        if coords is not None and len(coords):
            try:
                self.cmd.iterate(_PICK_SELE, "ids.append((model, index))", space={"ids": ids})
            except Exception:
                ids = []
        self._pick_coords = coords
        self._pick_ids = ids
        self._pick_time = time.monotonic()

    def _maybe_refresh_pick_cache(self, view):
        if self._hold:
            return
        if self._pick_coords is None or (time.monotonic() - self._pick_time) > 1.0:
            self.refresh_pick_cache()

    def follow(self):
        """Resume tracking the live view (used for click-and-drag translation)."""
        self._hold = False
        self._hold_pos = None
        self._seen_motion = False
        self._settle = 0

    def hold(self):
        """Keep the current placement until an interpolated camera move finishes."""
        self._begin_hold(None)

    def _begin_hold(self, pos):
        self._hold = True
        self._hold_pos = pos
        self._seen_motion = False
        self._settle = 0
        view = self._pick_view or self.prev_view
        if view is None:
            view = tuple(self.cmd.get_view())
        self.prev_view = tuple(view)
        self._hold_until = time.monotonic() + self._anim_duration
        if pos is not None:
            self._place(pos)

    def sync(self):
        self._apply_pending_snap()
        view = tuple(self.cmd.get_view())
        self._pick_view = view
        if self._hold:
            if self._hold_pos is not None:
                self._place(self._hold_pos)
            if view != self.prev_view:
                self._seen_motion = True
                self._settle = 0
                self.prev_view = view
                return False
            self.prev_view = view
            if not self._seen_motion or time.monotonic() < self._hold_until:
                return False
            self._settle += 1
            if self._settle >= 2:
                self.follow()
                self.prev_view = view
                self._place(_screen_center(view))
            return False
        if view == self.prev_view:
            self._maybe_refresh_pick_cache(view)
            return False
        self.prev_view = view
        self._place(_screen_center(view))
        return True

    def _place(self, center):
        self.cmd.set_object_ttt(
            self.name,
            [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                center[0], center[1], center[2], 1.0,
            ],
        )

    def close(self):
        self._closed = True
        try:
            self.cmd.delete(self.name)
        except Exception:
            pass


def _screen_center(view):
    """Model-space point currently in the middle of the viewer."""
    # Origin in camera space is view[9:12]. Screen center is camera (0, 0, z).
    dx = -float(view[9])
    dy = -float(view[10])
    rx = view[0] * dx + view[1] * dy
    ry = view[3] * dx + view[4] * dy
    rz = view[6] * dx + view[7] * dy
    return (float(view[12]) + rx, float(view[13]) + ry, float(view[14]) + rz)


def _qt_modules():
    try:
        from pymol.Qt import QtCore, QtGui, QtWidgets
        return QtCore, QtGui, QtWidgets
    except Exception:
        return None, None, None


def _find_viewer_widget(QtWidgets):
    app = QtWidgets.QApplication.instance()
    if app is None:
        return None
    ranked = []
    for widget in app.allWidgets():
        if not widget.isVisible() or widget.width() < 80 or widget.height() < 80:
            continue
        score = 0
        if hasattr(widget, "pymol") and hasattr(widget, "fb_scale"):
            score += 6
        elif hasattr(widget, "pymol"):
            score += 4
        if widget.inherits("QOpenGLWidget") or widget.inherits("QGLWidget"):
            score += 3
        cls = widget.metaObject().className()
        if any(token in cls for token in ("GLWidget", "OpenGL", "PyMOLGL", "CMol")):
            score += 2
        if score:
            ranked.append((score, widget.width() * widget.height(), widget))
    if not ranked:
        return None
    ranked.sort()
    return ranked[-1][-1]


def _widget_fb_scale(widget):
    scale = getattr(widget, "fb_scale", None)
    if scale:
        return float(scale)
    if hasattr(widget, "devicePixelRatioF"):
        return float(widget.devicePixelRatioF())
    if hasattr(widget, "devicePixelRatio"):
        return float(widget.devicePixelRatio())
    return 1.0


def _qt_to_pymol_xy(widget, x, y):
    """Match PyMOLGLWidget._event_x_y_mod: framebuffer pixels, Y from the bottom."""
    scale = _widget_fb_scale(widget)
    return int(scale * x), int(scale * (widget.height() - y))


def _set_single_middle(cmd_, action):
    try:
        cmd_.button("single_middle", "none", action)
    except Exception:
        pass


def _take_over_center_click(cmd_):
    """Wizard owns click-to-center so PyMOL's delayed 'cent' cannot be cancelled."""
    _set_single_middle(cmd_, "none")


def _restore_viewing_mouse(cmd_):
    """Restore default middle-click centering when the wizard closes."""
    _set_single_middle(cmd_, "cent")
    try:
        cmd_.unpick()
    except Exception:
        pass


_PICK_SELE = "visible and enabled and not hydro"


def _atom_sele(ids, index):
    if not ids or index < 0 or index >= len(ids):
        return None
    model, atm = ids[index]
    return "(%s)`%d" % (model, atm)


def _model_to_camera(view, pos):
    """OpenGL camera space: +X right, +Y up, look down -Z."""
    rel = (pos[0] - view[12], pos[1] - view[13], pos[2] - view[14])
    return (
        view[0] * rel[0] + view[3] * rel[1] + view[6] * rel[2] + view[9],
        view[1] * rel[0] + view[4] * rel[1] + view[7] * rel[2] + view[10],
        view[2] * rel[0] + view[5] * rel[1] + view[8] * rel[2] + view[11],
    )


def _pick_atom(view, coords, widget, x, y, viewport, fov, ortho, ids=None):
    """Nearest atom under the cursor using cached view/coords. No cmd calls."""
    click_x, click_y = _qt_to_pymol_xy(widget, x, y)
    try:
        width, height = float(viewport[0]), float(viewport[1])
    except Exception:
        width, height = 0.0, 0.0
    if width < 1 or height < 1:
        scale = _widget_fb_scale(widget)
        width, height = float(widget.width()) * scale, float(widget.height()) * scale
    if width < 1 or height < 1 or coords is None or len(coords) == 0:
        return None
    rect_bottom = max(_widget_fb_scale(widget) * widget.height() - height, 0.0)
    sx = float(click_x)
    sy = float(click_y) - rect_bottom
    fov_width = 2.0 * math.tan(math.radians(max(abs(float(fov)), 1.0)) / 2.0)
    origin_depth = max(abs(float(view[11])), 1e-6)
    best = None
    best_index = None
    best_key = None
    max_px = 24.0
    front = float(view[15])
    back = float(view[16])
    for index, pos in enumerate(coords):
        point = (float(pos[0]), float(pos[1]), float(pos[2]))
        cx, cy, cz = _model_to_camera(view, point)
        depth = origin_depth if ortho else -cz
        if depth < 1e-4:
            continue
        if front > 0.0 and back > front and (depth < front * 0.5 or depth > back * 1.5):
            continue
        angstrom_per_px = depth * fov_width / height
        if angstrom_per_px < 1e-8:
            continue
        dx = (width * 0.5 + cx / angstrom_per_px) - sx
        dy = (height * 0.5 + cy / angstrom_per_px) - sy
        dist2 = dx * dx + dy * dy
        if dist2 > max_px * max_px:
            continue
        key = (dist2, depth)
        if best_key is None or key < best_key:
            best_key = key
            best = point
            best_index = index
    if best is None:
        return None
    return best, _atom_sele(ids, best_index)


def _install_middle_click_filter(wizard):
    QtCore, QtGui, QtWidgets = _qt_modules()
    if QtCore is None:
        return None, None
    widget = _find_viewer_widget(QtWidgets)
    if widget is None:
        return None, None

    class _MiddleClickFilter(QtCore.QObject):
        _drag_px = 20

        def __init__(self, parent):
            QtCore.QObject.__init__(self, parent)
            self._press = None
            self._dragged = False

        def eventFilter(self, watched, event):
            if watched is not widget:
                return False
            middle = getattr(QtCore.Qt, "MiddleButton", None)
            if middle is None:
                middle = QtCore.Qt.MouseButton.MiddleButton
            etype = event.type()
            sphere = getattr(wizard, "camera_sphere", None)
            if sphere is None:
                return False

            if etype == QtCore.QEvent.MouseButtonPress and event.button() == middle:
                local = widget.mapFromGlobal(QtGui.QCursor.pos())
                self._press = (float(local.x()), float(local.y()))
                self._dragged = False
                return False

            if etype == QtCore.QEvent.MouseMove and self._press is not None:
                if event.buttons() & middle:
                    local = widget.mapFromGlobal(QtGui.QCursor.pos())
                    dx = local.x() - self._press[0]
                    dy = local.y() - self._press[1]
                    if (dx * dx + dy * dy) > (self._drag_px * self._drag_px):
                        if not self._dragged:
                            self._dragged = True
                            sphere.follow()
                        return False
                    # Keep tiny jitter from becoming a translate; we own click-to-center.
                    return True
                return False

            if etype == QtCore.QEvent.MouseButtonRelease and event.button() == middle:
                press = self._press
                dragged = self._dragged
                self._press = None
                self._dragged = False
                if press is not None and not dragged:
                    wizard._on_middle_click(press[0], press[1])
                return False

            return False

    filt = _MiddleClickFilter(widget)
    widget.installEventFilter(filt)
    return filt, widget


class PyMolVizWizard(Wizard):
    """Interactive PyMOL wizard with a side-panel menu."""

    def __init__(self):
        Wizard.__init__(self)
        self.prompt = ["PyMOLViz"]
        self.menu_items = [
            ("Item A", self.on_item_a),
            ("Item B", self.on_item_b),
            ("Item C", self.on_item_c),
        ]
        self.camera_sphere = _CameraCenterSphere(self.cmd)
        self._click_filter, self._click_widget = _install_middle_click_filter(self)
        _take_over_center_click(self.cmd)

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
        if sphere is not None:
            sphere.sync()

    def get_prompt(self):
        return self.prompt

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

    def on_item_a(self):
        self.prompt = ["Selected Item A"]

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
        QtCore, _, _ = _qt_modules()
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
        if sele:
            try:
                self.cmd.center(sele, animate=-1)
                return
            except Exception:
                try:
                    self.cmd.center(sele)
                    return
                except Exception:
                    pass
        if pos is None:
            return
        name = "_pmv_cent_tmp"
        try:
            self.cmd.pseudoatom(name, pos=list(pos))
            try:
                self.cmd.center(name, animate=-1)
            except Exception:
                self.cmd.center(name)
        finally:
            try:
                self.cmd.delete(name)
            except Exception:
                pass

    def cleanup(self):
        _restore_viewing_mouse(self.cmd)
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
    _restore_view(cmd, view)


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
