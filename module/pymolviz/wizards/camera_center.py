"""Wireframe cage locked to the current screen / view center."""

import time

from ..util.cgo import wireframe_sphere_cgo
from ..util.pymol_helpers import load_cgo_no_zoom, place_object, purge_objects, restore_view
from ..util.view import click_ray_points, screen_center
from .pick import pick_atom, qt_to_pymol_xy, widget_fb_scale


class CameraCenterSphere:
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
        self._anim_duration = 0.4
        self._pending_snap = None
        self._pending_center_sele = None
        self._pending_center_pos = None
        self._current_pos = None
        purge_objects(self.cmd, names=(self.name,), prefixes=("_pmv_cam_cb",))
        saved_view = tuple(self.cmd.get_view())
        cgo = wireframe_sphere_cgo((0.0, 0.0, 0.0), self.radius, self.color)
        load_cgo_no_zoom(self.cmd, cgo, self.name, 1)
        try:
            self.cmd.enable(self.name)
        except Exception:
            pass
        restore_view(self.cmd, saved_view)
        self.follow_view(saved_view)

    def ensure_object(self):
        """Recreate the CGO if it was removed (e.g. during session save)."""
        if self._closed:
            return
        try:
            names = self.cmd.get_names("objects")
            if self.name in names:
                return
        except Exception:
            pass
        self.prev_view = None
        self._current_pos = None
        saved_view = tuple(self.cmd.get_view())
        cgo = wireframe_sphere_cgo((0.0, 0.0, 0.0), self.radius, self.color)
        load_cgo_no_zoom(self.cmd, cgo, self.name, 1)
        try:
            self.cmd.enable(self.name)
        except Exception:
            pass
        restore_view(self.cmd, saved_view)
        self.follow_view(saved_view)

    def request_snap(self, widget, x, y):
        """Queue a click snap. No cmd calls on the mouse path."""
        self._pending_snap = (widget, float(x), float(y))

    def _apply_pending_snap(self):
        pending = self._pending_snap
        if pending is None:
            return None
        self._pending_snap = None
        widget, x, y = pending
        view = tuple(self.cmd.get_view())
        try:
            viewport = tuple(float(v) for v in self.cmd.get_viewport())
        except Exception:
            viewport = (float(widget.width()), float(widget.height()))
        try:
            fov = abs(float(self.cmd.get("field_of_view")))
        except Exception:
            fov = 20.0
        try:
            ortho = int(float(self.cmd.get("orthoscopic")))
        except Exception:
            ortho = 1 if float(view[17]) > 0.5 else 0
        coords = self._coords_along_click(widget, x, y, view, viewport, fov)
        target = pick_atom(view, coords, widget, x, y, viewport, fov, ortho)
        if target is None:
            self.hold()
            return None
        pos, sele = target
        if not sele:
            sele = self._sele_near(pos)
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

    def _coords_along_click(self, widget, x, y, view, viewport, fov):
        """Atoms in the click-ray bounding box — not the whole visible molecule."""
        click_x, click_y = qt_to_pymol_xy(widget, x, y)
        try:
            width, height = float(viewport[0]), float(viewport[1])
        except Exception:
            width, height = 0.0, 0.0
        if width < 1 or height < 1:
            scale = widget_fb_scale(widget)
            width = float(widget.width()) * scale
            height = float(widget.height()) * scale
        rect_bottom = max(widget_fb_scale(widget) * widget.height() - height, 0.0)
        sx = float(click_x)
        sy = float(click_y) - rect_bottom
        samples = click_ray_points(view, sx, sy, width, height, fov, n=8)
        try:
            state = int(self.cmd.get_state())
        except Exception:
            state = 1
        pad = 2.5
        xs = [point[0] for point in samples]
        ys = [point[1] for point in samples]
        zs = [point[2] for point in samples]
        sele = (
            "(visible and enabled) and "
            "x > %g and x < %g and y > %g and y < %g and z > %g and z < %g"
            % (
                min(xs) - pad,
                max(xs) + pad,
                min(ys) - pad,
                max(ys) + pad,
                min(zs) - pad,
                max(zs) + pad,
            )
        )
        try:
            coords = self.cmd.get_coords(sele, state)
        except Exception:
            coords = None
        if coords is not None and len(coords):
            return coords
        return None

    def _sele_near(self, pos):
        """Resolve model`index for a world-space hit without iterating the protein."""
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        name = "_pmv_pk"
        ids = []
        try:
            self.cmd.select(
                name,
                "(visible and enabled) within 0.15 of [%g,%g,%g]" % (x, y, z),
            )
            self.cmd.iterate(name, "ids.append((model, index))", space={"ids": ids})
        except Exception:
            ids = []
        try:
            self.cmd.delete(name)
        except Exception:
            pass
        if not ids:
            return None
        model, atm = ids[0]
        return "(%s)`%d" % (model, atm)

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
        view = self.prev_view
        if view is None:
            view = tuple(self.cmd.get_view())
        self.prev_view = tuple(view)
        self._hold_until = time.monotonic() + self._anim_duration
        if pos is not None:
            self._place(pos)

    def current_position(self):
        """Model-space position of the camera-center marker."""
        if self._current_pos is not None:
            return self._current_pos
        view = self.prev_view
        if view is None:
            try:
                view = tuple(self.cmd.get_view())
            except Exception:
                return None
        return screen_center(view)

    def follow_view(self, view):
        """Move the cage to the current screen center. No atom picking."""
        view = tuple(view)
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
                self._place(screen_center(view))
            return False
        if view == self.prev_view:
            return False
        self.prev_view = view
        self._place(screen_center(view))
        return True

    def _place(self, center):
        pos = (float(center[0]), float(center[1]), float(center[2]))
        prev = self._current_pos
        if prev is not None:
            dx = pos[0] - prev[0]
            dy = pos[1] - prev[1]
            dz = pos[2] - prev[2]
            if dx * dx + dy * dy + dz * dz < 1e-16:
                return
        self._current_pos = pos
        place_object(self.cmd, self.name, pos)

    def close(self):
        self._closed = True
        try:
            self.cmd.delete(self.name)
        except Exception:
            pass
