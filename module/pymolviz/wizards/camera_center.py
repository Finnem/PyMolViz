"""Wireframe cage locked to the current screen / view center."""

import time

from ..util.cgo import wireframe_sphere_cgo
from ..util.pymol_helpers import load_cgo_no_zoom, place_object, purge_objects, restore_view
from ..util.view import screen_center
from .pick import PICK_SELE, pick_atom


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
        purge_objects(self.cmd, names=(self.name,), prefixes=("_pmv_cam_cb",))
        saved_view = self.cmd.get_view()
        cgo = wireframe_sphere_cgo((0.0, 0.0, 0.0), self.radius, self.color)
        load_cgo_no_zoom(self.cmd, cgo, self.name, 1)
        self.cmd.enable(self.name)
        restore_view(self.cmd, saved_view)
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
        return pick_atom(
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
            coords = self.cmd.get_coords(PICK_SELE, state)
        except Exception:
            coords = None
        ids = []
        if coords is not None and len(coords):
            try:
                self.cmd.iterate(PICK_SELE, "ids.append((model, index))", space={"ids": ids})
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
                self._place(screen_center(view))
            return False
        if view == self.prev_view:
            self._maybe_refresh_pick_cache(view)
            return False
        self.prev_view = view
        self._place(screen_center(view))
        return True

    def _place(self, center):
        place_object(self.cmd, self.name, center)

    def close(self):
        self._closed = True
        try:
            self.cmd.delete(self.name)
        except Exception:
            pass
