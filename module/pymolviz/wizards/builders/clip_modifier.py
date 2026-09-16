"""Shared clip-plane list, pose dialog, and drag poll for every mesh editor."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from ...util.mesh_clip import (
    apply_clip_plane_pose,
    clip_plane_from_view,
    clip_planes_match,
    normalize_clip_planes,
)
from ..pick import (
    bind_tool_window,
    configure_tool_window,
    overlay_warning,
    overlay_window,
    qt_modules,
    qt_widget_alive,
)
from ..tooltips import (
    CLIP_NORMAL_TIP,
    CLIP_ORIGIN_TIP,
    DELETE_CLIP_TIP,
    EDIT_CLIP_TIP,
    FLIP_CLIP_TIP,
    SHOW_PLANE_GIZMO_TIP,
    apply_required_tooltips,
    install_item_widget_tooltips,
)
from ..widgets.ascii_locale import apply_ascii_float_locale
from ..widgets.spin_step import STEP_ANGSTROM, STEP_NORMAL, apply_spin_step
from ..widgets.switch import make_switch

ADD_CLIP_TIP = (
    "Insert a clipping plane through the mesh, parallel to the current view. "
    "It cuts immediately. PyMOL's Drag widget (arrows and rings) sits on the "
    "plane to move and tilt it."
)
# Remesh only after the manipulator stops. Live ticks move the gizmo CGO.
_CLIP_MESH_SETTLE_MS = 150


def as_xyz_span(span):
    """Normalize preview vertices or VisualPoint lists to an ``(N, 3)`` array."""
    if span is None:
        return None
    if isinstance(span, np.ndarray):
        arr = np.asarray(span, dtype=float).reshape(-1, 3)
        return arr if arr.size else None
    try:
        rows = []
        for item in span:
            if hasattr(item, "xyz"):
                if not getattr(item, "enabled", True):
                    continue
                rows.append(item.xyz())
            else:
                rows.append(tuple(item))
        if not rows:
            return None
        arr = np.asarray(rows, dtype=float).reshape(-1, 3)
        return arr if arr.size else None
    except TypeError:
        return None


class ClipModifierController:
    """Clip-plane state + gizmos used by Sphere, Box, Surface, and Arrow editors."""

    def __init__(
        self,
        *,
        cmd,
        page,
        context: str,
        preview,
        span_points: Callable,
        on_changed: Callable[[], None],
    ):
        self.cmd = cmd
        self._page = page
        self._context = context
        self._preview = preview
        self._span_points = span_points
        self._on_changed = on_changed
        self.clip_planes = []
        self._gizmos_shown = True
        self._list = None
        self._suspend = False
        self._drag_timer = None
        self._settle_timer = None
        self._mesh_clip_pending = False
        self._pose_dialog = None
        self._list_tip_filter = None

    def attach_list(self, list_widget) -> None:
        self._list = list_widget
        if list_widget is not None:
            list_widget.currentRowChanged.connect(lambda *_: self._on_selection_changed())
            self._list_tip_filter = install_item_widget_tooltips(list_widget)
            self._install_list_deselect(list_widget)
        self.sync_list()

    def load_planes(self, planes) -> None:
        self.clip_planes = [self.wizard_plane(p) for p in (planes or [])]
        self.sync_list()

    def set_planes(self, planes) -> None:
        self.load_planes(planes)

    def active_planes(self):
        return self.active_clip_planes()

    @property
    def planes(self):
        return self.clip_planes

    def _selected_index(self):
        return self.selected_index()

    def _sync_drag_poll(self) -> None:
        self.sync_drag_poll()

    def reset(self) -> None:
        self.clip_planes = []
        self.close_pose_dialog()
        self._cancel_mesh_commit()
        self.stop_drag_poll()
        if self._preview is not None and hasattr(self._preview, "release_clip_drag"):
            self._preview.release_clip_drag()
        self.sync_list()
        self.refresh_gizmos()

    def wizard_plane(self, stored) -> dict:
        plane = normalize_clip_planes([stored])
        if not plane:
            return {
                "origin": [0.0, 0.0, 0.0],
                "normal": [0.0, 0.0, 1.0],
                "scale": 5.0,
                "gizmo": True,
            }
        visible = True
        if isinstance(stored, dict) and "gizmo" in stored:
            visible = bool(stored.get("gizmo", True))
        return {
            "origin": list(plane[0]["origin"]),
            "normal": list(plane[0]["normal"]),
            "scale": float(plane[0]["scale"]),
            "gizmo": visible,
        }

    def active_clip_planes(self):
        return normalize_clip_planes(self.clip_planes)

    def selected_index(self):
        if self._list is None:
            return None
        row = int(self._list.currentRow())
        if row < 0 or row >= len(self.clip_planes):
            return None
        return row

    def modifier_count(self) -> int:
        return len(self.clip_planes)

    def span_points(self):
        span = None
        if self._preview is not None:
            getter = getattr(self._preview, "span_points", None)
            if callable(getter):
                span = getter()
        if span is None:
            span = self._span_points() if self._span_points is not None else None
        return as_xyz_span(span)

    def resolve_list_row(self, keep_row=None) -> int:
        """Row to select after rebuilding the list; ``-1`` means none."""
        if not self.clip_planes:
            return -1
        if keep_row is None:
            current = -1 if self._list is None else int(self._list.currentRow())
        else:
            current = int(keep_row)
        if current >= len(self.clip_planes):
            return len(self.clip_planes) - 1
        return current

    def sync_list(self, keep_row=None) -> None:
        if self._list is None:
            return
        _, _, QtWidgets = qt_modules()
        self._suspend = True
        try:
            current = self.resolve_list_row(keep_row)
            self._list.clear()
            for i, _plane in enumerate(self.clip_planes):
                item = QtWidgets.QListWidgetItem()
                row = self._clip_row_widget(i)
                item.setSizeHint(row.sizeHint())
                self._list.addItem(item)
                self._list.setItemWidget(item, row)
            self._list.setCurrentRow(current)
        finally:
            self._suspend = False

    def clear_selection(self) -> None:
        if self._list is None:
            return
        if self.selected_index() is None:
            return
        self._suspend = True
        try:
            self._list.setCurrentRow(-1)
        finally:
            self._suspend = False
        self._on_selection_changed()

    def _install_list_deselect(self, list_widget) -> None:
        QtCore, _, QtWidgets = qt_modules()
        if QtCore is None or QtWidgets is None:
            return
        original = list_widget.mousePressEvent

        def on_press(event):
            if event.button() == QtCore.Qt.LeftButton:
                item = list_widget.itemAt(event.pos())
                if item is None:
                    self.clear_selection()
                    event.accept()
                    return
            original(event)

        list_widget.mousePressEvent = on_press

    def _clip_symbol_button(self, QtCore, QtWidgets, text: str):
        btn = QtWidgets.QPushButton(text)
        btn.setFlat(True)
        btn.setFixedWidth(22)
        btn.setFixedHeight(22)
        btn.setCursor(QtCore.Qt.PointingHandCursor)
        btn.setFocusPolicy(QtCore.Qt.NoFocus)
        btn.setAutoDefault(False)
        btn.setMouseTracking(True)
        always = getattr(QtCore.Qt, "WA_AlwaysShowToolTips", None)
        if always is not None:
            btn.setAttribute(always, True)
        return btn

    def _clip_row_widget(self, index: int):
        QtCore, _, QtWidgets = qt_modules()
        row = QtWidgets.QWidget()
        row.setMouseTracking(True)
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(6, 1, 4, 1)
        layout.setSpacing(2)
        visible = make_switch("", compact=True)
        visible.blockSignals(True)
        visible.setChecked(bool(self.clip_planes[index].get("gizmo", True)))
        visible.setEnabled(bool(self._gizmos_shown))
        visible.blockSignals(False)
        visible.toggled.connect(lambda checked, i=index: self.set_plane_gizmo_visible(i, checked))
        layout.addWidget(visible)
        label = QtWidgets.QLabel("Plane %d" % (index + 1))
        layout.addWidget(label, stretch=1)
        edit = self._clip_symbol_button(QtCore, QtWidgets, "⚙")
        edit.clicked.connect(lambda _=False, i=index: self.edit_at(i))
        flip = self._clip_symbol_button(QtCore, QtWidgets, "⇄")
        flip.clicked.connect(lambda _=False, i=index: self.flip_at(i))
        delete = self._clip_symbol_button(QtCore, QtWidgets, "×")
        delete.clicked.connect(lambda _=False, i=index: self.delete_at(i))
        layout.addWidget(edit)
        layout.addWidget(flip)
        layout.addWidget(delete)

        def on_press(event, i=index):
            if event.button() == QtCore.Qt.LeftButton:
                if int(self._list.currentRow()) == i:
                    self.clear_selection()
                    event.accept()
                    return
                self._list.setCurrentRow(i)
            QtWidgets.QWidget.mousePressEvent(row, event)

        row.mousePressEvent = on_press
        apply_required_tooltips(
            [
                (visible, SHOW_PLANE_GIZMO_TIP, "Show clip plane"),
                (edit, EDIT_CLIP_TIP, "Edit clip plane"),
                (flip, FLIP_CLIP_TIP, "Flip clip plane"),
                (delete, DELETE_CLIP_TIP, "Delete clip plane"),
            ],
            context=self._context,
        )
        return row

    def clip_drag_live(self) -> bool:
        preview = self._preview
        if preview is None:
            return False
        fn = getattr(preview, "drag_is_live", None)
        if callable(fn):
            return bool(fn())
        gizmos = getattr(preview, "_gizmos", None)
        if gizmos is not None and hasattr(gizmos, "drag_is_live"):
            return bool(gizmos.drag_is_live())
        return False

    def gizmos_shown(self) -> bool:
        return bool(self._gizmos_shown)

    def set_gizmos_shown(self, shown: bool) -> None:
        shown = bool(shown)
        if shown == self._gizmos_shown:
            return
        self._gizmos_shown = shown
        if not shown:
            self._relatch_drag()
        self.refresh_gizmos()

    def set_plane_gizmo_visible(self, idx: int, visible: bool) -> None:
        if idx < 0 or idx >= len(self.clip_planes):
            return
        plane = self.clip_planes[idx]
        visible = bool(visible)
        if bool(plane.get("gizmo", True)) == visible:
            return
        plane["gizmo"] = visible
        if not visible and self.selected_index() == idx:
            self._relatch_drag()
        self.refresh_gizmos()

    def preview_gizmos(self):
        """Planes drawn in the viewer, with selected index remapped into that list."""
        if not self._gizmos_shown:
            return [], None
        visible = []
        selected = None
        sel = self.selected_index()
        for i, plane in enumerate(self.clip_planes):
            if not plane.get("gizmo", True):
                continue
            if sel is not None and i == sel:
                selected = len(visible)
            visible.append(plane)
        return visible, selected

    def refresh_gizmos(self, attach_drag=None) -> None:
        if self._preview is None or not hasattr(self._preview, "set_gizmos"):
            return
        planes, selected = self.preview_gizmos()
        if attach_drag is None:
            attach_drag = not self.clip_drag_live()
        if selected is None:
            attach_drag = False
        self._preview.set_gizmos(
            planes,
            selected_index=selected,
            span_points=self.span_points(),
            attach_drag=attach_drag,
        )
        self.sync_drag_poll()

    def sync_drag_poll(self) -> None:
        QtCore, _, _ = qt_modules()
        _planes, selected = self.preview_gizmos()
        has = selected is not None
        if not has:
            self.stop_drag_poll()
            self._commit_pending_clip_mesh()
            return
        if QtCore is None or not hasattr(QtCore, "QTimer"):
            return
        if self._page is None:
            return
        if self._drag_timer is None:
            timer = QtCore.QTimer(self._page)
            timer.setInterval(50)
            timer.timeout.connect(self._on_drag_tick)
            self._drag_timer = timer
        if not self._drag_timer.isActive():
            self._drag_timer.start()

    def stop_drag_poll(self) -> None:
        timer = self._drag_timer
        if timer is not None:
            try:
                timer.stop()
            except RuntimeError:
                pass

    def _ensure_settle_timer(self):
        QtCore, _, _ = qt_modules()
        if QtCore is None or not hasattr(QtCore, "QTimer") or self._page is None:
            return None
        if self._settle_timer is None:
            timer = QtCore.QTimer(self._page)
            timer.setSingleShot(True)
            timer.setInterval(_CLIP_MESH_SETTLE_MS)
            timer.timeout.connect(self._commit_pending_clip_mesh)
            self._settle_timer = timer
        return self._settle_timer

    def _cancel_mesh_commit(self) -> None:
        self._mesh_clip_pending = False
        timer = self._settle_timer
        if timer is not None:
            try:
                timer.stop()
            except RuntimeError:
                pass

    def _schedule_mesh_commit(self) -> None:
        self._mesh_clip_pending = True
        timer = self._ensure_settle_timer()
        if timer is not None:
            timer.start()

    def _commit_pending_clip_mesh(self) -> None:
        if not self._mesh_clip_pending:
            return
        if self._page is not None and not qt_widget_alive(self._page):
            self._mesh_clip_pending = False
            return
        self._mesh_clip_pending = False
        self._on_changed()

    def _on_drag_tick(self):
        if self._suspend:
            return
        idx = self.selected_index()
        if idx is None:
            self.stop_drag_poll()
            self._commit_pending_clip_mesh()
            return
        if self._preview is None or not hasattr(self._preview, "poll_clip_drag"):
            return
        pose = self._preview.poll_clip_drag()
        if pose is None:
            return
        origin, normal = pose
        plane = self.clip_planes[idx]
        proposed = {
            "origin": origin,
            "normal": normal,
            "scale": plane.get("scale", 5.0),
        }
        if clip_planes_match([plane], [proposed], ignore_scale=True):
            return
        apply_clip_plane_pose(plane, origin, normal)
        # Follow the manipulator with the plane CGO only. Remeshing the
        # clipped solid while the TTT is still changing is both laggy and
        # compounds motion if the dummy is restarted at the new origin.
        self.refresh_gizmos(attach_drag=False)
        self._schedule_mesh_commit()

    def _on_selection_changed(self):
        if self._suspend:
            return
        if self.selected_index() is None:
            self._relatch_drag()
            self.stop_drag_poll()
            self._commit_pending_clip_mesh()
        self.refresh_gizmos()

    def _relatch_drag(self):
        if self._preview is not None and hasattr(self._preview, "release_clip_drag"):
            self._preview.release_clip_drag()

    def add_plane(self) -> None:
        try:
            view = tuple(self.cmd.get_view())
        except Exception:
            return
        span = self.span_points()
        if span is None:
            return
        stored = clip_plane_from_view(view, span)
        self.clip_planes.append(self.wizard_plane(stored))
        self.sync_list(keep_row=len(self.clip_planes) - 1)
        self._cancel_mesh_commit()
        self._on_changed()

    def flip_at(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.clip_planes):
            return
        plane = self.clip_planes[idx]
        plane["normal"] = [-float(x) for x in plane["normal"]]
        if self._list is not None:
            self._list.setCurrentRow(idx)
        self._relatch_drag()
        self._cancel_mesh_commit()
        self._on_changed()

    def delete_at(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.clip_planes):
            return
        del self.clip_planes[idx]
        self.close_pose_dialog()
        self._relatch_drag()
        self._cancel_mesh_commit()
        self.sync_list(keep_row=min(idx, len(self.clip_planes) - 1))
        self._on_changed()

    def _xyz_spin_row(self, values, *, decimals, rng, tooltip):
        QtCore, _, QtWidgets = qt_modules()
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        spins = []
        for axis, value in zip("XYZ", values):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(decimals)
            spin.setRange(rng[0], rng[1])
            apply_spin_step(spin, STEP_ANGSTROM if decimals <= 3 else STEP_NORMAL)
            spin.setKeyboardTracking(True)
            spin.setValue(float(value))
            spin.setPrefix("%s " % axis)
            apply_ascii_float_locale(spin, QtCore)
            spins.append(spin)
            layout.addWidget(spin)
        apply_required_tooltips(
            [(spin, tooltip, "Clip %s" % axis) for spin, axis in zip(spins, "XYZ")],
            context=self._context,
        )
        return row, spins[0], spins[1], spins[2]

    def close_pose_dialog(self) -> None:
        dialog = self._pose_dialog
        self._pose_dialog = None
        if dialog is None:
            return
        try:
            dialog.close()
        except RuntimeError:
            pass

    def _forget_pose_dialog(self, dialog) -> None:
        if self._pose_dialog is dialog:
            self._pose_dialog = None

    def edit_at(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.clip_planes):
            return
        QtCore, _, QtWidgets = qt_modules()
        if QtWidgets is None:
            return
        if self._list is not None:
            self._list.setCurrentRow(idx)
        self.close_pose_dialog()
        plane = self.clip_planes[idx]
        overlay = overlay_window(self._page)
        dialog = QtWidgets.QDialog(overlay if overlay is not None else self._page)
        dialog.setWindowTitle("Clip plane %d" % (idx + 1))
        dialog.setModal(False)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        configure_tool_window(dialog)

        form = QtWidgets.QFormLayout()
        origin_row, ox, oy, oz = self._xyz_spin_row(
            plane["origin"], decimals=3, rng=(-1e6, 1e6), tooltip=CLIP_ORIGIN_TIP,
        )
        normal_row, nx, ny, nz = self._xyz_spin_row(
            plane["normal"], decimals=4, rng=(-1e3, 1e3), tooltip=CLIP_NORMAL_TIP,
        )
        form.addRow("Origin (Å)", origin_row)
        form.addRow("Keep-side normal", normal_row)
        hint = QtWidgets.QLabel("Normal is unit-normalized when applied.")
        hint.setWordWrap(True)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )

        def _accept():
            if idx < 0 or idx >= len(self.clip_planes):
                dialog.reject()
                return
            origin = (ox.value(), oy.value(), oz.value())
            normal = (nx.value(), ny.value(), nz.value())
            if not apply_clip_plane_pose(self.clip_planes[idx], origin, normal):
                overlay_warning(
                    dialog,
                    "Clip plane",
                    "The keep-side normal cannot be zero. Enter a direction such as 0, 0, 1.",
                )
                return
            self._relatch_drag()
            self._cancel_mesh_commit()
            self._on_changed()
            dialog.accept()

        buttons.accepted.connect(_accept)
        buttons.rejected.connect(dialog.reject)
        root = QtWidgets.QVBoxLayout(dialog)
        root.addLayout(form)
        root.addWidget(hint)
        root.addWidget(buttons)
        dialog.destroyed.connect(lambda *_a, d=dialog: self._forget_pose_dialog(d))
        self._pose_dialog = dialog
        dialog.show()
        bind_tool_window(dialog)
        dialog.raise_()
        dialog.activateWindow()


DEFAULT_CLIP_PREVIEW_NAME = "_pmv_prev_surface_clip"


class ClipGizmoPreview:
    """Clip-plane rectangles + native Drag widget, shared by every mesh preview."""

    def __init__(self, cmd_, name: str = DEFAULT_CLIP_PREVIEW_NAME):
        from .preview import RuntimeCollectionPreview

        self.cmd = cmd_
        self._name = name
        self._clip_preview = RuntimeCollectionPreview(cmd_, name)
        self._clip_drag_rest = None
        self._clip_drag_matrix = None
        self._clip_drag_button_mode = None
        self._axis_lock = False

    def set_gizmos(self, planes, selected_index=None, span_points=None, attach_drag=True, axis_lock=None):
        from ...meshes.CGOCollection import CGOCollection
        from ...meshes.ClipGizmo import ClipGizmo

        if axis_lock is not None:
            self._axis_lock = bool(axis_lock)
        planes = list(planes or [])
        if not planes:
            self._clip_preview.cleanup()
            self._stop_clip_drag()
            return
        children = []
        for i, plane in enumerate(planes):
            origin = plane.get("origin", (0.0, 0.0, 0.0))
            normal = plane.get("normal", (0.0, 0.0, 1.0))
            scale = float(plane.get("scale", 5.0) or 5.0)
            children.append(
                ClipGizmo(
                    origin, normal, scale,
                    points=span_points,
                    selected=(selected_index is not None and i == int(selected_index)),
                    draft=False,
                    axis_arrow=self._axis_lock,
                    bypass_colormap=True,
                )
            )
        collection = CGOCollection(children, name=self._name)
        self._clip_preview.update_collection(collection)
        if attach_drag:
            self._sync_clip_drag(planes, selected_index, span_points)

    def _clip_visual_center(self, plane, span_points):
        from ...util.clip_gizmo import fit_plane_rectangle

        origin = plane.get("origin", (0.0, 0.0, 0.0))
        normal = plane.get("normal", (0.0, 0.0, 1.0))
        scale = float(plane.get("scale", 5.0) or 5.0)
        center, _n, _u, _v = fit_plane_rectangle(
            origin, normal, points=span_points, scale=scale,
        )
        return (
            [float(origin[0]), float(origin[1]), float(origin[2])],
            [float(normal[0]), float(normal[1]), float(normal[2])],
            [float(center[0]), float(center[1]), float(center[2])],
        )

    def drag_is_live(self) -> bool:
        from ...util.clip_drag import CLIP_DRAG_NAME

        if self._clip_drag_rest is None:
            return False
        try:
            active = str(self.cmd.get_drag_object_name() or "")
        except Exception:
            active = ""
        return active == CLIP_DRAG_NAME

    def _sync_clip_drag(self, planes, selected_index, span_points):
        from ...util.clip_drag import CLIP_DRAG_NAME, read_object_matrix, start_clip_drag

        if selected_index is None or int(selected_index) < 0 or int(selected_index) >= len(planes):
            self._stop_clip_drag()
            return
        idx = int(selected_index)
        origin, normal, center = self._clip_visual_center(planes[idx], span_points)
        rest = self._clip_drag_rest
        if rest is not None:
            try:
                active = str(self.cmd.get_drag_object_name() or "")
            except Exception:
                active = ""
            if active == CLIP_DRAG_NAME and int(rest.get("index", idx)) == idx:
                return
        saved_mode = self._clip_drag_button_mode
        attached_mode = start_clip_drag(
            self.cmd, center, native_widget=not self._axis_lock,
        )
        self._clip_drag_button_mode = saved_mode if saved_mode is not None else attached_mode
        self._clip_drag_rest = {
            "origin": origin, "normal": normal, "center": center, "index": idx,
        }
        self._clip_drag_matrix = read_object_matrix(self.cmd, CLIP_DRAG_NAME)

    def poll_clip_drag(self):
        from ...util.clip_drag import (
            CLIP_DRAG_NAME,
            apply_axis_drag_matrix,
            apply_drag_matrix,
            matrix_changed,
            read_object_matrix,
        )

        rest = self._clip_drag_rest
        if rest is None:
            return None
        matrix = read_object_matrix(self.cmd, CLIP_DRAG_NAME)
        if not matrix_changed(self._clip_drag_matrix, matrix):
            return None
        self._clip_drag_matrix = matrix
        apply = apply_axis_drag_matrix if self._axis_lock else apply_drag_matrix
        origin, normal, _center = apply(
            rest["origin"], rest["normal"], rest["center"], matrix,
        )
        return origin, normal

    def release_clip_drag(self):
        self._stop_clip_drag()

    def _stop_clip_drag(self):
        from ...util.clip_drag import stop_clip_drag

        stop_clip_drag(self.cmd, button_mode=self._clip_drag_button_mode)
        self._clip_drag_rest = None
        self._clip_drag_matrix = None
        self._clip_drag_button_mode = None

    def cleanup(self):
        from ...util.clip_drag import CLIP_DRAG_NAME
        from ...util.pymol_helpers import purge_objects

        self._stop_clip_drag()
        self._clip_preview.cleanup()
        purge_objects(self.cmd, prefixes=(self._name, CLIP_DRAG_NAME))
