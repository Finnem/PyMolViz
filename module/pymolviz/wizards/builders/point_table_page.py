"""Shared points-table editor used by sphere, box, and surface pages."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from ..widgets.section import make_section
from .anchor_table import point_columns
from .appearance_section import (
    COLOR_ALL_LABEL,
    COLOR_SELECTION_LABEL,
    RESET_COLORS_LABEL,
    AppearanceSection,
)
from .base import BuilderPage
from .colors import bind_color_pick_result, pick_rgb
from .modifiers_section import ModifiersSection
from .point_insertion import PointInsertionWidget
from .point_list import PointListEditor
from .pairs import MULTI_CLICKED, take_single_selection_point
from ..pick import qt_widget_alive
from .points import (
    VisualPoint,
    apply_location,
    enabled_points,
    export_points_to_selection,
    hide_exported_point_labels,
    update_points_from_camera,
    update_points_from_selection,
)
from .zoom_selection import zoom_to_visual_points


def _list_delete_filter_type(QtCore):
    class _ListDeleteKeyFilter(QtCore.QObject):
        def __init__(self, owner):
            QtCore.QObject.__init__(self)
            self._owner = owner

        def eventFilter(self, obj, event):
            if event.type() != QtCore.QEvent.KeyPress:
                return False
            if event.key() == QtCore.Qt.Key_Escape:
                self._owner._abort_atom_pick()
                return True
            if event.key() not in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
                return False
            self._owner._delete_selected()
            return True

    return _ListDeleteKeyFilter


class PointTableBuilderPage(BuilderPage):
    """Editor shell composing geometry, Appearance, Points, Modifiers, and action bar."""

    COLS = point_columns()
    TABLE_STRONG_FOCUS = False

    def _appearance_config(self) -> dict:
        return {
            "show_wireframe": True,
            "show_quality": False,
            "quality_range": (1, 5),
            "quality_tooltip": "",
        }

    def _init_editor(self):
        self._points: List[VisualPoint] = []
        self._table = None
        self._list = None
        self._selected_index = None
        self._appearance = None
        self._insertion = None
        self._modifiers = None
        self._table_filter = None
        self._pick_index = None
        self._ignore_xyz = None
        self._poll_timer = None
        self._preview = self._make_preview()
        self._init_options_state()

    def _make_preview(self):
        raise NotImplementedError

    def _init_options_state(self):
        """Subclass fields that must exist before ``_build``."""

    def _cleanup_ephemeral(self):
        hide_exported_point_labels(self.cmd)
        self._clear_atom_pick()
        if self._insertion is not None:
            self._insertion.stop_timer()
        if self._modifiers is not None:
            self._modifiers.cleanup()

    def _can_commit(self) -> bool:
        return bool(enabled_points(self._points))

    def reset_for_create(self):
        self._apply_create_chrome()
        self._points = []
        self._reset_options()
        self._reset_preview_mode()
        if self._appearance is not None:
            self._appearance.bind_points(self._points)
        if self._modifiers is not None:
            self._modifiers.clip.reset()
            self._modifiers.refresh_summary()
        self._selected_index = None
        self._clear_atom_pick()
        if self._list is not None or self._table is not None:
            self._sync_table()
        if self._insertion is not None:
            self._insertion.start_preview_timer(self._page)
        self._preview.cleanup()

    def load_object(self, obj):
        from ..catalog import display_name

        self._editing_id = str(obj.id)
        name = display_name(obj) or self.DEFAULT_NAME
        self._points = self._points_from_object(obj)
        self._deferred.cancel()
        self._suspend_preview = True
        try:
            self._apply_edit_chrome(name)
            self._load_options(obj)
            self._load_preview_mode(obj)
            if self._appearance is not None:
                self._appearance.bind_points(self._points)
            self._selected_index = 0 if self._points else None
            if self._list is not None or self._table is not None:
                self._sync_table()
        finally:
            self._suspend_preview = False
        self._preview.adopt(obj)
        self._load_clip_planes(obj)
        self._after_load()

    def _reset_options(self):
        """Restore option widgets to create-mode defaults."""

    def _reset_preview_mode(self):
        from .preview_mode import DEFAULT_PREVIEW_MODE

        if self._appearance is not None:
            self._appearance.set_preview_mode(DEFAULT_PREVIEW_MODE)

    def _load_preview_mode(self, obj):
        from .preview_mode import read_preview_mode

        if self._appearance is not None:
            self._appearance.set_preview_mode(read_preview_mode(obj))

    def _preview_if_on(self):
        """Empty list, preview off, or dead widget → cleanup/clear; else return active mode."""
        from .preview_mode import preview_is_on

        if not qt_widget_alive(self._page):
            return None
        try:
            if not enabled_points(self._points):
                self._preview.cleanup()
                return None
            mode = self._current_preview_mode()
            if not preview_is_on(mode):
                self._preview.clear_meshes()
                if self._modifiers is not None:
                    self._modifiers.clip.refresh_gizmos()
                return None
            return mode
        except RuntimeError:
            return None

    def _load_options(self, obj):
        """Copy mesh-specific options from a persisted object."""

    def _load_clip_planes(self, obj):
        from ..catalog import editor_kind
        from .load_visual import arrow_options, box_options, sphere_options, surface_options

        kind = editor_kind(obj)
        opts = {}
        if kind == "surface":
            opts = surface_options(obj)
        elif kind == "sphere":
            opts = sphere_options(obj)
        elif kind == "box":
            opts = box_options(obj)
        elif kind == "arrow":
            opts = arrow_options(obj)
        if self._modifiers is not None:
            self._modifiers.clip.set_planes(opts.get("clip_planes") or [])
            self._modifiers.refresh_summary()

    def _points_from_object(self, obj) -> List[VisualPoint]:
        from .load_visual import points_from_mesh

        return points_from_mesh(obj)

    def _after_load(self):
        if self._insertion is not None:
            self._insertion.start_preview_timer(self._page)
        self._schedule_preview()

    def _build(self, parent):
        QtCore, QtGui, QtWidgets = self._require_qt()
        page, root, back = self._mount_shell(parent, QtWidgets)
        left, right = self._mount_editor_columns(root, QtWidgets)
        tips = []
        tips.extend(self._mount_points_section(left, QtCore, QtGui, QtWidgets) or ())
        tips.extend(self._mount_geometry(right, QtCore, QtGui, QtWidgets) or ())
        tips.extend(self._mount_options(right, QtCore, QtGui, QtWidgets) or ())
        tips.extend(self._mount_appearance(right, QtCore, QtGui, QtWidgets) or ())
        tips.extend(self._mount_modifiers(right, QtCore, QtGui, QtWidgets) or ())
        right.addStretch(1)
        self._mount_action_bar(page, root)
        self._poll_timer = QtCore.QTimer(page)
        self._poll_timer.setInterval(250)
        self._poll_timer.timeout.connect(self._poll_atom_pick)
        self._finish_build(page, back, tips)

    def _mount_geometry(self, root, QtCore, QtGui, QtWidgets) -> Sequence:
        """Geometry-only controls (subclass implements)."""
        return ()

    def _mount_options(self, root, QtCore, QtGui, QtWidgets) -> Sequence:
        """Type-specific options bound to the persisted data model."""
        return ()

    def _mount_appearance(self, root, QtCore, QtGui, QtWidgets) -> Sequence:
        cfg = self._appearance_config()
        self._appearance = AppearanceSection(
            self._page,
            self.cmd,
            self.CONTEXT,
            show_wireframe=cfg.get("show_wireframe", True),
            show_quality=cfg.get("show_quality", False),
            show_per_point=cfg.get("show_per_point", True),
            show_live_preview=cfg.get("show_live_preview", True),
            live_preview_tooltip=cfg.get("live_preview_tooltip", ""),
            quality_range=cfg.get("quality_range", (1, 5)),
            quality_tooltip=cfg.get("quality_tooltip", ""),
            on_changed=self._on_appearance_changed,
            on_preview=self._on_appearance_preview,
        )
        self._appearance.bind_points(self._points)
        self._appearance.set_selected_rows_provider(self._selected_rows)
        root.addWidget(self._appearance.widget)
        return self._appearance.tooltips()

    def _mount_modifiers(self, root, QtCore, QtGui, QtWidgets) -> Sequence:
        self._modifiers = ModifiersSection(
            root,
            self.cmd,
            self.CONTEXT,
            self._preview,
            get_span_points=lambda: self._points,
            on_changed=self._on_modifiers_changed,
            page=self._page,
        )
        root.addWidget(self._modifiers.widget)
        return self._modifiers.tooltips()

    def _on_appearance_preview(self):
        self._refresh_color_cells()
        self._schedule_preview()

    def _on_appearance_changed(self):
        self._sync_table(preview=True)

    def _with_look(self, collection):
        if collection is None:
            return collection
        collection.specular = True
        return collection

    def _refresh_color_cells(self):
        return

    def _sync_color_selection_button(self):
        if self._appearance is None:
            return
        self._appearance.set_color_selection_enabled(bool(self._selected_rows()))

    def _on_modifiers_changed(self):
        if self._modifiers is not None:
            self._modifiers.refresh_summary()
            self._modifiers.clip.refresh_gizmos()
        self._schedule_preview()

    def _clip_planes(self):
        if self._modifiers is None:
            return []
        return self._modifiers.clip.active_planes()

    def _configure_table(self, QtCore, QtGui, QtWidgets):
        return

    def _mount_points_section(self, root, QtCore, QtGui, QtWidgets) -> Sequence:
        pts = make_section("Points", expanding=True)
        pts_box = pts.body
        pts_layout = pts.layout

        self._list = PointListEditor(pts_box, self)
        self._insertion = PointInsertionWidget(
            pts_box,
            self.cmd,
            self.CONTEXT,
            get_existing=lambda: self._points,
            on_add=self._add_resolved_points,
            on_show_coords=self._on_show_coords,
            on_export=self._export_selection,
            hide_add=True,
            hide_coords=True,
            on_can_add_changed=self._list.set_add_enabled,
            on_wait_changed=self._list.set_add_waiting,
        )
        pts_layout.addWidget(self._insertion.widget)
        pts_layout.addWidget(self._list.widget, stretch=1)
        self._list.attach_add_to_section(pts)
        root.addWidget(pts.widget, stretch=1)

        filter_type = _list_delete_filter_type(QtCore)
        self._table_filter = filter_type(self)
        self._list.widget.installEventFilter(self._table_filter)
        self._insertion.start_preview_timer(self._page)
        return list(self._insertion.tooltips())

    def _selected_rows(self) -> List[int]:
        if self._selected_index is None:
            return []
        if 0 <= self._selected_index < len(self._points):
            return [self._selected_index]
        return []

    def _apply_color_to_selected(self):
        if self._appearance is not None:
            self._appearance._color_selection()

    def _pick_selected_color(self):
        rows = self._selected_rows()
        if not rows:
            return
        self._pick_rows_color(rows)

    def _pick_color_at_row(self, row: int):
        rows = self._selected_rows()
        if row not in rows:
            rows = [row]
        self._pick_rows_color(rows)

    def _pick_rows_color(self, rows):
        if not rows:
            return
        initial = self._points[rows[0]].color_choice()
        targets = list(rows)
        original = {row: self._points[row].color_choice() for row in targets}

        def on_preview(choice):
            if choice is None:
                return
            for row in targets:
                if 0 <= row < len(self._points):
                    self._points[row] = self._points[row].with_color_choice(choice)
            self._schedule_preview()

        def apply_choice(choice):
            for row in targets:
                if 0 <= row < len(self._points):
                    self._points[row] = self._points[row].with_color_choice(choice)
            self._sync_table()

        def restore_originals():
            for row, saved in original.items():
                self._points[row] = self._points[row].with_color_choice(saved)
            self._sync_table()

        pick_rgb(
            self._page,
            initial,
            on_change=on_preview,
            on_done=bind_color_pick_result(apply_choice, restore_originals),
            cmd=self.cmd,
        )

    def add_point(self):
        if self._insertion is None:
            return
        self._insertion._clicked_add()

    def select_point(self, index: int):
        if index < 0 or index >= len(self._points):
            return
        if self._selected_index == index:
            self._selected_index = None
        else:
            self._selected_index = index
        self._sync_table(preview=False)
        if (
            self._selected_index is not None
            and self._insertion is not None
            and self._insertion.zoom_checkbox.isChecked()
        ):
            zoom_to_visual_points(self.cmd, [self._points[self._selected_index]])

    def delete_point(self, index: int):
        if index < 0 or index >= len(self._points):
            return
        self._remove_preview_rows([index])
        if getattr(self, "_pick_index", None) == index:
            self._clear_atom_pick()
        elif getattr(self, "_pick_index", None) is not None and self._pick_index > index:
            self._pick_index -= 1
        del self._points[index]
        if self._selected_index is None:
            pass
        elif self._selected_index == index:
            if self._points:
                self._selected_index = min(index, len(self._points) - 1)
            else:
                self._selected_index = None
        elif self._selected_index > index:
            self._selected_index -= 1
        self._sync_table(preview=self._preview_after_delete())

    def set_point_enabled(self, index: int, checked: bool):
        if index < 0 or index >= len(self._points):
            return
        self._points[index] = self._points[index].with_enabled(checked)
        self._schedule_preview()

    def set_point_xyz(self, index: int, xyz):
        if index < 0 or index >= len(self._points):
            return
        self._points[index] = self._points[index].with_xyz(xyz)
        self._schedule_preview()

    def pick_point_atom(self, index: int):
        if index < 0 or index >= len(self._points):
            return
        if self._pick_index == index:
            self._abort_atom_pick()
            return
        self._selected_index = index
        self._clear_pymol_selection()
        self._ignore_xyz = self._points[index].xyz()
        self._pick_index = index
        self._start_atom_pick_timer()
        self._sync_table(preview=False)

    def _abort_atom_pick(self):
        if self._pick_index is None:
            return
        self._clear_atom_pick()
        self._sync_table(preview=False)

    def _clear_atom_pick(self):
        self._pick_index = None
        self._ignore_xyz = None
        self._stop_atom_pick_timer()

    def _start_atom_pick_timer(self):
        if not qt_widget_alive(self._poll_timer):
            return
        try:
            self._poll_timer.start()
        except RuntimeError:
            pass

    def _stop_atom_pick_timer(self):
        if not qt_widget_alive(self._poll_timer):
            return
        try:
            self._poll_timer.stop()
        except RuntimeError:
            pass

    def _same_as_ignored(self, point: VisualPoint) -> bool:
        if self._ignore_xyz is None or point is None:
            return False
        dx = point.x - self._ignore_xyz[0]
        dy = point.y - self._ignore_xyz[1]
        dz = point.z - self._ignore_xyz[2]
        return (dx * dx + dy * dy + dz * dz) < 1e-6

    def _clear_pymol_selection(self):
        try:
            self.cmd.select("sele", "none")
        except Exception:
            pass
        try:
            self.cmd.unpick()
        except Exception:
            pass

    def _poll_atom_pick(self):
        if self._pick_index is None:
            return
        if not qt_widget_alive(self._page):
            self._stop_atom_pick_timer()
            return
        hook = self._insertion.hook_checkbox.isChecked() if self._insertion else True
        point, status = take_single_selection_point(
            self.cmd,
            self._points,
            interactive_only=True,
            hook_to_selection=hook,
            multi_atom=MULTI_CLICKED,
        )
        if status != "one" or point is None or self._same_as_ignored(point):
            return
        idx = self._pick_index
        if not (0 <= idx < len(self._points)):
            self._clear_atom_pick()
            return
        self._points[idx] = apply_location(self._points[idx], point)
        self._clear_pymol_selection()
        self._clear_atom_pick()
        self._sync_table()

    def camera_point(self, index: int, snap=None):
        self._selected_index = index
        if self._pick_index == index:
            self._clear_atom_pick()
        if snap is None:
            snap = self._insertion.snap_checkbox.isChecked() if self._insertion else False
        hook = self._insertion.hook_checkbox.isChecked() if self._insertion else True
        self._points = update_points_from_camera(
            self.cmd, self._points, [index], bool(snap), hook_to_selection=hook,
        )
        self._sync_table()

    def set_point_anchor(self, index: int, checked: bool):
        if index < 0 or index >= len(self._points):
            return
        pt = self._points[index]
        if not pt.can_anchor():
            return
        self._points[index] = pt.with_anchor_intent(checked)

    def edit_point_color(self, index: int):
        self._selected_index = index
        self._pick_color_at_row(index)

    def point_editor_extras(self, index: int, pt: VisualPoint) -> Sequence:
        return ()

    def _context_color_action(self) -> bool:
        return True

    def _extend_points_context_menu(self, menu, rows):
        """Keep surface/radius overflow hooks on the shared overflow menu."""
        reset = menu.addAction(RESET_COLORS_LABEL)
        reset.triggered.connect(self._reset_colors)
        apply_all = menu.addAction(COLOR_ALL_LABEL)
        apply_all.triggered.connect(lambda: self._appearance and self._appearance._color_all())
        apply_sel = menu.addAction(COLOR_SELECTION_LABEL)
        apply_sel.setEnabled(bool(rows))
        apply_sel.triggered.connect(self._apply_color_to_selected)

    def _stamp_new_points(self, new_pts: List[VisualPoint]) -> List[VisualPoint]:
        if self._appearance is not None:
            return self._appearance.stamp_new_points(new_pts)
        from .colors import colors_for_new_points
        palette = colors_for_new_points(len(new_pts), start_index=len(self._points))
        return [pt.with_color(palette[i]) for i, pt in enumerate(new_pts)]

    def _on_enabled_toggled(self, row: int, checked: bool):
        self.set_point_enabled(row, checked)

    def _on_anchor_toggled(self, row: int, checked: bool):
        self.set_point_anchor(row, checked)

    def _on_show_coords(self, checked=False):
        return

    def _on_table_sync_begin(self):
        """Called at the start of ``_sync_table`` (e.g. quality caps)."""

    def _extra_row_values(self, pt: VisualPoint) -> Sequence[Tuple[int, str]]:
        return ()

    def _sync_table(self, preview=True):
        self._on_table_sync_begin()
        if self._selected_index is not None and not (
            0 <= self._selected_index < len(self._points)
        ):
            self._selected_index = None
        if self._list is not None:
            self._list.rebuild(self._points, self._selected_index, getattr(self, "_pick_index", None))
        if self._appearance is not None:
            self._appearance.bind_points(self._points)
        self._sync_color_selection_button()
        self._sync_commit_enabled()
        if preview:
            self._schedule_preview()

    def _update_selected_to_camera(self):
        rows = self._selected_rows()
        if not rows:
            return
        snap = self._insertion.snap_checkbox.isChecked() if self._insertion else False
        hook = self._insertion.hook_checkbox.isChecked() if self._insertion else True
        self._points = update_points_from_camera(
            self.cmd, self._points, rows, snap, hook_to_selection=hook,
        )
        self._sync_table()

    def _update_selected_to_selection(self):
        rows = self._selected_rows()
        if not rows:
            return
        hook = self._insertion.hook_checkbox.isChecked() if self._insertion else True
        updated = update_points_from_selection(
            self.cmd, self._points, rows, hook_to_selection=hook,
        )
        if updated is None:
            self._warn_empty_pymol_selection("Pick atom")
            return
        self._points = updated
        self._sync_table()

    def _add_resolved_points(self, new_pts=None):
        if new_pts is None:
            if self._insertion is None:
                return
            new_pts = self._insertion.resolve_points(existing=self._points)
        if not new_pts:
            return
        new_pts = self._stamp_new_points(new_pts)
        start = len(self._points)
        self._points.extend(new_pts)
        self._selected_index = start + len(new_pts) - 1
        self._add_preview_points(new_pts)
        if self._insertion is not None and self._insertion.zoom_checkbox.isChecked():
            zoom_to_visual_points(self.cmd, new_pts)

    def _add_preview_points(self, new_pts):
        self._sync_table()

    def _reset_colors(self):
        if self._appearance is not None:
            self._appearance._reset_colors()
            self._sync_table()

    def _remove_preview_rows(self, rows):
        pass

    def _preview_after_delete(self) -> bool:
        return True

    def _delete_selected(self):
        rows = self._selected_rows()
        if not rows:
            return
        self._remove_preview_rows(rows)
        for row in sorted(rows, reverse=True):
            if 0 <= row < len(self._points):
                del self._points[row]
        if self._points:
            self._selected_index = min(rows[0], len(self._points) - 1)
        else:
            self._selected_index = None
        self._sync_table(preview=self._preview_after_delete())

    def _export_selection(self):
        export_points_to_selection(self.cmd, self._points)
