"""Shared points-table editor used by sphere, box, and surface pages."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from ..pick import qt_modules
from ..tooltips import UPDATE_TO_CAMERA_TIP, UPDATE_TO_SELECTION_TIP
from ..widgets.scrolling import apply_expanding_list_policy
from ..widgets.section import make_section
from .anchor_table import (
    POINT_ANCHOR_COL,
    POINT_COLOR_COL,
    POINT_ENABLED_COL,
    POINT_NAME_COL,
    POINT_SOURCE_COL,
    POINT_X_COL,
    POINT_Y_COL,
    POINT_Z_COL,
    anchor_col_index,
    block_table_selection_signals,
    configure_point_table,
    enabled_col_index,
    point_columns,
    set_coordinate_columns_visible,
    sync_anchor_cell,
    sync_color_cell,
    sync_enabled_cell,
    unblock_table_selection_signals,
    update_color_cell,
)
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
from .points import (
    VisualPoint,
    assign_distinct_colors,
    enabled_points,
    export_points_to_selection,
    hide_exported_point_labels,
    update_points_from_camera,
    update_points_from_selection,
)
from .zoom_selection import points_from_rows, wire_zoom_to_selection


def _table_delete_filter_type(QtCore):
    class _TableDeleteKeyFilter(QtCore.QObject):
        def __init__(self, owner):
            QtCore.QObject.__init__(self)
            self._owner = owner

        def eventFilter(self, obj, event):
            if event.type() != QtCore.QEvent.KeyPress:
                return False
            if event.key() not in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
                return False
            table = self._owner._table
            _, _, QtWidgets = qt_modules()
            if QtWidgets is not None and table.state() == QtWidgets.QAbstractItemView.EditingState:
                return False
            self._owner._delete_selected()
            return True

    return _TableDeleteKeyFilter


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
        self._appearance = None
        self._insertion = None
        self._modifiers = None
        self._table_filter = None
        self._preview = self._make_preview()
        self._init_options_state()

    def _make_preview(self):
        raise NotImplementedError

    def _init_options_state(self):
        """Subclass fields that must exist before ``_build``."""

    def _cleanup_ephemeral(self):
        hide_exported_point_labels(self.cmd)
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
        if self._appearance is not None:
            self._appearance.bind_points(self._points)
        if self._modifiers is not None:
            self._modifiers.clip.reset()
            self._modifiers.refresh_summary()
        if self._table is not None:
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
            if self._appearance is not None:
                self._appearance.bind_points(self._points)
            if self._table is not None:
                self._sync_table()
        finally:
            self._suspend_preview = False
        self._preview.adopt(obj)
        self._load_clip_planes(obj)
        self._after_load()

    def _reset_options(self):
        """Restore option widgets to create-mode defaults."""

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
        tips = []
        tips.extend(self._mount_geometry(root, QtCore, QtGui, QtWidgets) or ())
        tips.extend(self._mount_appearance(root, QtCore, QtGui, QtWidgets) or ())
        tips.extend(self._mount_points_section(root, QtCore, QtGui, QtWidgets) or ())
        tips.extend(self._mount_modifiers(root, QtCore, QtGui, QtWidgets) or ())
        self._mount_action_bar(page, root)
        self._finish_build(page, back, tips)

    def _mount_geometry(self, root, QtCore, QtGui, QtWidgets) -> Sequence:
        """Geometry-only controls (subclass implements)."""
        return self._mount_options(root, QtCore, QtGui, QtWidgets)

    def _mount_options(self, root, QtCore, QtGui, QtWidgets) -> Sequence:
        return ()

    def _mount_appearance(self, root, QtCore, QtGui, QtWidgets) -> Sequence:
        cfg = self._appearance_config()
        self._appearance = AppearanceSection(
            self._page,
            self.cmd,
            self.CONTEXT,
            show_wireframe=cfg.get("show_wireframe", True),
            show_quality=cfg.get("show_quality", False),
            show_specular=cfg.get("show_specular", True),
            show_per_point=cfg.get("show_per_point", True),
            show_live_preview=cfg.get("show_live_preview", False),
            live_preview_tooltip=cfg.get("live_preview_tooltip", ""),
            quality_range=cfg.get("quality_range", (1, 5)),
            quality_tooltip=cfg.get("quality_tooltip", ""),
            on_changed=self._on_appearance_changed,
            on_preview=self._on_appearance_preview,
            on_look_changed=self._on_look_changed,
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

    def _specular_enabled(self) -> bool:
        if self._appearance is None:
            return True
        return self._appearance.specular()

    def _on_look_changed(self):
        if self._preview is not None and hasattr(self._preview, "set_specular"):
            self._preview.set_specular(self._specular_enabled())

    def _with_look(self, collection):
        if collection is None:
            return collection
        collection.specular = self._specular_enabled()
        return collection

    def _refresh_color_cells(self):
        if self._table is None:
            return
        _, _, QtWidgets = qt_modules()
        for row, pt in enumerate(self._points):
            update_color_cell(
                self._table, row, POINT_COLOR_COL, pt, QtWidgets,
            )

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
        if self.TABLE_STRONG_FOCUS:
            self._table.setFocusPolicy(QtCore.Qt.StrongFocus)

    def _mount_points_section(self, root, QtCore, QtGui, QtWidgets) -> Sequence:
        pts = make_section("Points", expanding=True)
        pts_box = pts.body
        pts_layout = pts.layout

        self._insertion = PointInsertionWidget(
            pts_box,
            self.cmd,
            self.CONTEXT,
            get_existing=lambda: self._points,
            on_add=self._add_resolved_points,
            on_show_coords=self._on_show_coords,
            on_export=self._export_selection,
        )
        pts_layout.addWidget(self._insertion.widget)

        self._table = QtWidgets.QTableWidget(0, len(self.COLS))
        configure_point_table(self._table, self.COLS, QtWidgets, QtCore)
        apply_expanding_list_policy(self._table, QtWidgets)
        self._configure_table(QtCore, QtGui, QtWidgets)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._table.cellChanged.connect(self._on_cell_changed)
        self._table.itemSelectionChanged.connect(self._sync_color_selection_button)
        self._table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._show_points_context_menu)
        wire_zoom_to_selection(
            self._table,
            self._insertion.zoom_checkbox,
            self.cmd,
            lambda rows: points_from_rows(self._points, rows),
        )

        filter_type = _table_delete_filter_type(QtCore)
        self._table_filter = filter_type(self)
        self._table.installEventFilter(self._table_filter)
        self._table.viewport().installEventFilter(self._table_filter)
        pts_layout.addWidget(self._table, stretch=1)
        root.addWidget(pts.widget, stretch=1)
        self._insertion.start_preview_timer(self._page)
        tips = list(self._insertion.tooltips())
        return tips

    def _selected_rows(self) -> List[int]:
        return sorted({i.row() for i in self._table.selectedIndexes()})

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
        _, _, QtWidgets = qt_modules()

        def on_preview(choice):
            if choice is None:
                return
            for row in targets:
                if 0 <= row < len(self._points):
                    self._points[row] = self._points[row].with_color_choice(choice)
            self._schedule_preview()
            for row in targets:
                if 0 <= row < len(self._points):
                    update_color_cell(
                        self._table, row, POINT_COLOR_COL, self._points[row], QtWidgets,
                    )

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

    def _context_color_action(self) -> bool:
        return True

    def _extend_points_context_menu(self, menu, rows):
        """Add mesh-specific actions after the shared camera/selection/delete items."""
        reset = menu.addAction(RESET_COLORS_LABEL)
        reset.triggered.connect(self._reset_colors)
        apply_all = menu.addAction(COLOR_ALL_LABEL)
        apply_all.triggered.connect(lambda: self._appearance and self._appearance._color_all())
        apply_sel = menu.addAction(COLOR_SELECTION_LABEL)
        apply_sel.setEnabled(bool(rows))
        apply_sel.triggered.connect(self._apply_color_to_selected)

    def _show_points_context_menu(self, pos):
        _, _, QtWidgets = qt_modules()
        index = self._table.indexAt(pos)
        if not index.isValid():
            return
        row = index.row()
        if row not in self._selected_rows():
            self._table.selectRow(row)

        rows = self._selected_rows()
        menu = QtWidgets.QMenu(self._table)
        if self._context_color_action():
            color_act = menu.addAction("Color selection…")
            color_act.setEnabled(bool(rows))
            color_act.triggered.connect(self._pick_selected_color)
        cam_act = menu.addAction("Update to camera center")
        cam_act.setToolTip(UPDATE_TO_CAMERA_TIP)
        cam_act.setEnabled(bool(rows))
        cam_act.triggered.connect(self._update_selected_to_camera)
        sel_act = menu.addAction("Update to selection")
        sel_act.setToolTip(UPDATE_TO_SELECTION_TIP)
        sel_act.setEnabled(bool(rows))
        sel_act.triggered.connect(self._update_selected_to_selection)
        del_act = menu.addAction("Delete selected")
        del_act.setEnabled(bool(rows))
        del_act.triggered.connect(self._delete_selected)
        self._extend_points_context_menu(menu, rows)
        menu.exec_(self._table.viewport().mapToGlobal(pos))

    def _stamp_new_points(self, new_pts: List[VisualPoint]) -> List[VisualPoint]:
        if self._appearance is not None:
            return self._appearance.stamp_new_points(new_pts)
        from .colors import colors_for_new_points
        palette = colors_for_new_points(len(new_pts), start_index=len(self._points))
        return [pt.with_color(palette[i]) for i, pt in enumerate(new_pts)]

    def _anchor_col(self) -> int:
        return anchor_col_index(self.COLS)

    def _enabled_col(self) -> int:
        return enabled_col_index(self.COLS)

    def _on_enabled_toggled(self, row: int, checked: bool):
        if row < 0 or row >= len(self._points):
            return
        self._points[row] = self._points[row].with_enabled(checked)
        self._sync_table()

    def _on_anchor_toggled(self, row: int, checked: bool):
        if row < 0 or row >= len(self._points):
            return
        pt = self._points[row]
        if not pt.can_anchor():
            return
        self._points[row] = pt.with_anchor_intent(checked)

    def _on_show_coords(self, checked=False):
        if self._table is None:
            return
        set_coordinate_columns_visible(self._table, bool(checked))

    def _on_table_sync_begin(self):
        """Called at the start of ``_sync_table`` (e.g. quality caps)."""

    def _extra_row_values(self, pt: VisualPoint) -> Sequence[Tuple[int, str]]:
        return ()

    def _style_cell(self, item, col: int, pt: VisualPoint):
        """Optional per-cell styling after the text is written."""

    def _sync_table(self, preview=True):
        self._on_table_sync_begin()
        QtCore, QtGui, QtWidgets = qt_modules()
        anchor_col = self._anchor_col()
        enabled_col = self._enabled_col()
        sel_blocked = block_table_selection_signals(self._table)
        self._table.blockSignals(True)
        try:
            self._table.setRowCount(len(self._points))
            for row, pt in enumerate(self._points):
                sync_enabled_cell(
                    self._table, row, enabled_col, pt,
                    self._on_enabled_toggled, QtWidgets, QtCore,
                )
                sync_anchor_cell(
                    self._table, row, anchor_col, pt,
                    self._on_anchor_toggled, QtWidgets, QtCore,
                )
                sync_color_cell(
                    self._table, row, POINT_COLOR_COL, pt,
                    self._pick_color_at_row, QtWidgets, QtCore,
                )
                values = (
                    (POINT_NAME_COL, pt.name),
                    (POINT_SOURCE_COL, pt.source),
                    (POINT_X_COL, "%.3f" % pt.x),
                    (POINT_Y_COL, "%.3f" % pt.y),
                    (POINT_Z_COL, "%.3f" % pt.z),
                ) + tuple(self._extra_row_values(pt))
                for col, text in values:
                    item = self._table.item(row, col)
                    if item is None:
                        item = QtWidgets.QTableWidgetItem()
                        self._table.setItem(row, col, item)
                    item.setText(text)
                    item.setBackground(QtGui.QBrush())
                    item.setForeground(QtGui.QBrush())
                    self._style_cell(item, col, pt)
        finally:
            self._table.blockSignals(False)
            unblock_table_selection_signals(self._table, sel_blocked)
        if self._appearance is not None:
            self._appearance.bind_points(self._points)
        self._sync_color_selection_button()
        self._sync_commit_enabled()
        if preview:
            self._schedule_preview()

    def _apply_extra_cell(self, pt: VisualPoint, col: int, text: str):
        return None

    def _on_cell_changed(self, row, col):
        if row < 0 or row >= len(self._points):
            return
        item = self._table.item(row, col)
        if item is None:
            return
        text = item.text()
        pt = self._points[row]
        if col in (self._anchor_col(), self._enabled_col(), POINT_COLOR_COL):
            return
        resync = False
        try:
            if col == POINT_NAME_COL:
                pt = pt.with_name(text)
            elif col == POINT_SOURCE_COL:
                pt = pt.with_source(text)
            elif col == POINT_X_COL:
                pt = pt.with_xyz((float(text), pt.y, pt.z))
            elif col == POINT_Y_COL:
                pt = pt.with_xyz((pt.x, float(text), pt.z))
            elif col == POINT_Z_COL:
                pt = pt.with_xyz((pt.x, pt.y, float(text)))
            else:
                extra = self._apply_extra_cell(pt, col, text)
                if extra is None:
                    return
                pt, resync = extra
            self._points[row] = pt
        except ValueError:
            self._sync_table()
            return
        if resync:
            self._sync_table()
            return
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
            self._warn_empty_pymol_selection("Update to selection")
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
        self._points.extend(new_pts)
        self._add_preview_points(new_pts)
        if self._insertion is not None and self._insertion.zoom_checkbox.isChecked():
            from .zoom_selection import zoom_to_visual_points
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
        rows = sorted({i.row() for i in self._table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        self._remove_preview_rows(rows)
        for row in rows:
            if 0 <= row < len(self._points):
                del self._points[row]
        self._sync_table(preview=self._preview_after_delete())

    def _export_selection(self):
        export_points_to_selection(self.cmd, self._points)
