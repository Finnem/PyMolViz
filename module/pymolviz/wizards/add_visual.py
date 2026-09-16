"""Standalone window for listing and adding visuals (CGOs)."""

from .builders.arrow_page import ArrowBuilderPage
from .builders.box_page import BoxBuilderPage
from .builders.points import hide_exported_point_labels
from .builders.preview import delete_visual, set_visual_enabled, visual_is_enabled
from .builders.sphere_page import SphereBuilderPage
from .builders.surface_page import SurfaceBuilderPage
from .catalog import editor_kind, object_rows, type_card_icon_rgb
from .pick import (
    bind_tool_window,
    configure_tool_window,
    overlay_information,
    overlay_question,
    overlay_warning,
    qt_modules,
)
from .pick import _qt_platform_name
from .tooltips import apply_required_tooltips
from .widgets.breadcrumb import (
    CRUMB_ADD_OBJECT,
    CRUMB_VISUALS,
    make_page_header,
)
from .widgets.section import make_section
from .widgets.scrolling import (
    WINDOW_DEFAULT_HEIGHT,
    WINDOW_DEFAULT_WIDTH,
    apply_expanding_list_policy,
    configure_resizable_window,
    make_scrolling_body,
)
from .widgets.catalog_chrome import (
    LIBRARY_ROW_MIN_HEIGHT,
    cell_band_css,
    make_kind_cell,
    make_lib_cell,
    make_row_icon_button,
)
from .widgets.sticky_add import (
    StickyAddOverlay,
    list_needs_sticky_add,
    sticky_add_overlay_rect,
)
from .widgets.theme import (
    apply_catalog_table_style,
    apply_page_layout,
    apply_type_card_style,
    apply_wizard_page_style,
    empty_title_css,
    mark_primary_button,
    muted_label_css,
    page_heading_css,
    selected_row_fill,
    type_card_subtitle_css,
    type_card_title_css,
)
from .widgets.switch import make_switch
from .widgets.catalog_window import (
    build_empty_library_page,
    build_type_picker_page,
    open_catalog_dialog,
)
from .widgets.type_icons import type_icon_pixmap


def objects_need_sticky_add(n_objects, viewport_height, row_height, add_height=None):
    return list_needs_sticky_add(n_objects, viewport_height, row_height, add_height=add_height)


def add_object_overlay_rect(
    table_width,
    table_height,
    viewport_x,
    viewport_y,
    viewport_height,
    n_objects,
    row_height,
    add_height,
    header_height=0,
    h_scrollbar_height=0,
    inset=6,
):
    return sticky_add_overlay_rect(
        table_width,
        table_height,
        viewport_x,
        viewport_y,
        viewport_height,
        n_objects,
        row_height,
        add_height,
        header_height=header_height,
        h_scrollbar_height=h_scrollbar_height,
        inset=inset,
    )

MESH_TYPES = (
    ("Spheres", "Sphere", "Points rendered as solid or wireframe spheres.", "sphere"),
    ("Boxes", "Box", "Axis-aligned or centered boxes around points.", "cube"),
    ("Surface", "Surface", "Gaussian, marching-cubes, or solvent-accessible surfaces.", "surface"),
    ("Arrows", "Arrows", "Arrows or lines between points, solid or dashed.", "arrow"),
)

ADD_VISUAL_LABEL = "Add Visual"
ADD_VISUAL_BUTTON = "+ Add Visual"
ADD_VISUAL_TIP = "Create a new visual."
OBJECT_COLUMNS = ("Name", "Type", "Visible", "Actions")
EMPTY_LIBRARY_TITLE = "No visual objects yet"
EMPTY_LIBRARY_HINT = "Add spheres, boxes, surfaces, or arrows"
_COL_NAME = 0
_COL_TYPE = 1
_COL_VISIBLE = 2
_COL_ACTIONS = 3
_PAGE_LIBRARY = 0
_PAGE_TYPES = 1
_LIBRARY_EMPTY = 0
_LIBRARY_LIST = 1
_ADD_ROW_ID = "__pmv_add_object__"
_EDIT_BUTTON_TEXT = "..."


def library_shows_empty_state(n_objects):
    return int(n_objects) <= 0


def _swatch_icon(QtGui, rgb):
    if QtGui is None or rgb is None or len(rgb) < 3:
        return None
    try:
        r, g, b = [max(0, min(255, int(round(float(c) * 255.0)))) for c in rgb[:3]]
    except (TypeError, ValueError):
        return None
    pix = QtGui.QPixmap(12, 12)
    pix.fill(QtGui.QColor(r, g, b))
    return QtGui.QIcon(pix)


class AddVisualWindow:
    """Owns a Qt window popped out from the wizard panel."""

    def __init__(self, wizard):
        self.wizard = wizard
        self._window = None
        self._stack = None
        self._mesh_choice = None
        self._sphere_page = None
        self._box_page = None
        self._arrow_page = None
        self._surface_page = None
        self._objects_table = None
        self._library_body = None
        self._add_object_overlay = None
        self._object_count = 0
        self._editing_obj = None
        self._sphere_stack_index = None
        self._box_stack_index = None
        self._arrow_stack_index = None
        self._surface_stack_index = None

    def show(self):
        QtCore, _, QtWidgets = qt_modules()
        if QtWidgets is None:
            self.wizard.prompt = ["Open 3D Objects Menu requires the PyMOL Qt UI"]
            return

        self._discard_window()

        try:
            self._open_window(QtCore, QtWidgets)
        except Exception as exc:
            self._reset_window()
            self.wizard.prompt = ["Open 3D Objects Menu failed: %s" % exc]
            try:
                overlay_warning(
                    None,
                    "PyMOLViz",
                    "Could not open Visuals:\n\n%s" % exc,
                )
            except Exception:
                pass

    def _discard_window(self):
        self._restore_editing_visual()
        hide_exported_point_labels(self.wizard.cmd)
        window = self._window
        self._reset_window()
        if window is None:
            return
        try:
            window.close()
            window.deleteLater()
        except RuntimeError:
            pass

    def _open_window(self, QtCore, QtWidgets):
        def _pages(stack):
            stack.addWidget(self._build_library_page(QtCore, QtWidgets))
            stack.addWidget(self._build_type_page(QtCore, QtWidgets))
            stack.setCurrentIndex(_PAGE_LIBRARY)

        window, stack = open_catalog_dialog(
            QtCore,
            QtWidgets,
            window_title="PyMOLViz Visuals",
            width=WINDOW_DEFAULT_WIDTH,
            height=WINDOW_DEFAULT_HEIGHT,
            build_root_pages=_pages,
            on_destroyed=self._on_destroyed,
        )
        self._stack = stack
        self._window = window
        self._refresh_objects_table()
        window.show()
        bind_tool_window(window)
        window.raise_()
        window.activateWindow()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents()
        self._sync_add_object_row()
        if not window.isVisible():
            raise RuntimeError(
                "Visuals window did not become visible "
                "(Qt platform=%r)" % (_qt_platform_name(),)
            )

    def _reset_window(self):
        self._window = None
        self._stack = None
        self._sphere_page = None
        self._box_page = None
        self._arrow_page = None
        self._surface_page = None
        self._objects_table = None
        self._library_body = None
        self._add_object_overlay = None
        self._object_count = 0
        self._editing_obj = None
        self._sphere_stack_index = None
        self._box_stack_index = None
        self._arrow_stack_index = None
        self._surface_stack_index = None

    def _ensure_sphere_page(self):
        if self._sphere_page is not None:
            return
        parent = self._window
        self._sphere_page = SphereBuilderPage(
            self.wizard.cmd,
            on_back=self._on_builder_back,
            on_create=self._on_builder_saved,
            parent=parent,
        )
        self._sphere_stack_index = self._stack.addWidget(self._sphere_page.widget)

    def _ensure_box_page(self):
        if self._box_page is not None:
            return
        parent = self._window
        self._box_page = BoxBuilderPage(
            self.wizard.cmd,
            on_back=self._on_builder_back,
            on_create=self._on_builder_saved,
            parent=parent,
        )
        self._box_stack_index = self._stack.addWidget(self._box_page.widget)

    def _ensure_arrow_page(self):
        if self._arrow_page is not None:
            return
        parent = self._window
        self._arrow_page = ArrowBuilderPage(
            self.wizard.cmd,
            on_back=self._on_builder_back,
            on_create=self._on_builder_saved,
            parent=parent,
        )
        self._arrow_stack_index = self._stack.addWidget(self._arrow_page.widget)

    def _ensure_surface_page(self):
        if self._surface_page is not None:
            return
        parent = self._window
        self._surface_page = SurfaceBuilderPage(
            self.wizard.cmd,
            on_back=self._on_builder_back,
            on_create=self._on_builder_saved,
            parent=parent,
        )
        self._surface_stack_index = self._stack.addWidget(self._surface_page.widget)

    def _build_library_page(self, QtCore, QtWidgets):
        page = QtWidgets.QWidget()
        apply_wizard_page_style(page)
        layout = QtWidgets.QVBoxLayout(page)
        apply_page_layout(layout)

        title = QtWidgets.QLabel("<b>Visuals</b>")
        title.setStyleSheet(page_heading_css())
        subtitle = QtWidgets.QLabel("Visual objects in this PyMOL session.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(muted_label_css())

        group = make_section("Objects", expanding=True)
        group_layout = group.layout
        group_layout.setContentsMargins(8, 8, 8, 8)
        group_layout.setSpacing(0)
        body = QtWidgets.QStackedWidget()
        self._library_body = body

        table_page = QtWidgets.QWidget()
        table_layout = QtWidgets.QVBoxLayout(table_page)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(0)
        table = QtWidgets.QTableWidget(0, len(OBJECT_COLUMNS))
        table.setHorizontalHeaderLabels(list(OBJECT_COLUMNS))
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        table.setAlternatingRowColors(False)
        table.setFrameShape(getattr(QtWidgets.QFrame, "NoFrame", 0))
        table.verticalHeader().setDefaultSectionSize(LIBRARY_ROW_MIN_HEIGHT)
        table.verticalHeader().setMinimumSectionSize(LIBRARY_ROW_MIN_HEIGHT)
        header = table.horizontalHeader()
        header.setVisible(True)
        header.setStretchLastSection(False)
        header.setHighlightSections(False)
        header.setSectionResizeMode(_COL_NAME, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(_COL_TYPE, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(_COL_VISIBLE, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(_COL_ACTIONS, QtWidgets.QHeaderView.ResizeToContents)
        apply_catalog_table_style(table)
        align_left = getattr(QtCore.Qt, "AlignLeft", None)
        vcenter = getattr(QtCore.Qt, "AlignVCenter", None)
        if align_left is not None and vcenter is not None:
            header.setDefaultAlignment(align_left | vcenter)
        table.cellClicked.connect(self._on_object_clicked)
        table.cellDoubleClicked.connect(self._on_object_activated)
        table.itemSelectionChanged.connect(self._sync_row_bands)
        table.setToolTip("Double-click a row to adjust that CGO.")
        table_layout.addWidget(table, stretch=1)
        apply_expanding_list_policy(table, QtWidgets)
        apply_expanding_list_policy(body, QtWidgets)
        self._objects_table = table
        self._add_object_overlay = StickyAddOverlay(
            table_page,
            table,
            text=ADD_VISUAL_BUTTON,
            tooltip=ADD_VISUAL_TIP,
            on_click=lambda: self._goto(_PAGE_TYPES),
            count=lambda: int(self._object_count),
            row_height=lambda: self._row_height(),
            add_height=lambda: self._add_button_height(),
            context="AddVisualWindow",
        )
        self._add_object_overlay.attach()

        body.addWidget(
            build_empty_library_page(
                QtCore,
                QtWidgets,
                empty_title=EMPTY_LIBRARY_TITLE,
                empty_hint=EMPTY_LIBRARY_HINT,
                add_button_text=ADD_VISUAL_BUTTON,
                add_tip=ADD_VISUAL_TIP,
                on_add=lambda: self._goto(_PAGE_TYPES),
                tooltip_context="AddVisualWindow",
                style_add_button=mark_primary_button,
            )
        )
        body.addWidget(table_page)
        body.setCurrentIndex(_LIBRARY_EMPTY)
        group_layout.addWidget(body, stretch=1)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        scroll, body_layout = make_scrolling_body(page)
        body_layout.addWidget(group.widget, stretch=1)
        layout.addWidget(scroll, stretch=1)
        return page

    def _build_type_page(self, QtCore, QtWidgets):
        return build_type_picker_page(
            QtCore,
            QtWidgets,
            title_parts=(CRUMB_VISUALS, CRUMB_ADD_OBJECT),
            back_tip="Return to the visual list.",
            subtitle="Choose a visual type.",
            entries=MESH_TYPES,
            on_pick=self._on_mesh_type,
            on_back=lambda: self._goto(_PAGE_LIBRARY),
            tooltip_context="AddVisualWindow",
            icon_rgb_fn=type_card_icon_rgb,
        )

    def _refresh_objects_table(self):
        table = self._objects_table
        if table is None:
            return
        from ..runtime.presence import sync_session_with_pymol
        from ..runtime.session import all_objects

        QtCore, QtGui, QtWidgets = qt_modules()
        sync_session_with_pymol(self.wizard.cmd)
        rows = object_rows(all_objects())
        self._object_count = len(rows)
        table.clearSpans()
        table.setRowCount(0)
        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            self._fill_object_row(table, i, row, QtCore, QtGui, QtWidgets)
        if rows:
            table.resizeRowsToContents()
            for i in range(table.rowCount()):
                if table.rowHeight(i) < LIBRARY_ROW_MIN_HEIGHT:
                    table.setRowHeight(i, LIBRARY_ROW_MIN_HEIGHT)
        self._sync_row_bands()
        self._sync_add_object_row()

    def _fill_object_row(self, table, index, row, QtCore, QtGui, QtWidgets):
        selected = table.currentRow() == index
        band = cell_band_css(selected_row_fill(selected=selected))
        name_item = QtWidgets.QTableWidgetItem(str(row["name"]))
        name_item.setData(QtCore.Qt.UserRole, row["id"])
        icon = _swatch_icon(QtGui, row.get("color"))
        if icon is not None:
            name_item.setIcon(icon)
        tip_parts = [row["type"]]
        if row.get("n_points"):
            tip_parts.append("%s points" % row["n_points"])
        if row.get("com"):
            tip_parts.append("COM %s" % row["com"])
        name_item.setToolTip(" · ".join(tip_parts))
        table.setItem(index, _COL_NAME, name_item)
        table.setCellWidget(
            index,
            _COL_TYPE,
            make_kind_cell(
                QtCore,
                QtWidgets,
                row.get("type"),
                band,
                min_height=LIBRARY_ROW_MIN_HEIGHT,
            ),
        )

        obj = self._session_object(row["id"])
        visible = True if obj is None else visual_is_enabled(self.wizard.cmd, obj)
        table.setCellWidget(
            index,
            _COL_VISIBLE,
            self._visibility_cell(QtCore, QtWidgets, row["id"], visible, band),
        )
        table.setCellWidget(
            index,
            _COL_ACTIONS,
            self._actions_cell(
                QtWidgets, row["id"], row["name"], row.get("editor"), band,
            ),
        )

    def _sync_row_bands(self):
        table = self._objects_table
        if table is None:
            return
        try:
            current = int(table.currentRow())
        except RuntimeError:
            return
        for index in range(table.rowCount()):
            band = cell_band_css(selected_row_fill(selected=(index == current)))
            for col in (_COL_TYPE, _COL_VISIBLE, _COL_ACTIONS):
                widget = table.cellWidget(index, col)
                if widget is None:
                    continue
                try:
                    widget.setStyleSheet(band)
                except RuntimeError:
                    pass

    def _visibility_cell(self, QtCore, QtWidgets, obj_id, checked, band):
        wrap = make_lib_cell(QtWidgets, band, min_height=LIBRARY_ROW_MIN_HEIGHT)
        layout = QtWidgets.QHBoxLayout(wrap)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        checkbox = make_switch(compact=True)
        checkbox.setChecked(bool(checked))
        checkbox.toggled.connect(
            lambda on, oid=obj_id: self._on_visibility_toggled(oid, on)
        )
        apply_required_tooltips(
            [(checkbox, "Show or hide this object in the viewer.", "Visible")],
            context="AddVisualWindow",
        )
        layout.addWidget(checkbox)
        return wrap

    def _actions_cell(self, QtWidgets, obj_id, name, editor, band):
        wrap = make_lib_cell(QtWidgets, band, min_height=LIBRARY_ROW_MIN_HEIGHT)
        layout = QtWidgets.QHBoxLayout(wrap)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        edit = make_row_icon_button(
            QtWidgets,
            _EDIT_BUTTON_TEXT,
            "Open the builder for this object.",
            "Edit",
            lambda *_a, oid=obj_id: self._edit_object_id(oid),
            "AddVisualWindow",
        )
        edit.setEnabled(bool(editor))
        delete = make_row_icon_button(
            QtWidgets,
            "",
            "Remove this object from the session.",
            "Delete",
            lambda *_a, oid=obj_id, label=name: self._on_delete_object(oid, label),
            "AddVisualWindow",
            icon="trash",
        )
        layout.addWidget(edit)
        layout.addWidget(delete)
        layout.addStretch(1)
        return wrap

    def _row_height(self):
        table = self._objects_table
        if table is None:
            return 0
        if table.rowCount():
            height = table.rowHeight(0)
            if height > 0:
                return height
        return max(int(table.verticalHeader().defaultSectionSize()), 1)

    def _row_marker(self, row):
        table = self._objects_table
        if table is None:
            return None
        item = table.item(row, _COL_NAME)
        if item is None:
            return None
        QtCore, _, _ = qt_modules()
        return item.data(QtCore.Qt.UserRole)

    def _add_button_height(self):
        return max(self._row_height() + 8, 30)

    def _sync_add_object_row(self):
        overlay = self._add_object_overlay
        table = self._objects_table
        body = self._library_body
        empty = library_shows_empty_state(self._object_count)
        if body is not None:
            body.setCurrentIndex(_LIBRARY_EMPTY if empty else _LIBRARY_LIST)
        if overlay is None or table is None:
            return
        if empty:
            try:
                overlay.widget.hide()
            except RuntimeError:
                pass
            return
        n_objects = int(self._object_count)
        if table.rowCount() != n_objects:
            table.clearSpans()
            table.setRowCount(n_objects)
        overlay.sync()

    def _on_object_clicked(self, row, _column):
        if self._row_marker(row) == _ADD_ROW_ID:
            self._goto(_PAGE_TYPES)

    def _session_object(self, obj_id):
        from ..runtime.session import get as session_get

        return session_get(obj_id)

    def _on_visibility_toggled(self, obj_id, checked):
        obj = self._session_object(obj_id)
        if obj is None:
            return
        set_visual_enabled(self.wizard.cmd, obj, bool(checked))

    def _on_delete_object(self, obj_id, name):
        obj = self._session_object(obj_id)
        if obj is None:
            return
        if not self._confirm_delete(name or "this object"):
            return
        delete_visual(self.wizard.cmd, obj)
        self._refresh_objects_table()

    def _confirm_delete(self, name):
        _, _, QtWidgets = qt_modules()
        if QtWidgets is None:
            return True
        result = overlay_question(
            self._window,
            "Delete object",
            "Delete %s from this session?" % name,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        return result == QtWidgets.QMessageBox.Yes

    def _edit_object_id(self, obj_id):
        obj = self._session_object(obj_id)
        if obj is None:
            return
        self._open_object_editor(obj)

    def _on_object_activated(self, row, column):
        if column in (_COL_VISIBLE, _COL_ACTIONS):
            return
        marker = self._row_marker(row)
        if marker == _ADD_ROW_ID:
            self._goto(_PAGE_TYPES)
            return
        obj = self._session_object(marker)
        if obj is None:
            return
        self._open_object_editor(obj)

    def _open_object_editor(self, obj):
        kind = editor_kind(obj)
        _, _, QtWidgets = qt_modules()
        if not kind:
            if QtWidgets is not None:
                overlay_information(
                    self._window,
                    "Edit object",
                    "No editor for %s yet." % type(obj).__name__,
                )
            return
        self._open_editor(kind, obj)

    def _open_editor(self, kind, obj=None):
        try:
            if kind == "Sphere":
                self._ensure_sphere_page()
                page = self._sphere_page
                index = self._sphere_stack_index
            elif kind == "Box":
                self._ensure_box_page()
                page = self._box_page
                index = self._box_stack_index
            elif kind == "Arrows":
                self._ensure_arrow_page()
                page = self._arrow_page
                index = self._arrow_stack_index
            elif kind == "Surface":
                self._ensure_surface_page()
                page = self._surface_page
                index = self._surface_stack_index
            else:
                return
        except Exception as exc:
            self.wizard.prompt = ["Mesh builder failed: %s" % exc]
            _, _, QtWidgets = qt_modules()
            if QtWidgets is not None:
                overlay_warning(
                    self._window,
                    "PyMOLViz",
                    "Could not open %s builder:\n\n%s" % (kind, exc),
                )
            return
        if obj is None:
            self._editing_obj = None
            page.reset_for_create()
        else:
            self._editing_obj = obj
            set_visual_enabled(self.wizard.cmd, obj, False)
            page.load_object(obj)
        self._goto(index)

    def _restore_editing_visual(self):
        obj = self._editing_obj
        self._editing_obj = None
        if obj is None:
            return
        set_visual_enabled(self.wizard.cmd, obj, True)

    def _on_builder_back(self):
        if self._editing_obj is not None:
            self._restore_editing_visual()
            self._refresh_objects_table()
            self._goto(_PAGE_LIBRARY)
            return
        self._goto(_PAGE_TYPES)

    def _on_builder_saved(self):
        hide_exported_point_labels(self.wizard.cmd)
        self._editing_obj = None
        self._refresh_objects_table()
        self._goto(_PAGE_LIBRARY)

    def _goto(self, index):
        if self._stack is not None:
            self._stack.setCurrentIndex(index)
        if index == _PAGE_LIBRARY:
            self._refresh_objects_table()

    def _on_mesh_type(self, name, kind=None):
        self._mesh_choice = name
        self.wizard.prompt = ["Mesh: %s" % name]
        try:
            self.wizard.cmd.refresh_wizard()
        except Exception:
            pass
        if kind:
            self._open_editor(kind, obj=None)
            return
        # Other mesh types remain stubs on the mesh list page.

    def _on_destroyed(self, *_args):
        self._restore_editing_visual()
        self._cleanup_builder_previews()
        self._reset_window()

    def _cleanup_builder_previews(self):
        hide_exported_point_labels(self.wizard.cmd)
        if self._sphere_page is not None:
            self._sphere_page.cleanup_preview()
        if self._box_page is not None:
            self._box_page.cleanup_preview()
        if self._arrow_page is not None:
            self._arrow_page.cleanup_preview()
        if self._surface_page is not None:
            self._surface_page.cleanup_preview()

    def close(self):
        self._restore_editing_visual()
        self._cleanup_builder_previews()
        window = self._window
        self._reset_window()
        if window is None:
            return
        try:
            window.close()
        except RuntimeError:
            pass
