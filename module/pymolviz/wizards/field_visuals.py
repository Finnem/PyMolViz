"""Standalone window for fields and their volumetric visuals."""

from .builders.field_page import FieldVisualBuilderPage
from .builders.field_visual import (
    delete_field_and_visuals,
    delete_field_visual,
    pymol_name_for,
)
from .builders.from_selection_page import FromSelectionFieldPage
from .builders.load_field import KIND_DERIVED, KIND_FROM_SELECTION, KIND_IMPLICIT, load_field_file
from .builders.preview import set_visual_enabled, visual_is_enabled
from .catalog import (
    KIND_ADD_VISUAL,
    KIND_FIELD,
    KIND_VISUAL,
    field_library_detail_text,
    field_library_field_count,
    field_library_rows,
    field_row_shows_edit,
    field_row_shows_symmetrize,
    parse_add_visual_row_id,
    row_indent_px,
    type_card_icon_rgb,
    visual_icon_kind,
    visual_icon_rgb,
)
from .pick import (
    overlay_get_open_file_name,
    overlay_question,
    overlay_warning,
    qt_modules,
)
from .tooltips import apply_required_tooltips
from .widgets.breadcrumb import (
    CRUMB_ADD_FIELD,
    CRUMB_ADD_OBJECT,
    CRUMB_FIELDS,
    make_page_header,
)
from .widgets.section import make_section
from .widgets.scrolling import (
    apply_expanding_list_policy,
    configure_resizable_window,
    make_scrolling_body,
)
from .widgets.catalog_chrome import (
    CATALOG_CHEVRON,
    CATALOG_INK,
    CATALOG_MUTED,
    apply_add_field_button_style,
    apply_add_visual_button_style,
    apply_fields_catalog_section_style,
    apply_fields_table_style,
    cell_band_css,
    library_footer_css,
    library_row_fill,
    make_field_rail,
    make_kind_cell,
    make_lib_cell,
    make_row_icon_button,
    rgb_css,
)
from .widgets.catalog_window import (
    build_empty_library_page,
    build_type_picker_page,
    close_catalog_window,
    show_catalog_library_window,
    transparent_for_mouse,
)
from .widgets.nest import make_nest_branch
from .widgets.switch import make_switch
from .widgets.sticky_add import (
    list_needs_sticky_add,
    sticky_add_overlay_rect,
)
from .widgets.theme import (
    apply_page_layout,
    apply_type_card_style,
    apply_wizard_page_style,
    empty_title_css,
    muted_label_css,
    type_card_subtitle_css,
    type_card_title_css,
)
from .widgets.type_icons import type_icon_pixmap


def fields_need_sticky_add(n_rows, viewport_height, row_height, add_height=None):
    return list_needs_sticky_add(n_rows, viewport_height, row_height, add_height=add_height)


def add_field_overlay_rect(
    table_width,
    table_height,
    viewport_x,
    viewport_y,
    viewport_height,
    n_rows,
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
        n_rows,
        row_height,
        add_height,
        header_height=header_height,
        h_scrollbar_height=h_scrollbar_height,
        inset=inset,
    )


FIELD_SOURCES = (
    (
        "From selection",
        "from_selection",
        "Gaussian density, distance, signed VDW, or nearest-atom property from selected atoms.",
        "surface",
    ),
    ("PyMOL map", "map", "Load CCP4, MRC, DX, or other maps PyMOL can read.", "volume"),
    ("PyMolViz file", "pmv", "Load a saved PyMolViz pack of fields and visuals.", "volume"),
    ("XYZ grid", "xyz", "Regular 3D values from an XYZ file.", "volume"),
    ("ORCA 3D", "orca", "ORCA 3D cube-style grid.", "volume"),
    ("MTZ map", "mtz", "Electron density from an MTZ reflection file.", "volume"),
    (
        "Derived field",
        "derived",
        "Combine existing fields with an expression. Coming later.",
        "volume",
    ),
)
DISABLED_FIELD_SOURCES = frozenset({"derived"})

FIELD_VISUAL_CARDS = (
    ("Volume", "Volume", "Translucent density cloud colored by value.", "volume"),
    ("IsoVolume", "IsoVolume", "Stacked transparent shells through the field.", "volume"),
    ("IsoSurface", "IsoSurface", "Solid surface at one field level.", "surface"),
    ("IsoMesh", "IsoMesh", "Wireframe mesh at one field level.", "isomesh"),
)

ADD_FIELD_LABEL = "Add Field"
ADD_FIELD_BUTTON = "+ Add Field"
ADD_FIELD_TIP = "Load a map or grid into this session."
ADD_VISUAL_LABEL = "Add Visual"
ADD_VISUAL_BUTTON = "+ Add Visual"
ADD_VISUAL_TIP = "Create a volume or isosurface for this field."
FIELD_COLUMNS = ("Name", "Kind", "Used by / Geometry", "Visible", "Edit", "Delete")
FIELD_ADD_VISUAL_SPAN = 3
LIBRARY_ROW_MIN_HEIGHT = 38
ADD_VISUAL_ROW_MIN_HEIGHT = 40
EMPTY_LIBRARY_TITLE = "No fields yet"
EMPTY_LIBRARY_HINT = "Load a map, or use a field already in PyMOL"
_FILE_FILTERS = {
    "map": "Maps (*.ccp4 *.mrc *.map *.dx *.xplor *.grd);;All files (*)",
    "xyz": "XYZ grids (*.xyz);;All files (*)",
    "orca": "ORCA 3D (*.txt *.cube);;All files (*)",
    "mtz": "MTZ maps (*.mtz);;All files (*)",
    "pmv": "PyMolViz (*.pmv);;All files (*)",
}
_COL_NAME = 0
_COL_KIND = 1
_COL_DETAIL = 2
_COL_VISIBLE = 3
_COL_EDIT = 4
_COL_DELETE = 5
_COL_VISIBLE_WIDTH = 56
_COL_EDIT_WIDTH = 48
_COL_DELETE_WIDTH = 52
_PAGE_LIBRARY = 0
_PAGE_FIELD_TYPES = 1
_PAGE_VISUAL_TYPES = 2
_LIBRARY_EMPTY = 0
_LIBRARY_LIST = 1
_EDIT_BUTTON_TEXT = "..."
SYMMETRIZE_BUTTON = "Symmetrize"
SYMMETRIZE_TIP = (
    "Choose an object or selection and extend this map's crystal cell around it. "
    "Covered targets already lie inside the current map."
)


def library_shows_empty_state(n_fields):
    return int(n_fields) <= 0


def _ignore_mouse(QtCore, widget):
    transparent_for_mouse(QtCore, widget)


def _row_kind_role(QtCore):
    base = 256
    if QtCore is not None:
        base = getattr(getattr(QtCore, "Qt", None), "UserRole", 256)
    try:
        return int(base) + 1
    except Exception:
        return 257


class FieldVisualsWindow:
    """Owns a Qt window for fields grouped with their volumetric visuals."""

    def __init__(self, wizard):
        self.wizard = wizard
        self._window = None
        self._stack = None
        self._fields_table = None
        self._library_body = None
        self._add_field_footer = None
        self._field_count = 0
        self._row_count = 0
        self._builder_page = None
        self._builder_stack_index = None
        self._from_selection_page = None
        self._from_selection_stack_index = None
        self._active_field_id = None
        self._editing_obj = None
        self._extend_dialog = None

    def show(self):
        QtCore, _, QtWidgets = qt_modules()
        if QtWidgets is None:
            self.wizard.prompt = ["Open Field Visuals Menu requires the PyMOL Qt UI"]
            return

        self._discard_window()

        try:
            self._open_window(QtCore, QtWidgets)
        except Exception as exc:
            self._reset_window()
            self.wizard.prompt = ["Open Field Visuals Menu failed: %s" % exc]
            try:
                overlay_warning(
                    None,
                    "PyMOLViz",
                    "Could not open Field Visuals:\n\n%s" % exc,
                )
            except Exception:
                pass

    def _discard_window(self):
        self._restore_editing_visual()
        dialog = getattr(self, "_extend_dialog", None)
        if dialog is not None:
            try:
                dialog.close()
            except Exception:
                pass
            self._extend_dialog = None
        window = self._window
        self._reset_window()
        if window is None:
            return
        close_catalog_window(window)

    def _open_window(self, QtCore, QtWidgets):
        def _pages(stack):
            stack.addWidget(self._build_library_page(QtCore, QtWidgets))
            stack.addWidget(self._build_field_type_page(QtCore, QtWidgets))
            stack.addWidget(self._build_visual_type_page(QtCore, QtWidgets))
            stack.setCurrentIndex(_PAGE_LIBRARY)

        def _after_open(_window, stack):
            self._stack = stack
            self._refresh_fields_table()
            self._sync_add_field_row()

        window, stack = show_catalog_library_window(
            QtCore,
            QtWidgets,
            window_title="PyMOLViz Fields",
            width=640,
            height=720,
            build_root_pages=_pages,
            on_destroyed=self._on_destroyed,
            after_open=_after_open,
            visibility_label="Field Visuals",
        )
        self._stack = stack
        self._window = window

    def _reset_window(self):
        self._window = None
        self._stack = None
        self._fields_table = None
        self._library_body = None
        self._add_field_footer = None
        self._field_count = 0
        self._row_count = 0
        self._builder_page = None
        self._builder_stack_index = None
        self._from_selection_page = None
        self._from_selection_stack_index = None
        self._active_field_id = None
        self._editing_obj = None
        self._extend_dialog = None

    def _ensure_stack(self):
        """Catalog-only windows may never have built a QStackedWidget; Edit still needs one."""
        if self._stack is not None:
            return
        _, _, QtWidgets = qt_modules()
        stacked_cls = getattr(QtWidgets, "QStackedWidget", None) if QtWidgets is not None else None
        if stacked_cls is None:
            return
        app_cls = getattr(QtWidgets, "QApplication", None)
        if app_cls is None:
            return
        try:
            if app_cls.instance() is None:
                return
        except Exception:
            return
        qwidget = getattr(QtWidgets, "QWidget", None)
        parent = self._window
        if parent is None or qwidget is None:
            return
        try:
            if not isinstance(parent, qwidget):
                return
        except TypeError:
            return
        try:
            stack = stacked_cls(parent)
        except Exception:
            return
        self._stack = stack
        getter = getattr(parent, "layout", None)
        layout = None
        if callable(getter):
            try:
                layout = getter()
            except Exception:
                layout = None
        if layout is None or not hasattr(layout, "addWidget"):
            return
        try:
            layout.addWidget(stack, 1)
        except TypeError:
            try:
                layout.addWidget(stack)
            except Exception:
                pass

    def _add_page_to_stack(self, page):
        if page is None:
            return None
        self._ensure_stack()
        stack = self._stack
        if stack is None:
            return None
        adder = getattr(stack, "addWidget", None)
        if not callable(adder):
            return None
        widget = getattr(page, "widget", page)
        return adder(widget)

    def _ensure_builder_page(self):
        if self._builder_page is None:
            self._builder_page = FieldVisualBuilderPage(
                self.wizard.cmd,
                on_back=self._on_builder_back,
                on_create=self._on_builder_saved,
                parent=self._window,
            )
        if self._builder_stack_index is None:
            self._builder_stack_index = self._add_page_to_stack(self._builder_page)

    def _ensure_from_selection_page(self):
        if self._from_selection_page is None:
            self._from_selection_page = FromSelectionFieldPage(
                self.wizard.cmd,
                on_back=lambda: self._goto(_PAGE_FIELD_TYPES),
                on_create=self._on_builder_saved,
                parent=self._window,
            )
        if self._from_selection_stack_index is None:
            self._from_selection_stack_index = self._add_page_to_stack(self._from_selection_page)

    def _build_library_page(self, QtCore, QtWidgets):
        page = QtWidgets.QWidget()
        apply_wizard_page_style(page)
        layout = QtWidgets.QVBoxLayout(page)
        apply_page_layout(layout)

        group = make_section("Fields", expanding=True)
        apply_fields_catalog_section_style(group.widget)
        group_layout = group.layout
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.setSpacing(0)
        body = QtWidgets.QStackedWidget()
        self._library_body = body

        table_page = QtWidgets.QWidget()
        table_layout = QtWidgets.QVBoxLayout(table_page)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(0)
        table = QtWidgets.QTableWidget(0, len(FIELD_COLUMNS))
        table.setHorizontalHeaderLabels(list(FIELD_COLUMNS))
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
        header.setSectionResizeMode(_COL_KIND, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(_COL_DETAIL, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(_COL_VISIBLE, QtWidgets.QHeaderView.Fixed)
        header.setSectionResizeMode(_COL_EDIT, QtWidgets.QHeaderView.Fixed)
        header.setSectionResizeMode(_COL_DELETE, QtWidgets.QHeaderView.Fixed)
        table.setColumnWidth(_COL_VISIBLE, _COL_VISIBLE_WIDTH)
        table.setColumnWidth(_COL_EDIT, _COL_EDIT_WIDTH)
        table.setColumnWidth(_COL_DELETE, _COL_DELETE_WIDTH)
        table.setWordWrap(False)
        apply_fields_table_style(table)
        align_left = getattr(QtCore.Qt, "AlignLeft", None)
        vcenter = getattr(QtCore.Qt, "AlignVCenter", None)
        if align_left is not None and vcenter is not None:
            header.setDefaultAlignment(align_left | vcenter)
        table.cellClicked.connect(self._on_row_clicked)
        table.cellDoubleClicked.connect(self._on_row_activated)
        table.itemSelectionChanged.connect(self._sync_row_bands)
        table.setToolTip("Double-click a visual to adjust it.")
        table_layout.addWidget(table, stretch=1)
        table_layout.addWidget(self._build_add_field_footer(QtCore, QtWidgets))
        apply_expanding_list_policy(table, QtWidgets)
        apply_expanding_list_policy(body, QtWidgets)
        apply_expanding_list_policy(table_page, QtWidgets)
        self._fields_table = table

        body.addWidget(
            build_empty_library_page(
                QtCore,
                QtWidgets,
                empty_title=EMPTY_LIBRARY_TITLE,
                empty_hint=EMPTY_LIBRARY_HINT,
                add_button_text=ADD_FIELD_BUTTON,
                add_tip=ADD_FIELD_TIP,
                on_add=lambda: self._goto(_PAGE_FIELD_TYPES),
                tooltip_context="FieldVisualsWindow",
                style_add_button=apply_add_field_button_style,
            )
        )
        body.addWidget(table_page)
        body.setCurrentIndex(_LIBRARY_EMPTY)
        group_layout.addWidget(body, stretch=1)

        layout.addWidget(group.widget, stretch=1)
        return page

    def _build_add_field_footer(self, QtCore, QtWidgets):
        footer = QtWidgets.QWidget()
        footer.setObjectName("pmvLibraryFooter")
        footer.setStyleSheet(library_footer_css())
        layout = QtWidgets.QHBoxLayout(footer)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(0)
        btn = QtWidgets.QPushButton(ADD_FIELD_BUTTON)
        btn.setAutoDefault(False)
        btn.setDefault(False)
        apply_add_field_button_style(btn)
        hand = getattr(QtCore.Qt, "PointingHandCursor", None)
        if hand is not None:
            btn.setCursor(hand)
        btn.clicked.connect(lambda: self._goto(_PAGE_FIELD_TYPES))
        apply_required_tooltips(
            [(btn, ADD_FIELD_TIP, ADD_FIELD_BUTTON)],
            context="FieldVisualsWindow",
        )
        layout.addWidget(btn, 1)
        preferred = getattr(QtWidgets.QSizePolicy, "Preferred", None)
        if preferred is not None:
            footer.setSizePolicy(preferred, preferred)
        self._add_field_footer = footer
        return footer

    def _build_field_type_page(self, QtCore, QtWidgets):
        return build_type_picker_page(
            QtCore,
            QtWidgets,
            title_parts=(CRUMB_FIELDS, CRUMB_ADD_FIELD),
            back_tip="Return to the field list.",
            subtitle="Choose how to load a field.",
            entries=FIELD_SOURCES,
            on_pick=self._on_field_source,
            on_back=lambda: self._goto(_PAGE_LIBRARY),
            tooltip_context="FieldVisualsWindow",
            icon_rgb_fn=type_card_icon_rgb,
            disabled_kinds=DISABLED_FIELD_SOURCES,
        )

    def _build_visual_type_page(self, QtCore, QtWidgets):
        return build_type_picker_page(
            QtCore,
            QtWidgets,
            title_parts=(CRUMB_FIELDS, CRUMB_ADD_OBJECT),
            back_tip="Return to the field list.",
            subtitle="Choose a visual for this field.",
            entries=FIELD_VISUAL_CARDS,
            on_pick=self._on_visual_type,
            on_back=lambda: self._goto(_PAGE_LIBRARY),
            tooltip_context="FieldVisualsWindow",
            icon_rgb_fn=type_card_icon_rgb,
        )

    def _refresh_fields_table(self):
        table = self._fields_table
        if table is None:
            return
        from ..runtime.presence import sync_session_with_pymol
        from ..runtime.session import all_objects

        QtCore, _, QtWidgets = qt_modules()
        sync_session_with_pymol(self.wizard.cmd)
        rows = field_library_rows(all_objects(), cmd=self.wizard.cmd)
        self._field_count = field_library_field_count(rows)
        self._row_count = len(rows)
        table.clearSpans()
        table.setRowCount(0)
        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            self._fill_library_row(table, i, row, QtCore, QtWidgets)
        if rows:
            table.resizeRowsToContents()
            for i, row in enumerate(rows):
                minimum = self._row_min_height(row.get("kind"))
                if table.rowHeight(i) < minimum:
                    table.setRowHeight(i, minimum)
        self._sync_row_bands()
        self._sync_add_field_row()

    def _row_min_height(self, kind):
        if kind == KIND_ADD_VISUAL:
            return ADD_VISUAL_ROW_MIN_HEIGHT
        return LIBRARY_ROW_MIN_HEIGHT

    def _fill_library_row(self, table, index, row, QtCore, QtWidgets):
        kind = row.get("kind")
        selected = table.currentRow() == index
        band = cell_band_css(library_row_fill(kind, selected=selected))
        if kind == KIND_ADD_VISUAL:
            table.setSpan(index, _COL_NAME, 1, FIELD_ADD_VISUAL_SPAN)
            name_item = QtWidgets.QTableWidgetItem("")
            name_item.setData(QtCore.Qt.UserRole, row["id"])
            name_item.setData(_row_kind_role(QtCore), kind)
            table.setItem(index, _COL_NAME, name_item)
            table.setCellWidget(
                index,
                _COL_NAME,
                self._add_visual_cell(QtCore, QtWidgets, row, band),
            )
            for col in (_COL_VISIBLE, _COL_EDIT, _COL_DELETE):
                table.setCellWidget(index, col, self._empty_cell(QtWidgets, band))
            return
        name_item = QtWidgets.QTableWidgetItem("")
        name_item.setData(QtCore.Qt.UserRole, row["id"])
        name_item.setData(_row_kind_role(QtCore), kind)
        table.setItem(index, _COL_NAME, name_item)
        table.setCellWidget(
            index,
            _COL_NAME,
            self._name_cell(QtCore, QtWidgets, row, band),
        )
        table.setCellWidget(
            index,
            _COL_KIND,
            self._kind_cell(QtCore, QtWidgets, row, band),
        )
        detail = field_library_detail_text(row)
        table.setCellWidget(
            index,
            _COL_DETAIL,
            self._text_cell(
                QtCore,
                QtWidgets,
                detail,
                band,
                muted=True,
                tooltip=detail or None,
            ),
        )
        visible = self._row_is_visible(row)
        table.setCellWidget(
            index,
            _COL_VISIBLE,
            self._visibility_cell(QtCore, QtWidgets, row, visible, band),
        )
        table.setCellWidget(
            index,
            _COL_EDIT,
            self._edit_cell(QtWidgets, row, band),
        )
        table.setCellWidget(
            index,
            _COL_DELETE,
            self._delete_cell(QtWidgets, row, band),
        )

    def _sync_row_bands(self):
        table = self._fields_table
        if table is None:
            return
        try:
            current = int(table.currentRow())
        except RuntimeError:
            return
        for index in range(table.rowCount()):
            item = table.item(index, _COL_NAME)
            kind = None
            if item is not None:
                kind = item.data(_row_kind_role(qt_modules()[0]))
            band = cell_band_css(library_row_fill(kind, selected=(index == current)))
            for col in range(table.columnCount()):
                widget = table.cellWidget(index, col)
                if widget is None:
                    continue
                try:
                    widget.setStyleSheet(band)
                except RuntimeError:
                    pass

    def _empty_cell(self, QtWidgets, band):
        return make_lib_cell(QtWidgets, band, min_height=LIBRARY_ROW_MIN_HEIGHT)

    def _text_cell(self, QtCore, QtWidgets, text, band, muted=False, tooltip=None):
        wrap = self._empty_cell(QtWidgets, band)
        layout = QtWidgets.QHBoxLayout(wrap)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(0)
        label = QtWidgets.QLabel(str(text or ""))
        if muted:
            label.setStyleSheet(
                "color: %s; background: transparent;" % rgb_css(CATALOG_MUTED)
            )
        else:
            label.setStyleSheet("background: transparent;")
        if tooltip:
            label.setToolTip(str(tooltip))
        label.setWordWrap(False)
        layout.addWidget(label, 1)
        return wrap

    def _kind_cell(self, QtCore, QtWidgets, row, band):
        return make_kind_cell(
            QtCore,
            QtWidgets,
            row.get("type"),
            band,
            min_height=LIBRARY_ROW_MIN_HEIGHT,
        )

    def _name_cell(self, QtCore, QtWidgets, row, band):
        wrap = self._empty_cell(QtWidgets, band)
        layout = QtWidgets.QHBoxLayout(wrap)
        kind = row.get("kind")
        if kind == KIND_FIELD:
            layout.setContentsMargins(0, 6, 8, 6)
            layout.setSpacing(6)
            rail = make_field_rail(QtWidgets, QtCore)
            layout.addWidget(rail)
            chevron = QtWidgets.QLabel("▾")
            chevron.setStyleSheet(
                "color: %s; background: transparent; font-size: 10px;" % rgb_css(CATALOG_CHEVRON)
            )
            _ignore_mouse(QtCore, chevron)
            layout.addWidget(chevron)
        else:
            layout.setContentsMargins(4, 6, 8, 6)
            layout.setSpacing(6)
            rail = make_nest_branch(row, wrap, width=max(row_indent_px(row), 22))
            if rail is not None:
                layout.addWidget(rail)
            icon_key = visual_icon_kind(row.get("type"))
            if icon_key:
                _, QtGui, _ = qt_modules()
                glyph = QtWidgets.QLabel()
                pix = type_icon_pixmap(
                    icon_key,
                    QtGui,
                    QtCore,
                    QtWidgets,
                    size=16,
                    color=visual_icon_rgb(row.get("type")),
                )
                if pix is not None:
                    glyph.setPixmap(pix)
                glyph.setFixedSize(16, 16)
                _ignore_mouse(QtCore, glyph)
                layout.addWidget(glyph)
        label = QtWidgets.QLabel(str(row.get("name") or ""))
        if kind == KIND_FIELD:
            label.setStyleSheet(
                "font-weight: 600; color: %s; background: transparent;" % rgb_css(CATALOG_INK)
            )
        else:
            label.setStyleSheet(
                "color: %s; background: transparent;" % rgb_css(CATALOG_INK)
            )
        layout.addWidget(label, 1)
        return wrap

    def _add_visual_cell(self, QtCore, QtWidgets, row, band):
        wrap = self._empty_cell(QtWidgets, band)
        wrap.setMinimumHeight(ADD_VISUAL_ROW_MIN_HEIGHT)
        layout = QtWidgets.QHBoxLayout(wrap)
        layout.setContentsMargins(4, 4, 8, 4)
        layout.setSpacing(6)
        rail = make_nest_branch(row, wrap, width=max(row_indent_px(row), 22))
        if rail is not None:
            layout.addWidget(rail)
        btn = QtWidgets.QPushButton(ADD_VISUAL_BUTTON)
        btn.setAutoDefault(False)
        btn.setDefault(False)
        apply_add_visual_button_style(btn)
        hand = getattr(QtCore.Qt, "PointingHandCursor", None)
        if hand is not None:
            btn.setCursor(hand)
        field_id = row.get("field_id")
        btn.clicked.connect(lambda *_a, fid=field_id: self._open_add_visual(fid))
        apply_required_tooltips(
            [(btn, ADD_VISUAL_TIP, ADD_VISUAL_BUTTON)],
            context="FieldVisualsWindow",
        )
        layout.addWidget(btn, 1)
        wrap._pmv_row_id = row["id"]
        return wrap

    def _visibility_cell(self, QtCore, QtWidgets, row, checked, band):
        wrap = self._empty_cell(QtWidgets, band)
        layout = QtWidgets.QHBoxLayout(wrap)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        checkbox = make_switch(compact=True)
        checkbox.setChecked(bool(checked))
        checkbox.toggled.connect(
            lambda on, r=row: self._on_visibility_toggled(r, on)
        )
        apply_required_tooltips(
            [(checkbox, "Show or hide this object in the viewer.", "Visible")],
            context="FieldVisualsWindow",
        )
        layout.addWidget(checkbox)
        return wrap

    def _edit_cell(self, QtWidgets, row, band):
        wrap = self._empty_cell(QtWidgets, band)
        if field_row_shows_edit(row):
            return self._icon_button_cell(
                QtWidgets,
                wrap,
                _EDIT_BUTTON_TEXT,
                "Open the builder for this visual.",
                "Edit",
                lambda *_a, r=row: self._edit_row(r),
            )
        if field_row_shows_symmetrize(row):
            return self._icon_button_cell(
                QtWidgets,
                wrap,
                SYMMETRIZE_BUTTON,
                SYMMETRIZE_TIP,
                SYMMETRIZE_BUTTON,
                lambda *_a, r=row: self._symmetrize_row(r),
                icon="lattice",
            )
        return wrap

    def _delete_cell(self, QtWidgets, row, band):
        wrap = self._empty_cell(QtWidgets, band)
        return self._icon_button_cell(
            QtWidgets,
            wrap,
            "",
            "Remove this field or visual from the session.",
            "Delete",
            lambda *_a, r=row: self._on_delete_row(r),
            icon="trash",
        )

    def _icon_button_cell(self, QtWidgets, wrap, text, tip, name, on_click, icon=None):
        QtCore, _, _ = qt_modules()
        layout = QtWidgets.QHBoxLayout(wrap)
        layout.setContentsMargins(4, 4, 4, 4)
        if QtCore is not None:
            layout.setAlignment(QtCore.Qt.AlignCenter)
        btn = make_row_icon_button(
            QtWidgets,
            text,
            tip,
            name,
            on_click,
            "FieldVisualsWindow",
            icon=icon,
        )
        layout.addWidget(btn)
        return wrap

    def _row_marker(self, row):
        table = self._fields_table
        if table is None:
            return None
        item = table.item(row, _COL_NAME)
        if item is not None:
            QtCore, _, _ = qt_modules()
            marker = item.data(QtCore.Qt.UserRole)
            if marker:
                return marker
        widget = table.cellWidget(row, _COL_NAME)
        return getattr(widget, "_pmv_row_id", None)

    def _sync_add_field_row(self):
        footer = self._add_field_footer
        table = self._fields_table
        body = self._library_body
        empty = library_shows_empty_state(self._field_count)
        if body is not None:
            body.setCurrentIndex(_LIBRARY_EMPTY if empty else _LIBRARY_LIST)
        if footer is None or table is None:
            return
        try:
            footer.setVisible(not empty)
        except RuntimeError:
            pass

    def _goto(self, index):
        if index is None:
            return
        if self._stack is not None:
            setter = getattr(self._stack, "setCurrentIndex", None)
            if callable(setter):
                setter(index)
        if index == _PAGE_LIBRARY:
            self._refresh_fields_table()

    def _open_add_visual(self, field_id):
        self._active_field_id = field_id
        self._goto(_PAGE_VISUAL_TYPES)

    def _on_row_clicked(self, row, _column):
        marker = self._row_marker(row)
        field_id = parse_add_visual_row_id(marker)
        if field_id:
            self._open_add_visual(field_id)

    def _on_row_activated(self, row, column):
        if column in (_COL_VISIBLE, _COL_EDIT, _COL_DELETE):
            return
        marker = self._row_marker(row)
        field_id = parse_add_visual_row_id(marker)
        if field_id:
            self._open_add_visual(field_id)
            return
        obj = self._session_object(marker)
        if obj is None:
            return
        from .catalog import is_field_visual

        if is_field_visual(obj):
            self._open_visual_editor(obj)

    def _session_object(self, obj_id):
        from ..runtime.session import get as session_get

        return session_get(obj_id)

    def _resolve_field(self, field_id):
        obj = self._session_object(field_id)
        if obj is not None:
            return obj
        from ..util.field_sample import discover_fields, resolve_grid_from_session

        grid = resolve_grid_from_session(field_id)
        if grid is not None:
            return grid
        for field in discover_fields(cmd=self.wizard.cmd):
            if str(getattr(field, "id", "") or "") == str(field_id):
                return field
        return None

    def _row_is_visible(self, row):
        kind = row.get("kind")
        if kind == KIND_VISUAL:
            obj = self._session_object(row["id"])
            if obj is not None:
                return visual_is_enabled(self.wizard.cmd, obj)
        name = self._pymol_name_for_row(row)
        return self._named_is_enabled(name)

    def _pymol_name_for_row(self, row):
        kind = row.get("kind")
        if kind == KIND_FIELD:
            field = self._resolve_field(row.get("id"))
            if field is not None:
                return pymol_name_for(field, self.wizard.cmd) or row.get("name")
            return row.get("name")
        obj = self._session_object(row.get("id"))
        if obj is not None:
            return pymol_name_for(obj, self.wizard.cmd)
        return row.get("name")

    def _named_is_enabled(self, name):
        if not name:
            return True
        cmd = self.wizard.cmd
        try:
            names = cmd.get_names("objects", enabled_only=1)
            return str(name) in names
        except TypeError:
            try:
                return str(name) in cmd.get_names("objects", 1)
            except Exception:
                return True
        except Exception:
            return True

    def _set_named_enabled(self, name, enabled):
        if not name:
            return
        try:
            if enabled:
                self.wizard.cmd.enable(name)
            else:
                self.wizard.cmd.disable(name)
        except Exception:
            pass

    def _on_visibility_toggled(self, row, checked):
        kind = row.get("kind")
        if kind == KIND_VISUAL:
            obj = self._session_object(row.get("id"))
            if obj is not None:
                set_visual_enabled(self.wizard.cmd, obj, bool(checked))
                return
        self._set_named_enabled(self._pymol_name_for_row(row), bool(checked))

    def _on_delete_row(self, row):
        kind = row.get("kind")
        name = row.get("name") or "this object"
        if kind == KIND_FIELD:
            field = self._resolve_field(row.get("id"))
            dependents = []
            if field is not None:
                from ..fields.dependents import dependent_labels, dependents_of_field

                dependents = dependents_of_field(field)
            if dependents:
                labels = dependent_labels(dependents)
                extra = "\n".join("- %s" % item for item in labels[:12])
                if len(labels) > 12:
                    extra += "\n- …"
                if not self._confirm_delete(
                    name,
                    title="Delete field",
                    message=(
                        "%s is used by:\n%s\n\nDelete this field and its dependents?"
                        % (name, extra)
                    ),
                ):
                    return
            elif not self._confirm_delete(name, title="Delete field"):
                return
            if field is None:
                return
            delete_field_and_visuals(self.wizard.cmd, field, dependents)
            self._refresh_fields_table()
            return
        if kind == KIND_VISUAL:
            if not self._confirm_delete(name, title="Delete visual"):
                return
            obj = self._session_object(row.get("id"))
            if obj is None:
                return
            delete_field_visual(self.wizard.cmd, obj)
            self._refresh_fields_table()

    def _confirm_delete(self, name, title="Delete object", message=None):
        _, _, QtWidgets = qt_modules()
        if QtWidgets is None:
            return True
        text = message or ("Delete %s from this session?" % name)
        result = overlay_question(
            self._window,
            title,
            text,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        return result == QtWidgets.QMessageBox.Yes

    def _edit_row(self, row):
        if row.get("kind") == KIND_FIELD:
            obj = self._resolve_field(row.get("id"))
            if obj is None:
                return
            self._open_field_editor(obj)
            return
        if row.get("kind") != KIND_VISUAL:
            return
        obj = self._session_object(row.get("id"))
        if obj is None:
            return
        self._open_visual_editor(obj)

    def _open_field_editor(self, obj):
        try:
            self._ensure_from_selection_page()
            self._from_selection_page.load_object(obj)
            self._goto(self._from_selection_stack_index)
        except Exception as exc:
            self.wizard.prompt = ["From Selection editor failed: %s" % exc]
            _, _, QtWidgets = qt_modules()
            if QtWidgets is not None:
                overlay_warning(
                    self._window,
                    "PyMOLViz",
                    "Could not open From Selection:\n\n%s" % exc,
                )

    def _symmetrize_row(self, row):
        if row.get("kind") != KIND_FIELD:
            return
        field = self._resolve_field(row.get("id"))
        if field is None:
            overlay_warning(self._window, "PyMOLViz", "Could not find that field.")
            return
        from ..fields.crystal import (
            CrystalError,
            default_extend_target_name,
            extend_target_rows,
            map_world_aabb,
        )
        from ..fields.field import ensure_brick
        from .builders.extend_cell_dialog import open_extend_cell_dialog
        from .builders.field_visual import symmetrize_field_to_selection

        cmd = self.wizard.cmd
        grid = ensure_brick(field, cmd=cmd)
        if grid is None:
            overlay_warning(self._window, "PyMOLViz", "This field has no sampleable map yet.")
            return
        map_name = getattr(grid, "_name", None) or getattr(grid, "name", None)
        targets = extend_target_rows(cmd, grid, skip_names=(map_name,))
        if not targets:
            overlay_warning(
                self._window,
                "PyMOLViz",
                "No objects or selections with atoms to extend around.",
            )
            return
        map_lo, map_hi = map_world_aabb(grid)
        current = default_extend_target_name(targets)
        existing = getattr(self, "_extend_dialog", None)
        if existing is not None:
            try:
                existing.close()
            except Exception:
                pass

        def apply_around(name):
            try:
                wrapped, info = symmetrize_field_to_selection(cmd, field, selection=name)
            except CrystalError as exc:
                overlay_warning(self._window, "PyMOLViz", str(exc))
                return
            except Exception as exc:
                overlay_warning(self._window, "PyMOLViz", "Could not symmetrize the map:\n\n%s" % exc)
                return
            self._after_symmetrize(row, wrapped, info)

        opened = open_extend_cell_dialog(
            self._window,
            targets,
            map_lo,
            map_hi,
            current=current,
            on_extend=apply_around,
        )
        if opened is None:
            apply_around(current)
            return
        self._extend_dialog = opened

    def _after_symmetrize(self, row, wrapped, info):
        copied = "copied neighboring cells" if info.get("copied") else "moved by a lattice vector"
        try:
            self.wizard.prompt = ["Map %s (%s)." % (copied, row.get("name") or "field")]
            self.wizard.cmd.refresh_wizard()
        except Exception:
            pass
        if self._builder_page is not None:
            try:
                widget = getattr(self._builder_page, "widget", self._builder_page)
                visible = True
                getter = getattr(widget, "isVisible", None)
                if callable(getter):
                    visible = bool(getter())
                if visible:
                    self._builder_page.notify_field_brick_changed(wrapped, info)
            except Exception:
                pass
        self._refresh_fields_table()

    def _open_visual_editor(self, obj):
        self._ensure_builder_page()
        self._editing_obj = obj
        set_visual_enabled(self.wizard.cmd, obj, False)
        if self._builder_page is not None:
            self._builder_page.load_object(obj)
        self._goto(self._builder_stack_index)

    def _open_visual_create(self, kind, field):
        self._ensure_builder_page()
        self._editing_obj = None
        if self._builder_page is not None:
            self._builder_page.reset_for_create(kind, field)
        self._goto(self._builder_stack_index)

    def _restore_editing_visual(self):
        obj = self._editing_obj
        self._editing_obj = None
        if obj is None:
            return
        set_visual_enabled(self.wizard.cmd, obj, True)

    def _on_builder_back(self):
        if self._editing_obj is not None:
            self._restore_editing_visual()
            self._goto(_PAGE_LIBRARY)
            return
        self._goto(_PAGE_VISUAL_TYPES)

    def _on_builder_saved(self):
        # Persist already materialized the updated visual; do not re-enable the
        # stale object handle kept for cancel/back.
        self._editing_obj = None
        self._goto(_PAGE_LIBRARY)

    def _on_field_source(self, name, kind):
        self.wizard.prompt = ["Field: %s" % name]
        try:
            self.wizard.cmd.refresh_wizard()
        except Exception:
            pass
        if kind in (KIND_FROM_SELECTION, KIND_IMPLICIT):
            try:
                self._ensure_from_selection_page()
                self._from_selection_page.reset_for_create()
                self._goto(self._from_selection_stack_index)
            except Exception as exc:
                self.wizard.prompt = ["From Selection builder failed: %s" % exc]
                _, _, QtWidgets = qt_modules()
                if QtWidgets is not None:
                    overlay_warning(
                        self._window,
                        "PyMOLViz",
                        "Could not open From Selection:\n\n%s" % exc,
                    )
            return
        if kind == KIND_DERIVED:
            return
        _, _, QtWidgets = qt_modules()
        if QtWidgets is None:
            return
        path, _ = overlay_get_open_file_name(
            self._window,
            "Load field",
            "",
            _FILE_FILTERS.get(kind, "All files (*)"),
        )
        if not path:
            return
        try:
            field = load_field_file(self.wizard.cmd, path, kind=kind)
        except Exception as exc:
            overlay_warning(
                self._window,
                "PyMOLViz",
                "Could not load field:\n\n%s" % exc,
            )
            return
        if field is None:
            overlay_warning(
                self._window,
                "PyMOLViz",
                "Could not load that file as a field.",
            )
            return
        self._goto(_PAGE_LIBRARY)

    def _on_visual_type(self, name, kind):
        field = self._resolve_field(self._active_field_id)
        if field is None:
            _, _, QtWidgets = qt_modules()
            if QtWidgets is not None:
                overlay_warning(
                    self._window,
                    "PyMOLViz",
                    "Select a field before adding a visual.",
                )
            self._goto(_PAGE_LIBRARY)
            return
        self.wizard.prompt = ["Field visual: %s" % name]
        try:
            self.wizard.cmd.refresh_wizard()
        except Exception:
            pass
        try:
            self._open_visual_create(kind, field)
        except Exception as exc:
            self.wizard.prompt = ["Field visual builder failed: %s" % exc]
            _, _, QtWidgets = qt_modules()
            if QtWidgets is not None:
                overlay_warning(
                    self._window,
                    "PyMOLViz",
                    "Could not open %s builder:\n\n%s" % (kind, exc),
                )

    def _on_destroyed(self, *_args):
        self._restore_editing_visual()
        if self._builder_page is not None:
            self._builder_page.cleanup_preview()
        if self._from_selection_page is not None:
            self._from_selection_page.cleanup_preview()
        self._reset_window()

    def close(self):
        self._restore_editing_visual()
        if self._builder_page is not None:
            self._builder_page.cleanup_preview()
        if self._from_selection_page is not None:
            self._from_selection_page.cleanup_preview()
        window = self._window
        self._reset_window()
        if window is None:
            return
        try:
            window.close()
        except RuntimeError:
            pass
