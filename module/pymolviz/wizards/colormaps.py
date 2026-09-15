"""Standalone window for listing and editing custom colormaps."""

from dataclasses import replace

from ..util.colormap_spec import (
    ColormapDefinition,
    FieldColorMapping,
    create_custom_preset,
    custom_colormap_rows,
    custom_preset_definition,
    delete_custom_preset,
    mapping_for_custom_preset,
    save_custom_preset,
)
from .builders.colormap_dialog import _ramp_image, open_colormap_editor
from .pick import (
    bind_tool_window,
    configure_tool_window,
    overlay_question,
    overlay_warning,
    qt_modules,
)
from .pick import _qt_platform_name
from .tooltips import apply_required_tooltips
from .widgets.catalog_chrome import (
    LIBRARY_ROW_MIN_HEIGHT,
    cell_band_css,
    make_lib_cell,
    make_row_icon_button,
)
from .widgets.section import make_section
from .widgets.scrolling import configure_resizable_window, make_scrolling_body
from .widgets.sticky_add import StickyAddOverlay
from .widgets.theme import (
    apply_catalog_table_style,
    apply_page_layout,
    apply_wizard_page_style,
    empty_title_css,
    mark_primary_button,
    muted_label_css,
    page_heading_css,
    selected_row_fill,
)


ADD_COLORMAP_LABEL = "+ Add Custom Colormap"
ADD_COLORMAP_TIP = "Create a new custom colormap and open the editor."
EDIT_COLORMAP_TIP = "Edit this custom colormap."
DELETE_COLORMAP_TIP = "Delete this custom colormap."
EMPTY_LIBRARY_TITLE = "No custom colormaps yet"
EMPTY_LIBRARY_HINT = "Add a colormap to reuse it across fields and visuals"
COLUMNS = ("Name", "Colormap", "Actions")
_COL_NAME = 0
_COL_RAMP = 1
_COL_ACTIONS = 2
_LIBRARY_EMPTY = 0
_LIBRARY_LIST = 1
_CONTEXT = "ColormapMenuWindow"


def _ignore_mouse(QtCore, widget):
    flag = getattr(QtCore.Qt, "WA_TransparentForMouseEvents", None)
    if flag is not None:
        widget.setAttribute(flag, True)


class ColormapMenuWindow:
    """Owns a Qt window popped out from the wizard panel."""

    def __init__(self, wizard):
        self.wizard = wizard
        self._window = None
        self._table = None
        self._library_body = None
        self._add_overlay = None
        self._map_count = 0
        self._editor = None

    def show(self):
        QtCore, _, QtWidgets = qt_modules()
        if QtWidgets is None:
            self.wizard.prompt = ["Open Colormap Menu requires the PyMOL Qt UI"]
            return

        self._discard_window()

        try:
            self._open_window(QtCore, QtWidgets)
        except Exception as exc:
            self._reset_window()
            self.wizard.prompt = ["Open Colormap Menu failed: %s" % exc]
            try:
                overlay_warning(
                    None,
                    "PyMOLViz",
                    "Could not open Colormaps:\n\n%s" % exc,
                )
            except Exception:
                pass

    def _discard_window(self):
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
        window = QtWidgets.QDialog()
        window.setWindowTitle("PyMOLViz Colormaps")
        window.setModal(False)
        configure_tool_window(window)
        window.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        window.resize(560, 560)
        configure_resizable_window(window)

        root = QtWidgets.QVBoxLayout(window)
        apply_page_layout(root)
        apply_wizard_page_style(window)
        root.addWidget(self._build_library_page(QtCore, QtWidgets), stretch=1)
        window.destroyed.connect(self._on_destroyed)
        self._window = window
        self._refresh_table()
        window.show()
        bind_tool_window(window)
        window.raise_()
        window.activateWindow()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents()
        if not window.isVisible():
            raise RuntimeError(
                "Colormap window did not become visible "
                "(Qt platform=%r)" % (_qt_platform_name(),)
            )

    def _reset_window(self):
        self._window = None
        self._table = None
        self._library_body = None
        self._add_overlay = None
        self._map_count = 0
        self._editor = None

    def _build_library_page(self, QtCore, QtWidgets):
        page = QtWidgets.QWidget()
        apply_wizard_page_style(page)
        layout = QtWidgets.QVBoxLayout(page)
        apply_page_layout(layout)

        title = QtWidgets.QLabel("<b>Colormaps</b>")
        title.setStyleSheet(page_heading_css())
        subtitle = QtWidgets.QLabel("Custom colormaps saved on this machine.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(muted_label_css())

        group = make_section("Custom colormaps", expanding=True)
        group_layout = group.layout
        group_layout.setContentsMargins(8, 8, 8, 8)
        group_layout.setSpacing(0)
        body = QtWidgets.QStackedWidget()
        self._library_body = body

        table_page = QtWidgets.QWidget()
        table_layout = QtWidgets.QVBoxLayout(table_page)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(0)
        table = QtWidgets.QTableWidget(0, len(COLUMNS))
        table.setObjectName("pmvColormapTable")
        table.setHorizontalHeaderLabels(list(COLUMNS))
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
        header.setSectionResizeMode(_COL_RAMP, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(_COL_ACTIONS, QtWidgets.QHeaderView.ResizeToContents)
        apply_catalog_table_style(table)
        align_left = getattr(QtCore.Qt, "AlignLeft", None)
        vcenter = getattr(QtCore.Qt, "AlignVCenter", None)
        if align_left is not None and vcenter is not None:
            header.setDefaultAlignment(align_left | vcenter)
        table.cellDoubleClicked.connect(self._on_row_activated)
        table.itemSelectionChanged.connect(self._sync_row_bands)
        table.setToolTip("Double-click a row to edit that colormap.")
        table_layout.addWidget(table, stretch=1)
        self._table = table
        self._add_overlay = StickyAddOverlay(
            table_page,
            table,
            text=ADD_COLORMAP_LABEL,
            tooltip=ADD_COLORMAP_TIP,
            on_click=self._add_colormap,
            count=lambda: int(self._map_count),
            row_height=lambda: self._row_height(),
            add_height=lambda: self._add_button_height(),
            context=_CONTEXT,
        )
        self._add_overlay.attach()

        body.addWidget(self._build_empty_state(QtCore, QtWidgets))
        body.addWidget(table_page)
        body.setCurrentIndex(_LIBRARY_EMPTY)
        group_layout.addWidget(body, stretch=1)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        scroll, body_layout = make_scrolling_body(page)
        body_layout.addWidget(group.widget, stretch=1)
        layout.addWidget(scroll, stretch=1)
        return page

    def _build_empty_state(self, QtCore, QtWidgets):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.addStretch(1)

        title = QtWidgets.QLabel(EMPTY_LIBRARY_TITLE)
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet(empty_title_css())
        hint = QtWidgets.QLabel(EMPTY_LIBRARY_HINT)
        hint.setAlignment(QtCore.Qt.AlignCenter)
        hint.setWordWrap(True)
        hint.setStyleSheet(muted_label_css())

        btn = QtWidgets.QPushButton(ADD_COLORMAP_LABEL)
        btn.setObjectName("pmvAddColormap")
        btn.setAutoDefault(False)
        btn.setDefault(False)
        btn.setMinimumWidth(220)
        mark_primary_button(btn)
        hand = getattr(QtCore.Qt, "PointingHandCursor", None)
        if hand is not None:
            btn.setCursor(hand)
        btn.clicked.connect(self._add_colormap)
        apply_required_tooltips(
            [(btn, ADD_COLORMAP_TIP, ADD_COLORMAP_LABEL)],
            context=_CONTEXT,
        )
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(btn)
        btn_row.addStretch(1)

        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addSpacing(12)
        layout.addLayout(btn_row)
        layout.addStretch(1)
        return page

    def _row_height(self):
        table = self._table
        if table is None:
            return LIBRARY_ROW_MIN_HEIGHT
        try:
            return max(int(table.verticalHeader().defaultSectionSize()), LIBRARY_ROW_MIN_HEIGHT)
        except Exception:
            return LIBRARY_ROW_MIN_HEIGHT

    def _add_button_height(self):
        overlay = self._add_overlay
        if overlay is None:
            return LIBRARY_ROW_MIN_HEIGHT
        try:
            return max(int(overlay.widget.height()), LIBRARY_ROW_MIN_HEIGHT)
        except Exception:
            return LIBRARY_ROW_MIN_HEIGHT

    def _refresh_table(self):
        table = self._table
        if table is None:
            return
        QtCore, QtGui, QtWidgets = qt_modules()
        rows = custom_colormap_rows()
        self._map_count = len(rows)
        table.clearSpans()
        table.setRowCount(0)
        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            self._fill_row(table, i, row, QtCore, QtGui, QtWidgets)
        if rows:
            if self._library_body is not None:
                self._library_body.setCurrentIndex(_LIBRARY_LIST)
            overlay = self._add_overlay
            if overlay is not None:
                try:
                    overlay.widget.show()
                except RuntimeError:
                    pass
                overlay.sync()
        elif self._library_body is not None:
            self._library_body.setCurrentIndex(_LIBRARY_EMPTY)
            overlay = self._add_overlay
            if overlay is not None:
                try:
                    overlay.widget.hide()
                except RuntimeError:
                    pass
        self._sync_row_bands()

    def _fill_row(self, table, index, row, QtCore, QtGui, QtWidgets):
        name = str(row["name"])
        band = cell_band_css(selected_row_fill(selected=False))
        name_item = QtWidgets.QTableWidgetItem(name)
        name_item.setData(QtCore.Qt.UserRole, name)
        name_item.setToolTip("Edit %s" % name)
        table.setItem(index, _COL_NAME, name_item)

        defn = ColormapDefinition.from_dict(row.get("definition") or {})
        ramp = QtWidgets.QLabel()
        pix = _ramp_image(QtGui, defn, 120, 14) if QtGui is not None else None
        if pix is not None:
            ramp.setPixmap(pix)
            ramp.setScaledContents(True)
        wrap = make_lib_cell(QtWidgets, band)
        inner = QtWidgets.QHBoxLayout(wrap)
        inner.setContentsMargins(8, 8, 8, 8)
        inner.addWidget(ramp, stretch=1)
        _ignore_mouse(QtCore, ramp)
        table.setCellWidget(index, _COL_RAMP, wrap)
        table.setCellWidget(
            index,
            _COL_ACTIONS,
            self._actions_cell(QtWidgets, name, band),
        )

    def _actions_cell(self, QtWidgets, name, band):
        wrap = make_lib_cell(QtWidgets, band)
        row = QtWidgets.QHBoxLayout(wrap)
        row.setContentsMargins(4, 4, 8, 4)
        row.setSpacing(4)
        row.addStretch(1)
        edit = make_row_icon_button(
            QtWidgets, "...", EDIT_COLORMAP_TIP, "Edit colormap",
            lambda _checked=False, n=name: self._edit_colormap(n),
            _CONTEXT, icon="gear",
        )
        delete = make_row_icon_button(
            QtWidgets, "⌫", DELETE_COLORMAP_TIP, "Delete colormap",
            lambda _checked=False, n=name: self._delete_colormap(n),
            _CONTEXT, icon="trash",
        )
        row.addWidget(edit)
        row.addWidget(delete)
        return wrap

    def _sync_row_bands(self):
        table = self._table
        if table is None:
            return
        selected = set()
        try:
            selected = {item.row() for item in table.selectedItems()}
        except Exception:
            pass
        for row in range(table.rowCount()):
            fill = selected_row_fill(selected=(row in selected))
            band = cell_band_css(fill)
            for col in (_COL_RAMP, _COL_ACTIONS):
                widget = table.cellWidget(row, col)
                if widget is not None:
                    widget.setStyleSheet(band)

    def _row_name(self, row):
        table = self._table
        if table is None:
            return None
        item = table.item(row, _COL_NAME)
        if item is None:
            return None
        QtCore, _, _ = qt_modules()
        name = item.data(QtCore.Qt.UserRole)
        return str(name) if name else None

    def _on_row_activated(self, row, column):
        if column == _COL_ACTIONS:
            return
        name = self._row_name(row)
        if name:
            self._edit_colormap(name)

    def _add_colormap(self):
        try:
            name = create_custom_preset()
        except Exception as exc:
            overlay_warning(self._window, "PyMOLViz", "Could not add colormap:\n\n%s" % exc)
            return
        self._refresh_table()
        self._edit_colormap(name)

    def _edit_colormap(self, name):
        mapping = mapping_for_custom_preset(name)
        if mapping is None:
            defn = custom_preset_definition(name)
            if defn is None:
                overlay_warning(self._window, "PyMOLViz", "Colormap %s was not found." % name)
                self._refresh_table()
                return
            mapping = FieldColorMapping(colormap=replace(defn, customized=True, preset=name))
        cmd = getattr(self.wizard, "cmd", None)

        def persist(updated):
            if updated is None:
                return
            preset = updated.colormap.preset or name
            save_custom_preset(preset, updated.colormap)
            self._refresh_table()

        self._editor = open_colormap_editor(
            self._window,
            mapping,
            cmd=cmd,
            on_apply=persist,
            on_done=lambda updated: persist(updated) if updated is not None else None,
        )

    def _delete_colormap(self, name):
        _, _, QtWidgets = qt_modules()
        if QtWidgets is not None:
            result = overlay_question(
                self._window,
                "Delete colormap",
                "Delete custom colormap %s?" % name,
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if result != QtWidgets.QMessageBox.Yes:
                return
        delete_custom_preset(name)
        self._refresh_table()

    def _on_destroyed(self, *_args):
        self._reset_window()

    def close(self):
        window = self._window
        self._reset_window()
        if window is None:
            return
        try:
            window.close()
        except RuntimeError:
            pass
