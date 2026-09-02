"""Standalone window for listing and adding visuals (CGOs)."""

from .builders.arrow_page import ArrowBuilderPage
from .builders.box_page import BoxBuilderPage
from .builders.preview import set_visual_enabled
from .builders.sphere_page import SphereBuilderPage
from .catalog import editor_kind, object_rows
from .pick import (
    bind_tool_window,
    configure_tool_window,
    find_pymol_window,
    qt_modules,
)
from .pick import _qt_platform_name

MESH_TYPES = (
    ("Sphere", "Solid or wireframe sphere"),
    ("Box", "Axis-aligned or centered box"),
    ("Surface", "Triangulated mesh surface"),
    ("Lines", "Line / polyline segments"),
    ("Arrows", "Directed arrow glyphs"),
)

_OBJECT_COLUMNS = ("Name", "Type", "# Points", "COM")
_PAGE_LIBRARY = 0
_PAGE_TYPES = 1


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
        self._objects_table = None
        self._editing_obj = None
        self._sphere_stack_index = None
        self._box_stack_index = None
        self._arrow_stack_index = None

    def show(self):
        QtCore, _, QtWidgets = qt_modules()
        if QtWidgets is None:
            self.wizard.prompt = ["Add Visual requires the PyMOL Qt UI"]
            return

        self._discard_window()

        try:
            self._open_window(QtCore, QtWidgets)
        except Exception as exc:
            self._reset_window()
            self.wizard.prompt = ["Add Visual failed: %s" % exc]
            try:
                QtWidgets.QMessageBox.warning(
                    None,
                    "PyMOLViz",
                    "Could not open Visuals:\n\n%s" % exc,
                )
            except Exception:
                pass

    def _discard_window(self):
        self._restore_editing_visual()
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
        anchor = find_pymol_window(QtWidgets)
        window = QtWidgets.QDialog(anchor)
        window.setWindowTitle("PyMOLViz Visuals")
        window.setModal(False)
        configure_tool_window(window, anchor=anchor)
        window.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        window.resize(640, 720)

        root = QtWidgets.QVBoxLayout(window)
        stack = QtWidgets.QStackedWidget()
        self._stack = stack

        stack.addWidget(self._build_library_page(QtCore, QtWidgets))
        stack.addWidget(self._build_type_page(QtWidgets))
        stack.setCurrentIndex(_PAGE_LIBRARY)

        root.addWidget(stack)
        window.destroyed.connect(self._on_destroyed)
        self._window = window
        self._refresh_objects_table()
        window.show()
        bind_tool_window(window)
        window.raise_()
        window.activateWindow()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents()
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
        self._objects_table = None
        self._editing_obj = None
        self._sphere_stack_index = None
        self._box_stack_index = None
        self._arrow_stack_index = None

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

    def _build_library_page(self, QtCore, QtWidgets):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        title = QtWidgets.QLabel("PyMOLViz")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        subtitle = QtWidgets.QLabel(
            "Objects added with PyMOLViz. Double-click a row to edit."
        )
        subtitle.setWordWrap(True)

        group = QtWidgets.QGroupBox("Objects")
        group_layout = QtWidgets.QVBoxLayout(group)
        table = QtWidgets.QTableWidget(0, len(_OBJECT_COLUMNS))
        table.setHorizontalHeaderLabels(list(_OBJECT_COLUMNS))
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        table.setAlternatingRowColors(True)
        header = table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        table.cellDoubleClicked.connect(self._on_object_activated)
        table.setToolTip("Double-click a row to adjust that CGO.")
        group_layout.addWidget(table)
        self._objects_table = table

        add_btn = QtWidgets.QPushButton("Add")
        add_btn.setToolTip("Create a new visual.")
        add_btn.clicked.connect(lambda: self._goto(_PAGE_TYPES))

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(group, stretch=1)
        layout.addWidget(add_btn)
        return page

    def _build_type_page(self, QtWidgets):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        header = QtWidgets.QHBoxLayout()
        back = QtWidgets.QPushButton("← Back")
        back.setFlat(True)
        back.setToolTip("Return to the object list.")
        back.clicked.connect(lambda: self._goto(_PAGE_LIBRARY))
        title = QtWidgets.QLabel("Add")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        header.addWidget(back)
        header.addWidget(title)
        header.addStretch(1)

        subtitle = QtWidgets.QLabel("Select a type to place.")
        subtitle.setWordWrap(True)

        layout.addLayout(header)
        layout.addWidget(subtitle)
        layout.addSpacing(8)

        for name, hint in MESH_TYPES:
            row = QtWidgets.QVBoxLayout()
            btn = QtWidgets.QPushButton(name)
            btn.clicked.connect(lambda _checked=False, n=name: self._on_mesh_type(n))
            label = QtWidgets.QLabel(hint)
            label.setStyleSheet("color: gray; margin-bottom: 4px;")
            row.addWidget(btn)
            row.addWidget(label)
            layout.addLayout(row)

        layout.addStretch(1)
        return page

    def _refresh_objects_table(self):
        table = self._objects_table
        if table is None:
            return
        from ..runtime.session import all_objects

        QtCore, _, QtWidgets = qt_modules()
        rows = object_rows(all_objects())
        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            values = (row["name"], row["type"], str(row["n_points"]), row["com"])
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                if col == 0:
                    item.setData(QtCore.Qt.UserRole, row["id"])
                table.setItem(i, col, item)

    def _session_object(self, obj_id):
        from ..runtime.session import get as session_get

        return session_get(obj_id)

    def _on_object_activated(self, row, _column):
        table = self._objects_table
        if table is None:
            return
        item = table.item(row, 0)
        if item is None:
            return
        QtCore, _, QtWidgets = qt_modules()
        obj_id = item.data(QtCore.Qt.UserRole)
        obj = self._session_object(obj_id)
        if obj is None:
            return
        kind = editor_kind(obj)
        if not kind:
            if QtWidgets is not None:
                QtWidgets.QMessageBox.information(
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
            else:
                return
        except Exception as exc:
            self.wizard.prompt = ["Mesh builder failed: %s" % exc]
            _, _, QtWidgets = qt_modules()
            if QtWidgets is not None:
                QtWidgets.QMessageBox.warning(
                    self._window,
                    "Add Visual",
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
        self._editing_obj = None
        self._refresh_objects_table()
        self._goto(_PAGE_LIBRARY)

    def _goto(self, index):
        if self._stack is not None:
            self._stack.setCurrentIndex(index)
        if index == _PAGE_LIBRARY:
            self._refresh_objects_table()

    def _on_mesh_type(self, name):
        self._mesh_choice = name
        self.wizard.prompt = ["Mesh: %s" % name]
        try:
            self.wizard.cmd.refresh_wizard()
        except Exception:
            pass
        if name in ("Sphere", "Box", "Arrows"):
            self._open_editor(name, obj=None)
            return
        # Other mesh types remain stubs on the mesh list page.

    def _on_destroyed(self, *_args):
        self._restore_editing_visual()
        self._cleanup_builder_previews()
        self._reset_window()

    def _cleanup_builder_previews(self):
        if self._sphere_page is not None:
            self._sphere_page.cleanup_preview()
        if self._box_page is not None:
            self._box_page.cleanup_preview()
        if self._arrow_page is not None:
            self._arrow_page.cleanup_preview()

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
