"""Standalone window for building / adding visuals (CGOs)."""

from .builders.arrow_page import ArrowBuilderPage
from .builders.box_page import BoxBuilderPage
from .builders.sphere_page import SphereBuilderPage
from .pick import (
    bind_tool_window,
    configure_tool_window,
    find_pymol_window,
    qt_modules,
)
from .pick import _qt_platform_name

# Category → mesh primitive chooser. Field is a placeholder for now.
MESH_TYPES = (
    ("Sphere", "Solid or wireframe sphere"),
    ("Box", "Axis-aligned or centered box"),
    ("Surface", "Triangulated mesh surface"),
    ("Lines", "Line / polyline segments"),
    ("Arrows", "Directed arrow glyphs"),
)


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
        self._mesh_page_index = 1
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
                    "Add Visual",
                    "Could not open Add Visual:\n\n%s" % exc,
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
        anchor = find_pymol_window(QtWidgets)
        window = QtWidgets.QDialog(anchor)
        window.setWindowTitle("Add Visual")
        window.setModal(False)
        configure_tool_window(window, anchor=anchor)
        window.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        window.resize(540, 720)

        root = QtWidgets.QVBoxLayout(window)
        stack = QtWidgets.QStackedWidget()
        self._stack = stack

        stack.addWidget(self._build_category_page(QtWidgets))
        stack.addWidget(self._build_mesh_page(QtWidgets))
        stack.addWidget(self._build_field_page(QtWidgets))
        stack.setCurrentIndex(0)

        root.addWidget(stack)
        window.destroyed.connect(self._on_destroyed)
        self._window = window
        window.show()
        bind_tool_window(window)
        window.raise_()
        window.activateWindow()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents()
        if not window.isVisible():
            raise RuntimeError(
                "Add Visual window did not become visible "
                "(Qt platform=%r)" % (_qt_platform_name(),)
            )

    def _reset_window(self):
        self._window = None
        self._stack = None
        self._sphere_page = None
        self._box_page = None
        self._arrow_page = None
        self._sphere_stack_index = None
        self._box_stack_index = None
        self._arrow_stack_index = None

    def _ensure_sphere_page(self):
        if self._sphere_page is not None:
            return
        parent = self._window
        self._sphere_page = SphereBuilderPage(
            self.wizard.cmd,
            on_back=lambda: self._goto(self._mesh_page_index),
            on_create=self.close,
            parent=parent,
        )
        self._sphere_stack_index = self._stack.addWidget(self._sphere_page.widget)

    def _ensure_box_page(self):
        if self._box_page is not None:
            return
        parent = self._window
        self._box_page = BoxBuilderPage(
            self.wizard.cmd,
            on_back=lambda: self._goto(self._mesh_page_index),
            on_create=self.close,
            parent=parent,
        )
        self._box_stack_index = self._stack.addWidget(self._box_page.widget)

    def _ensure_arrow_page(self):
        if self._arrow_page is not None:
            return
        parent = self._window
        self._arrow_page = ArrowBuilderPage(
            self.wizard.cmd,
            on_back=lambda: self._goto(self._mesh_page_index),
            on_create=self.close,
            parent=parent,
        )
        self._arrow_stack_index = self._stack.addWidget(self._arrow_page.widget)

    def _build_category_page(self, QtWidgets):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        title = QtWidgets.QLabel("Add Visual")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        subtitle = QtWidgets.QLabel("Choose what kind of object to create.")
        subtitle.setWordWrap(True)

        mesh_btn = QtWidgets.QPushButton("Mesh")
        mesh_btn.setToolTip("Geometric CGOs: spheres, boxes, surfaces, lines, arrows")
        mesh_hint = QtWidgets.QLabel("Spheres, boxes, surfaces, lines, arrows")
        mesh_hint.setStyleSheet("color: gray; margin-bottom: 8px;")

        field_btn = QtWidgets.QPushButton("Field")
        field_btn.setToolTip("Volumetric data: volumes, isosurfaces, isomeshes")
        field_hint = QtWidgets.QLabel("Volumes, isosurfaces, isomeshes (coming soon)")
        field_hint.setStyleSheet("color: gray;")

        mesh_btn.clicked.connect(lambda: self._goto(1))
        field_btn.clicked.connect(lambda: self._goto(2))

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(12)
        layout.addWidget(mesh_btn)
        layout.addWidget(mesh_hint)
        layout.addWidget(field_btn)
        layout.addWidget(field_hint)
        layout.addStretch(1)
        return page

    def _build_mesh_page(self, QtWidgets):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        header = QtWidgets.QHBoxLayout()
        back = QtWidgets.QPushButton("← Back")
        back.setFlat(True)
        back.clicked.connect(lambda: self._goto(0))
        title = QtWidgets.QLabel("Mesh")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        header.addWidget(back)
        header.addWidget(title)
        header.addStretch(1)

        subtitle = QtWidgets.QLabel("Select a mesh primitive to place.")
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

    def _build_field_page(self, QtWidgets):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        header = QtWidgets.QHBoxLayout()
        back = QtWidgets.QPushButton("← Back")
        back.setFlat(True)
        back.clicked.connect(lambda: self._goto(0))
        title = QtWidgets.QLabel("Field")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        header.addWidget(back)
        header.addWidget(title)
        header.addStretch(1)

        body = QtWidgets.QLabel(
            "Volumetric tools (volumes, isosurfaces, isomeshes) "
            "are not wired up yet."
        )
        body.setWordWrap(True)

        layout.addLayout(header)
        layout.addSpacing(8)
        layout.addWidget(body)
        layout.addStretch(1)
        return page

    def _goto(self, index):
        if self._stack is not None:
            self._stack.setCurrentIndex(index)

    def _on_mesh_type(self, name):
        self._mesh_choice = name
        self.wizard.prompt = ["Mesh: %s" % name]
        try:
            self.wizard.cmd.refresh_wizard()
        except Exception:
            pass
        try:
            if name == "Sphere":
                self._ensure_sphere_page()
                if self._stack is not None:
                    self._stack.setCurrentIndex(self._sphere_stack_index)
                return
            if name == "Box":
                self._ensure_box_page()
                if self._stack is not None:
                    self._stack.setCurrentIndex(self._box_stack_index)
                return
            if name == "Arrows":
                self._ensure_arrow_page()
                if self._stack is not None:
                    self._stack.setCurrentIndex(self._arrow_stack_index)
                return
        except Exception as exc:
            self.wizard.prompt = ["Mesh builder failed: %s" % exc]
            _, _, QtWidgets = qt_modules()
            if QtWidgets is not None:
                QtWidgets.QMessageBox.warning(
                    self._window,
                    "Add Visual",
                    "Could not open %s builder:\n\n%s" % (name, exc),
                )
        # Other mesh types remain stubs on the mesh list page.

    def _on_destroyed(self, *_args):
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
        self._cleanup_builder_previews()
        window = self._window
        self._reset_window()
        if window is None:
            return
        try:
            window.close()
        except RuntimeError:
            pass
