"""Standalone window for building / adding visuals (CGOs)."""

from .pick import qt_modules

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

    def show(self):
        QtCore, _, QtWidgets = qt_modules()
        if QtWidgets is None:
            self.wizard.prompt = ["Add Visual requires the PyMOL Qt UI"]
            return

        if self._window is not None:
            try:
                if self._window.isVisible():
                    self._window.raise_()
                    self._window.activateWindow()
                    return
            except RuntimeError:
                self._window = None

        window = QtWidgets.QWidget()
        window.setWindowTitle("Add Visual")
        window.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        window.resize(420, 360)

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
        window.raise_()
        window.activateWindow()

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

    def _on_destroyed(self, *_args):
        self._window = None
        self._stack = None

    def close(self):
        window = self._window
        self._window = None
        self._stack = None
        if window is None:
            return
        try:
            window.close()
        except RuntimeError:
            pass
