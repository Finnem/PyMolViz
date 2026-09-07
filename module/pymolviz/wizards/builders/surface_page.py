"""Surface mesh builder: SAS / MC / GAUSS / ASA around selected points."""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from ...util.mesh_clip import clip_plane_from_view, normalize_clip_planes, oriented_clip_normal
from ...util.solvent_surface import (
    DEFAULT_ALGORITHM,
    DEFAULT_ATOM_RADIUS,
    DEFAULT_PROBE_RADIUS,
    DEFAULT_QUALITY,
    DEFAULT_RADIUS_MODE,
    DEFAULT_VDW_SCALE,
    SURFACE_ALGORITHMS,
    lookup_source_vdw,
)
from ..pick import DeferredCallback, qt_modules, qt_widget_alive
from ..tooltips import (
    EMPTY_PYMOL_SELECTION_MSG,
    HOOK_TO_SELECTION_TIP,
    SNAP_TO_ATOM_TIP,
    UPDATE_TO_CAMERA_TIP,
    UPDATE_TO_SELECTION_TIP,
    apply_required_tooltips,
    warn_missing_setting_tooltips,
)
from ..widgets.log_slider import LogSegmentRadiusWidget
from ..widgets.sticky_add import StickyAddOverlay
from .anchor_table import (
    POINT_NAME_COL,
    POINT_SOURCE_COL,
    POINT_X_COL,
    POINT_Y_COL,
    POINT_Z_COL,
    SURFACE_RADIUS_COL,
    SURFACE_RADIUS_TIP,
    anchor_col_index,
    block_table_selection_signals,
    surface_point_columns,
    sync_anchor_cell,
    unblock_table_selection_signals,
)
from .colors import (
    DEFAULT_SURFACE_COLOR,
    normalize_rgba,
    pick_rgb,
    readable_text_color,
    rgb_to_css,
    rgb_to_hex,
    rgba_to_css,
)
from .load_visual import points_from_mesh, surface_options
from .object_names import unused_object_name
from .points import (
    VisualPoint,
    apply_global_color,
    camera_center_point,
    commit_point_anchors,
    export_points_to_selection,
    selection_points,
    update_points_from_camera,
    update_points_from_selection,
)
from .preview import SurfacePreview, persist_live_preview, retarget_surface_collection
from .runtime_helper import build_surface_collection
from .zoom_selection import ZOOM_TO_SELECTION_TIP, points_from_rows, wire_zoom_to_selection

COLS = surface_point_columns()
_DEFAULT_NAME = "pmv_surface"


class SurfaceBuilderPage:
    """Editor for a solvent-accessible surface around table points."""

    def __init__(
        self,
        cmd_,
        on_back: Callable[[], None],
        on_create: Optional[Callable[[], None]] = None,
        parent=None,
    ):
        self.cmd = cmd_
        self._on_back = on_back
        self._on_create = on_create
        self._points: List[VisualPoint] = []
        self._preview = SurfacePreview(cmd_)
        self._deferred = DeferredCallback()
        self._page = None
        self._table = None
        self._radius_widget = None
        self._probe_widget = None
        self._use_vdw = None
        self._vdw_scale = None
        self._algorithm = None
        self._quality = None
        self._wireframe = None
        self._color_btn = None
        self._color = (*DEFAULT_SURFACE_COLOR, 1.0)
        self._clip_planes = []
        self._suspend_clip_ui = False
        self._clip_list = None
        self._add_clip_btn = None
        self._commit_clip_btn = None
        self._clip_offset = None
        self._clip_tilt = None
        self._clip_turn = None
        self._flip_clip_btn = None
        self._delete_clip_btn = None
        self._snap_atom = None
        self._hook_selection = None
        self._zoom_selection = None
        self._object_name = None
        self._create_btn = None
        self._table_filter = None
        self._editing_id = None
        self._loaded_name = None
        self._suspend_preview = False
        self._sticky_add = None
        self._build(parent)

    @property
    def widget(self):
        return self._page

    def cleanup_preview(self):
        self._deferred.cancel()
        self._preview.cleanup()

    def reset_for_create(self):
        self._editing_id = None
        self._loaded_name = None
        self._points = []
        if self._object_name is not None:
            self._object_name.setText(unused_object_name(_DEFAULT_NAME, self.cmd))
        if self._create_btn is not None:
            self._create_btn.setText("Create CGO")
        if self._radius_widget is not None:
            self._radius_widget.set_value(DEFAULT_ATOM_RADIUS)
        if self._probe_widget is not None:
            self._probe_widget.set_value(DEFAULT_PROBE_RADIUS)
        if self._use_vdw is not None:
            self._use_vdw.setChecked(DEFAULT_RADIUS_MODE == "vdw")
        if self._vdw_scale is not None:
            self._vdw_scale.setValue(DEFAULT_VDW_SCALE)
            self._vdw_scale.setEnabled(DEFAULT_RADIUS_MODE == "vdw")
        if self._algorithm is not None:
            self._algorithm.setCurrentText(DEFAULT_ALGORITHM)
        if self._quality is not None:
            self._quality.setValue(DEFAULT_QUALITY)
        if self._wireframe is not None:
            self._wireframe.setChecked(False)
        self._color = (*DEFAULT_SURFACE_COLOR, 1.0)
        self._paint_color_button()
        self._clip_planes = []
        self._sync_clip_list()
        if self._table is not None:
            self._sync_table()
        self._preview.cleanup()

    def load_object(self, obj):
        from ..catalog import display_name

        self._editing_id = str(obj.id)
        self._loaded_name = display_name(obj) or _DEFAULT_NAME
        self._points = points_from_mesh(obj)
        opts = surface_options(obj)
        self._deferred.cancel()
        self._suspend_preview = True
        try:
            if self._object_name is not None:
                self._object_name.setText(display_name(obj) or _DEFAULT_NAME)
            if self._create_btn is not None:
                self._create_btn.setText("Update CGO")
            if self._radius_widget is not None:
                self._radius_widget.set_value(opts["radius"])
            if self._probe_widget is not None:
                self._probe_widget.set_value(opts["probe_radius"])
            use_vdw = str(opts.get("radius_mode") or DEFAULT_RADIUS_MODE).lower() == "vdw"
            if self._use_vdw is not None:
                self._use_vdw.setChecked(use_vdw)
            if self._vdw_scale is not None:
                self._vdw_scale.setValue(float(opts.get("vdw_scale") or DEFAULT_VDW_SCALE))
                self._vdw_scale.setEnabled(use_vdw)
            if self._algorithm is not None:
                index = self._algorithm.findText(str(opts["algorithm"]))
                if index >= 0:
                    self._algorithm.setCurrentIndex(index)
            if self._quality is not None:
                self._quality.setValue(int(opts["quality"]))
            if self._wireframe is not None:
                self._wireframe.setChecked(opts["wireframe"])
            if self._points:
                self._color = self._points[0].rgba()
            else:
                self._color = (*DEFAULT_SURFACE_COLOR, 1.0)
            self._paint_color_button()
            stored = opts.get("clip_planes") or []
            self._clip_planes = [self._wizard_plane(p, committed=True) for p in stored]
            self._sync_clip_list()
            if self._table is not None:
                self._sync_table()
        finally:
            self._suspend_preview = False
        self._preview.adopt(obj)
        self._schedule_preview()

    def _build(self, parent):
        QtCore, QtGui, QtWidgets = qt_modules()
        if QtWidgets is None:
            raise RuntimeError("PyMOL Qt UI required")

        page = QtWidgets.QWidget(parent)
        root = QtWidgets.QVBoxLayout(page)

        header = QtWidgets.QHBoxLayout()
        back = QtWidgets.QPushButton("← Back")
        back.setFlat(True)
        back.clicked.connect(self._go_back)
        title = QtWidgets.QLabel("Surface")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        header.addWidget(back)
        header.addWidget(title)
        header.addStretch(1)
        root.addLayout(header)

        opts = QtWidgets.QGroupBox("Options")
        opts_layout = QtWidgets.QFormLayout(opts)
        self._algorithm = QtWidgets.QComboBox()
        for name in SURFACE_ALGORITHMS:
            self._algorithm.addItem(name)
        self._algorithm.setCurrentText(DEFAULT_ALGORITHM)
        self._algorithm.currentIndexChanged.connect(self._schedule_preview)
        self._radius_widget = LogSegmentRadiusWidget(initial=DEFAULT_ATOM_RADIUS)
        self._radius_widget.connect_changed(self._on_radius_policy_changed)
        self._probe_widget = LogSegmentRadiusWidget(initial=DEFAULT_PROBE_RADIUS)
        self._probe_widget.connect_changed(self._schedule_preview)
        self._use_vdw = QtWidgets.QCheckBox("Scale by VDW")
        self._use_vdw.setChecked(DEFAULT_RADIUS_MODE == "vdw")
        self._vdw_scale = QtWidgets.QDoubleSpinBox()
        self._vdw_scale.setDecimals(2)
        self._vdw_scale.setRange(0.10, 5.00)
        self._vdw_scale.setSingleStep(0.05)
        self._vdw_scale.setValue(DEFAULT_VDW_SCALE)
        self._vdw_scale.setEnabled(DEFAULT_RADIUS_MODE == "vdw")
        self._use_vdw.toggled.connect(self._on_vdw_toggled)
        self._vdw_scale.valueChanged.connect(lambda *_: self._on_radius_policy_changed())
        vdw_row = QtWidgets.QWidget()
        vdw_layout = QtWidgets.QHBoxLayout(vdw_row)
        vdw_layout.setContentsMargins(0, 0, 0, 0)
        vdw_layout.addWidget(self._use_vdw)
        vdw_layout.addWidget(self._vdw_scale)
        vdw_layout.addStretch(1)
        self._quality = QtWidgets.QSpinBox()
        self._quality.setRange(1, 5)
        self._quality.setValue(DEFAULT_QUALITY)
        self._quality.valueChanged.connect(lambda *_: self._schedule_preview())
        self._wireframe = QtWidgets.QCheckBox("Wireframe")
        self._wireframe.toggled.connect(lambda *_: self._schedule_preview())
        self._color_btn = QtWidgets.QPushButton()
        self._color_btn.setFixedHeight(24)
        self._color_btn.setMinimumWidth(72)
        self._color_btn.clicked.connect(self._pick_surface_color)
        opts_layout.addRow("Algorithm", self._algorithm)
        opts_layout.addRow("Atom radius", self._radius_widget.widget)
        opts_layout.addRow("Probe radius", self._probe_widget.widget)
        opts_layout.addRow("", vdw_row)
        opts_layout.addRow("Quality", self._quality)
        opts_layout.addRow("Color", self._color_btn)
        opts_layout.addRow("", self._wireframe)
        self._paint_color_button()
        root.addWidget(opts)

        clip_box = QtWidgets.QGroupBox("Clip")
        clip_layout = QtWidgets.QVBoxLayout(clip_box)
        clip_btns = QtWidgets.QHBoxLayout()
        self._add_clip_btn = QtWidgets.QPushButton("Add clip plane")
        self._add_clip_btn.clicked.connect(self._add_clip_plane)
        self._commit_clip_btn = QtWidgets.QPushButton("Commit clip")
        self._commit_clip_btn.clicked.connect(self._commit_clip)
        clip_btns.addWidget(self._add_clip_btn)
        clip_btns.addWidget(self._commit_clip_btn)
        clip_layout.addLayout(clip_btns)
        self._clip_list = QtWidgets.QListWidget()
        self._clip_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self._clip_list.currentRowChanged.connect(lambda *_: self._on_clip_selection_changed())
        clip_layout.addWidget(self._clip_list)
        clip_edit = QtWidgets.QFormLayout()
        self._clip_offset = QtWidgets.QDoubleSpinBox()
        self._clip_offset.setDecimals(2)
        self._clip_offset.setRange(-80.0, 80.0)
        self._clip_offset.setSingleStep(0.25)
        self._clip_offset.setSuffix(" Å")
        self._clip_offset.valueChanged.connect(self._on_clip_offset_changed)
        self._clip_tilt = QtWidgets.QDoubleSpinBox()
        self._clip_tilt.setDecimals(0)
        self._clip_tilt.setRange(-180.0, 180.0)
        self._clip_tilt.setSingleStep(5.0)
        self._clip_tilt.setSuffix("°")
        self._clip_tilt.setWrapping(True)
        self._clip_tilt.valueChanged.connect(self._on_clip_tilt_changed)
        self._clip_turn = QtWidgets.QDoubleSpinBox()
        self._clip_turn.setDecimals(0)
        self._clip_turn.setRange(-180.0, 180.0)
        self._clip_turn.setSingleStep(5.0)
        self._clip_turn.setSuffix("°")
        self._clip_turn.setWrapping(True)
        self._clip_turn.valueChanged.connect(self._on_clip_turn_changed)
        clip_edit.addRow("Offset", self._clip_offset)
        clip_edit.addRow("Tilt", self._clip_tilt)
        clip_edit.addRow("Turn", self._clip_turn)
        clip_layout.addLayout(clip_edit)
        clip_edit_btns = QtWidgets.QHBoxLayout()
        self._flip_clip_btn = QtWidgets.QPushButton("Flip keep side")
        self._flip_clip_btn.clicked.connect(self._flip_clip_plane)
        self._delete_clip_btn = QtWidgets.QPushButton("Delete plane")
        self._delete_clip_btn.clicked.connect(self._delete_clip_plane)
        clip_edit_btns.addWidget(self._flip_clip_btn)
        clip_edit_btns.addWidget(self._delete_clip_btn)
        clip_layout.addLayout(clip_edit_btns)
        root.addWidget(clip_box)
        self._sync_clip_list()

        pts_box = QtWidgets.QGroupBox("Points")
        pts_layout = QtWidgets.QVBoxLayout(pts_box)
        self._snap_atom = QtWidgets.QCheckBox("Snap to atom")
        self._snap_atom.setChecked(True)
        self._hook_selection = QtWidgets.QCheckBox("Anchor new points")
        self._hook_selection.setChecked(True)
        self._zoom_selection = QtWidgets.QCheckBox("Zoom to selection")
        export_sel = QtWidgets.QPushButton("Export to selection")
        export_sel.clicked.connect(self._export_selection)
        flags = QtWidgets.QHBoxLayout()
        flags.addWidget(self._snap_atom)
        flags.addWidget(self._hook_selection)
        flags.addWidget(self._zoom_selection)
        flags.addStretch(1)
        flags.addWidget(export_sel)
        pts_layout.addLayout(flags)

        self._table = QtWidgets.QTableWidget(0, len(COLS))
        self._table.setHorizontalHeaderLabels(list(COLS))
        radius_header = self._table.horizontalHeaderItem(SURFACE_RADIUS_COL)
        if radius_header is not None:
            radius_header.setToolTip(SURFACE_RADIUS_TIP)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._table.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.cellChanged.connect(self._on_cell_changed)
        self._table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._show_points_context_menu)
        wire_zoom_to_selection(
            self._table,
            self._zoom_selection,
            self.cmd,
            lambda rows: points_from_rows(self._points, rows),
        )

        class _TableKeyFilter(QtCore.QObject):
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

        self._table_filter = _TableKeyFilter(self)
        self._table.installEventFilter(self._table_filter)
        self._table.viewport().installEventFilter(self._table_filter)
        pts_layout.addWidget(self._table)
        self._sticky_add = StickyAddOverlay(
            pts_box,
            self._table,
            text="+ Add point",
            tooltip=(
                "Add selected atoms, or the camera-center marker if nothing is selected. "
                "With Snap to atom, uses a visible atom within 2 Å of that marker."
            ),
            on_click=self._add_point,
            count=lambda: len(self._points),
            context="SurfaceBuilderPage",
        )
        self._sticky_add.attach()
        root.addWidget(pts_box, stretch=1)

        actions = QtWidgets.QHBoxLayout()
        self._object_name = QtWidgets.QLineEdit()
        self._object_name.setPlaceholderText("Object name")
        self._object_name.setText(unused_object_name(_DEFAULT_NAME, self.cmd))
        self._create_btn = QtWidgets.QPushButton("Create CGO")
        self._create_btn.clicked.connect(self._create_cgo)
        export_btn = QtWidgets.QPushButton("Export CGO")
        export_btn.clicked.connect(self._export_cgo)
        actions.addWidget(self._object_name, stretch=2)
        actions.addWidget(self._create_btn)
        actions.addWidget(export_btn)
        root.addLayout(actions)

        apply_required_tooltips(
            [
                (back, "Return to the mesh type list."),
                (
                    self._algorithm,
                    "SAS: rolling-ball solvent-excluded surface (MSMS / Connolly) "
                    "— geodesic van der Waals caps with probe-radius fillets. "
                    "MC: the same SES from an EDT of the accessible union, extracted "
                    "with marching cubes (watertight grid, no cap/saddle stitches). "
                    "GAUSS: PyMOL map_new gaussian + isosurface — atomic scattering "
                    "Gaussians, B-factor floor, contour at 1σ (blob envelope). "
                    "ASA: solvent-accessible patches on the expanded spheres "
                    "(atom radius + probe).",
                ),
                (self._radius_widget, "Default atom radius in Ångströms for points without a custom R value."),
                (
                    self._probe_widget,
                    "Radius of the rolling solvent sphere in Ångströms "
                    "(1.4 Å is water). Larger probes carve smoother, shallower valleys. "
                    "GAUSS ignores the probe (it uses scattering factors and a B-floor).",
                ),
                (
                    self._use_vdw,
                    "When checked, atom-anchored points use that atom's van der Waals "
                    "radius times the scale instead of the global Atom radius. "
                    "Manual points still use Atom radius unless they have a custom R.",
                ),
                (
                    self._vdw_scale,
                    "Multiplier applied to each atom's van der Waals radius when "
                    "Scale by VDW is on. 1.00 is the unscaled VDW radius.",
                ),
                (self._quality,
                    "Mesh detail from 1 (coarse, fast) to 5 (fine). "
                    "SAS densifies the geodesic template caps and probe fillets; "
                    "MC and GAUSS use a finer voxel grid; ASA uses a denser sphere sampling.",
                ),
                (
                    self._color_btn,
                    "Color and opacity of the whole surface.",
                ),
                (self._wireframe, "Draw the surface as a triangle cage instead of a filled mesh."),
                (
                    self._add_clip_btn,
                    "Insert a clipping plane through the current view: a rectangle "
                    "and an eye on the keep side. The surface is not cut until you "
                    "Commit clip.",
                ),
                (
                    self._commit_clip_btn,
                    "Cut the surface with every listed plane. Geometry on the eye "
                    "side is kept; the other side is discarded. You can still rotate, "
                    "flip, offset, or delete planes afterward.",
                ),
                (
                    self._clip_offset,
                    "Slide the selected plane along its keep-side normal, in Ångströms.",
                ),
                (
                    self._clip_tilt,
                    "Rotate the selected plane around its in-plane horizontal axis.",
                ),
                (
                    self._clip_turn,
                    "Rotate the selected plane around its in-plane vertical axis.",
                ),
                (
                    self._flip_clip_btn,
                    "Swap which side of the selected plane is kept. The eye jumps "
                    "to the new keep side.",
                ),
                (
                    self._delete_clip_btn,
                    "Remove the selected plane. If it was already committed, the "
                    "surface is recut without it.",
                ),
                (self._snap_atom, SNAP_TO_ATOM_TIP),
                (self._hook_selection, HOOK_TO_SELECTION_TIP),
                (self._zoom_selection, ZOOM_TO_SELECTION_TIP),
                (
                    export_sel,
                    "Create a PyMOL selection covering the table points as pseudoatoms.",
                ),
                (self._object_name, "Name of the PyMOL CGO object created or exported."),
                (self._create_btn, "Commit the surface to the session as a named CGO object."),
                (export_btn, "Write a Python script that rebuilds this CGO."),
            ],
            context="SurfaceBuilderPage",
        )
        self._page = page
        warn_missing_setting_tooltips(page, context="SurfaceBuilderPage")

    def _wizard_plane(self, stored, committed=False) -> dict:
        plane = normalize_clip_planes([stored])
        if not plane:
            origin = [0.0, 0.0, 0.0]
            normal = [0.0, 0.0, 1.0]
            scale = 5.0
        else:
            origin = list(plane[0]["origin"])
            normal = list(plane[0]["normal"])
            scale = float(plane[0]["scale"])
        return {
            "origin": origin,
            "normal": normal,
            "scale": scale,
            "base": list(origin),
            "base_normal": list(normal),
            "offset": 0.0,
            "tilt": 0.0,
            "turn": 0.0,
            "committed": bool(committed),
        }

    def _committed_clip_planes(self):
        return normalize_clip_planes(
            [p for p in self._clip_planes if p.get("committed")]
        )

    def _selected_clip_index(self):
        if self._clip_list is None:
            return None
        row = int(self._clip_list.currentRow())
        if row < 0 or row >= len(self._clip_planes):
            return None
        return row

    def _clip_origin_from_offset(self, plane) -> None:
        base = np.asarray(plane.get("base", plane["origin"]), dtype=float).reshape(3)
        normal = np.asarray(plane["normal"], dtype=float).reshape(3)
        ln = float(np.linalg.norm(normal))
        if ln > 1e-12:
            normal = normal / ln
        origin = base + normal * float(plane.get("offset", 0.0))
        plane["origin"] = [float(origin[0]), float(origin[1]), float(origin[2])]
        plane["normal"] = [float(normal[0]), float(normal[1]), float(normal[2])]

    def _apply_clip_orientation(self, plane) -> None:
        base_n = plane.get("base_normal", plane["normal"])
        tilted = oriented_clip_normal(
            base_n,
            float(plane.get("tilt", 0.0)),
            float(plane.get("turn", 0.0)),
        )
        plane["normal"] = [float(tilted[0]), float(tilted[1]), float(tilted[2])]
        self._clip_origin_from_offset(plane)

    def _sync_clip_fields(self):
        idx = self._selected_clip_index()
        has = idx is not None
        if self._clip_offset is not None:
            self._clip_offset.setEnabled(has)
        if self._clip_tilt is not None:
            self._clip_tilt.setEnabled(has)
        if self._clip_turn is not None:
            self._clip_turn.setEnabled(has)
        if self._flip_clip_btn is not None:
            self._flip_clip_btn.setEnabled(has)
        if self._delete_clip_btn is not None:
            self._delete_clip_btn.setEnabled(has)
        if self._commit_clip_btn is not None:
            self._commit_clip_btn.setEnabled(
                any(not p.get("committed") for p in self._clip_planes)
            )
        if not has:
            return
        plane = self._clip_planes[idx]
        if self._clip_offset is not None:
            self._clip_offset.setValue(float(plane.get("offset", 0.0)))
        if self._clip_tilt is not None:
            self._clip_tilt.setValue(float(plane.get("tilt", 0.0)))
        if self._clip_turn is not None:
            self._clip_turn.setValue(float(plane.get("turn", 0.0)))

    def _sync_clip_list(self, keep_row=None):
        if self._clip_list is None:
            return
        self._suspend_clip_ui = True
        try:
            current = self._clip_list.currentRow() if keep_row is None else int(keep_row)
            self._clip_list.clear()
            for i, plane in enumerate(self._clip_planes):
                status = "draft" if not plane.get("committed") else "cut"
                self._clip_list.addItem("Plane %d · %s" % (i + 1, status))
            if self._clip_planes:
                row = current
                if row < 0:
                    row = 0
                if row >= len(self._clip_planes):
                    row = len(self._clip_planes) - 1
                self._clip_list.setCurrentRow(row)
            self._sync_clip_fields()
        finally:
            self._suspend_clip_ui = False

    def _refresh_clip_gizmos(self):
        span = self._preview.span_points()
        if span is None:
            span = self._clip_fallback_points()
        self._preview.set_gizmos(
            self._clip_planes,
            selected_index=self._selected_clip_index(),
            span_points=span,
        )

    def _clip_fallback_points(self):
        if not self._points:
            return None
        radius, probe, *_rest = self._params()
        pad = max(float(radius) + float(probe), 1.0)
        corners = []
        for pt in self._points:
            x, y, z = float(pt.x), float(pt.y), float(pt.z)
            for dx in (-pad, pad):
                for dy in (-pad, pad):
                    for dz in (-pad, pad):
                        corners.append((x + dx, y + dy, z + dz))
        return np.asarray(corners, dtype=float)

    def _clip_changed(self, recut: bool):
        if recut:
            self._schedule_preview()
        else:
            self._refresh_clip_gizmos()

    def _on_clip_selection_changed(self):
        if self._suspend_clip_ui:
            return
        self._suspend_clip_ui = True
        try:
            self._sync_clip_fields()
        finally:
            self._suspend_clip_ui = False
        self._refresh_clip_gizmos()

    def _add_clip_plane(self):
        if not self._points:
            return
        try:
            view = tuple(self.cmd.get_view())
        except Exception:
            return
        xyz = [(pt.x, pt.y, pt.z) for pt in self._points]
        stored = clip_plane_from_view(view, xyz)
        self._clip_planes.append(self._wizard_plane(stored, committed=False))
        self._sync_clip_list(keep_row=len(self._clip_planes) - 1)
        self._refresh_clip_gizmos()

    def _commit_clip(self):
        if not self._clip_planes:
            return
        for plane in self._clip_planes:
            plane["committed"] = True
        self._sync_clip_list()
        self._schedule_preview()

    def _flip_clip_plane(self):
        idx = self._selected_clip_index()
        if idx is None:
            return
        plane = self._clip_planes[idx]
        plane["base_normal"] = [
            -float(x) for x in plane.get("base_normal", plane["normal"])
        ]
        plane["offset"] = -float(plane.get("offset", 0.0))
        self._apply_clip_orientation(plane)
        self._suspend_clip_ui = True
        try:
            self._sync_clip_fields()
        finally:
            self._suspend_clip_ui = False
        self._clip_changed(recut=bool(plane.get("committed")))

    def _delete_clip_plane(self):
        idx = self._selected_clip_index()
        if idx is None:
            return
        was_committed = bool(self._clip_planes[idx].get("committed"))
        del self._clip_planes[idx]
        self._sync_clip_list(keep_row=min(idx, len(self._clip_planes) - 1))
        if was_committed:
            self._schedule_preview()
        else:
            self._refresh_clip_gizmos()

    def _on_clip_offset_changed(self, value):
        if self._suspend_clip_ui:
            return
        idx = self._selected_clip_index()
        if idx is None:
            return
        plane = self._clip_planes[idx]
        plane["offset"] = float(value)
        self._clip_origin_from_offset(plane)
        self._clip_changed(recut=bool(plane.get("committed")))

    def _on_clip_tilt_changed(self, value):
        if self._suspend_clip_ui:
            return
        idx = self._selected_clip_index()
        if idx is None:
            return
        plane = self._clip_planes[idx]
        plane["tilt"] = float(value)
        self._apply_clip_orientation(plane)
        self._clip_changed(recut=bool(plane.get("committed")))

    def _on_clip_turn_changed(self, value):
        if self._suspend_clip_ui:
            return
        idx = self._selected_clip_index()
        if idx is None:
            return
        plane = self._clip_planes[idx]
        plane["turn"] = float(value)
        self._apply_clip_orientation(plane)
        self._clip_changed(recut=bool(plane.get("committed")))

    def _params(self):
        use_vdw = bool(self._use_vdw.isChecked()) if self._use_vdw is not None else False
        vdw_scale = float(self._vdw_scale.value()) if self._vdw_scale is not None else DEFAULT_VDW_SCALE
        return (
            self._radius_widget.value(),
            self._probe_widget.value(),
            self._algorithm.currentText(),
            int(self._quality.value()),
            self._wireframe.isChecked(),
            use_vdw,
            vdw_scale,
        )

    def _radius_mode(self) -> str:
        return "vdw" if self._params()[5] else "uniform"

    def _inherited_radius(self, pt) -> float:
        atom_radius, _probe, _alg, _qual, _wire, use_vdw, vdw_scale = self._params()
        if use_vdw:
            vdw = lookup_source_vdw(pt.point_source, None)
            if vdw is not None:
                return float(vdw) * float(vdw_scale)
        return float(atom_radius)

    def _on_vdw_toggled(self, checked: bool):
        if self._vdw_scale is not None:
            self._vdw_scale.setEnabled(bool(checked))
        self._on_radius_policy_changed()

    def _on_radius_policy_changed(self):
        if self._table is not None:
            self._sync_table(preview=True)
        else:
            self._schedule_preview()

    def _selected_rows(self) -> List[int]:
        return sorted({i.row() for i in self._table.selectedIndexes()})

    def _paint_color_button(self):
        if self._color_btn is None:
            return
        rgba = normalize_rgba(self._color)
        rgb = rgba[:3]
        fg = readable_text_color(rgb)
        self._color_btn.setText(rgb_to_hex(rgb))
        self._color_btn.setStyleSheet(
            "QPushButton { background-color: %s; color: %s; border: 1px solid #666; }"
            % (rgba_to_css(rgba), rgb_to_css(fg))
        )

    def _set_surface_color(self, rgba, preview=True):
        self._color = normalize_rgba(rgba)
        apply_global_color(self._points, self._color)
        self._paint_color_button()
        if preview:
            self._schedule_preview()

    def _pick_surface_color(self):
        initial = normalize_rgba(self._color)
        original = initial

        def on_preview(rgba):
            self._set_surface_color(rgba, preview=True)

        def on_done(rgba):
            if rgba is None:
                self._set_surface_color(original, preview=True)
            else:
                self._set_surface_color(rgba, preview=True)

        pick_rgb(self._page, initial, on_change=on_preview, on_done=on_done)

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
        reset_r = menu.addAction("Reset radius to default")
        reset_r.setEnabled(bool(rows))
        reset_r.triggered.connect(self._reset_selected_radii)
        menu.exec_(self._table.viewport().mapToGlobal(pos))

    def _stamp_new_points(self, new_pts: List[VisualPoint]) -> List[VisualPoint]:
        color = normalize_rgba(self._color)
        return [pt.with_color(color) for pt in new_pts]

    def _go_back(self):
        self._deferred.cancel()
        self._preview.cleanup()
        self._on_back()

    def _schedule_preview(self):
        if self._suspend_preview:
            return
        self._deferred.schedule(self._refresh_preview, page=self._page)

    def _refresh_preview(self):
        if not qt_widget_alive(self._page):
            return
        try:
            if not self._points:
                self._preview.cleanup()
                return
            radius, probe, algorithm, quality, wireframe, _use_vdw, vdw_scale = self._params()
            self._preview.update(
                self._points, radius, probe, algorithm, quality, wireframe,
                radius_mode=self._radius_mode(), vdw_scale=vdw_scale,
                clip_planes=self._committed_clip_planes(),
                gizmo_planes=self._clip_planes,
                gizmo_selected=self._selected_clip_index(),
            )
        except Exception:
            pass

    def _anchor_col(self) -> int:
        return anchor_col_index(COLS)

    def _on_anchor_toggled(self, row: int, checked: bool):
        if row < 0 or row >= len(self._points):
            return
        pt = self._points[row]
        if not pt.can_anchor():
            return
        self._points[row] = pt.with_anchor_intent(checked)

    def _sync_table(self, preview=True):
        QtCore, QtGui, QtWidgets = qt_modules()
        anchor_col = self._anchor_col()
        sel_blocked = block_table_selection_signals(self._table)
        self._table.blockSignals(True)
        try:
            self._table.setRowCount(len(self._points))
            for row, pt in enumerate(self._points):
                sync_anchor_cell(
                    self._table, row, anchor_col, pt,
                    self._on_anchor_toggled, QtWidgets, QtCore,
                )
                values = (
                    (POINT_NAME_COL, pt.name),
                    (POINT_SOURCE_COL, pt.source),
                    (POINT_X_COL, "%.3f" % pt.x),
                    (POINT_Y_COL, "%.3f" % pt.y),
                    (POINT_Z_COL, "%.3f" % pt.z),
                    (SURFACE_RADIUS_COL, "%.2f" % (
                        float(pt.radius) if pt.radius is not None else self._inherited_radius(pt)
                    )),
                )
                for col, text in values:
                    item = self._table.item(row, col)
                    if item is None:
                        item = QtWidgets.QTableWidgetItem()
                        self._table.setItem(row, col, item)
                    item.setText(text)
                    item.setToolTip(SURFACE_RADIUS_TIP if col == SURFACE_RADIUS_COL else "")
                    if col == SURFACE_RADIUS_COL:
                        font = item.font()
                        font.setItalic(pt.radius is None)
                        item.setFont(font)
                        item.setBackground(QtGui.QBrush())
                        item.setForeground(QtGui.QBrush())
                    else:
                        item.setBackground(QtGui.QBrush())
                        item.setForeground(QtGui.QBrush())
        finally:
            self._table.blockSignals(False)
            unblock_table_selection_signals(self._table, sel_blocked)
        if getattr(self, "_sticky_add", None) is not None:
            self._sticky_add.sync()
        if preview:
            self._schedule_preview()

    def _on_cell_changed(self, row, col):
        if row < 0 or row >= len(self._points):
            return
        item = self._table.item(row, col)
        if item is None:
            return
        text = item.text()
        pt = self._points[row]
        if col == anchor_col_index(COLS):
            return
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
            elif col == SURFACE_RADIUS_COL:
                stripped = text.strip()
                if not stripped:
                    pt = pt.with_radius(None)
                else:
                    pt = pt.with_radius(float(stripped))
            else:
                return
            self._points[row] = pt
        except ValueError:
            self._sync_table()
            return
        if col == SURFACE_RADIUS_COL:
            self._sync_table()
            return
        self._schedule_preview()

    def _warn_empty_pymol_selection(self, title):
        _, _, QtWidgets = qt_modules()
        if QtWidgets is not None:
            QtWidgets.QMessageBox.information(self._page, title, EMPTY_PYMOL_SELECTION_MSG)

    def _update_selected_to_camera(self):
        rows = self._selected_rows()
        if not rows:
            return
        self._points = update_points_from_camera(
            self.cmd, self._points, rows, self._snap_atom.isChecked(),
            hook_to_selection=self._hook_selection.isChecked(),
        )
        self._sync_table()

    def _update_selected_to_selection(self):
        rows = self._selected_rows()
        if not rows:
            return
        updated = update_points_from_selection(
            self.cmd, self._points, rows,
            hook_to_selection=self._hook_selection.isChecked(),
        )
        if updated is None:
            self._warn_empty_pymol_selection("Update to selection")
            return
        self._points = updated
        self._sync_table()

    def _add_point(self):
        new_pts = selection_points(
            self.cmd, self._points, hook_to_selection=self._hook_selection.isChecked(),
        )
        if new_pts:
            self._points.extend(self._stamp_new_points(new_pts))
            self._sync_table()
            return
        self._add_camera_center()

    def _add_camera_center(self):
        pt = camera_center_point(
            self.cmd, self._snap_atom.isChecked(), self._points,
            hook_to_selection=self._hook_selection.isChecked(),
        )
        self._points.extend(self._stamp_new_points([pt]))
        self._sync_table()

    def _delete_selected(self):
        rows = sorted({i.row() for i in self._table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        self._deferred.cancel()
        for row in rows:
            if 0 <= row < len(self._points):
                del self._points[row]
        self._sync_table()

    def _export_selection(self):
        export_points_to_selection(self.cmd, self._points)

    def _reset_selected_radii(self):
        rows = self._selected_rows()
        if not rows:
            return
        for row in rows:
            if 0 <= row < len(self._points):
                self._points[row] = self._points[row].with_radius(None)
        self._sync_table()

    def _collection(self, name: str):
        radius, probe, algorithm, quality, wireframe, _use_vdw, vdw_scale = self._params()
        return build_surface_collection(
            commit_point_anchors(self._points),
            radius, probe, algorithm, quality, wireframe, name,
            radius_mode=self._radius_mode(), vdw_scale=vdw_scale,
            clip_planes=self._committed_clip_planes(),
        )

    def _create_cgo(self):
        if not self._points:
            return
        for plane in self._clip_planes:
            plane["committed"] = True
        self._sync_clip_list()
        typed = self._object_name.text().strip() or _DEFAULT_NAME
        name = unused_object_name(typed, self.cmd, keep=self._loaded_name)
        radius, probe, algorithm, quality, wireframe, _use_vdw, vdw_scale = self._params()
        persist_live_preview(
            self.cmd,
            self._preview,
            name,
            obj_id=self._editing_id,
            retarget=lambda coll: retarget_surface_collection(
                coll, commit_point_anchors(self._points),
                radius, probe, algorithm, quality, wireframe,
                radius_mode=self._radius_mode(), vdw_scale=vdw_scale,
                clip_planes=self._committed_clip_planes(),
            ),
            fallback=lambda: self._collection(name),
        )
        if self._on_create is not None:
            self._on_create()

    def _export_cgo(self):
        if not self._points:
            return
        _, _, QtWidgets = qt_modules()
        name = self._object_name.text().strip() or _DEFAULT_NAME
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self._page, "Export CGO script", "%s.py" % name, "Python (*.py)",
        )
        if not path:
            return
        self._collection(name).write(path)
