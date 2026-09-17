"""Builder: Field from the current atom selection (Gaussian, distance, VDW, property)."""

from __future__ import annotations

from ...fields.domain import BOUNDS_AROUND_SELECTION, BOUNDS_CUSTOM_BOX, BOUNDS_OBJECT, Domain
from ...fields.identity import GEN_GAUSSIAN, GEN_NEAREST_PROP
from ...util.gaussian_map import DEFAULT_GAUSSIAN_ISOLEVEL, DEFAULT_GAUSSIAN_RESOLUTION
from ...util.solvent_surface import (
    DEFAULT_QUALITY,
    estimate_surface_job,
    gauss_spacing,
)
from .heavy_job import ask_heavy_job
from ..pick import (
    overlay_question,
    overlay_warning,
    qt_modules,
    qt_widget_alive,
)
from ..widgets.ascii_locale import apply_ascii_float_locale, configure_committed_spin
from ..widgets.spin_step import (
    STEP_ANGSTROM,
    STEP_ANGSTROM_COARSE,
    STEP_RESOLUTION,
    STEP_SPACING,
    apply_spin_step,
)
from ..widgets.breadcrumb import CRUMB_FROM_SELECTION, create_field_crumbs, edit_field_crumbs
from ..widgets.section import make_section
from .field_params import (
    BOUNDS_MODE_LABELS,
    FIELD_MODEL_LABELS,
    FIELD_MODEL_ORDER,
    NEAREST_PROPERTIES,
    field_model_shows,
    iso_spin_range,
    normalize_field_model,
)
from .object_names import unused_object_name
from .domain_schematic import (
    SCHEMATIC_TIP,
    DomainSchematicWidget,
    domain_schematic_scene,
    wrap_domain_knobs,
)
from .field_preview import (
    build_field_iso_preview_visual,
    default_field_iso_level,
    field_supports_iso_preview,
    iso_preview_key,
)
from .preview_mode import (
    DEFAULT_PREVIEW_MODE,
    PREVIEW_OFF,
    read_preview_mode,
    stamp_preview_mode,
    preview_domain,
    preview_is_on,
    preview_iso_kind,
    preview_is_simple,
)
from .appearance_section import COLOR_ALL_LABEL
from .load_field import (
    atom_records_from_points,
    commit_from_selection_preset,
    field_from_selection,
    field_options,
    points_from_field,
    remember_default_color_field,
)
from .point_table_page import PointTableBuilderPage
from .points import enabled_points
from .preview import FromSelectionFieldPreview
from .export import export_objects
from .surface_params import COLOR_MODE_FIELD, COLOR_MODE_UNIFORM

_MARKER_RADIUS = 0.40
_QUALITY_TIP = (
    "Voxel spacing of the Gaussian map from 1 (draft) to 5 (fine). "
    "Same ladder as Gaussian Spheres on the explicit Surface builder."
)
_RESOLUTION_TIP = (
    "PyMOL gaussian_resolution in Ångströms. Smaller values tighten the blobs "
    "(default 2.0, matching map_new gaussian)."
)
_ISO_TIP = (
    "Isovalue for the live isosurface preview and the IsoSurface created on Done. "
    "Gaussian uses normalized map units (default 1.0, matching PyMOL isosurface "
    "on map_new gaussian). Distance is in Å (default 1.5). Signed VDW is 0 at "
    "the VDW surface. Nearest-atom property uses native property units."
)
_LIVE_PREVIEW_TIP = (
    "No preview keeps cheap atom markers only. Simple preview rebuilds a "
    "coarser IsoMesh. Full preview is the native isosurface Done creates."
)
_MODEL_TIP = "How voxel values are computed from the selected atoms."
_PROPERTY_TIP = "Atom property copied onto the nearest voxel. Element and chain are categorical."
_BOUNDS_TIP = "How the voxel box is placed around the source atoms."
_PADDING_TIP = "Extra padding in Ångströms added around the bounds."
_SPACING_TIP = "Grid step in Ångströms. Smaller values make a finer field."


class FromSelectionFieldPage(PointTableBuilderPage):
    """Select atoms, pick a field model, then commit a reusable Field."""

    DEFAULT_NAME = "pmv_field"
    CRUMB_LEAF = CRUMB_FROM_SELECTION
    CONTEXT = "FromSelectionFieldPage"

    def _appearance_config(self) -> dict:
        return {
            "show_wireframe": False,
            "show_quality": False,
            "show_per_point": False,
            "show_live_preview": True,
            "live_preview_tooltip": _LIVE_PREVIEW_TIP,
        }

    def _create_breadcrumb(self):
        return create_field_crumbs(self.CRUMB_LEAF)

    def _edit_breadcrumb(self):
        return edit_field_crumbs(self.CRUMB_LEAF)

    def _make_preview(self):
        return FromSelectionFieldPreview(self.cmd)

    def _init_options_state(self):
        self._algorithm = None
        self._quality = None
        self._resolution = None
        self._iso_value = None
        self._property = None
        self._bounds_mode = None
        self._padding = None
        self._spacing = None
        self._domain_object = None
        self._box_lo = None
        self._box_hi = None
        self._geom_form = None
        self._opt_form = None
        self._domain_form = None
        self._domain_schematic = None
        self._quality_row = None
        self._resolution_row = None
        self._iso_row = None
        self._property_row = None
        self._object_row = None
        self._box_row = None
        self._heavy_ok = None
        self._heavy_denied = None
        self._pending_field_opts = None

    def _mount_modifiers(self, root, QtCore, QtGui, QtWidgets):
        return ()

    def load_object(self, obj):
        from ..catalog import display_name

        if type(obj).__name__ != "Field":
            return super().load_object(obj)
        self._editing_id = str(getattr(obj, "id", "") or "")
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
        finally:
            self._suspend_preview = False
        if self._preview is not None:
            self._preview.cleanup()
        self._after_load()

    def _points_from_object(self, obj):
        if type(obj).__name__ == "Field":
            return points_from_field(obj)
        return super()._points_from_object(obj)

    def _load_options(self, obj):
        opts = field_options(obj)
        self._pending_field_opts = opts
        algo = opts.get("algorithm")
        if self._algorithm is not None and algo:
            index = self._algorithm.findData(algo)
            self._algorithm.blockSignals(True)
            try:
                if index >= 0:
                    self._algorithm.setCurrentIndex(index)
            finally:
                self._algorithm.blockSignals(False)
        if self._quality is not None:
            self._quality.blockSignals(True)
            try:
                self._quality.setValue(int(opts.get("quality") or DEFAULT_QUALITY))
            finally:
                self._quality.blockSignals(False)
        if self._resolution is not None:
            self._resolution.blockSignals(True)
            try:
                self._resolution.setValue(float(opts.get("resolution") or DEFAULT_GAUSSIAN_RESOLUTION))
            finally:
                self._resolution.blockSignals(False)
        self._apply_iso_spin()
        iso = opts.get("iso_level")
        if self._iso_value is not None and iso is not None:
            self._iso_value.blockSignals(True)
            try:
                self._iso_value.setValue(float(iso))
            finally:
                self._iso_value.blockSignals(False)
        prop = opts.get("property")
        if self._property is not None and prop:
            index = self._property.findData(prop)
            if index >= 0:
                self._property.setCurrentIndex(index)
        domain = opts.get("domain") or {}
        mode = domain.get("bounds_mode") if isinstance(domain, dict) else None
        if self._bounds_mode is not None and mode:
            index = self._bounds_mode.findData(mode)
            self._bounds_mode.blockSignals(True)
            try:
                if index >= 0:
                    self._bounds_mode.setCurrentIndex(index)
            finally:
                self._bounds_mode.blockSignals(False)
        if self._padding is not None and domain.get("padding") is not None:
            self._padding.setValue(float(domain["padding"]))
        if self._spacing is not None and domain.get("spacing") is not None:
            self._spacing.setValue(float(domain["spacing"]))
        if self._domain_object is not None:
            self._domain_object.setText(str(domain.get("object_name") or ""))
        aabb = domain.get("aabb") if isinstance(domain, dict) else None
        if aabb and self._box_lo and self._box_hi:
            try:
                lo, hi = aabb[0], aabb[1]
                for i in range(3):
                    self._box_lo[i].setValue(float(lo[i]))
                    self._box_hi[i].setValue(float(hi[i]))
            except (TypeError, IndexError, ValueError):
                pass
        self._sync_algorithm_visibility()
        self._sync_domain_visibility()

    def _after_load(self):
        opts = getattr(self, "_pending_field_opts", None) or {}
        if self._appearance is not None:
            fid = opts.get("color_field_id")
            if fid:
                self._appearance._set_mode_ui(COLOR_MODE_FIELD)
                picker = getattr(self._appearance, "_field_picker", None)
                if picker is not None:
                    picker.refresh(fid)
                cmap = opts.get("colormap")
                editor = getattr(self._appearance, "_cmap_editor", None)
                if cmap and editor is not None:
                    editor.set_colormap(str(cmap))
            else:
                self._appearance._set_mode_ui(COLOR_MODE_UNIFORM)
            self._appearance._sync_mode_widgets()
        super()._after_load()

    def _context_color_action(self) -> bool:
        return False

    def _extend_points_context_menu(self, menu, rows):
        del rows
        apply_all = menu.addAction(COLOR_ALL_LABEL)
        apply_all.triggered.connect(lambda: self._appearance and self._appearance._color_all())

    def _reset_options(self):
        if self._algorithm is not None:
            self._algorithm.setCurrentIndex(0)
        if self._quality is not None:
            self._quality.setValue(DEFAULT_QUALITY)
        if self._resolution is not None:
            self._resolution.setValue(DEFAULT_GAUSSIAN_RESOLUTION)
        self._apply_iso_spin()
        if self._iso_value is not None:
            self._iso_value.setValue(default_field_iso_level(self._current_algorithm()))
        if self._appearance is not None:
            self._appearance.set_preview_mode(DEFAULT_PREVIEW_MODE)
        if self._property is not None:
            self._property.setCurrentIndex(0)
        if self._bounds_mode is not None:
            self._bounds_mode.setCurrentIndex(0)
        if self._padding is not None:
            self._padding.setValue(0.0)
        if self._spacing is not None:
            self._spacing.setValue(gauss_spacing(DEFAULT_QUALITY))
        self._heavy_ok = None
        self._heavy_denied = None
        self._pending_field_opts = None
        if self._appearance is not None:
            self._appearance._set_mode_ui(COLOR_MODE_UNIFORM)
            self._appearance._sync_mode_widgets()
        self._sync_algorithm_visibility()
        self._sync_domain_visibility()

    def _current_algorithm(self) -> str:
        if self._algorithm is None:
            return GEN_GAUSSIAN
        data = self._algorithm.currentData()
        return normalize_field_model(data or self._algorithm.currentText())

    def _mount_geometry(self, root, QtCore, QtGui, QtWidgets):
        geom = make_section("Geometry", form=True)
        layout = geom.layout
        self._algorithm = QtWidgets.QComboBox()
        for key in FIELD_MODEL_ORDER:
            self._algorithm.addItem(FIELD_MODEL_LABELS[key], key)
        self._algorithm.currentIndexChanged.connect(lambda *_: self._on_algorithm_changed())
        self._geom_form = layout
        layout.addRow("Algorithm", self._algorithm)
        root.addWidget(geom.widget)

        domain = make_section("Domain", form=False)
        knobs = QtWidgets.QWidget()
        dlayout = QtWidgets.QFormLayout(knobs)
        dlayout.setContentsMargins(0, 0, 0, 0)
        dlayout.setHorizontalSpacing(12)
        dlayout.setVerticalSpacing(8)
        self._bounds_mode = QtWidgets.QComboBox()
        for key, label in (
            (BOUNDS_AROUND_SELECTION, BOUNDS_MODE_LABELS[BOUNDS_AROUND_SELECTION]),
            (BOUNDS_OBJECT, BOUNDS_MODE_LABELS[BOUNDS_OBJECT]),
            (BOUNDS_CUSTOM_BOX, BOUNDS_MODE_LABELS[BOUNDS_CUSTOM_BOX]),
        ):
            self._bounds_mode.addItem(label, key)
        self._bounds_mode.currentIndexChanged.connect(self._on_domain_changed)
        self._padding = QtWidgets.QDoubleSpinBox()
        self._padding.setRange(0.0, 50.0)
        apply_spin_step(self._padding, STEP_ANGSTROM_COARSE, decimals=2)
        self._padding.setValue(0.0)
        apply_ascii_float_locale(self._padding, QtCore)
        self._padding.valueChanged.connect(lambda *_: self._schedule_preview())
        self._spacing = QtWidgets.QDoubleSpinBox()
        self._spacing.setRange(0.05, 8.0)
        apply_spin_step(self._spacing, STEP_SPACING, decimals=3)
        self._spacing.setValue(gauss_spacing(DEFAULT_QUALITY))
        apply_ascii_float_locale(self._spacing, QtCore)
        self._spacing.valueChanged.connect(lambda *_: self._schedule_preview())
        self._domain_object = QtWidgets.QLineEdit()
        self._domain_object.textChanged.connect(lambda *_: self._schedule_preview())
        self._domain_object.setPlaceholderText("PyMOL object name")
        box = QtWidgets.QWidget()
        box_layout = QtWidgets.QGridLayout(box)
        box_layout.setContentsMargins(0, 0, 0, 0)
        self._box_lo = []
        self._box_hi = []
        for i, axis in enumerate("XYZ"):
            lo = QtWidgets.QDoubleSpinBox()
            hi = QtWidgets.QDoubleSpinBox()
            for spin in (lo, hi):
                spin.setRange(-1e4, 1e4)
                apply_spin_step(spin, STEP_ANGSTROM, decimals=2)
                apply_ascii_float_locale(spin, QtCore)
                spin.valueChanged.connect(lambda *_: self._schedule_preview())
            box_layout.addWidget(QtWidgets.QLabel(axis), i, 0)
            box_layout.addWidget(lo, i, 1)
            box_layout.addWidget(hi, i, 2)
            self._box_lo.append(lo)
            self._box_hi.append(hi)
        self._domain_form = dlayout
        dlayout.addRow("Bounds", self._bounds_mode)
        dlayout.addRow("Padding (Å)", self._padding)
        dlayout.addRow("Spacing (Å)", self._spacing)
        self._object_row = dlayout.rowCount()
        dlayout.addRow("Object", self._domain_object)
        self._box_row = dlayout.rowCount()
        dlayout.addRow("Box min / max", box)
        self._domain_schematic = DomainSchematicWidget(domain.body)
        row = wrap_domain_knobs(QtWidgets, self._domain_schematic, knobs)
        domain.layout.addWidget(row)
        root.addWidget(domain.widget)
        return (
            (self._algorithm, _MODEL_TIP),
            (self._bounds_mode, _BOUNDS_TIP),
            (self._padding, _PADDING_TIP),
            (self._spacing, _SPACING_TIP),
            (self._domain_schematic.widget, SCHEMATIC_TIP, "Domain schematic"),
        )

    def _mount_options(self, root, QtCore, QtGui, QtWidgets):
        opts = make_section("Options", form=True)
        layout = opts.layout
        self._quality = QtWidgets.QSpinBox()
        self._quality.setRange(1, 5)
        self._quality.setValue(DEFAULT_QUALITY)
        configure_committed_spin(self._quality)
        self._quality.valueChanged.connect(lambda *_: self._on_quality_changed())
        self._resolution = QtWidgets.QDoubleSpinBox()
        self._resolution.setRange(1.00, 8.00)
        apply_spin_step(self._resolution, STEP_RESOLUTION, decimals=2)
        self._resolution.setValue(DEFAULT_GAUSSIAN_RESOLUTION)
        apply_ascii_float_locale(self._resolution, QtCore)
        self._resolution.valueChanged.connect(lambda *_: self._schedule_preview())
        self._iso_value = QtWidgets.QDoubleSpinBox()
        self._iso_value.setRange(0.05, 8.00)
        apply_spin_step(self._iso_value, 0.05, decimals=2)
        self._iso_value.setValue(DEFAULT_GAUSSIAN_ISOLEVEL)
        apply_ascii_float_locale(self._iso_value, QtCore)
        self._iso_value.valueChanged.connect(lambda *_: self._schedule_preview())
        self._property = QtWidgets.QComboBox()
        for key, label in NEAREST_PROPERTIES:
            self._property.addItem(label, key)
        self._opt_form = layout
        self._quality_row = layout.rowCount()
        layout.addRow("Quality", self._quality)
        self._resolution_row = layout.rowCount()
        layout.addRow("Resolution (Å)", self._resolution)
        self._iso_row = layout.rowCount()
        layout.addRow("Iso value", self._iso_value)
        self._property_row = layout.rowCount()
        layout.addRow("Property", self._property)
        root.addWidget(opts.widget)
        self._sync_algorithm_visibility()
        self._sync_domain_visibility()
        return (
            (self._quality, _QUALITY_TIP),
            (self._resolution, _RESOLUTION_TIP),
            (self._iso_value, _ISO_TIP),
            (self._property, _PROPERTY_TIP),
        )

    def _set_form_row_visible(self, layout, row, visible):
        if layout is None or row is None:
            return
        setter = getattr(layout, "setRowVisible", None)
        if setter is not None:
            setter(row, bool(visible))
            return
        _, _, QtWidgets = qt_modules()
        if QtWidgets is None:
            return
        for role in (QtWidgets.QFormLayout.LabelRole, QtWidgets.QFormLayout.FieldRole):
            item = layout.itemAt(int(row), role)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.setVisible(bool(visible))

    def _sync_algorithm_visibility(self):
        algo = self._current_algorithm()
        self._set_form_row_visible(self._opt_form, self._quality_row, field_model_shows(algo, "quality"))
        self._set_form_row_visible(self._opt_form, self._resolution_row, field_model_shows(algo, "resolution"))
        self._set_form_row_visible(self._opt_form, self._iso_row, field_model_shows(algo, "iso_value"))
        self._set_form_row_visible(self._opt_form, self._property_row, field_model_shows(algo, "property"))
        self._apply_iso_spin()

    def _apply_iso_spin(self):
        if self._iso_value is None:
            return
        lo, hi, step, decimals = iso_spin_range(self._current_algorithm())
        self._iso_value.blockSignals(True)
        try:
            self._iso_value.setDecimals(int(decimals))
            self._iso_value.setRange(float(lo), float(hi))
            self._iso_value.setSingleStep(float(step))
        finally:
            self._iso_value.blockSignals(False)

    def _sync_domain_visibility(self):
        mode = BOUNDS_AROUND_SELECTION
        if self._bounds_mode is not None:
            mode = str(self._bounds_mode.currentData() or BOUNDS_AROUND_SELECTION)
        self._set_form_row_visible(self._domain_form, self._object_row, mode == BOUNDS_OBJECT)
        self._set_form_row_visible(self._domain_form, self._box_row, mode == BOUNDS_CUSTOM_BOX)

    def _on_algorithm_changed(self):
        self._sync_algorithm_visibility()
        if self._iso_value is not None:
            self._iso_value.blockSignals(True)
            try:
                self._iso_value.setValue(default_field_iso_level(self._current_algorithm()))
            finally:
                self._iso_value.blockSignals(False)
        self._schedule_preview()

    def _on_domain_changed(self):
        self._sync_domain_visibility()
        self._schedule_preview()

    def _on_quality_changed(self):
        if self._spacing is not None:
            self._spacing.blockSignals(True)
            self._spacing.setValue(gauss_spacing(self._quality_value()))
            self._spacing.blockSignals(False)
        self._schedule_preview()

    def _quality_value(self) -> int:
        if self._quality is None:
            return DEFAULT_QUALITY
        return int(self._quality.value())

    def _resolution_value(self) -> float:
        if self._resolution is None:
            return DEFAULT_GAUSSIAN_RESOLUTION
        return float(self._resolution.value())

    def _iso_value_amount(self) -> float:
        if self._iso_value is None:
            return default_field_iso_level(self._current_algorithm())
        return float(self._iso_value.value())

    def _live_preview_enabled(self) -> bool:
        if self._appearance is None:
            return False
        return preview_is_on(self._appearance.preview_mode())

    def _preview_mode(self) -> str:
        if self._appearance is None:
            return PREVIEW_OFF
        return self._appearance.preview_mode()

    def _property_key(self) -> str:
        if self._property is None:
            return "b_factor"
        return str(self._property.currentData() or "b_factor")

    def _domain_centers(self):
        centers = []
        for pt in enabled_points(self._points):
            try:
                xyz = pt.resolve()
            except Exception:
                xyz = pt.xyz()
            centers.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
        return centers

    def _sync_domain_schematic(self, domain, centers):
        if self._domain_schematic is None:
            return
        self._domain_schematic.set_scene(domain_schematic_scene(domain, centers))

    def _domain_from_form(self) -> Domain:
        mode = BOUNDS_AROUND_SELECTION
        if self._bounds_mode is not None:
            mode = str(self._bounds_mode.currentData() or BOUNDS_AROUND_SELECTION)
        padding = float(self._padding.value()) if self._padding is not None else 0.0
        spacing = float(self._spacing.value()) if self._spacing is not None else gauss_spacing(self._quality_value())
        object_name = None
        aabb = None
        if mode == BOUNDS_OBJECT and self._domain_object is not None:
            object_name = self._domain_object.text().strip() or None
            if object_name:
                try:
                    extent = self.cmd.get_extent(object_name)
                    aabb = [list(extent[0]), list(extent[1])]
                except Exception:
                    aabb = None
        if mode == BOUNDS_CUSTOM_BOX and self._box_lo and self._box_hi:
            aabb = [
                [float(spin.value()) for spin in self._box_lo],
                [float(spin.value()) for spin in self._box_hi],
            ]
        return Domain(
            bounds_mode=mode,
            padding=padding,
            spacing=spacing,
            aabb=aabb,
            object_name=object_name,
        )

    def _refresh_preview(self):
        if not qt_widget_alive(self._page):
            return
        try:
            domain = self._domain_from_form()
            centers = self._domain_centers()
            self._sync_domain_schematic(domain, centers)
            aabb = domain.resolve_aabb(centers)
            mode = self._effective_preview_mode(self._preview_mode())
            show_domain = preview_is_on(mode)
            if not enabled_points(self._points):
                self._preview.update(
                    [],
                    _MARKER_RADIUS,
                    domain_aabb=aabb,
                    show_domain=show_domain,
                )
                return
            live_iso = (
                preview_is_on(mode)
                and field_supports_iso_preview(self._current_algorithm())
            )
            preview_domain_spec = preview_domain(domain, mode) if live_iso else domain
            if live_iso and not preview_is_simple(mode) and not self._confirm_heavy_map():
                self._preview.update(
                    self._points,
                    _MARKER_RADIUS,
                    live_iso=False,
                    domain_aabb=aabb,
                    show_domain=show_domain,
                )
                return
            iso_visual = None
            volume_visual = None
            iso_key = None
            if live_iso:
                color_mode = self._builder_color_mode()
                color_field_id = None
                colormap = None
                colormap_spec = None
                clims = None
                if (
                    not preview_is_simple(mode)
                    and color_mode == COLOR_MODE_FIELD
                    and self._appearance is not None
                ):
                    color_field_id = self._appearance.color_field_id()
                    colormap = self._appearance.colormap_name()
                    colormap_spec = self._appearance.colormap_spec()
                    clims = self._appearance.custom_clims()
                iso_kind = preview_iso_kind("IsoSurface", mode)
                iso_key = iso_preview_key(
                    self._points,
                    algorithm=self._current_algorithm(),
                    domain=preview_domain_spec,
                    quality=self._quality_value(),
                    resolution=self._resolution_value(),
                    property_key=self._property_key(),
                    iso_level=self._iso_value_amount(),
                    color_mode=color_mode if not preview_is_simple(mode) else COLOR_MODE_UNIFORM,
                    color_field_id=color_field_id,
                    colormap=colormap,
                    colormap_spec=colormap_spec,
                    clims=clims,
                )
                iso_key = tuple(iso_key) + (mode, iso_kind) if iso_key is not None else None
                built = build_field_iso_preview_visual(
                    self.cmd,
                    self._points,
                    algorithm=self._current_algorithm(),
                    domain=preview_domain_spec,
                    quality=self._quality_value(),
                    resolution=self._resolution_value(),
                    property_key=self._property_key(),
                    iso_level=self._iso_value_amount(),
                    color_mode=COLOR_MODE_UNIFORM if preview_is_simple(mode) else color_mode,
                    color_field_id=color_field_id,
                    colormap=colormap,
                    colormap_spec=colormap_spec,
                    clims=clims,
                    kind=iso_kind,
                )
                if isinstance(built, tuple):
                    iso_visual, volume_visual = built
                else:
                    iso_visual = built
            self._preview.update(
                self._points,
                _MARKER_RADIUS,
                live_iso=live_iso,
                iso_visual=iso_visual,
                volume_visual=volume_visual,
                iso_key=iso_key,
                domain_aabb=aabb,
                show_domain=show_domain,
            )
        except RuntimeError:
            pass

    def _confirm_heavy_map(self) -> bool:
        records = atom_records_from_points(self._points)
        centers = [row["xyz"] for row in records]
        elements = [row.get("elem", "C") for row in records]
        job = estimate_surface_job(
            centers,
            algorithm="GAUSS",
            quality=self._quality_value(),
            elements=elements,
        )
        seconds = max(2, int(round(float(job.get("seconds") or 0.0))))
        n = int(job.get("n_atoms") or 0)
        q = int(job.get("quality") or 0)
        voxels = int(job.get("voxels") or 0)
        extra = " (%s voxels)" % format(voxels, ",") if voxels else ""
        message = (
            "Building this field at quality %s for %s points "
            "may take about %s seconds%s. PyMOL will not respond until it finishes. "
            "Build it anyway?" % (q, n, seconds, extra)
        )
        allowed, ok_fp, denied_fp = ask_heavy_job(
            self._page,
            job,
            title="Heavy field",
            message=message,
            previous_ok=self._heavy_ok,
            previous_denied=self._heavy_denied,
        )
        if ok_fp is not None:
            self._heavy_ok = ok_fp
        if denied_fp is not None:
            self._heavy_denied = denied_fp
        return allowed

    def _builder_color_mode(self) -> str:
        if self._appearance is None:
            return COLOR_MODE_UNIFORM
        return self._appearance.color_mode()

    def _build_field(self, name):
        return field_from_selection(
            self.cmd,
            self._points,
            name,
            algorithm=self._current_algorithm(),
            domain=self._domain_from_form(),
            quality=self._quality_value(),
            resolution=self._resolution_value(),
            property_key=self._property_key(),
        )

    def _attach_color_to_geometry(self, field, name):
        del name
        if field is None:
            return field
        mode = self._builder_color_mode()
        if mode == COLOR_MODE_FIELD:
            fid = self._appearance.color_field_id() if self._appearance is not None else None
            cmap = self._appearance.colormap_name() if self._appearance is not None else None
            return remember_default_color_field(field, color_field_id=fid, colormap=cmap)
        return remember_default_color_field(field, color_field_id=None)

    def _preset_iso_level(self):
        if field_model_shows(self._current_algorithm(), "iso_value"):
            return self._iso_value_amount()
        return None

    def _create_cgo(self):
        if not self._can_commit():
            return
        if self._current_algorithm() == GEN_GAUSSIAN and not self._confirm_heavy_map():
            return
        name = unused_object_name(self._typed_name(), self.cmd, keep=self._loaded_name)
        mode = self._builder_color_mode()
        color_field_id = None
        colormap = None
        if self._appearance is not None:
            colormap = self._appearance.colormap_name()
            if mode == COLOR_MODE_FIELD:
                color_field_id = self._appearance.color_field_id()
        field, _color, _visual = commit_from_selection_preset(
            self.cmd,
            self._points,
            name,
            algorithm=self._current_algorithm(),
            domain=self._domain_from_form(),
            quality=self._quality_value(),
            resolution=self._resolution_value(),
            property_key=self._property_key(),
            color_mode=mode,
            color_field_id=color_field_id,
            colormap=colormap,
            iso_level=self._preset_iso_level(),
        )
        if field is None:
            _, _, QtWidgets = qt_modules()
            if QtWidgets is not None:
                overlay_warning(
                    self._page,
                    "From Selection",
                    "Could not build a field from these points.",
                )
            return
        mode = self._preview_mode()
        stamp_preview_mode(field, mode)
        if _visual is not None:
            stamp_preview_mode(_visual, mode)
        self._preview.cleanup()
        if self._on_create is not None:
            self._on_create()

    def _export_cgo(self):
        if not self._can_commit():
            return
        if self._current_algorithm() == GEN_GAUSSIAN and not self._confirm_heavy_map():
            return
        name = unused_object_name(self._typed_name(), self.cmd, keep=self._loaded_name)
        field = self._build_field(name)
        if field is None:
            return
        self._attach_color_to_geometry(field, name)
        export_objects(self._page, field, name, title="Export field")

    def _collection(self, name: str):
        return None
