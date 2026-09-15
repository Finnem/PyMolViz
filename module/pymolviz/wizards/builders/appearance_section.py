"""Shared Appearance section: color mode, per-point colors, wireframe, quality."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

from ...util.field_sample import DEFAULT_SURFACE_COLORMAP
from ..pick import qt_modules
from ..widgets.ascii_locale import configure_committed_spin
from ..widgets.section import make_section
from ..widgets.switch import make_switch
from ..widgets.theme import apply_secondary_button_style
from .colormap_editor import ColormapEditor, colormap_range_mode_labels
from .colors import (
    CLIM_MODE_AUTO,
    CLIM_MODE_CUSTOM,
    ColorChoice,
    _dialog_parent,
    bind_color_pick_result,
    colors_for_new_points,
    normalize_clim_mode,
    pick_rgb,
)
from .field_picker import FieldPickerWidget
from .points import VisualPoint, apply_global_color, assign_distinct_colors, infer_color_mode
from .preview_mode import (
    DEFAULT_PREVIEW_MODE,
    PREVIEW_OFF,
    PreviewModeRadios,
    preview_is_on,
)
from .surface_params import COLOR_MODE_FIELD, COLOR_MODE_PER_POINT, COLOR_MODE_UNIFORM, color_mode_shows

COLOR_ALL_LABEL = "Color All"
COLOR_SELECTION_LABEL = "Color Selection"
RESET_COLORS_LABEL = "Reset Colors"
COLOR_MODE_UNIFORM_LABEL = "Uniform"
COLOR_MODE_PER_POINT_LABEL = "Per-point"
COLOR_MODE_FIELD_LABEL = "From field"
CLIM_MODE_AUTO_LABEL, CLIM_MODE_CUSTOM_LABEL, CLIM_MODE_SYMMETRIC_LABEL, CLIM_MODE_PERCENTILE_LABEL = colormap_range_mode_labels()
DEFAULT_LIVE_PREVIEW_TIP = (
    "No preview hides the live object. Simple preview uses a cheaper stand-in. "
    "Full preview matches what Done will create."
)


def appearance_color_action_labels() -> Tuple[str, str, str]:
    """Button labels for the Appearance color row (testable without Qt)."""
    return (COLOR_ALL_LABEL, COLOR_SELECTION_LABEL, RESET_COLORS_LABEL)


def appearance_color_mode_labels(*, show_per_point: bool = True):
    if show_per_point:
        return (COLOR_MODE_UNIFORM_LABEL, COLOR_MODE_PER_POINT_LABEL, COLOR_MODE_FIELD_LABEL)
    return (COLOR_MODE_UNIFORM_LABEL, COLOR_MODE_FIELD_LABEL)


def appearance_clim_mode_labels() -> Tuple[str, str, str, str]:
    return colormap_range_mode_labels()


def apply_color_mode_to_points(
    points: List[VisualPoint],
    mode: str,
    *,
    field_id=None,
    colormap=DEFAULT_SURFACE_COLORMAP,
    clims=None,
    clim_mode=CLIM_MODE_AUTO,
    colormap_spec=None,
) -> List[VisualPoint]:
    """Mutate ``points`` in place to Uniform / Per-point / From field. Returns the list."""
    if not points:
        return points
    key = str(mode or COLOR_MODE_UNIFORM)
    if key == COLOR_MODE_UNIFORM:
        first = points[0]
        rgba = first.rgba()
        for i, pt in enumerate(points):
            points[i] = pt.with_color(rgba)
        return points
    if key == COLOR_MODE_PER_POINT:
        for i, pt in enumerate(points):
            if pt.field_id:
                points[i] = pt.with_color(pt.rgba())
        return points
    fid = str(field_id) if field_id else None
    if not fid:
        return points
    cmap = str(colormap or DEFAULT_SURFACE_COLORMAP)
    mode_key = normalize_clim_mode(clim_mode)
    choice = ColorChoice(
        rgba=points[0].rgba(),
        field_id=fid,
        colormap=cmap,
        clims=clims,
        clim_mode=mode_key,
        colormap_spec=colormap_spec,
    )
    for i, pt in enumerate(points):
        points[i] = pt.with_color_choice(choice)
    return points


class AppearanceSection:
    """Color mode, per-point color actions, plus optional wireframe/quality controls."""

    def __init__(
        self,
        parent,
        cmd,
        context: str,
        *,
        show_wireframe: bool = True,
        show_quality: bool = True,
        show_per_point: bool = True,
        show_live_preview: bool = True,
        live_preview_tooltip: str = "",
        quality_range: Tuple[int, int] = (1, 5),
        quality_tooltip: str = "",
        on_changed: Optional[Callable[[], None]] = None,
        on_preview: Optional[Callable[[], None]] = None,
        on_look_changed: Optional[Callable[[], None]] = None,
    ):
        self._parent = parent
        self.cmd = cmd
        self._context = context
        self._on_changed = on_changed or (lambda: None)
        self._on_preview = on_preview
        self._show_wireframe = bool(show_wireframe)
        self._show_quality = bool(show_quality)
        self._show_per_point = bool(show_per_point)
        self._show_live_preview = bool(show_live_preview)
        self._live_preview_tooltip = str(live_preview_tooltip or DEFAULT_LIVE_PREVIEW_TIP)
        self._quality_range = quality_range
        self._quality_tooltip = quality_tooltip
        self._points: List[VisualPoint] = []
        self._selected_rows_provider: Optional[Callable[[], Sequence[int]]] = None
        self._widget = None
        self._wireframe = None
        self._quality = None
        self._color_all_btn = None
        self._color_sel_btn = None
        self._reset_colors_btn = None
        self._mode_uniform = None
        self._mode_per_point = None
        self._mode_field = None
        self._field_picker = None
        self._cmap_editor = None
        self._live_preview = None
        self._preview_mode = None
        self._field_row = None
        self._syncing = False
        self._build()

    @property
    def widget(self):
        return self._widget

    @property
    def wireframe_checkbox(self):
        return self._wireframe

    @property
    def quality_spin(self):
        return self._quality

    def wireframe(self) -> bool:
        if self._wireframe is None:
            return False
        return bool(self._wireframe.isChecked())

    def set_wireframe(self, checked: bool) -> None:
        if self._wireframe is not None:
            self._wireframe.setChecked(bool(checked))

    def quality(self) -> int:
        if self._quality is None:
            return 3
        return int(self._quality.value())

    def set_quality(self, value: int) -> None:
        if self._quality is not None:
            self._quality.setValue(int(value))

    def set_quality_maximum(self, allowed: int) -> None:
        if self._quality is None:
            return
        allowed = int(allowed)
        current = self._quality.value()
        self._quality.blockSignals(True)
        try:
            self._quality.setMaximum(allowed)
            if current > allowed:
                self._quality.setValue(allowed)
        finally:
            self._quality.blockSignals(False)

    def set_quality_tooltip(self, text: str) -> None:
        self._quality_tooltip = str(text)
        if self._quality is not None:
            self._quality.setToolTip(str(text))

    def set_points(self, points: Sequence[VisualPoint]) -> None:
        self._points = list(points)
        self.sync_from_points()

    def bind_points(self, points: List[VisualPoint]) -> None:
        self._points = points
        self.sync_from_points()

    def set_selected_rows_provider(self, provider: Callable[[], Sequence[int]]) -> None:
        self._selected_rows_provider = provider

    def set_apply_selected_handler(self, handler: Callable[[], None]) -> None:
        """Compatibility: treat the handler as a selected-row provider if it returns rows."""
        self._selected_rows_provider = handler

    def set_color_selection_enabled(self, enabled: bool) -> None:
        if self._color_sel_btn is not None:
            per_point = self.color_mode() == COLOR_MODE_PER_POINT
            self._color_sel_btn.setEnabled(bool(enabled) and per_point)

    def color_mode(self) -> str:
        if getattr(self, "_mode_field", None) is not None and self._mode_field.isChecked():
            return COLOR_MODE_FIELD
        if getattr(self, "_mode_per_point", None) is not None and self._mode_per_point.isChecked():
            return COLOR_MODE_PER_POINT
        if getattr(self, "_mode_uniform", None) is not None and self._mode_uniform.isChecked():
            return COLOR_MODE_UNIFORM
        if self._points:
            return infer_color_mode(self._points)
        return COLOR_MODE_UNIFORM

    def color_field_id(self):
        return self._selected_field_id()

    def colormap_name(self) -> str:
        return self._colormap_name()

    def colormap_spec(self):
        return self._colormap_spec()

    def preview_mode(self) -> str:
        if self._preview_mode is not None:
            return self._preview_mode.mode()
        return DEFAULT_PREVIEW_MODE

    def set_preview_mode(self, mode) -> None:
        if self._preview_mode is not None:
            self._preview_mode.set_mode(mode)

    def live_preview(self) -> bool:
        return preview_is_on(self.preview_mode())

    def set_live_preview(self, checked: bool) -> None:
        self.set_preview_mode(DEFAULT_PREVIEW_MODE if checked else PREVIEW_OFF)

    @property
    def live_preview_checkbox(self):
        if self._preview_mode is not None:
            return self._preview_mode.widget
        return None

    def clim_mode(self) -> str:
        return self._clim_mode_value()

    def custom_clims(self):
        return self._custom_clims()

    def stamp_new_points(self, new_pts: List[VisualPoint]) -> List[VisualPoint]:
        if self.color_mode() == COLOR_MODE_FIELD:
            fid = self._selected_field_id()
            if fid:
                stamped = list(new_pts)
                apply_color_mode_to_points(
                    stamped,
                    COLOR_MODE_FIELD,
                    field_id=fid,
                    colormap=self._colormap_name(),
                    clims=self._clims_for_apply(),
                    clim_mode=self._clim_mode_value(),
                    colormap_spec=self._colormap_spec(),
                )
                return stamped
        palette = colors_for_new_points(len(new_pts), start_index=len(self._points))
        return [pt.with_color(palette[i]) for i, pt in enumerate(new_pts)]

    def sync_from_points(self) -> None:
        if not self._points:
            return
        mode = infer_color_mode(self._points)
        if not self._show_per_point and mode == COLOR_MODE_PER_POINT:
            mode = COLOR_MODE_UNIFORM
        self._set_mode_ui(mode)
        if mode == COLOR_MODE_FIELD:
            pt0 = next((pt for pt in self._points if pt.field_id), None)
            if pt0 is not None and self._field_picker is not None:
                self._syncing = True
                try:
                    self._field_picker.refresh(pt0.field_id)
                    if self._cmap_editor is not None and (pt0.field_colormap or pt0.field_clims is not None or pt0.field_clim_mode):
                        mode_key = normalize_clim_mode(pt0.field_clim_mode) if pt0.field_clim_mode else (
                            CLIM_MODE_CUSTOM if pt0.field_clims is not None else CLIM_MODE_AUTO
                        )
                        self._cmap_editor.set_colormap(
                            pt0.field_colormap or DEFAULT_SURFACE_COLORMAP,
                            range_mode=mode_key,
                            clims=pt0.field_clims,
                            spec=getattr(pt0, "field_colormap_spec", None),
                        )
                finally:
                    self._syncing = False
        self._sync_mode_widgets()

    def tooltips(self) -> Sequence[Tuple[object, str]]:
        tips = [
            (self._color_all_btn, "Open the color picker and apply the choice to every point."),
            (self._color_sel_btn, "Open the color picker and apply the choice to selected table rows."),
            (self._reset_colors_btn, "Reassign distinct palette colors to enabled points."),
        ]
        if self._mode_uniform is not None:
            tips.append((self._mode_uniform, "One solid color for the whole object."))
        if self._mode_per_point is not None:
            tips.append((self._mode_per_point, "Each point keeps its own color."))
        if self._mode_field is not None:
            tips.append((self._mode_field, "Sample a Field at this object's points. The object stores a field id, not a voxel copy."))
        if self._field_picker is not None:
            tips.append((self._field_picker.widget, "Field used to color this object."))
        if self._cmap_editor is not None:
            tips.extend((w, t) for w, t, *_rest in self._cmap_editor.tooltips())
        if self._preview_mode is not None:
            tips.extend((w, t) for w, t, *_rest in self._preview_mode.tooltips())
        if self._wireframe is not None:
            tips.append((self._wireframe, "Draw triangle meshes as wireframe cages instead of filled surfaces."))
        if self._quality is not None:
            tips.append((self._quality, self._quality_tooltip or "Mesh detail level."))
        return tips

    def _selected_field_id(self):
        if self._field_picker is None:
            return None
        return self._field_picker.field_id()

    def _colormap_name(self) -> str:
        if self._cmap_editor is None:
            return DEFAULT_SURFACE_COLORMAP
        return self._cmap_editor.colormap_name()

    def _colormap_spec(self):
        if self._cmap_editor is None:
            return None
        return self._cmap_editor.colormap_spec()

    def _clim_mode_value(self) -> str:
        if self._cmap_editor is None:
            return CLIM_MODE_AUTO
        return normalize_clim_mode(self._cmap_editor.range_mode())

    def _custom_clims(self):
        if self._cmap_editor is None:
            return None
        return self._cmap_editor.custom_clims()

    def _clims_for_apply(self):
        mode = self._clim_mode_value()
        if mode == CLIM_MODE_AUTO:
            return None
        if mode == CLIM_MODE_CUSTOM:
            return self._custom_clims()
        if self._cmap_editor is None:
            return self._custom_clims()
        from ...util.colormap_spec import field_values_for_stats

        values = field_values_for_stats(self._selected_field_id())
        return self._cmap_editor.resolved_clims(values)

    def _set_mode_ui(self, mode: str) -> None:
        radios = {
            COLOR_MODE_UNIFORM: self._mode_uniform,
            COLOR_MODE_FIELD: self._mode_field,
        }
        if self._mode_per_point is not None:
            radios[COLOR_MODE_PER_POINT] = self._mode_per_point
        target = radios.get(str(mode)) or self._mode_uniform
        if target is None:
            return
        self._syncing = True
        try:
            target.setChecked(True)
        finally:
            self._syncing = False

    def _sync_mode_widgets(self) -> None:
        mode = self.color_mode()
        field_on = color_mode_shows(mode, "field")
        if self._field_row is not None:
            self._field_row.setVisible(field_on)
        if self._color_all_btn is not None:
            self._color_all_btn.setEnabled(not field_on)
        if self._reset_colors_btn is not None:
            self._reset_colors_btn.setEnabled(mode == COLOR_MODE_PER_POINT)
        if self._color_sel_btn is not None:
            selected = bool(self._selected_rows()) if mode == COLOR_MODE_PER_POINT else False
            self._color_sel_btn.setEnabled(selected)
        if self._cmap_editor is not None:
            self._cmap_editor.widget.setEnabled(field_on)

    def _apply_current_mode(self) -> None:
        if self._syncing or not self._points:
            return
        apply_color_mode_to_points(
            self._points,
            self.color_mode(),
            field_id=self._selected_field_id(),
            colormap=self._colormap_name(),
            clims=self._clims_for_apply(),
            clim_mode=self._clim_mode_value(),
            colormap_spec=self._colormap_spec(),
        )
        self._sync_mode_widgets()
        self._on_changed()

    def _on_mode_toggled(self) -> None:
        if self._syncing:
            return
        self._sync_mode_widgets()
        self._apply_current_mode()

    def _on_field_settings_changed(self) -> None:
        if self._syncing:
            return
        self._sync_mode_widgets()
        if self.color_mode() == COLOR_MODE_FIELD:
            self._apply_current_mode()

    def _build(self):
        QtCore, _, QtWidgets = qt_modules()
        section = make_section("Appearance", self._parent)
        layout = section.layout

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.setSpacing(8)
        self._mode_uniform = QtWidgets.QRadioButton(COLOR_MODE_UNIFORM_LABEL)
        self._mode_field = QtWidgets.QRadioButton(COLOR_MODE_FIELD_LABEL)
        self._mode_uniform.setChecked(True)
        self._mode_uniform.toggled.connect(lambda on: on and self._on_mode_toggled())
        self._mode_field.toggled.connect(lambda on: on and self._on_mode_toggled())
        mode_row.addWidget(self._mode_uniform)
        if self._show_per_point:
            self._mode_per_point = QtWidgets.QRadioButton(COLOR_MODE_PER_POINT_LABEL)
            self._mode_per_point.toggled.connect(lambda on: on and self._on_mode_toggled())
            mode_row.addWidget(self._mode_per_point)
        mode_row.addWidget(self._mode_field)
        mode_row.addStretch(1)

        if self._show_live_preview:
            self._preview_mode = PreviewModeRadios(on_changed=lambda *_: self._emit_preview())
            layout.addWidget(self._preview_mode.widget)

        layout.addLayout(mode_row)

        self._field_row = QtWidgets.QWidget()
        field_form = QtWidgets.QFormLayout(self._field_row)
        field_form.setContentsMargins(0, 0, 0, 0)
        self._field_picker = FieldPickerWidget(
            self._field_row,
            self.cmd,
            self._context,
            on_changed=self._on_field_settings_changed,
            empty_label="Select a field",
        )
        self._cmap_editor = ColormapEditor(
            self._field_row,
            on_changed=self._on_field_settings_changed,
            cmd=self.cmd,
            field_id_provider=self._selected_field_id,
        )
        field_form.addRow("Field", self._field_picker.widget)
        field_form.addRow(self._cmap_editor.widget)
        layout.addWidget(self._field_row)

        action_row = QtWidgets.QHBoxLayout()
        action_row.setSpacing(8)
        self._color_all_btn = QtWidgets.QPushButton(COLOR_ALL_LABEL)
        self._color_sel_btn = QtWidgets.QPushButton(COLOR_SELECTION_LABEL)
        self._reset_colors_btn = QtWidgets.QPushButton(RESET_COLORS_LABEL)
        apply_secondary_button_style(self._color_all_btn, "pmvColorAll")
        apply_secondary_button_style(self._color_sel_btn, "pmvColorSelection")
        apply_secondary_button_style(self._reset_colors_btn, "pmvResetColors")
        self._color_all_btn.clicked.connect(lambda *_: self._color_all())
        self._color_sel_btn.clicked.connect(lambda *_: self._color_selection())
        self._reset_colors_btn.clicked.connect(self._reset_colors)
        self._color_sel_btn.setEnabled(False)
        action_row.addWidget(self._color_all_btn)
        action_row.addWidget(self._color_sel_btn)
        action_row.addWidget(self._reset_colors_btn)
        action_row.addStretch(1)
        layout.addLayout(action_row)
        if not self._show_per_point:
            self._color_sel_btn.setVisible(False)
            self._reset_colors_btn.setVisible(False)

        look_row = QtWidgets.QHBoxLayout()
        look_row.setSpacing(12)
        if self._show_wireframe:
            self._wireframe = make_switch("Wireframe")
            self._wireframe.toggled.connect(lambda *_: self._on_changed())
            look_row.addWidget(self._wireframe)

        look_row.addStretch(1)

        if self._show_quality:
            look_row.addWidget(QtWidgets.QLabel("Quality"))
            self._quality = QtWidgets.QSpinBox()
            lo, hi = self._quality_range
            self._quality.setRange(int(lo), int(hi))
            self._quality.setValue(int(lo) if int(lo) <= 3 <= int(hi) else int(lo))
            configure_committed_spin(self._quality)
            self._quality.valueChanged.connect(lambda *_: self._on_changed())
            look_row.addWidget(self._quality)

        if self._show_wireframe or self._show_quality:
            layout.addLayout(look_row)
        self._widget = section.widget
        self._sync_mode_widgets()

    def _selected_rows(self) -> List[int]:
        if self._selected_rows_provider is None:
            return []
        rows = self._selected_rows_provider()
        if rows is None:
            return []
        if isinstance(rows, (list, tuple)):
            return [int(i) for i in rows]
        return []

    def _color_all(self) -> None:
        self._pick_for_rows(list(range(len(self._points))))

    def _color_selection(self) -> None:
        rows = self._selected_rows()
        if not rows:
            return
        self._pick_for_rows(rows)

    def _emit_preview(self) -> None:
        if self._on_preview is not None:
            self._on_preview()
            return
        self._on_changed()

    def _pick_for_rows(self, rows: Sequence[int]) -> None:
        targets = [int(row) for row in rows if 0 <= int(row) < len(self._points)]
        if not targets:
            return
        initial = self._points[targets[0]].color_choice()
        original = {row: self._points[row].color_choice() for row in targets}

        def on_preview(choice):
            if choice is None:
                return
            apply_global_color(self._points, choice, rows=targets)
            self._emit_preview()

        def apply_choice(choice):
            apply_global_color(self._points, choice, rows=targets)
            self.sync_from_points()
            self._on_changed()

        def restore_originals():
            for row, saved in original.items():
                self._points[row] = self._points[row].with_color_choice(saved)
            self.sync_from_points()
            self._on_changed()

        pick_rgb(
            _dialog_parent(self._widget) or _dialog_parent(self._parent),
            initial,
            on_change=on_preview,
            on_done=bind_color_pick_result(apply_choice, restore_originals),
            cmd=self.cmd,
        )

    def _reset_colors(self):
        rows = [i for i, pt in enumerate(self._points) if getattr(pt, "enabled", True)]
        subset = [self._points[i] for i in rows]
        assign_distinct_colors(subset)
        for row, pt in zip(rows, subset):
            self._points[row] = pt
        self.sync_from_points()
        self._on_changed()
