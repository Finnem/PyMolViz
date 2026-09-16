"""Compact Appearance colormap row: preset, reverse, range, add/adjust custom maps."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Optional, Sequence, Tuple

from ...ColorMap import (
    RANGE_MODE_AUTO,
    RANGE_MODE_CUSTOM,
    RANGE_MODE_SYMMETRIC,
    apply_colormap_reverse,
    parse_colormap_reverse,
)
from ...util.colormap_spec import (
    RANGE_MODE_PERCENTILE,
    ColormapDefinition,
    FieldColorMapping,
    Normalization,
    builtin_preset_names,
    custom_preset_definition,
    definition_from_preset,
    is_custom_preset_name,
    load_custom_presets,
    normalize_range_mode_full,
    ramp_rgba,
    resolve_limits,
    reverse_definition,
    save_custom_preset,
    unused_custom_preset_name,
    stored_colormap_spec,
    normalization_from_stored_spec,
)
from ...util.field_sample import DEFAULT_SURFACE_COLORMAP, FIELD_COLORMAPS
from ..pick import qt_modules, qt_widget_alive
from ..widgets.ascii_locale import apply_ascii_float_locale
from ..widgets.spin_step import bind_peer_steps
from ..widgets.switch import make_switch
from ..widgets.theme import apply_shrinking_combo, row_icon_css

RANGE_AUTO_LABEL = "Auto"
RANGE_CUSTOM_LABEL = "Custom"
RANGE_SYMMETRIC_LABEL = "Symmetric"
RANGE_PERCENTILE_LABEL = "Percentile"
ADD_CUSTOM_COLORMAP_LABEL = "+"
ADD_CUSTOM_COLORMAP_TIP = "add custom colormap"
ADJUST_CUSTOM_COLORMAP_TIP = "Adjust this custom colormap"
KIND_BUILTIN = "builtin"
KIND_CUSTOM = "custom"

COLORMAP_PRESET_TIP = "Named colormap used to map field values."
COLORMAP_REVERSE_TIP = "Flip the ramp without swapping min and max."
COLORMAP_RANGE_TIP = "Map field values with auto, custom, symmetric, or percentile limits."
COLORMAP_CLIM_TIP = "Manual color limits. Used when Range is Custom or Symmetric."
COLORMAP_STRIP_TIP = "Preview of the current colormap ramp."


def colormap_range_mode_labels() -> Tuple[str, str, str, str]:
    return (RANGE_AUTO_LABEL, RANGE_CUSTOM_LABEL, RANGE_SYMMETRIC_LABEL, RANGE_PERCENTILE_LABEL)


def colormap_add_custom_tip() -> str:
    return ADD_CUSTOM_COLORMAP_TIP


def colormap_adjust_custom_tip() -> str:
    return ADJUST_CUSTOM_COLORMAP_TIP


def _kind_role(QtCore):
    return int(getattr(QtCore.Qt, "UserRole", 256)) + 1


COLORMAP_COMBO_ICON_WIDTH = 72
COLORMAP_COMBO_ICON_HEIGHT = 14


def definition_for_preset_label(name: str) -> ColormapDefinition:
    text = str(name or "").strip()
    custom = custom_preset_definition(text)
    if custom is not None:
        return custom
    preset, reverse = parse_colormap_reverse(text, presets=FIELD_COLORMAPS)
    return definition_from_preset(preset, reverse=reverse)


def colormap_preview_pixmap(QtGui, QtCore, definition, width=COLORMAP_COMBO_ICON_WIDTH, height=COLORMAP_COMBO_ICON_HEIGHT):
    """Small horizontal ramp for combo rows and the closed combo display."""
    if QtGui is None or definition is None:
        return None
    try:
        rgba = ramp_rgba(definition, n=max(8, int(width)))
    except Exception:
        return None
    n = int(rgba.shape[0])
    fmt = getattr(QtGui.QImage, "Format_RGB32", None)
    qrgb = getattr(QtGui, "qRgb", None)
    if fmt is None or qrgb is None:
        return None
    image = QtGui.QImage(n, 1, fmt)
    for i in range(n):
        r, g, b = [max(0, min(255, int(round(float(c) * 255.0)))) for c in rgba[i][:3]]
        image.setPixel(i, 0, qrgb(r, g, b))
    return QtGui.QPixmap.fromImage(image).scaled(
        int(width),
        int(height),
        getattr(QtCore.Qt, "IgnoreAspectRatio", 0),
        getattr(QtCore.Qt, "SmoothTransformation", 0),
    )


def _icon_button(QtWidgets, QtGui, QtCore, kind, text, tip, object_name, on_click):
    from ..widgets.type_icons import action_icon_pixmap

    btn = QtWidgets.QPushButton(text)
    btn.setObjectName(object_name)
    btn.setStyleSheet(row_icon_css(object_name))
    btn.setAutoDefault(False)
    btn.setDefault(False)
    btn.setFixedSize(28, 28)
    btn.setToolTip(tip)
    no_focus = getattr(QtCore.Qt, "NoFocus", None)
    if no_focus is not None:
        btn.setFocusPolicy(no_focus)
    hand = getattr(QtCore.Qt, "PointingHandCursor", None)
    if hand is not None:
        btn.setCursor(hand)
    pix = action_icon_pixmap(kind, QtGui, QtCore, QtWidgets, size=14)
    if pix is not None and QtGui is not None:
        btn.setIcon(QtGui.QIcon(pix))
        btn.setIconSize(pix.size())
        btn.setText("")
    btn.clicked.connect(lambda *_: on_click())
    return btn


class ColormapPresetPicker:
    """Preset combo with a plus button and a gear on every custom colormap."""

    def __init__(
        self,
        parent,
        *,
        on_selected: Optional[Callable[[str], None]] = None,
        on_add: Optional[Callable[[], None]] = None,
        on_edit: Optional[Callable[[str], None]] = None,
        default_name: str = DEFAULT_SURFACE_COLORMAP,
    ):
        QtCore, QtGui, QtWidgets = qt_modules()
        if QtWidgets is None:
            raise RuntimeError("PyMOL Qt UI required")
        self._on_selected = on_selected or (lambda _name: None)
        self._on_add = on_add or (lambda: None)
        self._on_edit = on_edit or (lambda _name: None)
        self._default_name = str(default_name or DEFAULT_SURFACE_COLORMAP)
        self._syncing = False

        box = QtWidgets.QWidget(parent)
        row = QtWidgets.QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        combo = QtWidgets.QComboBox()
        combo.setObjectName("pmvColormapPreset")
        combo.setToolTip(COLORMAP_PRESET_TIP)
        combo.currentIndexChanged.connect(lambda *_: self._emit_selected())
        apply_shrinking_combo(combo, QtWidgets)
        combo.setIconSize(QtCore.QSize(COLORMAP_COMBO_ICON_WIDTH, COLORMAP_COMBO_ICON_HEIGHT))
        self._combo = combo

        self._gear = _icon_button(
            QtWidgets, QtGui, QtCore, "gear", "⚙", ADJUST_CUSTOM_COLORMAP_TIP,
            "pmvColormapAdjust", self._edit_current,
        )
        self._plus = _icon_button(
            QtWidgets, QtGui, QtCore, "plus", "+", ADD_CUSTOM_COLORMAP_TIP,
            "pmvColormapAdd", lambda: self._on_add(),
        )
        row.addWidget(combo, stretch=1)
        row.addWidget(self._gear)
        row.addWidget(self._plus)
        self._widget = box
        self.reload()

    @property
    def widget(self):
        return self._widget

    @property
    def combo(self):
        return self._combo

    @property
    def plus_button(self):
        return self._plus

    @property
    def gear_button(self):
        return self._gear

    def current_name(self) -> str:
        if not qt_widget_alive(self._combo):
            return self._default_name
        text = str(self._combo.currentText() or "").strip()
        return text or self._default_name

    def current_is_custom(self) -> bool:
        if not qt_widget_alive(self._combo):
            return is_custom_preset_name(self._default_name)
        QtCore, _, _ = qt_modules()
        idx = self._combo.currentIndex()
        kind = self._combo.itemData(idx, _kind_role(QtCore)) if idx >= 0 else None
        if kind:
            return kind == KIND_CUSTOM
        return is_custom_preset_name(self.current_name())

    def set_current(self, name) -> None:
        if not qt_widget_alive(self._combo):
            return
        text = str(name or self._default_name)
        idx = self._combo.findText(text)
        self._syncing = True
        try:
            if idx < 0:
                self.reload(select=text)
                self._syncing = True
                idx = self._combo.findText(text)
            if idx >= 0:
                self._combo.setCurrentIndex(idx)
        finally:
            self._syncing = False
        self._sync_gear()

    def reload(self, select=None) -> None:
        QtCore, QtGui, _ = qt_modules()
        if not qt_widget_alive(self._combo):
            return
        keep = str(select or self.current_name() or self._default_name)
        role = _kind_role(QtCore)
        tip_role = getattr(QtCore.Qt, "ToolTipRole", 3)
        builtins = set(builtin_preset_names())
        self._syncing = True
        self._combo.blockSignals(True)
        try:
            self._combo.clear()
            for name in builtin_preset_names():
                self._combo.addItem(name, name)
                idx = self._combo.count() - 1
                self._combo.setItemData(idx, KIND_BUILTIN, role)
                self._set_item_icon(idx, name, QtGui, QtCore)
            customs = load_custom_presets()
            labels = [str(item.get("name") or "") for item in customs if item.get("name")]
            ephemeral = keep and keep not in builtins and keep not in labels
            if labels or ephemeral:
                inserter = getattr(self._combo, "insertSeparator", None)
                if callable(inserter):
                    inserter(self._combo.count())
                for label in labels:
                    self._add_custom_item(label, role, tip_role, QtGui, QtCore)
                if ephemeral:
                    self._add_custom_item(keep, role, tip_role, QtGui, QtCore)
            idx = self._combo.findText(keep)
            if idx < 0:
                idx = self._combo.findText(self._default_name)
            if idx >= 0:
                self._combo.setCurrentIndex(idx)
        finally:
            self._combo.blockSignals(False)
            self._syncing = False
        self._sync_gear()

    def set_item_icon_from_definition(self, index: int, definition: ColormapDefinition) -> None:
        QtCore, QtGui, _ = qt_modules()
        if not qt_widget_alive(self._combo) or QtGui is None:
            return
        self._set_item_icon(int(index), None, QtGui, QtCore, definition=definition)

    def _set_item_icon(self, index, name, QtGui, QtCore, definition=None) -> None:
        if not qt_widget_alive(self._combo) or QtGui is None or index < 0:
            return
        defn = definition
        if defn is None:
            label = str(name or self._combo.itemText(index) or "").strip()
            if not label:
                return
            defn = definition_for_preset_label(label)
        pix = colormap_preview_pixmap(QtGui, QtCore, defn)
        if pix is not None:
            self._combo.setItemIcon(int(index), QtGui.QIcon(pix))

    def _add_custom_item(self, label, role, tip_role, QtGui=None, QtCore=None) -> None:
        self._combo.addItem(label, label)
        idx = self._combo.count() - 1
        self._combo.setItemData(idx, KIND_CUSTOM, role)
        self._combo.setItemData(idx, ADJUST_CUSTOM_COLORMAP_TIP, tip_role)
        if QtGui is not None and QtCore is not None:
            self._set_item_icon(idx, label, QtGui, QtCore)

    def tooltips(self) -> Sequence[Tuple[object, str, str]]:
        return (
            (self._combo, COLORMAP_PRESET_TIP, "Colormap"),
            (self._plus, ADD_CUSTOM_COLORMAP_TIP, "Add custom colormap"),
            (self._gear, ADJUST_CUSTOM_COLORMAP_TIP, "Adjust custom colormap"),
        )

    def _sync_gear(self) -> None:
        if not qt_widget_alive(self._combo) or not qt_widget_alive(self._gear):
            return
        custom = self.current_is_custom()
        self._gear.setVisible(custom)
        self._gear.setEnabled(custom)

    def _emit_selected(self) -> None:
        self._sync_gear()
        if self._syncing:
            return
        self._on_selected(self.current_name())

    def _edit_current(self) -> None:
        if not self.current_is_custom():
            return
        self._on_edit(self.current_name())


class ColormapEditor:
    """Preset combo, Reverse, range, min/max, strip, plus, and custom-map gears."""

    def __init__(
        self,
        parent,
        *,
        on_changed: Optional[Callable[[], None]] = None,
        default_name: str = DEFAULT_SURFACE_COLORMAP,
        field_id_provider: Optional[Callable[[], Optional[str]]] = None,
        cmd=None,
    ):
        self._on_changed = on_changed or (lambda: None)
        self._field_id_provider = field_id_provider
        self.cmd = cmd
        self._syncing = False
        self._picker = None
        self._reverse = None
        self._range = None
        self._lo = None
        self._hi = None
        self._widget = None
        self._default_name = str(default_name or DEFAULT_SURFACE_COLORMAP)
        self._definition = definition_from_preset(self._default_name)
        self._norm_extra = Normalization()
        self._build(parent)

    @property
    def widget(self):
        return self._widget

    @property
    def preset_combo(self):
        if self._picker is None:
            return None
        return self._picker.combo

    @property
    def reverse_checkbox(self):
        return self._reverse

    @property
    def range_combo(self):
        return self._range

    def colormap_name(self) -> str:
        return apply_colormap_reverse(self._preset_name(), self.reverse())

    def preset_name(self) -> str:
        return self._preset_name()

    def reverse(self) -> bool:
        if not qt_widget_alive(self._reverse):
            return False
        return bool(self._reverse.isChecked())

    def range_mode(self) -> str:
        if not qt_widget_alive(self._range):
            return RANGE_MODE_AUTO
        return normalize_range_mode_full(self._range.currentData())

    def custom_clims(self):
        if not qt_widget_alive(self._lo) or not qt_widget_alive(self._hi):
            return None
        if self.range_mode() not in (RANGE_MODE_CUSTOM, RANGE_MODE_SYMMETRIC):
            return None
        return (float(self._lo.value()), float(self._hi.value()))

    def resolved_clims(self, values=None):
        return resolve_limits(self.normalization(), values)

    def definition(self) -> ColormapDefinition:
        return self._definition

    def colormap_spec(self):
        return stored_colormap_spec(self._definition, self.normalization())

    def normalization(self) -> Normalization:
        clims = self.custom_clims()
        extra = self._norm_extra
        return Normalization(
            mode=self.range_mode(),
            vmin=None if clims is None else clims[0],
            vmax=None if clims is None else clims[1],
            center=extra.center,
            percentile_low=extra.percentile_low,
            percentile_high=extra.percentile_high,
            link_center_zero=extra.link_center_zero,
        )

    def mapping(self, field_id=None) -> FieldColorMapping:
        fid = field_id if field_id is not None else self._field_id()
        return FieldColorMapping(
            field_id=str(fid) if fid else None,
            colormap=self._definition,
            normalization=self.normalization(),
        )

    def _ui_alive(self) -> bool:
        if not qt_widget_alive(self._widget):
            return False
        if self._picker is not None and not qt_widget_alive(self._picker.combo):
            return False
        return True

    def set_colormap(self, name, *, reverse=None, range_mode=None, clims=None, spec=None) -> None:
        stored_norm = normalization_from_stored_spec(spec)
        if spec:
            self._definition = ColormapDefinition.from_dict(spec)
            preset, parsed_reverse = parse_colormap_reverse(
                self._definition.preset or name or self._default_name,
                presets=FIELD_COLORMAPS,
            )
        else:
            preset, parsed_reverse = parse_colormap_reverse(
                name or self._default_name, presets=FIELD_COLORMAPS,
            )
            if reverse is not None:
                parsed_reverse = bool(reverse)
            self._definition = definition_from_preset(preset, reverse=parsed_reverse)
        if reverse is not None:
            parsed_reverse = bool(reverse)
        if stored_norm is not None:
            self._norm_extra = stored_norm
            if range_mode is None:
                range_mode = stored_norm.mode
            if clims is None and stored_norm.vmin is not None and stored_norm.vmax is not None:
                clims = (float(stored_norm.vmin), float(stored_norm.vmax))
        self._syncing = True
        try:
            if not self._ui_alive():
                return
            if self._picker is not None:
                self._picker.reload(select=str(preset))
                self._picker.set_current(preset)
            if qt_widget_alive(self._reverse):
                self._reverse.setChecked(bool(parsed_reverse) and not self._definition.customized)
            if range_mode is not None and qt_widget_alive(self._range):
                idx = self._range.findData(normalize_range_mode_full(range_mode))
                if idx >= 0:
                    self._range.setCurrentIndex(idx)
            if clims is not None and qt_widget_alive(self._lo) and qt_widget_alive(self._hi):
                self._lo.setValue(float(clims[0]))
                self._hi.setValue(float(clims[1]))
        finally:
            self._syncing = False
        if self._ui_alive():
            self._sync_enabled()
            self._refresh_strip()

    def set_mapping(self, mapping: FieldColorMapping) -> None:
        self._norm_extra = mapping.normalization
        norm = mapping.normalization
        clims = None
        if norm.mode in (RANGE_MODE_CUSTOM, RANGE_MODE_SYMMETRIC):
            if norm.vmin is not None and norm.vmax is not None:
                clims = (float(norm.vmin), float(norm.vmax))
        else:
            values = None
            fid = self._field_id()
            if fid:
                try:
                    from ...util.colormap_spec import field_values_for_stats

                    values = field_values_for_stats(fid)
                except Exception:
                    values = None
            resolved = resolve_limits(norm, values)
            if resolved is not None:
                clims = (float(resolved[0]), float(resolved[1]))
        self.set_colormap(
            mapping.colormap.preset or self._default_name,
            range_mode=norm.mode,
            clims=clims,
            spec=stored_colormap_spec(mapping.colormap, mapping.normalization),
        )

    def set_range_mode(self, mode: str) -> None:
        if not qt_widget_alive(self._range):
            return
        idx = self._range.findData(normalize_range_mode_full(mode))
        if idx >= 0:
            self._range.setCurrentIndex(idx)

    def tooltips(self) -> Sequence[Tuple[object, str, str]]:
        tips = []
        if self._picker is not None:
            tips.extend(self._picker.tooltips())
        tips.extend((
            (self._reverse, COLORMAP_REVERSE_TIP, "Reverse"),
            (self._range, COLORMAP_RANGE_TIP, "Range"),
            (self._lo, COLORMAP_CLIM_TIP, "Min"),
            (self._hi, COLORMAP_CLIM_TIP, "Max"),
        ))
        if self._picker is not None and qt_widget_alive(self._picker.combo):
            tips.append((self._picker.combo, COLORMAP_STRIP_TIP, "Colormap preview"))
        return tuple(tips)

    def _field_id(self):
        provider = getattr(self, "_field_id_provider", None)
        if provider is None:
            return None
        try:
            return provider()
        except Exception:
            return None

    def _preset_name(self) -> str:
        if self._picker is None:
            return self._default_name
        return self._picker.current_name()

    def _overlay_parent(self):
        widget = self._widget
        if not qt_widget_alive(widget):
            return None
        window = widget.window() if hasattr(widget, "window") else widget
        return window if qt_widget_alive(window) else widget

    def _load_named_preset(self, name: str) -> None:
        if self._syncing:
            return
        defn = custom_preset_definition(name)
        if defn is not None:
            self._definition = defn
            if qt_widget_alive(self._reverse):
                self._syncing = True
                try:
                    self._reverse.setChecked(False)
                finally:
                    self._syncing = False
        else:
            self._definition = definition_from_preset(name, reverse=self.reverse())
        self._sync_enabled()
        self._refresh_strip()
        self._on_changed()

    def _emit(self) -> None:
        if self._syncing:
            return
        self._sync_enabled()
        self._refresh_strip()
        self._on_changed()

    def _on_reverse(self) -> None:
        if self._syncing:
            return
        if self._definition.customized:
            self._definition = reverse_definition(self._definition)
        else:
            self._definition = definition_from_preset(self._preset_name(), reverse=self.reverse())
        self._sync_enabled()
        self._refresh_strip()
        self._on_changed()

    def _sync_enabled(self) -> None:
        mode = self.range_mode()
        manual = mode in (RANGE_MODE_CUSTOM, RANGE_MODE_SYMMETRIC)
        if qt_widget_alive(self._lo):
            self._lo.setEnabled(manual)
        if qt_widget_alive(self._hi):
            self._hi.setEnabled(manual)

    def _refresh_strip(self) -> None:
        if self._picker is None or not qt_widget_alive(self._picker.combo):
            return
        idx = self._picker.combo.currentIndex()
        if idx >= 0:
            self._picker.set_item_icon_from_definition(idx, self._definition)

    def _open_editor(self, mapping=None, *, save_name=None):
        from .colormap_dialog import open_colormap_editor

        mapping = mapping or self.mapping()
        original = self.mapping()

        def apply_mapping(updated, persist=True):
            if updated is None:
                return
            name = None
            if updated.colormap.customized and updated.colormap.preset:
                name = updated.colormap.preset
            elif save_name and (
                updated.colormap.customized or is_custom_preset_name(save_name)
            ):
                name = save_name
            if persist and name:
                save_custom_preset(name, updated.colormap)
            if self._picker is not None:
                self._picker.reload(select=name or updated.colormap.preset)
            self.set_mapping(updated)
            if not self._ui_alive():
                return
            try:
                self._on_changed()
            except RuntimeError:
                pass

        def live(updated):
            apply_mapping(updated, persist=False)

        def done(updated):
            if updated is not None:
                return
            self.set_mapping(original)
            if not self._ui_alive():
                return
            try:
                self._on_changed()
            except RuntimeError:
                pass

        open_colormap_editor(
            self._widget,
            mapping,
            cmd=self.cmd,
            on_apply=lambda updated: apply_mapping(updated, persist=True),
            on_change=live,
            on_done=done,
        )

    def _add_custom(self):
        name = unused_custom_preset_name()
        mapping = self.mapping()
        mapping = FieldColorMapping(
            field_id=mapping.field_id,
            colormap=replace(mapping.colormap, customized=True, preset=name),
            normalization=mapping.normalization,
            title=mapping.title,
            units=mapping.units,
        )
        self._open_editor(mapping, save_name=name)

    def _adjust_custom(self, name: str):
        defn = custom_preset_definition(name)
        if defn is None:
            self._open_editor(save_name=name)
            return
        mapping = self.mapping()
        mapping = FieldColorMapping(
            field_id=mapping.field_id,
            colormap=defn,
            normalization=mapping.normalization,
            title=mapping.title,
            units=mapping.units,
        )
        if self._picker is not None:
            self._picker.set_current(name)
        self._definition = defn
        if self._ui_alive():
            self._refresh_strip()
        self._open_editor(mapping, save_name=name)

    def _build(self, parent):
        QtCore, _, QtWidgets = qt_modules()
        if QtWidgets is None:
            raise RuntimeError("PyMOL Qt UI required")
        box = QtWidgets.QWidget(parent)
        preferred = getattr(QtWidgets.QSizePolicy, "Preferred", None)
        expanding = getattr(QtWidgets.QSizePolicy, "Expanding", None)
        if expanding is not None and preferred is not None:
            box.setSizePolicy(expanding, preferred)
        form = QtWidgets.QFormLayout(box)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)
        grow = getattr(QtWidgets.QFormLayout, "ExpandingFieldsGrow", None)
        setter = getattr(form, "setFieldGrowthPolicy", None)
        if callable(setter) and grow is not None:
            form.setFieldGrowthPolicy(grow)

        self._picker = ColormapPresetPicker(
            box,
            on_selected=self._load_named_preset,
            on_add=self._add_custom,
            on_edit=self._adjust_custom,
            default_name=self._default_name,
        )
        form.addRow("Colormap", self._picker.widget)

        range_row = QtWidgets.QWidget()
        range_layout = QtWidgets.QHBoxLayout(range_row)
        range_layout.setContentsMargins(0, 0, 0, 0)
        range_layout.setSpacing(8)
        self._range = QtWidgets.QComboBox()
        self._range.setObjectName("pmvColormapRange")
        apply_shrinking_combo(self._range, QtWidgets, min_chars=6)
        self._range.addItem(RANGE_AUTO_LABEL, RANGE_MODE_AUTO)
        self._range.addItem(RANGE_CUSTOM_LABEL, RANGE_MODE_CUSTOM)
        self._range.addItem(RANGE_SYMMETRIC_LABEL, RANGE_MODE_SYMMETRIC)
        self._range.addItem(RANGE_PERCENTILE_LABEL, RANGE_MODE_PERCENTILE)
        self._range.currentIndexChanged.connect(lambda *_: self._emit())
        self._reverse = make_switch("Reverse")
        self._reverse.setObjectName("pmvColormapReverse")
        self._reverse.toggled.connect(lambda *_: self._on_reverse())
        range_layout.addWidget(self._range, stretch=1)
        range_layout.addWidget(self._reverse)
        form.addRow("Range", range_row)

        clim_wrap = QtWidgets.QWidget()
        clim_row = QtWidgets.QHBoxLayout(clim_wrap)
        clim_row.setContentsMargins(0, 0, 0, 0)
        self._lo = QtWidgets.QDoubleSpinBox()
        self._hi = QtWidgets.QDoubleSpinBox()
        for spin in (self._lo, self._hi):
            spin.setDecimals(4)
            spin.setRange(-1e6, 1e6)
            apply_ascii_float_locale(spin, QtCore)
            expanding = getattr(QtWidgets.QSizePolicy, "Expanding", None)
            fixed = getattr(QtWidgets.QSizePolicy, "Fixed", None)
            if expanding is not None and fixed is not None:
                spin.setSizePolicy(expanding, fixed)
            spin.valueChanged.connect(lambda *_: self._emit())
        bind_peer_steps(self._lo, self._hi, min_decimals=4)
        clim_row.addWidget(self._lo, stretch=1)
        clim_row.addWidget(self._hi, stretch=1)
        form.addRow("Min / max", clim_wrap)

        self._widget = box
        self._sync_enabled()
        self._refresh_strip()
