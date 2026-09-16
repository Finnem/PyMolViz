"""Carve a field visual to a PyMOL object/selection and radius (Å)."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from ..pick import qt_modules
from ..widgets.ascii_locale import apply_ascii_float_locale
from ..widgets.spin_step import STEP_ANGSTROM, apply_spin_step
from ..widgets.switch import make_switch
from ..widgets.theme import apply_shrinking_combo


CARVE_ENABLE_TIP = (
    "Limit the iso or volume to a neighborhood of a PyMOL object or named selection."
)
CARVE_TARGET_TIP = "Object or selection name passed to PyMOL as the carve target."
CARVE_RADIUS_TIP = "Radius in Ångströms around that target (PyMOL carve parameter)."
EMPTY_CARVE_LABEL = "(choose target)"


def is_carve_target_name(selection) -> bool:
    """True when ``selection`` is a real PyMOL name, not the combo placeholder."""
    sel = str(selection or "").strip()
    if not sel:
        return False
    if sel == EMPTY_CARVE_LABEL:
        return False
    if sel.startswith("(") and sel.endswith(")"):
        return False
    return True


def normalize_carve_args(selection, carve) -> Tuple[Optional[str], Optional[float]]:
    """Return ``(selection, carve)`` for volumetric visuals, or ``(None, None)`` when off."""
    sel = str(selection or "").strip()
    if not is_carve_target_name(sel):
        return None, None
    if carve is None:
        return sel, None
    try:
        radius = float(carve)
    except (TypeError, ValueError):
        return None, None
    if radius <= 0:
        return None, None
    return sel, radius


def iter_carve_targets(cmd_) -> Sequence[str]:
    """Enabled objects and named selections suitable for ``cmd.isosurface(..., carve=)``."""
    names = []
    seen = set()
    for listing in (
        ("objects", True),
        ("objects", False),
        ("selections", True),
        ("selections", False),
    ):
        kind, enabled_only = listing
        try:
            if enabled_only:
                batch = cmd_.get_names(kind, enabled_only=1)
            else:
                batch = cmd_.get_names(kind, 1)
        except TypeError:
            try:
                batch = cmd_.get_names(kind)
            except Exception:
                batch = ()
        except Exception:
            batch = ()
        for item in batch or ():
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            names.append(text)
    return names


def populate_carve_target_combo(combo, cmd_, selected: Optional[str] = None, empty_label=EMPTY_CARVE_LABEL):
    _, _, QtWidgets = qt_modules()
    combo.blockSignals(True)
    try:
        combo.clear()
        combo.addItem(empty_label, None)
        for name in iter_carve_targets(cmd_):
            combo.addItem(name, name)
        if selected:
            idx = combo.findData(str(selected))
            if idx >= 0:
                combo.setCurrentIndex(idx)
            else:
                combo.addItem(str(selected), str(selected))
                combo.setCurrentIndex(combo.count() - 1)
    finally:
        combo.blockSignals(False)
    return combo


class CarveAroundWidget:
    """Enable + target combo + radius spin for field-visual carve."""

    def __init__(self, parent, cmd_, context: str, *, on_changed=None):
        self.cmd = cmd_
        self._context = context
        self._on_changed = on_changed
        QtCore, _, QtWidgets = qt_modules()
        row = QtWidgets.QWidget(parent)
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        self._enable = make_switch("Carve")
        self._target = QtWidgets.QComboBox(row)
        apply_shrinking_combo(self._target, QtWidgets, min_chars=12)
        self._radius = QtWidgets.QDoubleSpinBox(row)
        self._radius.setRange(0.1, 500.0)
        apply_spin_step(self._radius, STEP_ANGSTROM, decimals=2)
        self._radius.setValue(2.0)
        apply_ascii_float_locale(self._radius, QtCore)
        layout.addWidget(self._enable)
        layout.addWidget(self._target, stretch=1)
        layout.addWidget(self._radius)
        self._row = row
        populate_carve_target_combo(self._target, cmd_)
        self._enable.toggled.connect(self._on_toggle)
        self._target.currentIndexChanged.connect(self._emit_changed)
        self._radius.valueChanged.connect(self._emit_changed)
        self._sync_enabled()

    @property
    def widget(self):
        return self._row

    def refresh(self, selected: Optional[str] = None):
        populate_carve_target_combo(self._target, self.cmd, selected=selected or self.selection())

    def _emit_changed(self, *_args):
        if self._on_changed is not None:
            self._on_changed()

    def _on_toggle(self, *_args):
        self._sync_enabled()
        self._emit_changed()

    def _sync_enabled(self):
        on = self.is_enabled()
        self._target.setEnabled(on)
        self._radius.setEnabled(on)

    def is_enabled(self) -> bool:
        return bool(self._enable.isChecked())

    def selection(self) -> Optional[str]:
        if not self.is_enabled():
            return None
        data = self._target.currentData()
        if data is None:
            return None
        text = str(data).strip()
        return text if is_carve_target_name(text) else None

    def carve_radius(self) -> Optional[float]:
        if not self.is_enabled():
            return None
        return float(self._radius.value())

    def carve_args(self) -> Tuple[Optional[str], Optional[float]]:
        return normalize_carve_args(self.selection(), self.carve_radius())

    def set_carve(self, selection=None, carve=None):
        enabled = False
        sel = str(selection or "").strip() if selection else ""
        radius = None
        if carve is not None:
            try:
                radius = float(carve)
            except (TypeError, ValueError):
                radius = None
        if sel and radius is not None and radius > 0:
            enabled = True
        self._enable.blockSignals(True)
        self._enable.setChecked(enabled)
        self._enable.blockSignals(False)
        if radius is not None and radius > 0:
            self._radius.setValue(radius)
        populate_carve_target_combo(self._target, self.cmd, selected=sel or None)
        self._sync_enabled()

    def reset(self):
        self.set_carve(selection=None, carve=None)

    def tooltips(self):
        return [
            (self._enable, CARVE_ENABLE_TIP, "Carve"),
            (self._target, CARVE_TARGET_TIP, "Carve target"),
            (self._radius, CARVE_RADIUS_TIP, "Carve radius"),
        ]
