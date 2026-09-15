"""Reusable Field combo: label, kind, grid size, units — ids stay in item data."""

from __future__ import annotations

from typing import Optional, Sequence

from ...util.field_sample import discover_fields, field_picker_label
from ..pick import qt_modules
from ..widgets.theme import apply_shrinking_combo


def iter_picker_entries(cmd=None, kinds=None):
    """``[(field_id, label), ...]`` with no raw ids in the label."""
    kinds = None if kinds is None else {str(k) for k in kinds}
    out = []
    seen = set()
    for field in discover_fields(cmd=cmd):
        kind = str(getattr(field, "kind", "scalar") or "scalar")
        if kinds is not None and kind not in kinds:
            continue
        oid = str(getattr(field, "id", "") or "")
        if not oid or oid in seen:
            continue
        seen.add(oid)
        out.append((oid, field_picker_label(field)))
    return out


def populate_field_combo(combo, cmd=None, selected_id=None, kinds=None, empty_label="Select a field"):
    combo.blockSignals(True)
    try:
        combo.clear()
        entries = iter_picker_entries(cmd=cmd, kinds=kinds)
        if not entries:
            combo.addItem(empty_label, None)
        else:
            combo.addItem(empty_label, None)
            for fid, label in entries:
                combo.addItem(label, fid)
        if selected_id:
            idx = combo.findData(str(selected_id))
            if idx >= 0:
                combo.setCurrentIndex(idx)
    finally:
        combo.blockSignals(False)
    return combo


class FieldPickerWidget:
    """Labeled combo of session Fields."""

    def __init__(self, parent, cmd, context: str, *, kinds=None, on_changed=None, empty_label="Select a field"):
        self.cmd = cmd
        self._kinds = kinds
        self._on_changed = on_changed
        self._empty_label = empty_label
        _, _, QtWidgets = qt_modules()
        self._combo = QtWidgets.QComboBox(parent)
        apply_shrinking_combo(self._combo, QtWidgets, min_chars=10)
        populate_field_combo(self._combo, cmd=cmd, kinds=kinds, empty_label=empty_label)
        if on_changed is not None:
            self._combo.currentIndexChanged.connect(lambda *_: on_changed())

    @property
    def widget(self):
        return self._combo

    def field_id(self) -> Optional[str]:
        data = self._combo.currentData()
        return str(data) if data else None

    def set_field_id(self, field_id):
        populate_field_combo(
            self._combo,
            cmd=self.cmd,
            selected_id=field_id,
            kinds=self._kinds,
            empty_label=self._empty_label,
        )

    def refresh(self, selected_id=None):
        current = selected_id if selected_id is not None else self.field_id()
        populate_field_combo(
            self._combo,
            cmd=self.cmd,
            selected_id=current,
            kinds=self._kinds,
            empty_label=self._empty_label,
        )
