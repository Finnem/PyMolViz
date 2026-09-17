"""Picker: choose which MTZ amplitude/phase maps to load as fields."""

from __future__ import annotations

from pathlib import Path

from ...util.io import mtz_map_id, mtz_map_title
from ..pick import (
    bind_tool_window,
    configure_overlay_dialog,
    overlay_window,
    qt_modules,
)
from ..tooltips import apply_required_tooltips
from ..widgets.theme import (
    apply_wizard_page_style,
    catalog_table_css,
    mark_primary_button,
    muted_label_css,
)

MTZ_DIALOG_TITLE = "MTZ maps"
LOAD_BUTTON = "Load"
CANCEL_BUTTON = "Cancel"
MAPS_TIP = "Check each map to import as its own field. Amplitude and phase columns are paired for the FFT."
COLUMNS_TIP = "Every column in the MTZ, including indices, sigmas, and free-R flags."
AMPLITUDE_TIP = "Structure-factor or intensity column for a custom map."
PHASE_TIP = "Phase column for a custom map."
ADD_MAP_TIP = "Add this amplitude/phase pair to the list of maps to load."
LOAD_TIP = "Import the checked maps as separate fields."
CANCEL_TIP = "Close without loading the MTZ."


def _cell_text(cell) -> str:
    if not cell or len(cell) < 6:
        return ""
    a, b, c, alpha, beta, gamma = [float(v) for v in cell[:6]]
    if max(abs(a), abs(b), abs(c)) < 1e-8:
        return ""
    return "Cell  %.2f  %.2f  %.2f Å    %.1f  %.1f  %.1f°" % (a, b, c, alpha, beta, gamma)


def _header_text(inventory) -> str:
    inv = inventory or {}
    path = str(inv.get("path") or "")
    name = Path(path).name if path else "MTZ"
    bits = [name]
    title = str(inv.get("title") or "").strip()
    if title and title.lower() not in name.lower():
        bits.append(title)
    sg = str(inv.get("space_group") or "").strip()
    if sg:
        bits.append(sg)
    nref = inv.get("n_reflections")
    if nref is not None:
        bits.append("%s reflections" % nref)
    cell = _cell_text(inv.get("cell"))
    lines = ["  ·  ".join(bits)]
    if cell:
        lines.append(cell)
    return "\n".join(lines)


def _amplitude_labels(inventory) -> list:
    labels = []
    seen = set()
    for col in (inventory or {}).get("columns") or ():
        ctype = str(col.get("type") or "")
        label = str(col.get("label") or "")
        if not label or ctype not in "FGDIJKM":
            continue
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def _phase_labels(inventory) -> list:
    labels = []
    seen = set()
    for col in (inventory or {}).get("columns") or ():
        if str(col.get("type") or "") != "P":
            continue
        label = str(col.get("label") or "")
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def show_mtz_import_dialog(parent, inventory, *, on_load=None):
    """Show maps and columns. ``on_load(selected_maps)`` when the user accepts."""
    QtCore, _, QtWidgets = qt_modules()
    if QtWidgets is None or not hasattr(QtWidgets, "QDialog"):
        return None
    overlay = overlay_window(parent)
    dialog = QtWidgets.QDialog(overlay if overlay is not None else parent)
    dialog.setWindowTitle(MTZ_DIALOG_TITLE)
    dialog.setModal(False)
    dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
    configure_overlay_dialog(dialog, overlay if overlay is not None else parent)
    apply_wizard_page_style(dialog)
    dialog.resize(560, 520)

    inv = dict(inventory or {})
    maps = [dict(row) for row in inv.get("maps") or ()]
    user_role = getattr(QtCore.Qt, "UserRole", 256)
    check_on = getattr(QtCore.Qt, "Checked", 2)
    check_off = getattr(QtCore.Qt, "Unchecked", 0)

    header = QtWidgets.QLabel(_header_text(inv))
    header.setWordWrap(True)
    header.setStyleSheet(muted_label_css())

    map_table = QtWidgets.QTableWidget(0, 4)
    map_table.setObjectName("pmvMtzMapTable")
    map_table.setHorizontalHeaderLabels(("Load", "Map", "Amplitude", "Phase"))
    map_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    map_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
    map_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    map_table.verticalHeader().setVisible(False)
    map_table.setShowGrid(False)
    map_table.setStyleSheet(catalog_table_css("pmvMtzMapTable"))
    header_view = map_table.horizontalHeader()
    header_view.setStretchLastSection(True)
    map_table.setColumnWidth(0, 52)
    map_table.setColumnWidth(1, 140)
    map_table.setColumnWidth(2, 120)

    col_table = QtWidgets.QTableWidget(0, 3)
    col_table.setObjectName("pmvMtzColumnTable")
    col_table.setHorizontalHeaderLabels(("Column", "Type", "Dataset"))
    col_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
    col_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    col_table.verticalHeader().setVisible(False)
    col_table.setShowGrid(False)
    col_table.setStyleSheet(catalog_table_css("pmvMtzColumnTable"))
    col_table.horizontalHeader().setStretchLastSection(True)

    for col in inv.get("columns") or ():
        row = col_table.rowCount()
        col_table.insertRow(row)
        col_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(col.get("label") or "")))
        type_text = str(col.get("type") or "")
        detail = str(col.get("type_label") or "")
        if type_text and detail and detail != type_text:
            type_text = "%s  ·  %s" % (type_text, detail)
        col_table.setItem(row, 1, QtWidgets.QTableWidgetItem(type_text))
        col_table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(col.get("dataset") or "")))
    col_table.resizeRowsToContents()

    amp_combo = QtWidgets.QComboBox()
    phase_combo = QtWidgets.QComboBox()
    for label in _amplitude_labels(inv):
        amp_combo.addItem(label, label)
    for label in _phase_labels(inv):
        phase_combo.addItem(label, label)
    add_btn = QtWidgets.QPushButton("Add map")
    add_btn.setAutoDefault(False)
    add_btn.setDefault(False)
    have_custom = amp_combo.count() > 0 and phase_combo.count() > 0
    amp_combo.setEnabled(have_custom)
    phase_combo.setEnabled(have_custom)
    add_btn.setEnabled(have_custom)
    custom = QtWidgets.QHBoxLayout()
    custom.addWidget(QtWidgets.QLabel("Custom"))
    custom.addWidget(amp_combo, stretch=1)
    custom.addWidget(phase_combo, stretch=1)
    custom.addWidget(add_btn)

    buttons = QtWidgets.QDialogButtonBox()
    load_btn = buttons.addButton(LOAD_BUTTON, QtWidgets.QDialogButtonBox.AcceptRole)
    cancel_btn = buttons.addButton(CANCEL_BUTTON, QtWidgets.QDialogButtonBox.RejectRole)
    load_btn.setAutoDefault(False)
    cancel_btn.setAutoDefault(False)
    cancel_btn.setDefault(False)
    mark_primary_button(load_btn)

    apply_required_tooltips(
        [
            (map_table, MAPS_TIP, "MTZ maps"),
            (col_table, COLUMNS_TIP, "MTZ columns"),
            (amp_combo, AMPLITUDE_TIP, "Amplitude"),
            (phase_combo, PHASE_TIP, "Phase"),
            (add_btn, ADD_MAP_TIP, "Add map"),
            (load_btn, LOAD_TIP, LOAD_BUTTON),
            (cancel_btn, CANCEL_TIP, CANCEL_BUTTON),
        ],
        context="MtzImportDialog",
    )

    def _map_ids():
        ids = []
        for row in range(map_table.rowCount()):
            item = map_table.item(row, 1)
            if item is None:
                continue
            data = item.data(user_role)
            if isinstance(data, dict) and data.get("id"):
                ids.append(str(data["id"]))
        return ids

    def _append_map(product, *, checked=None):
        product = dict(product or {})
        factor = str(product.get("factor") or "").strip()
        phase = str(product.get("phase") or "").strip()
        if not factor or not phase:
            return
        product["id"] = product.get("id") or mtz_map_id(factor, phase)
        product["title"] = product.get("title") or mtz_map_title(factor, phase)
        if product["id"] in _map_ids():
            for row in range(map_table.rowCount()):
                item = map_table.item(row, 1)
                data = item.data(user_role) if item is not None else None
                if isinstance(data, dict) and data.get("id") == product["id"]:
                    check = map_table.item(row, 0)
                    if check is not None:
                        check.setCheckState(check_on)
                    map_table.selectRow(row)
                    break
            return
        row = map_table.rowCount()
        map_table.insertRow(row)
        check = QtWidgets.QTableWidgetItem()
        check.setFlags(
            getattr(QtCore.Qt, "ItemIsUserCheckable", 0)
            | getattr(QtCore.Qt, "ItemIsEnabled", 0)
            | getattr(QtCore.Qt, "ItemIsSelectable", 0)
        )
        if checked is None:
            checked = bool(product.get("default"))
        check.setCheckState(check_on if checked else check_off)
        map_table.setItem(row, 0, check)
        name_item = QtWidgets.QTableWidgetItem(str(product.get("title") or ""))
        name_item.setData(user_role, product)
        name_item.setFlags(name_item.flags() & ~getattr(QtCore.Qt, "ItemIsEditable", 0))
        map_table.setItem(row, 1, name_item)
        amp_item = QtWidgets.QTableWidgetItem(factor)
        amp_item.setFlags(amp_item.flags() & ~getattr(QtCore.Qt, "ItemIsEditable", 0))
        map_table.setItem(row, 2, amp_item)
        phase_item = QtWidgets.QTableWidgetItem(phase)
        phase_item.setFlags(phase_item.flags() & ~getattr(QtCore.Qt, "ItemIsEditable", 0))
        map_table.setItem(row, 3, phase_item)
        map_table.selectRow(row)
        _sync_load()

    def _checked_maps():
        selected = []
        for row in range(map_table.rowCount()):
            check = map_table.item(row, 0)
            item = map_table.item(row, 1)
            if check is None or item is None:
                continue
            if int(check.checkState()) != int(check_on):
                continue
            data = item.data(user_role)
            if isinstance(data, dict):
                selected.append(dict(data))
        return selected

    def _sync_load():
        n = len(_checked_maps())
        load_btn.setEnabled(n > 0)
        load_btn.setText("Load %d map%s" % (n, "" if n == 1 else "s") if n else LOAD_BUTTON)

    def _add_custom():
        factor = str(amp_combo.currentData() or amp_combo.currentText() or "").strip()
        phase = str(phase_combo.currentData() or phase_combo.currentText() or "").strip()
        _append_map(
            {"factor": factor, "phase": phase, "title": mtz_map_title(factor, phase)},
            checked=True,
        )

    def _accept():
        selected = _checked_maps()
        if not selected:
            return
        if on_load is not None:
            on_load(selected)
        dialog.accept()

    map_table.itemChanged.connect(lambda *_: _sync_load())
    add_btn.clicked.connect(_add_custom)
    load_btn.clicked.connect(_accept)
    cancel_btn.clicked.connect(dialog.reject)

    map_table.blockSignals(True)
    any_default = False
    for product in maps:
        _append_map(product)
        if product.get("default"):
            any_default = True
    if maps and not any_default:
        check = map_table.item(0, 0)
        if check is not None:
            check.setCheckState(check_on)
    map_table.blockSignals(False)
    _sync_load()

    maps_label = QtWidgets.QLabel("Maps")
    cols_label = QtWidgets.QLabel("Columns")
    root = QtWidgets.QVBoxLayout(dialog)
    root.addWidget(header)
    root.addWidget(maps_label)
    root.addWidget(map_table, stretch=2)
    root.addLayout(custom)
    root.addWidget(cols_label)
    root.addWidget(col_table, stretch=2)
    root.addWidget(buttons)
    dialog.show()
    bind_tool_window(dialog)
    dialog.raise_()
    dialog.activateWindow()
    return dialog
