"""Picker: extend a crystal map around an object or selection."""

from __future__ import annotations

from ...fields.crystal import DEFAULT_FIT_MARGIN, coverage_sketch, fit_result_aabb
from ..pick import (
    bind_tool_window,
    configure_overlay_dialog,
    overlay_window,
    qt_modules,
    qt_widget_alive,
)
from ..tooltips import apply_required_tooltips
from ..widgets.ascii_locale import apply_ascii_float_locale
from ..widgets.spin_step import STEP_ANGSTROM, apply_spin_step
from ..widgets.switch import make_switch
from ..widgets.theme import apply_wizard_page_style, mark_primary_button, muted_label_css

EXTEND_DIALOG_TITLE = "Extend map around"
EXTEND_BUTTON = "Apply"
EXTEND_CANCEL = "Cancel"
EXTEND_LIST_TIP = "Object or selection the map cell should cover."
EXTEND_SKETCH_TIP = (
    "Map (blue), target, and the shrink box (dashed) in the plane of largest offset."
)
EXTEND_APPLY_TIP = (
    "Place the map on this target, then optionally shrink to the object and/or "
    "double the sampling (same box, finer grid)."
)
SHRINK_LABEL = "Shrink to object"
DOUBLE_MAP_LABEL = "Double map"
MARGIN_LABEL = "Margin (Å)"
SHRINK_TIP = (
    "Crop the map to the target bounding box plus the margin after placing it."
)
DOUBLE_MAP_TIP = (
    "Same as PyMOL map_double: keep the current box and insert mid-samples "
    "(2n − 1 points, half the step) so the density is finer, not larger."
)
MARGIN_TIP = "Extra Ångströms kept on each side when shrinking to the object."
STATUS_LABEL = {
    "inside": "Covered",
    "partial": "Partly covered",
    "outside": "Not covered",
    "empty": "No atoms",
}
KIND_LABEL = {"object": "Object", "selection": "Selection"}
MAP_FILL = (186, 214, 246, 160)
MAP_EDGE = (24, 82, 158)
FIT_FILL = (196, 176, 232, 70)
FIT_EDGE = (92, 48, 168)
TARGET_FILL = {
    "inside": (186, 232, 214, 140),
    "partial": (252, 226, 196, 150),
    "outside": (252, 210, 196, 150),
    "empty": (226, 232, 238, 120),
}
TARGET_EDGE = {
    "inside": (22, 110, 78),
    "partial": (176, 88, 28),
    "outside": (176, 48, 28),
    "empty": (80, 90, 102),
}


def _row_label(row) -> str:
    kind = KIND_LABEL.get(row.get("kind"), "Target")
    status = STATUS_LABEL.get(row.get("status"), "Unknown")
    n = int(row.get("n_atoms") or 0)
    atoms = "%d atom%s" % (n, "" if n == 1 else "s")
    return "%s    ·    %s    ·    %s    ·    %s" % (row.get("name"), kind, status, atoms)


class _CoverageSketch:
    """Overlay of the map AABB and target in the plane of largest offset."""

    def __init__(self, parent, map_lo, map_hi):
        QtCore, QtGui, QtWidgets = qt_modules()
        self._map_lo = map_lo
        self._map_hi = map_hi
        self._target_lo = None
        self._target_hi = None
        self._fit_lo = None
        self._fit_hi = None
        self._status = "empty"
        widget = QtWidgets.QWidget(parent)
        widget.setObjectName("pmvExtendCoverageSketch")
        widget.setToolTip(EXTEND_SKETCH_TIP)
        widget.setMinimumSize(220, 168)
        widget.setMaximumHeight(210)
        widget.paintEvent = self._paint
        self.widget = widget
        self._QtCore = QtCore
        self._QtGui = QtGui

    def set_target(self, lo, hi, status, fit_lo=None, fit_hi=None):
        self._target_lo = lo
        self._target_hi = hi
        self._fit_lo = fit_lo
        self._fit_hi = fit_hi
        self._status = str(status or "empty")
        try:
            self.widget.update()
        except RuntimeError:
            pass

    def _paint(self, event):
        QtGui = self._QtGui
        painter = QtGui.QPainter(self.widget)
        try:
            painter.setRenderHint(getattr(QtGui.QPainter, "Antialiasing", 0), True)
            rect = self.widget.rect()
            painter.fillRect(rect, QtGui.QColor(255, 255, 255))
            inset = 22
            box = rect.adjusted(inset, 8, -8, -inset)
            sketch = coverage_sketch(
                self._map_lo,
                self._map_hi,
                self._target_lo,
                self._target_hi,
                fit_lo=self._fit_lo,
                fit_hi=self._fit_hi,
            )
            self._draw_rect(painter, QtGui, box, sketch["map"], MAP_FILL, MAP_EDGE, 2)
            fit = sketch.get("fit")
            if fit is not None:
                self._draw_rect(painter, QtGui, box, fit, FIT_FILL, FIT_EDGE, 2, dashed=True)
            target = sketch.get("target")
            if target is not None:
                fill = TARGET_FILL.get(self._status, TARGET_FILL["empty"])
                edge = TARGET_EDGE.get(self._status, TARGET_EDGE["empty"])
                self._draw_rect(painter, QtGui, box, target, fill, edge, 2)
            names = sketch.get("axis_names") or ("X", "Y")
            self._draw_axis_labels(painter, QtGui, box, names)
        finally:
            painter.end()

    def _draw_rect(self, painter, QtGui, box, spec, fill, edge, width, dashed=False):
        x = box.x() + spec["x"] * box.width()
        y = box.y() + spec["y"] * box.height()
        w = max(spec["w"] * box.width(), 2.0)
        h = max(spec["h"] * box.height(), 2.0)
        painter.setBrush(QtGui.QColor(*fill))
        pen = QtGui.QPen(QtGui.QColor(*edge), width)
        if dashed:
            style = getattr(getattr(self._QtCore, "Qt", None), "DashLine", None)
            if style is not None:
                pen.setStyle(style)
        painter.setPen(pen)
        painter.drawRect(int(x), int(y), int(w), int(h))

    def _draw_axis_labels(self, painter, QtGui, box, names):
        hx, vy = str(names[0]), str(names[1])
        painter.setPen(QtGui.QPen(QtGui.QColor(80, 90, 102), 1))
        painter.setBrush(QtGui.QBrush())
        x0, y0 = int(box.left()), int(box.bottom())
        x1, y1 = int(box.right()), int(box.top())
        painter.drawLine(x0, y0, x1, y0)
        painter.drawLine(x0, y0, x0, y1)
        painter.drawLine(x1 - 6, y0 - 4, x1, y0)
        painter.drawLine(x1 - 6, y0 + 4, x1, y0)
        painter.drawLine(x0 - 4, y1 + 6, x0, y1)
        painter.drawLine(x0 + 4, y1 + 6, x0, y1)
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(x1 - 12, y0 + 16, hx)
        painter.drawText(x0 - 16, y1 + 12, vy)


def open_extend_cell_dialog(
    parent,
    rows,
    map_lo,
    map_hi,
    *,
    current=None,
    on_extend=None,
    on_apply=None,
    on_cancel=None,
):
    """Show the object/selection picker. Returns the dialog or None."""
    QtCore, _, QtWidgets = qt_modules()
    if QtWidgets is None or not hasattr(QtWidgets, "QDialog"):
        return None
    overlay = overlay_window(parent)
    dialog = QtWidgets.QDialog(overlay if overlay is not None else parent)
    dialog.setWindowTitle(EXTEND_DIALOG_TITLE)
    dialog.setModal(False)
    dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
    configure_overlay_dialog(dialog, overlay if overlay is not None else parent)
    apply_wizard_page_style(dialog)
    dialog.resize(480, 500)

    rows = list(rows or [])
    sketch = _CoverageSketch(dialog, map_lo, map_hi)
    listing = QtWidgets.QListWidget()
    listing.setObjectName("pmvExtendTargetList")
    listing.setToolTip(EXTEND_LIST_TIP)
    for row in rows:
        item = QtWidgets.QListWidgetItem(_row_label(row))
        item.setData(getattr(QtCore.Qt, "UserRole", 256), dict(row))
        listing.addItem(item)

    shrink = make_switch(SHRINK_LABEL)
    shrink.setObjectName("pmvExtendShrink")
    double_map = make_switch(DOUBLE_MAP_LABEL)
    double_map.setObjectName("pmvExtendDoubleMap")
    margin = QtWidgets.QDoubleSpinBox()
    margin.setObjectName("pmvExtendMargin")
    margin.setRange(0.0, 50.0)
    margin.setDecimals(2)
    margin.setValue(float(DEFAULT_FIT_MARGIN))
    apply_spin_step(margin, STEP_ANGSTROM, decimals=2)
    apply_ascii_float_locale(margin, QtCore)
    options = QtWidgets.QHBoxLayout()
    options.addWidget(shrink)
    options.addWidget(double_map)
    options.addSpacing(12)
    options.addWidget(QtWidgets.QLabel(MARGIN_LABEL))
    options.addWidget(margin)
    options.addStretch(1)

    hint = QtWidgets.QLabel(
        "Blue is the current map. The solid box is the selected object. "
        "Dashed purple is the shrink / double-map result. "
        "The view uses the two axes with the largest offset (labeled on the sketch). "
        "Covered means every atom already sits inside the map."
    )
    hint.setWordWrap(True)
    hint.setStyleSheet(muted_label_css())

    legend = QtWidgets.QLabel(
        "Blue: map   ·   Green: covered   ·   Orange: needs extend   ·   "
        "Dashed: shrink / double-map   ·   Axes labeled on the sketch"
    )
    legend.setStyleSheet(muted_label_css())

    buttons = QtWidgets.QDialogButtonBox()
    extend_btn = buttons.addButton(EXTEND_BUTTON, QtWidgets.QDialogButtonBox.AcceptRole)
    cancel_btn = buttons.addButton(EXTEND_CANCEL, QtWidgets.QDialogButtonBox.RejectRole)
    mark_primary_button(extend_btn)
    apply_required_tooltips(
        [
            (listing, EXTEND_LIST_TIP, "Extend target"),
            (sketch.widget, EXTEND_SKETCH_TIP, "Coverage sketch"),
            (shrink, SHRINK_TIP, SHRINK_LABEL),
            (double_map, DOUBLE_MAP_TIP, DOUBLE_MAP_LABEL),
            (margin, MARGIN_TIP, MARGIN_LABEL),
            (extend_btn, EXTEND_APPLY_TIP, EXTEND_BUTTON),
            (cancel_btn, "Close without changing the map.", EXTEND_CANCEL),
        ],
        context="ExtendCellDialog",
    )

    def _current_row():
        item = listing.currentItem()
        if item is None:
            return None
        data = item.data(getattr(QtCore.Qt, "UserRole", 256))
        return data if isinstance(data, dict) else None

    def _fit_box(row):
        if row is None:
            return None
        return fit_result_aabb(
            map_lo,
            map_hi,
            row.get("lo"),
            row.get("hi"),
            shrink=bool(shrink.isChecked()),
            margin=float(margin.value()),
            double_map=bool(double_map.isChecked()),
        )

    def _sync_sketch():
        row = _current_row()
        if row is None:
            sketch.set_target(None, None, "empty")
            extend_btn.setEnabled(False)
            return
        fit = _fit_box(row)
        fit_lo = fit_hi = None
        if fit is not None:
            fit_lo, fit_hi = fit
        sketch.set_target(row.get("lo"), row.get("hi"), row.get("status"), fit_lo, fit_hi)
        extend_btn.setEnabled(True)
        margin.setEnabled(bool(shrink.isChecked()))

    def _emit(name, **opts):
        if on_apply is not None:
            on_apply(name, **opts)
            return
        if on_extend is not None:
            on_extend(name)

    def _accept():
        row = _current_row()
        if row is None or (on_apply is None and on_extend is None):
            dialog.reject()
            return
        _emit(
            str(row.get("name") or ""),
            shrink=bool(shrink.isChecked()),
            margin=float(margin.value()),
            double_map=bool(double_map.isChecked()),
        )
        dialog.accept()

    listing.currentRowChanged.connect(lambda *_: _sync_sketch())
    shrink.toggled.connect(lambda *_: _sync_sketch())
    double_map.toggled.connect(lambda *_: _sync_sketch())
    margin.valueChanged.connect(lambda *_: _sync_sketch())
    extend_btn.clicked.connect(_accept)
    cancel_btn.clicked.connect(dialog.reject)
    if on_cancel is not None:
        dialog.rejected.connect(lambda: on_cancel())

    choose = None
    names = [str(row.get("name") or "") for row in rows]
    if current and current in names:
        choose = names.index(current)
    elif rows:
        choose = 0
    if choose is not None:
        listing.setCurrentRow(choose)
    _sync_sketch()

    root = QtWidgets.QVBoxLayout(dialog)
    root.addWidget(listing, stretch=1)
    root.addWidget(sketch.widget)
    root.addLayout(options)
    root.addWidget(legend)
    root.addWidget(hint)
    root.addWidget(buttons)
    dialog.show()
    bind_tool_window(dialog)
    dialog.raise_()
    dialog.activateWindow()
    return dialog
