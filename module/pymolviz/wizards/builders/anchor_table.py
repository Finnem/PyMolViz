"""Per-row anchor checkbox helpers for point builder tables."""

from __future__ import annotations

from typing import Callable, Tuple

from .colors import rgba_to_css
from .points import VisualPoint
from ..widgets.theme import swatch_button_css

ENABLED_COL = "Enabled"
ANCHOR_COL = "Attach"
COLOR_COL = "Color"
NAME_COL = "Atom / Label"
SOURCE_COL = "Source"
X_COL = "X"
Y_COL = "Y"
Z_COL = "Z"
RADIUS_COL = "R (Å)"

ENABLED_TIP = (
    "Include this point in the preview and committed object. "
    "Disabled points stay in the table but are omitted from rendering."
)

ANCHOR_START_COL = "Anch S"
ANCHOR_END_COL = "Anch E"

ANCHOR_TIP = (
    "When checked, this point stays attached to its atom and follows if the atom moves. "
    "When unchecked, the position stays fixed at the current xyz."
)
ANCHOR_DISABLED_TIP = "No atom reference — add from selection or snap to an atom first."
COLOR_TIP = (
    "Color of this point in PyMOL. Click the swatch to pick a solid color and opacity."
)
NAME_TIP = "Atom identifier or a custom label for this point."
SOURCE_TIP = "How this point was placed (atom selection, camera center, or manual)."
COORD_TIP = "Position in Ångströms. Hidden unless Show coordinates is on."

POINT_DATA_COLS: Tuple[str, ...] = (COLOR_COL, NAME_COL, SOURCE_COL, X_COL, Y_COL, Z_COL)
POINT_ENABLED_COL = 0
POINT_ANCHOR_COL = 1
POINT_COLOR_COL = 2
POINT_NAME_COL = 3
POINT_SOURCE_COL = 4
POINT_X_COL = 5
POINT_Y_COL = 6
POINT_Z_COL = 7
SURFACE_RADIUS_COL = 8
COORDINATE_COLS: Tuple[int, ...] = (POINT_X_COL, POINT_Y_COL, POINT_Z_COL)

SURFACE_RADIUS_TIP = (
    "Leave blank to inherit the global Atom radius or VDW scale. "
    "Type a value in Ångströms to override this point."
)

HEADER_TIPS = {
    ENABLED_COL: ENABLED_TIP,
    ANCHOR_COL: ANCHOR_TIP,
    COLOR_COL: COLOR_TIP,
    NAME_COL: NAME_TIP,
    SOURCE_COL: SOURCE_TIP,
    X_COL: COORD_TIP,
    Y_COL: COORD_TIP,
    Z_COL: COORD_TIP,
    RADIUS_COL: SURFACE_RADIUS_TIP,
}


def point_columns():
    return (ENABLED_COL, ANCHOR_COL) + POINT_DATA_COLS


def surface_point_columns():
    return point_columns() + (RADIUS_COL,)


def arrow_columns():
    return (
        ANCHOR_START_COL,
        ANCHOR_END_COL,
        "Start",
        "Start src",
        "End",
        "End src",
        "X0",
        "Y0",
        "Z0",
        "X1",
        "Y1",
        "Z1",
    )


ARROW_START_NAME_COL = 2
ARROW_START_SRC_COL = 3
ARROW_END_NAME_COL = 4
ARROW_END_SRC_COL = 5
ARROW_X0_COL = 6
ARROW_Y0_COL = 7
ARROW_Z0_COL = 8
ARROW_X1_COL = 9
ARROW_Y1_COL = 10
ARROW_Z1_COL = 11


def anchor_col_index(columns) -> int:
    return list(columns).index(ANCHOR_COL)


def enabled_col_index(columns) -> int:
    return list(columns).index(ENABLED_COL)


def arrow_anchor_col_indices(columns):
    cols = list(columns)
    return cols.index(ANCHOR_START_COL), cols.index(ANCHOR_END_COL)


def configure_point_table(table, columns, QtWidgets, QtCore, *, show_coords=False):
    """Headers, resize modes, and default-hidden coordinate columns."""
    labels = list(columns)
    table.setColumnCount(len(labels))
    table.setHorizontalHeaderLabels(labels)
    table.verticalHeader().setVisible(False)
    table.setAlternatingRowColors(True)
    table.setShowGrid(True)
    header = table.horizontalHeader()
    header.setVisible(True)
    header.setStretchLastSection(False)
    header.setHighlightSections(False)
    try:
        header.setDefaultAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
    except Exception:
        pass

    n_cols = len(labels)
    modes = QtWidgets.QHeaderView
    def _resize(col, mode):
        if 0 <= col < n_cols:
            header.setSectionResizeMode(col, mode)

    _resize(0, modes.ResizeToContents)
    if ENABLED_COL in labels:
        _resize(labels.index(ENABLED_COL), modes.ResizeToContents)
    if ANCHOR_COL in labels:
        _resize(labels.index(ANCHOR_COL), modes.ResizeToContents)
    if COLOR_COL in labels:
        _resize(labels.index(COLOR_COL), modes.ResizeToContents)
    if NAME_COL in labels:
        _resize(labels.index(NAME_COL), modes.Stretch)
    if SOURCE_COL in labels:
        _resize(labels.index(SOURCE_COL), modes.Stretch)
    for name in (X_COL, Y_COL, Z_COL, RADIUS_COL):
        if name in labels:
            _resize(labels.index(name), modes.ResizeToContents)

    for index, name in enumerate(labels):
        item = table.horizontalHeaderItem(index)
        if item is None:
            continue
        tip = HEADER_TIPS.get(name)
        if tip:
            item.setToolTip(tip)

    set_coordinate_columns_visible(table, show_coords)


def set_coordinate_columns_visible(table, visible: bool) -> None:
    n_cols = int(table.columnCount())
    for col in COORDINATE_COLS:
        if 0 <= col < n_cols:
            table.setColumnHidden(col, not bool(visible))


def sync_enabled_cell(
    table,
    row: int,
    col: int,
    pt: VisualPoint,
    on_toggled: Callable[[int, bool], None],
    QtWidgets,
    QtCore,
) -> None:
    table.removeCellWidget(row, col)
    cb = QtWidgets.QCheckBox()
    cb.setChecked(bool(getattr(pt, "enabled", True)))
    cb.setToolTip(ENABLED_TIP)
    cb.toggled.connect(lambda checked, r=row: on_toggled(r, checked))

    wrapper = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(wrapper)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setAlignment(QtCore.Qt.AlignCenter)
    layout.addWidget(cb)
    table.setCellWidget(row, col, wrapper)


def sync_anchor_cell(
    table,
    row: int,
    col: int,
    pt: VisualPoint,
    on_toggled: Callable[[int, bool], None],
    QtWidgets,
    QtCore,
) -> None:
    """Replace the anchor checkbox for one table cell."""
    table.removeCellWidget(row, col)
    cb = QtWidgets.QCheckBox()
    cb.setChecked(pt.wants_anchor())
    cb.setEnabled(pt.can_anchor())
    cb.setToolTip(ANCHOR_TIP if pt.can_anchor() else ANCHOR_DISABLED_TIP)
    cb.toggled.connect(lambda checked, r=row: on_toggled(r, checked))

    wrapper = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(wrapper)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setAlignment(QtCore.Qt.AlignCenter)
    layout.addWidget(cb)
    table.setCellWidget(row, col, wrapper)


def _color_button_style(pt: VisualPoint) -> str:
    if getattr(pt, "field_id", None):
        from ...util.field_sample import rgb_from_scalars
        from .colors import rgb_to_hex
        cmap = pt.field_colormap or "RdYlBu_r"
        rgb, _ = rgb_from_scalars([0.0, 1.0], cmap, clims=(0.0, 1.0))
        lo = rgb_to_hex((float(rgb[0, 0]), float(rgb[0, 1]), float(rgb[0, 2])))
        hi = rgb_to_hex((float(rgb[1, 0]), float(rgb[1, 1]), float(rgb[1, 2])))
        fill = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 %s, stop:1 %s)" % (lo, hi)
        return swatch_button_css(
            fill,
            extra=" min-width: 18px; max-width: 22px; min-height: 18px; max-height: 22px;",
        )
    return swatch_button_css(
        rgba_to_css(pt.rgba()),
        extra=" min-width: 18px; max-width: 22px; min-height: 18px; max-height: 22px;",
    )


def _color_button_tip(pt: VisualPoint) -> str:
    if getattr(pt, "field_id", None):
        label = str(pt.field_id)
        prefix = "pymol_map:"
        if label.startswith(prefix):
            label = label[len(prefix):]
        return "Colored by field: %s. Click to change." % label
    return COLOR_TIP


def sync_color_cell(
    table,
    row: int,
    col: int,
    pt: VisualPoint,
    on_pick: Callable[[int], None],
    QtWidgets,
    QtCore,
) -> None:
    """Color swatch button, separate from the atom/label text."""
    table.removeCellWidget(row, col)
    btn = QtWidgets.QPushButton()
    btn.setFlat(True)
    btn.setToolTip(_color_button_tip(pt))
    btn.setStyleSheet(_color_button_style(pt))
    btn.clicked.connect(lambda *_args, r=row: on_pick(r))

    wrapper = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(wrapper)
    layout.setContentsMargins(4, 2, 4, 2)
    layout.setAlignment(QtCore.Qt.AlignCenter)
    layout.addWidget(btn)
    table.setCellWidget(row, col, wrapper)

    item = table.item(row, col)
    if item is None:
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
        table.setItem(row, col, item)


def update_color_cell(table, row: int, col: int, pt: VisualPoint, QtWidgets) -> None:
    wrap = table.cellWidget(row, col)
    if wrap is None:
        return
    btn = wrap.findChild(QtWidgets.QPushButton)
    if btn is not None:
        btn.setStyleSheet(_color_button_style(pt))
        btn.setToolTip(_color_button_tip(pt))


def block_table_selection_signals(table) -> bool:
    """Block selection-model signals while rebuilding table rows."""
    sm = table.selectionModel()
    if sm is None:
        return False
    sm.blockSignals(True)
    return True


def unblock_table_selection_signals(table, blocked: bool) -> None:
    if not blocked:
        return
    sm = table.selectionModel()
    if sm is not None:
        sm.blockSignals(False)
