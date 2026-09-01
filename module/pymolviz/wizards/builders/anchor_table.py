"""Per-row anchor checkbox helpers for point builder tables."""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

from .points import VisualPoint

ANCHOR_COL = "Anchor"
ANCHOR_START_COL = "Anch S"
ANCHOR_END_COL = "Anch E"

ANCHOR_TIP = (
    "When checked, the CGO follows this atom if it moves. "
    "When unchecked, the position stays fixed at the current xyz."
)
ANCHOR_DISABLED_TIP = "No atom reference — add from selection or snap to atom first."

POINT_DATA_COLS: Tuple[str, ...] = ("Name", "Source", "X", "Y", "Z")
POINT_NAME_COL = 1
POINT_SOURCE_COL = 2
POINT_X_COL = 3
POINT_Y_COL = 4
POINT_Z_COL = 5


def point_columns():
    return (ANCHOR_COL,) + POINT_DATA_COLS


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


def arrow_anchor_col_indices(columns):
    cols = list(columns)
    return cols.index(ANCHOR_START_COL), cols.index(ANCHOR_END_COL)


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
    cb.setChecked(pt.is_anchored())
    cb.setEnabled(pt.can_anchor())
    cb.setToolTip(ANCHOR_TIP if pt.can_anchor() else ANCHOR_DISABLED_TIP)
    cb.toggled.connect(lambda checked, r=row: on_toggled(r, checked))

    wrapper = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(wrapper)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setAlignment(QtCore.Qt.AlignCenter)
    layout.addWidget(cb)
    table.setCellWidget(row, col, wrapper)


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
