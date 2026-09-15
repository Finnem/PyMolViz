"""Placement-option labels shared with tests (implementation lives in PointInsertionWidget)."""

from __future__ import annotations

from .point_insertion import (
    ADD_POINT_HEADER_LABEL,
    ADD_POINT_TIP,
    EXPORT_SEL_LABEL,
    EXPORT_SEL_TIP,
    HOOK_LABEL,
    SHOW_COORDS_LABEL,
    SHOW_COORDS_TIP,
    SNAP_LABEL,
    ZOOM_LABEL,
)

# Re-export historical builder used only by older tests.
from typing import Callable, NamedTuple, Optional


class PointOptionControls(NamedTuple):
    snap: object
    hook: object
    zoom: object
    show_coords: object
    export_sel: object
    add_header: object
    layout: object


def build_point_option_controls(
    QtWidgets,
    *,
    on_export: Callable[[], None],
    on_add: Callable[[], None],
    on_show_coords: Optional[Callable[[bool], None]] = None,
) -> PointOptionControls:
    snap = QtWidgets.QCheckBox(SNAP_LABEL)
    snap.setChecked(True)
    hook = QtWidgets.QCheckBox(HOOK_LABEL)
    hook.setChecked(True)
    zoom = QtWidgets.QCheckBox(ZOOM_LABEL)
    show_coords = QtWidgets.QCheckBox(SHOW_COORDS_LABEL)
    show_coords.setChecked(False)
    if on_show_coords is not None:
        show_coords.toggled.connect(on_show_coords)
    export_sel = QtWidgets.QPushButton(EXPORT_SEL_LABEL)
    export_sel.clicked.connect(lambda *_args: on_export())
    add_header = QtWidgets.QPushButton(ADD_POINT_HEADER_LABEL)
    add_header.setAutoDefault(False)
    add_header.setDefault(False)
    add_header.clicked.connect(lambda *_args: on_add())
    layout = QtWidgets.QVBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    row1 = QtWidgets.QHBoxLayout()
    row1.addWidget(snap)
    row1.addWidget(hook)
    row1.addWidget(zoom)
    row1.addStretch(1)
    row1.addWidget(add_header)
    row2 = QtWidgets.QHBoxLayout()
    row2.addWidget(show_coords)
    row2.addStretch(1)
    row2.addWidget(export_sel)
    layout.addLayout(row1)
    layout.addLayout(row2)
    return PointOptionControls(
        snap, hook, zoom, show_coords, export_sel, add_header, layout,
    )
