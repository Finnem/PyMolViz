"""Expandable arrow list using shared spatial row + point editors."""

from __future__ import annotations

from .pairs import PENDING_END, endpoint_label, free_point_display_names
from .point_insertion import (
    INSERT_SOURCE_CAMERA,
    INSERT_SOURCE_FRESH,
    INSERT_SOURCE_SELECTION,
    InsertionAddBar,
)
from .spatial_item_list import (
    ARROW_LIST_COLUMNS,
    SpatialItemList,
    make_spatial_detail,
    make_spatial_item_row,
)
from .spatial_point_editor import (
    ATTACH_LABEL,
    CAMERA_CENTER_LABEL,
    CAMERA_CENTER_TIP,
    ENDPOINT_HELP,
    PICK_ATOM_LABEL,
    PICK_ATOM_TIP,
    UPDATE_POSITION_LABEL,
    XYZ_LABEL,
    SpatialPointEditor,
)
from .spatial_source import arrow_source_summary
from ..pick import qt_modules
from ..tooltips import warn_missing_setting_tooltips
from ..widgets.theme import muted_label_css
from ..widgets.type_icons import action_icon_pixmap

ADD_ARROW_TIP = (
    "Add Camera Center starts an arrow at the view. Add Current Selection "
    "uses atoms selected now. Add Clicked Atoms waits for picks in PyMOL."
)
CANCEL_PICK_TIP = "Cancel the current start/end pick and drop any incomplete arrow."
DELETE_ARROW_TIP = "Remove this arrow from the list."
PICK_START_TIP = PICK_ATOM_TIP
PICK_END_TIP = PICK_ATOM_TIP
CAMERA_ENDPOINT_TIP = CAMERA_CENTER_TIP
SWAP_ARROW_TIP = "Swap the start and end of this arrow."
COLOR_ARROW_TIP = (
    "Start color (upper left) and end color (lower right). "
    "Click to set one color for the whole arrow."
)
ANCHOR_START_TIP = ATTACH_LABEL
ANCHOR_END_TIP = ATTACH_LABEL
FOCUS_ENDPOINT_TIP = "Select and center this atom (or frame this point) in PyMOL."


class ArrowPairEditor:
    """Scrollable pair rows with two shared SpatialPointEditor cards when expanded."""

    def __init__(self, parent, host):
        self._host = host
        self._list = SpatialItemList(
            parent,
            add_text="+ Add arrow",
            add_tip=ADD_ARROW_TIP,
            on_add=self._host.add_arrow,
            columns=ARROW_LIST_COLUMNS,
            empty_hint="No arrows yet.",
            context="ArrowPairEditor",
            show_add=False,
        )
        self._box = self._list.widget
        self._add_btn = self._list.add_button
        self._picking = False
        self._add_bar = InsertionAddBar(
            on_clicked_atoms=self._host.set_add_clicked_atoms,
            on_camera=lambda: self._host.add_arrow(INSERT_SOURCE_CAMERA),
            on_selection=lambda: self._host.add_arrow(INSERT_SOURCE_SELECTION),
        )

    @property
    def widget(self):
        return self._box

    def attach_add_to_section(self, section) -> None:
        if section is not None and hasattr(section, "add_header_widget"):
            section.add_header_widget(self._add_bar.widget)
        self._list.attach_add_to_section(section)

    def add_source(self) -> str:
        if self.clicked_atoms_checked():
            return INSERT_SOURCE_FRESH
        return INSERT_SOURCE_SELECTION

    def clicked_atoms_checked(self) -> bool:
        return self._add_bar.clicked_atoms_checked()

    def set_clicked_atoms(self, checked: bool, notify: bool = True) -> None:
        self._add_bar.set_clicked_atoms(checked, notify=notify)

    def sync_add_bar(self, cmd) -> None:
        self._add_bar.sync_from_cmd(cmd)

    def set_picking(self, picking: bool) -> None:
        self._picking = bool(picking)

    def _sync_add_chrome(self) -> None:
        return

    def rebuild(self, pairs, selected_id, pick_pair_id, pick_role, context=None):
        QtCore, QtGui, QtWidgets = qt_modules()
        self._list.clear()
        free_names = free_point_display_names(pairs)
        self.set_picking(pick_role is not None)
        for i, pair in enumerate(pairs):
            selected = pair.pair_id == selected_id
            self._list.add_block(
                self._make_block(
                    QtCore, QtGui, QtWidgets, pair, i, selected,
                    pick_pair_id, pick_role, free_names, context,
                )
            )
        self._list.set_empty(len(pairs) == 0)
        warn_missing_setting_tooltips(self._box, context="ArrowPairEditor")

    def _make_block(
        self, QtCore, QtGui, QtWidgets, pair, index, selected,
        pick_pair_id, pick_role, free_names, context,
    ):
        host = self._host
        pid = pair.pair_id
        block = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(block)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        start = endpoint_label(pair.start, pending="[pick start…]", free_names=free_names)
        end = endpoint_label(pair.end, pending=PENDING_END, free_names=free_names)
        layout.addWidget(make_spatial_item_row(
            columns=ARROW_LIST_COLUMNS,
            selected=selected,
            enabled=pair.enabled,
            index_text=str(index + 1),
            values=(start, end, arrow_source_summary(pair.start, pair.end)),
            color=pair.start.color,
            end_color=(pair.end.color if pair.end is not None else pair.start.color),
            on_enabled=lambda checked, i=pid: host.set_pair_enabled(i, checked),
            on_select=lambda i=pid: host.select_arrow(i),
            on_delete=lambda i=pid: host.delete_arrow(i),
            on_color=lambda i=pid: host.edit_color(i),
            color_tip=COLOR_ARROW_TIP,
            overflow_actions=(
                ("Color…", lambda i=pid: host.edit_color(i), True),
                ("Swap start and end", lambda i=pid: host.swap_arrow(i), pair.is_complete()),
                ("Delete", lambda i=pid: host.delete_arrow(i), True),
            ),
            context="ArrowPairEditor",
        ))
        if selected:
            layout.addWidget(self._details(
                QtCore, QtGui, QtWidgets, pair, pick_pair_id, pick_role, free_names,
            ))
        return block

    def _details(self, QtCore, QtGui, QtWidgets, pair, pick_pair_id, pick_role, free_names):
        host = self._host
        pid = pair.pair_id
        box, layout = make_spatial_detail(QtWidgets)
        ends = QtWidgets.QHBoxLayout()
        ends.setSpacing(4)
        start_card = self._endpoint_card(
            pair, "start", pick_pair_id, pick_role, free_names,
        ).widget
        end_card = self._endpoint_card(
            pair, "end", pick_pair_id, pick_role, free_names,
        ).widget
        ends.addWidget(start_card, stretch=1)
        swap = QtWidgets.QToolButton()
        swap.setAutoRaise(True)
        swap.setToolTip(SWAP_ARROW_TIP)
        swap.setFixedSize(22, 22)
        pix = action_icon_pixmap("forward", QtGui, QtCore, QtWidgets, size=16)
        if pix is not None and QtGui is not None:
            swap.setIcon(QtGui.QIcon(pix))
            swap.setIconSize(QtCore.QSize(16, 16))
        else:
            swap.setText("→")
        swap.setEnabled(pair.is_complete())
        swap.clicked.connect(lambda *_ , i=pid: host.swap_arrow(i))
        ends.addWidget(swap)
        ends.addWidget(end_card, stretch=1)
        layout.addLayout(ends)
        help_row = QtWidgets.QLabel(ENDPOINT_HELP)
        help_row.setWordWrap(True)
        help_row.setStyleSheet(muted_label_css())
        layout.addWidget(help_row)
        return box

    def _endpoint_card(self, pair, role, pick_pair_id, pick_role, free_names):
        host = self._host
        pid = pair.pair_id
        pt = pair.start if role == "start" else pair.end
        picking = pick_pair_id == pid and pick_role == role
        return SpatialPointEditor(
            pt,
            title="Start point" if role == "start" else "End point",
            color=(pt.color if pt is not None else pair.color),
            picking=picking,
            on_xyz=lambda xyz, i=pid, r=role: host.set_endpoint_xyz(i, r, xyz),
            on_pick_atom=lambda i=pid, r=role: host.pick_endpoint(i, r),
            on_camera=lambda i=pid, r=role: host.camera_endpoint(i, r, snap=False),
            on_attach=lambda checked, i=pid, r=role: host.set_endpoint_anchor(i, r, checked),
            on_color=lambda i=pid, r=role: host.edit_endpoint_color(i, r),
        )
