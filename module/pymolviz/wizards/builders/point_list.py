"""Expandable point list using shared spatial row + point editor."""

from __future__ import annotations

from .spatial_item_list import POINT_LIST_COLUMNS, SpatialItemList, make_spatial_detail, make_spatial_item_row
from .spatial_point_editor import SpatialPointEditor
from .spatial_source import point_primary_label, point_source_summary
from ..pick import qt_modules
from ..tooltips import warn_missing_setting_tooltips

ADD_POINT_TIP = (
    "Insert points from the chosen source. Current selection uses atoms "
    "selected now. Fresh selection waits for a new pick after Add."
)


class PointListEditor:
    """Scrollable point rows with one shared SpatialPointEditor when expanded."""

    def __init__(self, parent, host):
        self._host = host
        self._list = SpatialItemList(
            parent,
            add_text="+ Add point",
            add_tip=ADD_POINT_TIP,
            on_add=self._host.add_point,
            columns=POINT_LIST_COLUMNS,
            empty_hint="No points yet.",
            context="PointListEditor",
        )
        self._box = self._list.widget
        self._add_btn = self._list.add_button

    @property
    def widget(self):
        return self._box

    @property
    def add_button(self):
        return self._add_btn

    def attach_add_to_section(self, section) -> None:
        self._list.attach_add_to_section(section)

    def set_add_enabled(self, enabled: bool) -> None:
        self._list.set_add_enabled(enabled)

    def set_add_waiting(self, waiting: bool) -> None:
        from .point_insertion import CANCEL_FRESH_TIP

        if waiting:
            self._list.set_add_chrome("Cancel pick", CANCEL_FRESH_TIP)
        else:
            self._list.reset_add_chrome()

    def set_add_source_icon(self, source: str) -> None:
        self._list.set_add_source_icon(source)

    def rebuild(self, points, selected_index, pick_index=None):
        QtCore, QtGui, QtWidgets = qt_modules()
        self._list.clear()
        for i, pt in enumerate(points):
            selected = selected_index == i
            self._list.add_block(self._make_block(QtWidgets, pt, i, selected, pick_index))
        self._list.set_empty(len(points) == 0)
        warn_missing_setting_tooltips(self._box, context="PointListEditor")

    def _make_block(self, QtWidgets, pt, index, selected, pick_index=None):
        host = self._host
        xyz = pt.xyz()
        block = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(block)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(make_spatial_item_row(
            columns=POINT_LIST_COLUMNS,
            selected=selected,
            enabled=pt.enabled,
            index_text=str(index + 1),
            values=(
                point_primary_label(pt),
                point_source_summary(pt),
                "%.2f" % xyz[0],
                "%.2f" % xyz[1],
                "%.2f" % xyz[2],
            ),
            color=pt.color,
            on_enabled=lambda checked, i=index: host.set_point_enabled(i, checked),
            on_select=lambda i=index: host.select_point(i),
            on_delete=lambda i=index: host.delete_point(i),
            on_color=lambda i=index: host.edit_point_color(i),
            overflow_actions=(
                ("Color…", lambda i=index: host.edit_point_color(i), True),
                ("Use camera center", lambda i=index: host.camera_point(i), True),
                ("Pick atom", lambda i=index: host.pick_point_atom(i), True),
                ("Delete", lambda i=index: host.delete_point(i), True),
            ),
            context="PointListEditor",
        ))
        if selected:
            extras = host.point_editor_extras(index, pt)
            editor = SpatialPointEditor(
                pt,
                title="Point %d: %s" % (index + 1, point_primary_label(pt)),
                color=pt.color,
                picking=pick_index == index,
                on_xyz=lambda xyz, i=index: host.set_point_xyz(i, xyz),
                on_pick_atom=lambda i=index: host.pick_point_atom(i),
                on_camera=lambda i=index: host.camera_point(i, snap=False),
                on_attach=lambda checked, i=index: host.set_point_anchor(i, checked),
                on_color=lambda i=index: host.edit_point_color(i),
                extras=extras,
            )
            wrap, inner = make_spatial_detail(QtWidgets)
            inner.addWidget(editor.widget)
            layout.addWidget(wrap)
        return block
