"""Shared inline editor for one spatial point (used by Points and Arrows)."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from .colors import rgb_to_css
from .points import VisualPoint
from .spatial_source import point_attach_status, point_kind_label
from ..pick import qt_modules
from ..tooltips import apply_required_tooltips
from ..widgets.switch import make_switch
from ..widgets.theme import (
    BORDER,
    DETAIL,
    INK,
    apply_secondary_button_style,
    muted_label_css,
    rgb_css,
    swatch_button_css,
)
from ..widgets.ascii_locale import apply_ascii_float_locale
from ..widgets.type_icons import apply_source_icon

XYZ_LABEL = "XYZ coordinates (Å)"
UPDATE_POSITION_LABEL = "Update position"
PICK_ATOM_LABEL = "Pick atom"
CAMERA_CENTER_LABEL = "Use camera center"
ATTACH_LABEL = "Anchor to atom"
ATTACH_HINT = "Keep anchored if atom moves"
SOURCE_STATUS_LABEL = "Source / status"
COLOR_LABEL = "Color"

PICK_ATOM_TIP = (
    "Click Pick atom, then click an atom in PyMOL. The current selection "
    "is ignored; the clicked atom replaces this point or endpoint."
)
CAMERA_CENTER_TIP = (
    "Use camera center: place at the current view center without snapping to a "
    "nearby atom (unlike Add point with Snap new points to atoms)."
)
ATTACH_TIP = (
    "Keep this point anchored to its atom so it follows if the atom moves."
)
COLOR_TIP = "Choose the color (and opacity) of this point."
ENDPOINT_HELP = (
    "Pick atom waits for a new click in PyMOL and uses that atom. "
    "Use camera center places the point at the view center without atom snap."
)


class SpatialPointEditor:
    """One XYZ / Update position / Anchor / Color card."""

    def __init__(
        self,
        pt: Optional[VisualPoint],
        *,
        title: Optional[str] = None,
        color=None,
        picking: bool = False,
        on_xyz: Optional[Callable] = None,
        on_pick_atom: Optional[Callable] = None,
        on_camera: Optional[Callable] = None,
        on_attach: Optional[Callable] = None,
        on_color: Optional[Callable] = None,
        extras: Sequence = (),
    ):
        QtCore, QtGui, QtWidgets = qt_modules()
        card = QtWidgets.QFrame()
        card.setObjectName("pmvEndpointCard")
        card.setMinimumWidth(0)
        preferred = getattr(QtWidgets.QSizePolicy, "Preferred", None)
        ignored = getattr(QtWidgets.QSizePolicy, "Ignored", None)
        if preferred is not None and ignored is not None:
            card.setSizePolicy(ignored, preferred)
        card.setStyleSheet(
            "QFrame#pmvEndpointCard { background: %s; border: 1px solid %s;"
            " border-radius: 6px; }" % (rgb_css(DETAIL), rgb_css(BORDER))
        )
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)

        if title:
            heading = QtWidgets.QLabel(title)
            heading.setStyleSheet("font-weight: 600; color: %s;" % rgb_css(INK))
            layout.addWidget(heading)

        layout.addWidget(_caption(QtWidgets, XYZ_LABEL))
        xyz_row = QtWidgets.QHBoxLayout()
        xyz_row.setSpacing(6)
        spins = []
        xyz = pt.xyz() if pt is not None else (0.0, 0.0, 0.0)
        for axis, value in zip(("X", "Y", "Z"), xyz):
            col = QtWidgets.QVBoxLayout()
            col.setSpacing(2)
            col.addWidget(_caption(QtWidgets, axis))
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(-9999.0, 9999.0)
            spin.setSingleStep(0.1)
            spin.setValue(float(value))
            spin.setEnabled(pt is not None)
            apply_ascii_float_locale(spin, QtCore)
            spins.append(spin)
            col.addWidget(spin)
            xyz_row.addLayout(col)
        layout.addLayout(xyz_row)

        def emit_xyz(*_args):
            if on_xyz is None or len(spins) != 3:
                return
            on_xyz((float(spins[0].value()), float(spins[1].value()), float(spins[2].value())))

        for spin in spins:
            spin.valueChanged.connect(emit_xyz)

        layout.addWidget(_caption(QtWidgets, UPDATE_POSITION_LABEL))
        actions = QtWidgets.QHBoxLayout()
        actions.setSpacing(6)
        pick = _action_button(
            QtWidgets,
            "Picking…" if picking else PICK_ATOM_LABEL,
            PICK_ATOM_TIP,
        )
        pick.setCheckable(True)
        pick.setChecked(bool(picking))
        apply_source_icon(pick, "selection", QtGui, QtCore, QtWidgets)
        if on_pick_atom is not None:
            pick.clicked.connect(lambda *_: on_pick_atom())
        camera = _action_button(QtWidgets, CAMERA_CENTER_LABEL, CAMERA_CENTER_TIP)
        apply_source_icon(camera, "camera", QtGui, QtCore, QtWidgets)
        if on_camera is not None:
            camera.clicked.connect(lambda *_: on_camera())
        actions.addWidget(pick, stretch=1)
        actions.addWidget(camera, stretch=1)
        layout.addLayout(actions)

        attach_row = QtWidgets.QHBoxLayout()
        attach_row.setSpacing(8)
        pin = make_switch(ATTACH_LABEL, icon="anchor")
        can_attach = pt is not None and pt.can_anchor()
        pin.setEnabled(can_attach)
        pin.setChecked(bool(pt is not None and pt.wants_anchor()))
        if on_attach is not None:
            pin.toggled.connect(lambda checked: on_attach(bool(checked)))
        hint = QtWidgets.QLabel(ATTACH_HINT)
        hint.setStyleSheet(muted_label_css())
        attach_row.addWidget(pin)
        attach_row.addWidget(hint, stretch=1)
        layout.addLayout(attach_row)

        source_color = QtWidgets.QHBoxLayout()
        source_color.setSpacing(12)
        source_col = QtWidgets.QVBoxLayout()
        source_col.setSpacing(2)
        source_col.addWidget(_caption(QtWidgets, SOURCE_STATUS_LABEL))
        kind = QtWidgets.QLabel(point_kind_label(pt))
        kind.setStyleSheet("font-weight: 600; color: %s;" % rgb_css(INK))
        status = QtWidgets.QLabel(point_attach_status(pt))
        status.setStyleSheet(muted_label_css())
        source_col.addWidget(kind)
        source_col.addWidget(status)
        source_color.addLayout(source_col, stretch=1)
        color_col = QtWidgets.QVBoxLayout()
        color_col.setSpacing(2)
        color_col.addWidget(_caption(QtWidgets, COLOR_LABEL))
        color_btn = QtWidgets.QPushButton("Color")
        color_btn.setMinimumWidth(72)
        color_btn.setFlat(False)
        rgb = color if color is not None else (getattr(pt, "color", (0.2, 0.6, 0.9)) if pt is not None else (0.2, 0.6, 0.9))
        color_btn.setStyleSheet(
            swatch_button_css(rgb_to_css(rgb), extra=" min-height: 22px; color: %s;" % rgb_css(INK))
        )
        if on_color is not None:
            color_btn.clicked.connect(lambda *_: on_color())
        color_col.addWidget(color_btn)
        source_color.addLayout(color_col)
        layout.addLayout(source_color)

        for extra in extras or ():
            if extra is not None:
                layout.addWidget(extra)

        apply_required_tooltips(
            [
                (pick, PICK_ATOM_TIP, PICK_ATOM_LABEL),
                (camera, CAMERA_CENTER_TIP, CAMERA_CENTER_LABEL),
                (pin, ATTACH_TIP, ATTACH_LABEL),
                (color_btn, COLOR_TIP, COLOR_LABEL),
            ],
            context="SpatialPointEditor",
        )
        self.widget = card
        self._spins = spins
        self._pick = pick


def _caption(QtWidgets, text):
    label = QtWidgets.QLabel(text)
    label.setStyleSheet(muted_label_css())
    return label


def _action_button(QtWidgets, text, tooltip):
    btn = QtWidgets.QPushButton(text)
    btn.setAutoDefault(False)
    btn.setDefault(False)
    btn.setFlat(False)
    btn.setMinimumWidth(0)
    apply_secondary_button_style(btn, "pmvArrowAction")
    btn.setToolTip(tooltip)
    return btn
