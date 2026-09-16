"""X/Y/Z cardinal clip plane row for field visual Modifiers."""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

from ...fields.clip import default_cardinal_plane, normalize_cardinal_planes, opposite_cardinal_plane
from ..pick import qt_modules
from ..tooltips import (
    ADD_AXIS_PLANE_TIP,
    FLIP_CLIP_TIP,
    REMOVE_AXIS_PLANE_TIP,
    SHOW_CLIP_VISUALS_TIP,
    SHOW_PLANE_GIZMO_TIP,
    apply_required_tooltips,
)
from ..widgets.ascii_locale import apply_ascii_float_locale
from ..widgets.spin_step import STEP_ANGSTROM, apply_spin_step
from ..widgets.switch import make_switch

CLIP_TIP = (
    "Clip the field with planes perpendicular to X, Y, and Z. Enable each "
    "axis separately. Drag that plane along its axis, or edit the position. "
    "Flip chooses which half to keep. Use + to clip both sides of an axis "
    "(at most two planes), then − to remove the extra one. Hide a plane's "
    "rectangle without turning the clip off."
)
CLIP_ENABLE_TIPS = (
    "Enable an X clipping plane. Drag it along X, or edit the position.",
    "Enable a Y clipping plane. Drag it along Y, or edit the position.",
    "Enable a Z clipping plane. Drag it along Z, or edit the position.",
)
CLIP_POS_TIPS = (
    "Position of the X clip plane in Ångströms.",
    "Position of the Y clip plane in Ångströms.",
    "Position of the Z clip plane in Ångströms.",
)
CLIP_POS2_TIPS = (
    "Position of the second X clip plane in Ångströms.",
    "Position of the second Y clip plane in Ångströms.",
    "Position of the second Z clip plane in Ångströms.",
)


class CardinalClipSection:
    """Three axis switches with up to two faces each (lo and hi)."""

    def __init__(
        self,
        parent,
        *,
        on_changed: Optional[Callable[[], None]] = None,
        seed_plane: Optional[Callable[[int], dict]] = None,
        seed_opposite: Optional[Callable[[int, dict], dict]] = None,
        on_axis_enabled: Optional[Callable[[int], None]] = None,
        on_axis_disabled: Optional[Callable[[int], None]] = None,
        on_axis_focus: Optional[Callable[..., None]] = None,
        on_axis_flip: Optional[Callable[[int], None]] = None,
        on_gizmo_changed: Optional[Callable[[], None]] = None,
        context: str = "FieldVisualBuilderPage",
    ):
        self._on_changed = on_changed or (lambda: None)
        self._seed_plane = seed_plane or (lambda _axis: default_cardinal_plane(0, None))
        self._seed_opposite = seed_opposite or (
            lambda _axis, existing: opposite_cardinal_plane(existing, None)
        )
        self._on_axis_enabled = on_axis_enabled or (lambda _axis: None)
        self._on_axis_disabled = on_axis_disabled or (lambda _axis: None)
        self._on_axis_focus = on_axis_focus or (lambda *_a, **_k: None)
        self._on_axis_flip = on_axis_flip or (lambda _axis: None)
        self._on_gizmo_changed = on_gizmo_changed or (lambda: None)
        self._context = context
        self._clip_axis_on = [None, None, None]
        self._clip_axis_gizmo = [None, None, None]
        self._clip_axis_pair = [None, None, None]
        self._clip_pair_row = [None, None, None]
        self._clip_face_pos = [[None, None], [None, None], [None, None]]
        self._clip_face_flip = [[None, None], [None, None], [None, None]]
        self._clip_face_hi = [[False, True], [False, True], [False, True]]
        self._clip_face_on = [[False, False], [False, False], [False, False]]
        self._show_gizmos = None
        self._spin_suspend = False
        self._widget = None
        self._build(parent)

    @property
    def widget(self):
        return self._widget

    def planes(self):
        planes = []
        for axis in range(3):
            box = self._clip_axis_on[axis]
            if box is None or not box.isChecked():
                continue
            for face in range(2):
                if not self._clip_face_on[axis][face]:
                    continue
                spin = self._clip_face_pos[axis][face]
                position = float(spin.value()) if spin is not None else 0.0
                planes.append({
                    "axis": axis,
                    "position": position,
                    "hi": bool(self._clip_face_hi[axis][face]),
                })
        return normalize_cardinal_planes(planes)

    def set_planes(self, planes) -> None:
        grouped = {0: [], 1: [], 2: []}
        for plane in normalize_cardinal_planes(planes):
            grouped[int(plane["axis"])].append(plane)
        self._spin_suspend = True
        try:
            for axis in range(3):
                faces = grouped[axis]
                on = bool(faces)
                if self._clip_axis_on[axis] is not None:
                    self._clip_axis_on[axis].setChecked(on)
                self._clip_face_on[axis] = [False, False]
                if not faces:
                    continue
                self._apply_face(axis, 0, faces[0])
                if len(faces) > 1:
                    self._apply_face(axis, 1, faces[1])
        finally:
            self._spin_suspend = False
        self.sync_widgets()

    def reset(self) -> None:
        self._spin_suspend = True
        try:
            self._clip_face_hi = [[False, True], [False, True], [False, True]]
            self._clip_face_on = [[False, False], [False, False], [False, False]]
            for axis in range(3):
                if self._clip_axis_on[axis] is not None:
                    self._clip_axis_on[axis].setChecked(False)
                if self._clip_axis_gizmo[axis] is not None:
                    self._clip_axis_gizmo[axis].setChecked(True)
            if self._show_gizmos is not None:
                self._show_gizmos.setChecked(True)
        finally:
            self._spin_suspend = False
        self.sync_widgets()

    def any_enabled(self) -> bool:
        return any(
            box is not None and box.isChecked()
            for box in self._clip_axis_on
        )

    def enabled_axes(self):
        return [
            i for i in range(3)
            if self._clip_axis_on[i] is not None and self._clip_axis_on[i].isChecked()
        ]

    def gizmos_shown(self) -> bool:
        box = self._show_gizmos
        if box is None:
            return True
        return bool(box.isChecked())

    def gizmo_state(self):
        from ...util.clip_gizmo import normalize_clip_gizmo_state

        axes = []
        for i in range(3):
            box = self._clip_axis_gizmo[i]
            axes.append(True if box is None else bool(box.isChecked()))
        return normalize_clip_gizmo_state({"shown": self.gizmos_shown(), "axes": axes})

    def set_gizmo_state(self, state) -> None:
        from ...util.clip_gizmo import normalize_clip_gizmo_state

        state = normalize_clip_gizmo_state(state)
        self._spin_suspend = True
        try:
            if self._show_gizmos is not None:
                self._show_gizmos.setChecked(bool(state["shown"]))
            for i, on in enumerate(state["axes"]):
                box = self._clip_axis_gizmo[i]
                if box is not None:
                    box.setChecked(bool(on))
        finally:
            self._spin_suspend = False
        self.sync_widgets()
        self._on_gizmo_changed()

    def axis_gizmo_visible(self, axis: int) -> bool:
        if not self.gizmos_shown():
            return False
        box = self._clip_axis_gizmo[int(axis)]
        if box is None:
            return True
        return bool(box.isChecked())

    def axis_face_count(self, axis: int) -> int:
        if self._clip_axis_on[axis] is None or not self._clip_axis_on[axis].isChecked():
            return 0
        return sum(1 for on in self._clip_face_on[axis] if on)

    def sync_widgets(self) -> None:
        master = bool(self.gizmos_shown())
        for axis in range(3):
            on = bool(self._clip_axis_on[axis].isChecked()) if self._clip_axis_on[axis] is not None else False
            paired = on and self._clip_face_on[axis][1]
            if self._clip_axis_gizmo[axis] is not None:
                self._clip_axis_gizmo[axis].setEnabled(on and master)
            pair = self._clip_axis_pair[axis]
            if pair is not None:
                pair.setEnabled(on)
                pair.setText("−" if paired else "+")
                pair.setToolTip(REMOVE_AXIS_PLANE_TIP if paired else ADD_AXIS_PLANE_TIP)
            extra = self._clip_pair_row[axis]
            if extra is not None:
                extra.setVisible(bool(paired))
            for face in range(2):
                face_on = on and self._clip_face_on[axis][face]
                spin = self._clip_face_pos[axis][face]
                flip = self._clip_face_flip[axis][face]
                if spin is not None:
                    spin.setEnabled(face_on)
                if flip is not None:
                    flip.setEnabled(face_on and not paired)

    def tooltips(self) -> Sequence[Tuple[object, str, str]]:
        tips = [
            (self._show_gizmos, SHOW_CLIP_VISUALS_TIP, "Show clip visuals"),
        ]
        for i, axis in enumerate("XYZ"):
            tips.extend([
                (self._clip_axis_on[i], CLIP_ENABLE_TIPS[i], "Clip %s" % axis),
                (self._clip_axis_gizmo[i], SHOW_PLANE_GIZMO_TIP, "Show clip %s" % axis),
                (self._clip_axis_pair[i], ADD_AXIS_PLANE_TIP, "Add or remove %s clip" % axis),
                (self._clip_face_pos[i][0], CLIP_POS_TIPS[i], "Clip %s position" % axis),
                (self._clip_face_flip[i][0], FLIP_CLIP_TIP, "Flip clip %s" % axis),
                (self._clip_face_pos[i][1], CLIP_POS2_TIPS[i], "Clip %s second position" % axis),
                (self._clip_face_flip[i][1], FLIP_CLIP_TIP, "Flip clip %s extra" % axis),
            ])
        return tips

    def _apply_face(self, axis: int, face: int, plane: dict) -> None:
        self._clip_face_on[axis][face] = True
        self._clip_face_hi[axis][face] = bool(plane["hi"])
        spin = self._clip_face_pos[axis][face]
        if spin is not None:
            spin.setValue(float(plane["position"]))

    def _face_plane(self, axis: int, face: int) -> dict:
        spin = self._clip_face_pos[axis][face]
        position = float(spin.value()) if spin is not None else 0.0
        return {
            "axis": int(axis),
            "position": position,
            "hi": bool(self._clip_face_hi[axis][face]),
        }

    def _symbol_button(self, QtCore, QtWidgets, text: str):
        btn = QtWidgets.QPushButton(text)
        btn.setAutoDefault(False)
        btn.setDefault(False)
        btn.setFixedWidth(22)
        btn.setFixedHeight(22)
        btn.setCursor(QtCore.Qt.PointingHandCursor)
        btn.setFocusPolicy(QtCore.Qt.NoFocus)
        return btn

    def _build(self, parent):
        QtCore, _, QtWidgets = qt_modules()
        axes = QtWidgets.QWidget(parent)
        root = QtWidgets.QVBoxLayout(axes)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)
        self._spin_suspend = True
        try:
            self._show_gizmos = make_switch("Show clip visuals")
            self._show_gizmos.setChecked(True)
            self._show_gizmos.toggled.connect(lambda *_: self._on_gizmo_toggle())
            root.addWidget(self._show_gizmos)
            for i, axis in enumerate("XYZ"):
                root.addWidget(self._build_axis(i, axis, QtCore, QtWidgets))
        finally:
            self._spin_suspend = False
        apply_required_tooltips(self.tooltips(), context=self._context)
        self.sync_widgets()
        self._widget = axes

    def _build_axis(self, axis: int, label: str, QtCore, QtWidgets):
        box = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(box)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)
        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)
        enable = make_switch(label)
        enable.toggled.connect(lambda *_a, ax=axis: self._on_toggle(ax))
        gizmo = make_switch("", compact=True)
        gizmo.setChecked(True)
        gizmo.toggled.connect(lambda *_: self._on_gizmo_toggle())
        spin = self._make_spin(QtCore, QtWidgets, axis, 0)
        flip = QtWidgets.QPushButton("Flip")
        flip.setAutoDefault(False)
        flip.setDefault(False)
        flip.clicked.connect(lambda *_a, ax=axis, face=0: self._on_flip(ax, face))
        pair = self._symbol_button(QtCore, QtWidgets, "+")
        pair.clicked.connect(lambda *_a, ax=axis: self._on_pair(ax))
        top.addWidget(enable)
        top.addWidget(gizmo)
        top.addWidget(spin, stretch=1)
        top.addWidget(flip)
        top.addWidget(pair)
        col.addLayout(top)
        extra = QtWidgets.QWidget()
        extra_row = QtWidgets.QHBoxLayout(extra)
        extra_row.setContentsMargins(28, 0, 0, 0)
        extra_row.setSpacing(8)
        spin2 = self._make_spin(QtCore, QtWidgets, axis, 1)
        flip2 = QtWidgets.QPushButton("Flip")
        flip2.setAutoDefault(False)
        flip2.setDefault(False)
        flip2.clicked.connect(lambda *_a, ax=axis, face=1: self._on_flip(ax, face))
        extra_row.addWidget(spin2, stretch=1)
        extra_row.addWidget(flip2)
        extra.setVisible(False)
        col.addWidget(extra)
        self._clip_axis_on[axis] = enable
        self._clip_axis_gizmo[axis] = gizmo
        self._clip_axis_pair[axis] = pair
        self._clip_pair_row[axis] = extra
        self._clip_face_pos[axis][0] = spin
        self._clip_face_pos[axis][1] = spin2
        self._clip_face_flip[axis][0] = flip
        self._clip_face_flip[axis][1] = flip2
        return box

    def _make_spin(self, QtCore, QtWidgets, axis: int, face: int):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(-1e4, 1e4)
        apply_spin_step(spin, STEP_ANGSTROM, decimals=2)
        apply_ascii_float_locale(spin, QtCore)
        spin.valueChanged.connect(lambda *_a, ax=axis, fa=face: self._on_pos(ax, fa))
        return spin

    def _on_gizmo_toggle(self) -> None:
        if self._spin_suspend:
            return
        self.sync_widgets()
        self._on_gizmo_changed()

    def _on_toggle(self, axis: int) -> None:
        if self._spin_suspend:
            return
        box = self._clip_axis_on[axis]
        on = bool(box.isChecked()) if box is not None else False
        if on:
            seeded = self._seed_plane(int(axis))
            self._spin_suspend = True
            try:
                self._clip_face_on[axis] = [True, False]
                self._apply_face(axis, 0, seeded)
            finally:
                self._spin_suspend = False
            self._on_axis_enabled(int(axis))
        else:
            self._clip_face_on[axis] = [False, False]
            self._on_axis_disabled(int(axis))
        self.sync_widgets()
        self._on_changed()

    def _on_pos(self, axis: int, face: int) -> None:
        if self._spin_suspend:
            return
        if self.any_enabled() and self._clip_face_on[axis][face]:
            self._on_axis_focus(int(axis), bool(self._clip_face_hi[axis][face]))
        self._on_changed()

    def _on_flip(self, axis: int, face: int) -> None:
        if self._spin_suspend:
            return
        box = self._clip_axis_on[axis]
        if box is None or not box.isChecked() or not self._clip_face_on[axis][face]:
            return
        if self._clip_face_on[axis][1]:
            return
        self._clip_face_hi[axis][face] = not bool(self._clip_face_hi[axis][face])
        self._on_axis_focus(int(axis), bool(self._clip_face_hi[axis][face]))
        self._on_axis_flip(int(axis))
        self._on_changed()

    def _on_pair(self, axis: int) -> None:
        if self._spin_suspend:
            return
        box = self._clip_axis_on[axis]
        if box is None or not box.isChecked() or not self._clip_face_on[axis][0]:
            return
        if self._clip_face_on[axis][1]:
            self._clip_face_on[axis][1] = False
            self._on_axis_focus(int(axis), bool(self._clip_face_hi[axis][0]))
            self.sync_widgets()
            self._on_changed()
            return
        existing = self._face_plane(axis, 0)
        seeded = self._seed_opposite(int(axis), existing) or opposite_cardinal_plane(existing, None)
        self._spin_suspend = True
        try:
            self._apply_face(axis, 1, seeded)
        finally:
            self._spin_suspend = False
        self._on_axis_focus(int(axis), bool(self._clip_face_hi[axis][1]))
        self.sync_widgets()
        self._on_changed()
