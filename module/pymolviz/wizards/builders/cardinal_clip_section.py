"""X/Y/Z cardinal clip plane row for field visual Modifiers."""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

from ...fields.clip import default_cardinal_plane, normalize_cardinal_planes
from ..pick import qt_modules
from ..tooltips import FLIP_CLIP_TIP
from ..widgets.ascii_locale import apply_ascii_float_locale
from ..widgets.switch import make_switch

CLIP_TIP = (
    "Clip the field with planes perpendicular to X, Y, and Z. Enable each "
    "axis separately. Drag that plane along its axis, or edit the position. "
    "Flip chooses which half to keep."
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


class CardinalClipSection:
    """Three axis switches with position spins and flip buttons."""

    def __init__(
        self,
        parent,
        *,
        on_changed: Optional[Callable[[], None]] = None,
        seed_plane: Optional[Callable[[int], dict]] = None,
        on_axis_enabled: Optional[Callable[[int], None]] = None,
        on_axis_disabled: Optional[Callable[[int], None]] = None,
        on_axis_focus: Optional[Callable[[int], None]] = None,
        on_axis_flip: Optional[Callable[[int], None]] = None,
    ):
        self._on_changed = on_changed or (lambda: None)
        self._seed_plane = seed_plane or (lambda _axis: default_cardinal_plane(0, None))
        self._on_axis_enabled = on_axis_enabled or (lambda _axis: None)
        self._on_axis_disabled = on_axis_disabled or (lambda _axis: None)
        self._on_axis_focus = on_axis_focus or (lambda _axis: None)
        self._on_axis_flip = on_axis_flip or (lambda _axis: None)
        self._clip_axis_on = [None, None, None]
        self._clip_axis_pos = [None, None, None]
        self._clip_axis_flip = [None, None, None]
        self._clip_axis_hi = [False, False, False]
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
            spin = self._clip_axis_pos[axis]
            position = float(spin.value()) if spin is not None else 0.0
            planes.append({
                "axis": axis,
                "position": position,
                "hi": bool(self._clip_axis_hi[axis]),
            })
        return normalize_cardinal_planes(planes)

    def set_planes(self, planes) -> None:
        by_axis = {int(p["axis"]): p for p in normalize_cardinal_planes(planes)}
        self._spin_suspend = True
        try:
            for axis in range(3):
                plane = by_axis.get(axis)
                on = plane is not None
                if self._clip_axis_on[axis] is not None:
                    self._clip_axis_on[axis].setChecked(on)
                if plane is None:
                    continue
                self._clip_axis_hi[axis] = bool(plane["hi"])
                if self._clip_axis_pos[axis] is not None:
                    self._clip_axis_pos[axis].setValue(float(plane["position"]))
        finally:
            self._spin_suspend = False
        self.sync_widgets()

    def reset(self) -> None:
        self._spin_suspend = True
        try:
            self._clip_axis_hi = [False, False, False]
            for axis in range(3):
                if self._clip_axis_on[axis] is not None:
                    self._clip_axis_on[axis].setChecked(False)
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

    def sync_widgets(self) -> None:
        for axis in range(3):
            on = bool(self._clip_axis_on[axis].isChecked()) if self._clip_axis_on[axis] is not None else False
            if self._clip_axis_pos[axis] is not None:
                self._clip_axis_pos[axis].setEnabled(on)
            if self._clip_axis_flip[axis] is not None:
                self._clip_axis_flip[axis].setEnabled(on)

    def tooltips(self) -> Sequence[Tuple[object, str, str]]:
        tips = []
        for i, axis in enumerate("XYZ"):
            tips.extend([
                (self._clip_axis_on[i], CLIP_ENABLE_TIPS[i], "Clip %s" % axis),
                (self._clip_axis_pos[i], CLIP_POS_TIPS[i], "Clip %s position" % axis),
                (self._clip_axis_flip[i], FLIP_CLIP_TIP, "Flip clip %s" % axis),
            ])
        return tips

    def _build(self, parent):
        QtCore, _, QtWidgets = qt_modules()
        axes = QtWidgets.QWidget(parent)
        grid = QtWidgets.QGridLayout(axes)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        for i, axis in enumerate("XYZ"):
            enable = make_switch(axis)
            enable.toggled.connect(lambda *_a, ax=i: self._on_toggle(ax))
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(2)
            spin.setRange(-1e4, 1e4)
            spin.setSingleStep(0.5)
            apply_ascii_float_locale(spin, QtCore)
            spin.valueChanged.connect(lambda *_a, ax=i: self._on_pos(ax))
            flip = QtWidgets.QPushButton("Flip")
            flip.setAutoDefault(False)
            flip.setDefault(False)
            flip.clicked.connect(lambda *_a, ax=i: self._on_flip(ax))
            grid.addWidget(enable, i, 0)
            grid.addWidget(spin, i, 1)
            grid.addWidget(flip, i, 2)
            self._clip_axis_on[i] = enable
            self._clip_axis_pos[i] = spin
            self._clip_axis_flip[i] = flip
        self._widget = axes

    def _on_toggle(self, axis: int) -> None:
        if self._spin_suspend:
            return
        box = self._clip_axis_on[axis]
        on = bool(box.isChecked()) if box is not None else False
        if on:
            seeded = self._seed_plane(int(axis))
            self._spin_suspend = True
            try:
                self._clip_axis_hi[axis] = bool(seeded["hi"])
                if self._clip_axis_pos[axis] is not None:
                    self._clip_axis_pos[axis].setValue(float(seeded["position"]))
            finally:
                self._spin_suspend = False
            self._on_axis_enabled(int(axis))
        else:
            self._on_axis_disabled(int(axis))
        self.sync_widgets()
        self._on_changed()

    def _on_pos(self, axis: int) -> None:
        if self._spin_suspend:
            return
        if self.any_enabled():
            self._on_axis_focus(int(axis))
        self._on_changed()

    def _on_flip(self, axis: int) -> None:
        if self._spin_suspend:
            return
        box = self._clip_axis_on[axis]
        if box is None or not box.isChecked():
            return
        self._clip_axis_hi[axis] = not bool(self._clip_axis_hi[axis])
        self._on_axis_flip(int(axis))
        self._on_changed()
