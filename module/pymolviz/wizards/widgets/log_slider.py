"""Piecewise-linear radius control: [0,1], [1,10], [10,100]."""

from ..pick import qt_modules

SLIDER_MAX = 300
SEGMENTS = (
    (0.0, 1.0, 0, 100),
    (1.0, 10.0, 100, 200),
    (10.0, 100.0, 200, 300),
)


def slider_to_value(slider_pos: int) -> float:
    pos = max(0, min(SLIDER_MAX, int(slider_pos)))
    for lo, hi, s0, s1 in SEGMENTS:
        if pos <= s1:
            if s1 == s0:
                return lo
            t = (pos - s0) / float(s1 - s0)
            return lo + t * (hi - lo)
    return 100.0


def value_to_slider(value: float) -> int:
    val = max(0.0, min(100.0, float(value)))
    for lo, hi, s0, s1 in SEGMENTS:
        if val <= hi or hi == 100.0:
            if hi == lo:
                return s0
            t = (val - lo) / (hi - lo)
            return int(round(s0 + t * (s1 - s0)))
    return SLIDER_MAX


class LogSegmentRadiusWidget:
    """Horizontal slider + double spinbox for radius 0–100."""

    valueChanged = None

    def __init__(self, parent=None, initial=1.0):
        QtCore, _, QtWidgets = qt_modules()
        if QtWidgets is None:
            raise RuntimeError("PyMOL Qt UI required")
        self._widget = QtWidgets.QWidget(parent)
        self._blocking = False
        layout = QtWidgets.QHBoxLayout(self._widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(SLIDER_MAX)
        self._spin = QtWidgets.QDoubleSpinBox()
        self._spin.setDecimals(3)
        self._spin.setMinimum(0.0)
        self._spin.setMaximum(100.0)
        self._spin.setSingleStep(0.1)
        layout.addWidget(self._slider, stretch=3)
        layout.addWidget(self._spin, stretch=1)
        self._slider.valueChanged.connect(self._on_slider)
        self._spin.valueChanged.connect(self._on_spin)
        self.set_value(initial)

    @property
    def widget(self):
        return self._widget

    def value(self) -> float:
        return float(self._spin.value())

    def set_value(self, value: float):
        self._blocking = True
        try:
            self._spin.setValue(float(value))
            self._slider.setValue(value_to_slider(value))
        finally:
            self._blocking = False

    def setToolTip(self, text):
        text = str(text or "")
        self._widget.setToolTip(text)
        self._slider.setToolTip(text)
        self._spin.setToolTip(text)

    def toolTip(self):
        return self._widget.toolTip()

    def connect_changed(self, callback):
        self._slider.valueChanged.connect(lambda *_: callback())
        self._spin.valueChanged.connect(lambda *_: callback())

    def _on_slider(self, pos):
        if self._blocking:
            return
        self._blocking = True
        try:
            val = slider_to_value(pos)
            self._spin.setValue(val)
        finally:
            self._blocking = False

    def _on_spin(self, val):
        if self._blocking:
            return
        self._blocking = True
        try:
            self._slider.setValue(value_to_slider(val))
        finally:
            self._blocking = False
