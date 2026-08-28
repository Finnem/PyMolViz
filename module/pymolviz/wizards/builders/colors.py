"""Color helpers for wizard UI and CGO builders."""

from __future__ import annotations

import json
import os
from typing import Callable, List, Optional, Sequence, Tuple

RGB = Tuple[float, float, float]
RGBA = Tuple[float, float, float, float]

DEFAULT_SPHERE_COLOR: RGB = (1.0, 0.85, 0.15)
DEFAULT_SPHERE_ALPHA = 1.0
RECENT_COLOR_LIMIT = 100
QT_CUSTOM_COLOR_SLOTS = 16


def readable_text_color(rgb: RGB) -> RGB:
    """Black or white text for contrast on the given background."""
    r, g, b = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return (0.0, 0.0, 0.0) if lum > 0.55 else (1.0, 1.0, 1.0)


def distinct_color(index: int, n_hint: int = 20) -> RGB:
    """Nth color from the library distinct palette."""
    from ...util.colors import get_distinct_colors

    palette = get_distinct_colors(max(int(n_hint), index + 1))
    c = palette[index % len(palette)]
    return (float(c[0]), float(c[1]), float(c[2]))


def colors_for_new_points(count: int, start_index: int = 0) -> list:
    from ...util.colors import get_distinct_colors

    n = max(count + start_index, 20)
    palette = get_distinct_colors(n)
    return [
        (float(palette[(start_index + i) % len(palette)][0]),
         float(palette[(start_index + i) % len(palette)][1]),
         float(palette[(start_index + i) % len(palette)][2]))
        for i in range(count)
    ]


def qcolor_to_rgb(color) -> RGB:
    return (color.red() / 255.0, color.green() / 255.0, color.blue() / 255.0)


def qcolor_to_rgba(color) -> RGBA:
    return (
        color.red() / 255.0,
        color.green() / 255.0,
        color.blue() / 255.0,
        color.alpha() / 255.0,
    )


def normalize_rgba(value: Sequence[float], default_alpha: float = DEFAULT_SPHERE_ALPHA) -> RGBA:
    if len(value) >= 4:
        return (
            float(value[0]),
            float(value[1]),
            float(value[2]),
            float(value[3]),
        )
    if len(value) == 3:
        return (float(value[0]), float(value[1]), float(value[2]), float(default_alpha))
    raise ValueError("Expected RGB or RGBA sequence")


def rgba_to_css(rgba: RGBA) -> str:
    r, g, b, a = rgba
    return "rgba(%d, %d, %d, %.3f)" % (int(r * 255), int(g * 255), int(b * 255), float(a))


def rgb_to_css(rgb: RGB) -> str:
    r, g, b = rgb
    return "rgb(%d, %d, %d)" % (int(r * 255), int(g * 255), int(b * 255))


def rgb_to_hex(rgb: RGB) -> str:
    r, g, b = rgb
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def rgba_to_hex(rgba: RGBA) -> str:
    r, g, b, a = rgba
    return "#%02x%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255), int(a * 255))


def hex_to_rgba(value: str) -> Optional[RGBA]:
    text = value.strip()
    if not text.startswith("#"):
        text = "#" + text
    if len(text) not in (7, 9):
        return None
    try:
        r = int(text[1:3], 16) / 255.0
        g = int(text[3:5], 16) / 255.0
        b = int(text[5:7], 16) / 255.0
        a = int(text[7:9], 16) / 255.0 if len(text) == 9 else 1.0
    except ValueError:
        return None
    return (r, g, b, a)


def hex_to_rgb(value: str) -> Optional[RGB]:
    rgba = hex_to_rgba(value)
    if rgba is None:
        return None
    return (rgba[0], rgba[1], rgba[2])


def _rgba_key(rgba: RGBA) -> Tuple[int, int, int, int]:
    return (
        int(rgba[0] * 255),
        int(rgba[1] * 255),
        int(rgba[2] * 255),
        int(rgba[3] * 255),
    )


class RecentColorHistory:
    """Persistently stores the most recently confirmed manual color picks."""

    _shared = None

    @classmethod
    def shared(cls) -> "RecentColorHistory":
        if cls._shared is None:
            cls._shared = cls()
        return cls._shared

    def __init__(self):
        self._colors: List[RGBA] = []
        self._load()

    @staticmethod
    def _config_path() -> str:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        if xdg:
            base = os.path.join(xdg, "pymolviz")
        else:
            base = os.path.join(os.path.expanduser("~"), ".config", "pymolviz")
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, "recent_colors.json")

    def colors(self) -> List[RGBA]:
        return list(self._colors)

    def remember(self, rgba: RGBA) -> None:
        key = _rgba_key(rgba)
        self._colors = [c for c in self._colors if _rgba_key(c) != key]
        self._colors.insert(0, normalize_rgba(rgba))
        self._colors = self._colors[:RECENT_COLOR_LIMIT]
        self._save()

    def _load(self) -> None:
        path = self._config_path()
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            self._colors = []
            return
        entries = payload.get("recent_colors", [])
        colors = []
        seen = set()
        for entry in entries:
            rgba = hex_to_rgba(entry) if isinstance(entry, str) else None
            if rgba is None and isinstance(entry, (list, tuple)) and len(entry) in (3, 4):
                rgba = normalize_rgba(entry)
            if rgba is None:
                continue
            key = _rgba_key(rgba)
            if key in seen:
                continue
            seen.add(key)
            colors.append(rgba)
            if len(colors) >= RECENT_COLOR_LIMIT:
                break
        self._colors = colors

    def _save(self) -> None:
        path = self._config_path()
        payload = {"recent_colors": [rgba_to_hex(c) for c in self._colors]}
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
        except OSError:
            pass


def _rgba_to_qcolor(QtGui, rgba: RGBA):
    r, g, b, a = normalize_rgba(rgba)
    return QtGui.QColor(int(r * 255), int(g * 255), int(b * 255), int(a * 255))


def _apply_qt_custom_colors(QtGui, QtWidgets, colors: Sequence[RGBA]) -> None:
    """Fill QColorDialog's custom-color grid from recent manual picks."""
    for index in range(QT_CUSTOM_COLOR_SLOTS):
        if index < len(colors):
            QtWidgets.QColorDialog.setCustomColor(
                index, _rgba_to_qcolor(QtGui, colors[index])
            )


def _color_swatch_button(QtWidgets, rgba: RGBA, tooltip: str, on_pick):
    btn = QtWidgets.QPushButton()
    btn.setFixedSize(18, 18)
    btn.setFlat(True)
    btn.setToolTip(tooltip)
    btn.setStyleSheet(
        "QPushButton { background-color: %s; border: 1px solid #666; }"
        "QPushButton:hover { border: 1px solid #222; }"
        % rgba_to_css(rgba)
    )
    btn.clicked.connect(lambda checked=False, value=rgba: on_pick(value))
    return btn


def _color_reference_row(QtWidgets, label: str, rgba: RGBA, on_pick):
    row = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(QtWidgets.QLabel(label))
    layout.addWidget(_color_swatch_button(QtWidgets, rgba, rgba_to_hex(rgba), on_pick))
    layout.addWidget(QtWidgets.QLabel(rgba_to_hex(rgba)))
    layout.addStretch(1)
    return row


def _opacity_controls(QtCore, QtWidgets, initial_alpha: float, on_alpha_changed):
    """Always-visible opacity slider (PyMOL Qt often hides QColorDialog alpha)."""
    row = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(QtWidgets.QLabel("Opacity"))
    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider.setRange(0, 100)
    pct = int(round(max(0.0, min(1.0, float(initial_alpha))) * 100))
    slider.setValue(pct)
    spin = QtWidgets.QSpinBox()
    spin.setRange(0, 100)
    spin.setSuffix(" %")
    spin.setValue(pct)

    def _set_pct(value: int):
        pct = max(0, min(100, int(value)))
        if slider.value() != pct:
            slider.blockSignals(True)
            slider.setValue(pct)
            slider.blockSignals(False)
        if spin.value() != pct:
            spin.blockSignals(True)
            spin.setValue(pct)
            spin.blockSignals(False)
        on_alpha_changed(pct / 100.0)

    slider.valueChanged.connect(_set_pct)
    spin.valueChanged.connect(_set_pct)
    layout.addWidget(slider, stretch=1)
    layout.addWidget(spin)
    return row, _set_pct


def pick_rgb(
    parent,
    initial: Sequence[float] = DEFAULT_SPHERE_COLOR,
    on_change: Optional[Callable[[RGBA], None]] = None,
    on_done: Optional[Callable[[Optional[RGBA]], None]] = None,
) -> bool:
    """Open a non-modal Qt color dialog with alpha.

    ``initial`` is the object's current RGBA (shown for reference). The dialog
    opens on the last manually confirmed pick when available.

    ``on_change`` is called while the dialog is open whenever the current color
    changes (for live preview). ``on_done`` is called when the dialog closes;
    it receives the chosen RGBA, or None if cancelled.

    Confirmed picks are stored in a persistent recent-color history (100 max).
    The dialog's custom-color grid is filled automatically from that history.

    Returns True if the dialog was shown.
    """
    from ..pick import bind_tool_window, configure_tool_window, qt_modules

    QtCore, QtGui, QtWidgets = qt_modules()
    if QtWidgets is None:
        if on_done is not None:
            on_done(None)
        return False

    history = RecentColorHistory.shared()
    object_rgba = normalize_rgba(initial)
    recent_colors = history.colors()
    if recent_colors:
        last = recent_colors[0]
        start_rgba = (last[0], last[1], last[2], object_rgba[3])
    else:
        start_rgba = object_rgba
    start = _rgba_to_qcolor(QtGui, start_rgba)

    dialog = QtWidgets.QColorDialog(start, parent)
    dialog.setWindowTitle("Choose color")
    dialog.setOption(QtWidgets.QColorDialog.DontUseNativeDialog, True)
    dialog.setOption(QtWidgets.QColorDialog.ShowAlphaChannel, True)
    try:
        dialog.setOptions(
            QtWidgets.QColorDialog.DontUseNativeDialog
            | QtWidgets.QColorDialog.ShowAlphaChannel
        )
    except Exception:
        pass
    dialog.setModal(False)
    dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
    configure_tool_window(dialog)
    _apply_qt_custom_colors(QtGui, QtWidgets, recent_colors)

    syncing = {"active": False}
    opacity_sync = {"set_pct": None}
    opacity = {"value": float(start_rgba[3])}

    def _composed_rgba() -> RGBA:
        color = dialog.currentColor()
        if color.isValid():
            r, g, b = qcolor_to_rgb(color)
        else:
            r, g, b = object_rgba[:3]
        return (float(r), float(g), float(b), float(opacity["value"]))

    def _notify():
        if on_change is not None:
            on_change(_composed_rgba())

    def apply_color(rgba: RGBA):
        rgba = normalize_rgba(rgba)
        opacity["value"] = float(rgba[3])
        dialog.setCurrentColor(_rgba_to_qcolor(QtGui, rgba))
        set_pct = opacity_sync["set_pct"]
        if set_pct is not None:
            syncing["active"] = True
            try:
                set_pct(int(round(rgba[3] * 100)))
            finally:
                syncing["active"] = False
        _notify()

    def _apply_alpha(alpha: float):
        if syncing["active"]:
            return
        opacity["value"] = max(0.0, min(1.0, float(alpha)))
        _notify()

    layout = dialog.layout()
    if layout is not None:
        current_box = QtWidgets.QGroupBox("Current color")
        current_layout = QtWidgets.QVBoxLayout(current_box)
        current_layout.addWidget(
            _color_reference_row(QtWidgets, "Object", object_rgba, apply_color)
        )
        layout.addWidget(current_box)

        opacity_box = QtWidgets.QGroupBox("Transparency")
        opacity_layout = QtWidgets.QVBoxLayout(opacity_box)
        opacity_row, set_opacity_pct = _opacity_controls(
            QtCore,
            QtWidgets,
            start_rgba[3],
            _apply_alpha,
        )
        opacity_layout.addWidget(opacity_row)
        layout.addWidget(opacity_box)
        opacity_sync["set_pct"] = set_opacity_pct

    if on_change is not None:
        def _emit(color):
            if syncing["active"] or not color.isValid():
                return
            _notify()

        dialog.currentColorChanged.connect(_emit)
        _notify()

    def _finish(result):
        if result == QtWidgets.QDialog.Accepted:
            rgba = _composed_rgba()
            history.remember(rgba)
            if on_done is not None:
                on_done(rgba)
            return
        if on_done is not None:
            on_done(None)

    dialog.finished.connect(_finish)
    dialog.show()
    bind_tool_window(dialog)
    dialog.raise_()
    dialog.activateWindow()
    return True
