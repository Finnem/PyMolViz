"""Color helpers for wizard UI and CGO builders."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from ...util.field_sample import DEFAULT_SURFACE_COLORMAP, field_choices
from ..widgets.section import make_section
from ..widgets.theme import apply_shrinking_combo, swatch_button_css
from .colormap_editor import ColormapEditor

RGB = Tuple[float, float, float]
RGBA = Tuple[float, float, float, float]

DEFAULT_SPHERE_COLOR: RGB = (1.0, 0.85, 0.15)
DEFAULT_SURFACE_COLOR: RGB = (0.20, 0.60, 0.90)
DEFAULT_SPHERE_ALPHA = 1.0
RECENT_COLOR_LIMIT = 100
QT_CUSTOM_COLOR_SLOTS = 16


CLIM_MODE_AUTO = "auto"
CLIM_MODE_CUSTOM = "custom"
CLIM_MODE_SYMMETRIC = "symmetric"
CLIM_MODE_PERCENTILE = "percentile"
CLIM_MODES = (CLIM_MODE_AUTO, CLIM_MODE_CUSTOM, CLIM_MODE_SYMMETRIC, CLIM_MODE_PERCENTILE)


@dataclass(frozen=True)
class ColorChoice:
    """Solid RGBA, or a field sampled through a colormap."""

    rgba: RGBA
    field_id: Optional[str] = None
    colormap: str = "RdYlBu_r"
    clims: Optional[Tuple[float, float]] = None
    clim_mode: Optional[str] = None
    colormap_spec: Optional[dict] = None

    @property
    def is_field(self) -> bool:
        return bool(self.field_id)

    @property
    def rgb(self) -> RGB:
        return (float(self.rgba[0]), float(self.rgba[1]), float(self.rgba[2]))


def normalize_clim_mode(mode) -> str:
    text = str(mode or "").strip().lower().replace("-", "_")
    if text in ("custom", "manual"):
        return CLIM_MODE_CUSTOM
    if text in ("symmetric", "sym", "about_zero", "aboutzero"):
        return CLIM_MODE_SYMMETRIC
    if text in ("percentile", "pct"):
        return CLIM_MODE_PERCENTILE
    return CLIM_MODE_AUTO


def effective_clim_mode(choice: ColorChoice) -> str:
    if getattr(choice, "clim_mode", None):
        return normalize_clim_mode(choice.clim_mode)
    return CLIM_MODE_CUSTOM if choice.clims is not None else CLIM_MODE_AUTO


def resolve_color_clims(clim_mode, custom=None, field_id=None, values=None):
    """Auto → None; custom → (lo, hi); symmetric / percentile from field values."""
    from ...fields.isovalues import clim_range

    mode = normalize_clim_mode(clim_mode)
    if mode == CLIM_MODE_CUSTOM:
        resolved = clim_range("custom", values, custom=custom)
        if resolved is None:
            return None
        return (float(resolved[0]), float(resolved[1]))
    if mode == CLIM_MODE_AUTO:
        return None
    if values is None and field_id:
        from ...util.field_sample import resolve_grid_from_session

        grid = resolve_grid_from_session(field_id)
        values = getattr(grid, "values", None) if grid is not None else None
    if mode == CLIM_MODE_PERCENTILE:
        from ...util.colormap_spec import Normalization, RANGE_MODE_PERCENTILE, resolve_limits

        resolved = resolve_limits(Normalization(mode=RANGE_MODE_PERCENTILE), values)
        if resolved is None:
            return None
        return (float(resolved[0]), float(resolved[1]))
    resolved = clim_range("symmetric", values)
    if resolved is None:
        return None
    return (float(resolved[0]), float(resolved[1]))


def as_color_choice(value, default_alpha: float = DEFAULT_SPHERE_ALPHA) -> ColorChoice:
    if isinstance(value, ColorChoice):
        return value
    if value is None:
        rgba = normalize_rgba(DEFAULT_SPHERE_COLOR, default_alpha)
        return ColorChoice(rgba=rgba)
    return ColorChoice(rgba=normalize_rgba(value, default_alpha))


def choice_with_alpha(choice: ColorChoice, alpha: float) -> ColorChoice:
    rgba = normalize_rgba(choice.rgba)
    return ColorChoice(
        rgba=(rgba[0], rgba[1], rgba[2], float(alpha)),
        field_id=choice.field_id,
        colormap=choice.colormap,
        clims=choice.clims,
        clim_mode=getattr(choice, "clim_mode", None),
        colormap_spec=getattr(choice, "colormap_spec", None),
    )


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


def _spec_key(spec) -> str:
    if not spec:
        return ""
    try:
        return json.dumps(spec, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(spec)


def color_choice_signature(choice: ColorChoice):
    """Hashable snapshot used to skip no-op picker emissions."""
    return (
        choice.field_id,
        choice.colormap,
        choice.clims,
        getattr(choice, "clim_mode", None),
        _spec_key(getattr(choice, "colormap_spec", None)),
        _rgba_key(choice.rgba),
    )


def picker_start_rgba(object_rgba: Sequence[float], recent_colors=None) -> RGBA:
    """Opening color is the object being edited, not the last custom pick.

    Recent colors belong in QColorDialog's custom slots. Using the latest
    history entry as ``currentColor`` would paint the live preview before
    the user changes anything.
    """
    return normalize_rgba(object_rgba)


def picker_close_should_accept(*, visible: bool, already_accepted: bool) -> bool:
    """Title-bar close keeps the live preview; Cancel already hid via reject()."""
    return bool(visible) and not already_accepted


def picker_compose_or_last(compose, last):
    """Read the dialog color, or the last previewed choice if Qt already tore it down."""
    try:
        choice = compose()
    except RuntimeError:
        return last
    return last if choice is None else choice


def make_picker_finish(*, accepted, compose, last, remember, on_done):
    """OK / ``colorSelected`` commits and stays committed.

    A later ``finished(Rejected)`` from teardown or a buried dialog must not
    restore the pre-picker colors over that submit.
    """
    submitted = {"value": False}
    reverted = {"value": False}

    def _finish(result):
        if result == accepted:
            if submitted["value"]:
                return
            submitted["value"] = True
            choice = picker_compose_or_last(compose, last["value"])
            last["value"] = choice
            if choice is not None and not getattr(choice, "field_id", None):
                remember(choice.rgba)
            if on_done is not None:
                on_done(choice)
            return
        if submitted["value"] or reverted["value"]:
            return
        reverted["value"] = True
        if on_done is not None:
            on_done(None)

    return _finish


def bind_color_pick_result(apply_choice, restore_originals):
    """Apply OK once; ignore a later cancel so it cannot restore over the submit."""
    submitted = {"value": False}

    def on_done(choice):
        if choice is not None:
            submitted["value"] = True
            apply_choice(choice)
            return
        if submitted["value"]:
            return
        restore_originals()

    return on_done


_ACTIVE_COLOR_DIALOG = {"widget": None}
_COLOR_DIALOG_STACK = []


def live_color_dialog(is_alive, holder=None):
    """The topmost open picker, if its Qt object is still valid."""
    if holder is not None:
        widget = holder.get("widget")
        if widget is None:
            return None
        try:
            alive = True if is_alive is None else bool(is_alive(widget))
        except RuntimeError:
            alive = False
        if not alive:
            holder["widget"] = None
            return None
        return widget
    while _COLOR_DIALOG_STACK:
        widget = _COLOR_DIALOG_STACK[-1]
        try:
            alive = True if is_alive is None else bool(is_alive(widget))
        except RuntimeError:
            alive = False
        if alive:
            _ACTIVE_COLOR_DIALOG["widget"] = widget
            return widget
        _COLOR_DIALOG_STACK.pop()
    _ACTIVE_COLOR_DIALOG["widget"] = None
    return None


def remember_color_dialog(widget, holder=None):
    if holder is not None:
        holder["widget"] = widget
        return
    if widget not in _COLOR_DIALOG_STACK:
        _COLOR_DIALOG_STACK.append(widget)
    _ACTIVE_COLOR_DIALOG["widget"] = widget


def forget_color_dialog(widget, holder=None):
    if holder is not None:
        if holder.get("widget") is widget:
            holder["widget"] = None
        return
    try:
        _COLOR_DIALOG_STACK.remove(widget)
    except ValueError:
        pass
    if _ACTIVE_COLOR_DIALOG.get("widget") is widget:
        _ACTIVE_COLOR_DIALOG["widget"] = _COLOR_DIALOG_STACK[-1] if _COLOR_DIALOG_STACK else None


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
    btn.setStyleSheet(swatch_button_css(rgba_to_css(rgba)))
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


def picker_color_choice(
    rgba: Sequence[float],
    field_id: Optional[str] = None,
    colormap: Optional[str] = None,
    clims=None,
    clim_mode=None,
    colormap_spec=None,
) -> ColorChoice:
    """ColorChoice emitted by the picker: solid RGB or sample-from-field."""
    normalized = normalize_rgba(rgba)
    if field_id:
        if clim_mode:
            mode = normalize_clim_mode(clim_mode)
        elif clims is not None:
            mode = CLIM_MODE_CUSTOM
        else:
            mode = CLIM_MODE_AUTO
        return ColorChoice(
            rgba=normalized,
            field_id=str(field_id),
            colormap=str(colormap or DEFAULT_SURFACE_COLORMAP),
            clims=clims,
            clim_mode=mode,
            colormap_spec=colormap_spec,
        )
    return ColorChoice(rgba=normalized)


def _mount_field_picker(layout, QtWidgets, cmd, initial: ColorChoice, state: dict, notify):
    """Compact Color-by-field controls inside the color dialog."""
    section = make_section("Color by field", form=True)
    form = section.layout
    field_combo = QtWidgets.QComboBox()
    apply_shrinking_combo(field_combo, QtWidgets, min_chars=10)

    def _selected_field_id():
        fid = field_combo.currentData()
        return str(fid) if fid else None

    editor = ColormapEditor(
        section.widget,
        on_changed=lambda: None,
        cmd=cmd,
        field_id_provider=_selected_field_id,
    )
    editor.set_colormap(
        initial.colormap or DEFAULT_SURFACE_COLORMAP,
        range_mode=effective_clim_mode(initial),
        clims=initial.clims,
        spec=getattr(initial, "colormap_spec", None),
    )
    field_combo.addItem("Solid color", None)
    entries = field_choices(cmd=cmd)
    for fid, label in entries:
        field_combo.addItem(label, fid)
    if not entries:
        field_combo.setItemText(0, "Solid color (no fields)")
    if initial.field_id:
        fidx = field_combo.findData(initial.field_id)
        if fidx >= 0:
            field_combo.setCurrentIndex(fidx)

    def _sync_state(emit=True):
        fid = field_combo.currentData()
        state["field_id"] = str(fid) if fid else None
        state["colormap"] = editor.colormap_name()
        state["colormap_spec"] = editor.colormap_spec()
        mode = editor.range_mode()
        state["clim_mode"] = mode
        enabled = bool(fid)
        editor.widget.setEnabled(enabled)
        if not fid:
            state["clims"] = None
        elif mode == CLIM_MODE_CUSTOM:
            state["clims"] = editor.custom_clims()
        elif mode == CLIM_MODE_SYMMETRIC:
            state["clims"] = resolve_color_clims(CLIM_MODE_SYMMETRIC, field_id=str(fid))
        elif mode == CLIM_MODE_PERCENTILE:
            state["clims"] = resolve_color_clims(CLIM_MODE_PERCENTILE, field_id=str(fid))
        else:
            state["clims"] = None
        if emit:
            notify()

    editor._on_changed = lambda: _sync_state()
    field_combo.currentIndexChanged.connect(lambda *_: _sync_state())
    form.addRow("Field", field_combo)
    form.addRow(editor.widget)
    layout.addWidget(section.widget)
    _sync_state(emit=False)


def _dialog_parent(obj):
    """QWidget suitable as a QColorDialog parent, or None.

    QColorDialog requires a QWidget. Callers sometimes pass a QVBoxLayout
    (the page layout). Layouts expose the owning widget via parentWidget().
    """
    if obj is None:
        return None
    is_widget_type = getattr(obj, "isWidgetType", None)
    if callable(is_widget_type):
        try:
            if is_widget_type():
                return obj
        except Exception:
            pass
    getter = getattr(obj, "parentWidget", None)
    if not callable(getter):
        return None
    try:
        widget = getter()
    except Exception:
        return None
    if widget is None:
        return None
    nested = getattr(widget, "isWidgetType", None)
    if callable(nested):
        try:
            if nested():
                return widget
        except Exception:
            return None
        return None
    return widget


def _configure_color_dialog_window(widget, anchor=None):
    """Stay above PyMOL without Qt.Tool (tool windows hide when the viewer focuses).

    ``anchor`` is the window this picker is transient for. Use the colormap
    editor (not the PyMOL viewer) when picking a stop color so X11 does not
    unmap the editor as the previous transient-for-PyMOL dialog.
    """
    from ..pick import find_pymol_window, qt_modules

    QtCore, _, QtWidgets = qt_modules()
    if QtCore is None:
        return
    if anchor is None:
        anchor = find_pymol_window(QtWidgets)
    widget._pmv_window_anchor = anchor
    widget._pmv_raise_last = True
    widget.setWindowFlags(
        widget.windowFlags() | QtCore.Qt.WindowStaysOnTopHint
    )
    non_modal = getattr(getattr(QtCore, "Qt", None), "NonModal", None)
    if non_modal is not None:
        try:
            widget.setWindowModality(non_modal)
        except Exception:
            pass


def pick_rgb(
    parent,
    initial: Sequence[float] = DEFAULT_SPHERE_COLOR,
    on_change: Optional[Callable[[ColorChoice], None]] = None,
    on_done: Optional[Callable[[Optional[ColorChoice]], None]] = None,
    cmd=None,
    allow_field: bool = True,
) -> bool:
    """Open a non-modal color dialog: solid RGB/alpha, or Color by field.

    Field sampling is a picker strategy stored on each point, not an
    Appearance color mode. ``cmd`` is used to list sampleable fields.
    ``allow_field=False`` is required when picking a colormap stop (or NaN /
    out-of-range color): a nested colormap editor recurses and can crash.
    Pickers opened from the colormap editor stay transient to that editor so
    the editor is not closed or unmapped. A stop-color picker stacks on top of
    an already-open sphere "Choose color" dialog instead of replacing it.

    Only one *sphere* picker is reused at a time. Nested colormap stop pickers
    are additional dialogs. Live preview applies as the user picks; the
    dialog is raised again so PyMOL cannot bury it. OK and the window close
    button keep that color; Cancel restores the colors from before open.
    A later cancel after OK is ignored.
    """
    from ..pick import bind_tool_window, overlay_window, qt_modules, qt_widget_alive

    QtCore, QtGui, QtWidgets = qt_modules()
    if QtWidgets is None:
        if on_done is not None:
            on_done(None)
        return False
    host = _dialog_parent(parent)
    # StayOnTop dialogs (colormap editor / stop popup) must own the picker as
    # a transient, not as a nested QDialog under the Fields overlay. Nesting
    # under Fields / PyMOL unmaps the editor; nesting QDialog-in-QDialog
    # can reject the editor when the picker closes.
    if getattr(host, "_pmv_raise_last", False):
        parent = None
        window_anchor = host
    else:
        parent = overlay_window(host)
        window_anchor = None

    allow_field = bool(allow_field)
    existing = live_color_dialog(qt_widget_alive)
    # Nested stop-color pickers must not close the sphere Choose color dialog
    # (that dialog hosts the compact colormap row that opened this editor).
    if existing is not None and window_anchor is None:
        same_mode = getattr(existing, "_pmv_allow_field", True) == allow_field
        if same_mode:
            try:
                existing.show()
                existing.raise_()
                existing.activateWindow()
                return True
            except RuntimeError:
                forget_color_dialog(existing)
        else:
            try:
                existing.close()
            except Exception:
                pass
            forget_color_dialog(existing)

    initial_choice = as_color_choice(initial)
    if not allow_field:
        initial_choice = ColorChoice(rgba=normalize_rgba(initial_choice.rgba))
    history = RecentColorHistory.shared()
    object_rgba = normalize_rgba(initial_choice.rgba)
    recent_colors = history.colors()
    # Fill custom slots before constructing the dialog so Qt does not treat
    # setCustomColor as a current-color change on an already-open picker.
    _apply_qt_custom_colors(QtGui, QtWidgets, recent_colors)
    start_rgba = picker_start_rgba(object_rgba, recent_colors)
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
    dialog._pmv_allow_field = allow_field
    _configure_color_dialog_window(dialog, anchor=window_anchor)
    # Avoid setTransientParent + X11 EWMH pinning in bind_tool_window(); that path
    # segfaults immediately on some Unix sessions when opening Appearance pickers.
    dialog._pmv_no_transient = True

    syncing = {"active": False}
    ready = {"value": False}
    opacity_sync = {"set_pct": None}
    opacity = {"value": float(start_rgba[3])}
    field_state = {
        "field_id": initial_choice.field_id,
        "colormap": initial_choice.colormap or DEFAULT_SURFACE_COLORMAP,
        "clims": initial_choice.clims,
        "clim_mode": effective_clim_mode(initial_choice) if initial_choice.field_id else CLIM_MODE_AUTO,
        "colormap_spec": getattr(initial_choice, "colormap_spec", None),
    }
    last_emitted = {"sig": color_choice_signature(initial_choice)}
    last_choice = {"value": initial_choice}

    def _composed_rgba() -> RGBA:
        color = dialog.currentColor()
        if color.isValid():
            r, g, b = qcolor_to_rgb(color)
        else:
            r, g, b = object_rgba[:3]
        return (float(r), float(g), float(b), float(opacity["value"]))

    def _composed_choice() -> ColorChoice:
        if not allow_field:
            return picker_color_choice(_composed_rgba())
        return picker_color_choice(
            _composed_rgba(),
            field_id=field_state.get("field_id"),
            colormap=field_state.get("colormap"),
            clims=field_state.get("clims"),
            clim_mode=field_state.get("clim_mode"),
            colormap_spec=field_state.get("colormap_spec"),
        )

    def _keep_front():
        try:
            if dialog.isVisible():
                dialog.raise_()
                dialog.activateWindow()
        except RuntimeError:
            pass

    def _notify():
        choice = _composed_choice()
        last_choice["value"] = choice
        if on_change is not None:
            sig = color_choice_signature(choice)
            if sig != last_emitted["sig"]:
                last_emitted["sig"] = sig
                on_change(choice)
        _keep_front()

    def _restore_current():
        syncing["active"] = True
        try:
            dialog.setCurrentColor(start)
        finally:
            syncing["active"] = False

    def apply_color(rgba: RGBA):
        rgba = normalize_rgba(rgba)
        opacity["value"] = float(rgba[3])
        syncing["active"] = True
        try:
            dialog.setCurrentColor(_rgba_to_qcolor(QtGui, rgba))
            set_pct = opacity_sync["set_pct"]
            if set_pct is not None:
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
        current_section = make_section("Current color")
        current_section.layout.addWidget(
            _color_reference_row(QtWidgets, "Object", object_rgba, apply_color)
        )
        layout.addWidget(current_section.widget)

        opacity_section = make_section("Transparency")
        opacity_row, set_opacity_pct = _opacity_controls(
            QtCore,
            QtWidgets,
            start_rgba[3],
            _apply_alpha,
        )
        opacity_section.layout.addWidget(opacity_row)
        layout.addWidget(opacity_section.widget)
        opacity_sync["set_pct"] = set_opacity_pct
        if allow_field:
            _mount_field_picker(layout, QtWidgets, cmd, initial_choice, field_state, _notify)

    def _emit(color):
        if not ready["value"] or syncing["active"] or not color.isValid():
            return
        _notify()

    dialog.currentColorChanged.connect(_emit)

    def _on_done(choice):
        forget_color_dialog(dialog)
        if on_done is not None:
            on_done(choice)

    _finish = make_picker_finish(
        accepted=QtWidgets.QDialog.Accepted,
        compose=_composed_choice,
        last=last_choice,
        remember=history.remember,
        on_done=_on_done,
    )
    # OK must commit even if destroy later emits finished(Rejected).
    dialog.colorSelected.connect(lambda *_: _finish(QtWidgets.QDialog.Accepted))
    dialog.accepted.connect(lambda: _finish(QtWidgets.QDialog.Accepted))
    dialog.finished.connect(_finish)
    dialog.destroyed.connect(lambda *_: forget_color_dialog(dialog))
    remember_color_dialog(dialog)

    class _CommitOnCloseFilter(QtCore.QObject):
        def eventFilter(inner, obj, event):
            close_type = getattr(QtCore.QEvent, "Close", None)
            if obj is not dialog or event.type() != close_type:
                return False
            try:
                visible = dialog.isVisible()
                already_accepted = dialog.result() == QtWidgets.QDialog.Accepted
            except RuntimeError:
                return False
            if picker_close_should_accept(
                visible=visible, already_accepted=already_accepted
            ):
                dialog.accept()
            return False

    close_filter = _CommitOnCloseFilter(dialog)
    dialog.installEventFilter(close_filter)
    dialog._pmv_close_filter = close_filter
    dialog.show()

    def _bind_tool_window_when_mapped():
        if qt_widget_alive(dialog):
            bind_tool_window(dialog)

    QtCore.QTimer.singleShot(0, _bind_tool_window_when_mapped)
    dialog.raise_()
    dialog.activateWindow()
    _restore_current()
    ready["value"] = True
    return True


def field_combo_colormap(combo) -> str:
    data = combo.currentData()
    if data:
        return str(data)
    text = combo.currentText()
    return str(text) if text else "RdYlBu_r"
