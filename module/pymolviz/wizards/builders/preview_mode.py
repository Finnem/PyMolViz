"""Viewer preview fidelity: off, cheap stand-in, or the committed look."""

from __future__ import annotations

from typing import Optional

PREVIEW_OFF = "off"
PREVIEW_SIMPLE = "simple"
PREVIEW_FULL = "full"
PREVIEW_MODES = (PREVIEW_OFF, PREVIEW_SIMPLE, PREVIEW_FULL)

PREVIEW_OFF_LABEL = "No preview"
PREVIEW_SIMPLE_LABEL = "Simple preview"
PREVIEW_FULL_LABEL = "Full preview"

PREVIEW_OFF_TIP = "Hide the live viewer object. Clip gizmos and the domain box can still show."
PREVIEW_SIMPLE_TIP = (
    "Faster stand-in: coarser field sampling, lower mesh quality, and IsoMesh "
    "instead of a filled isosurface. Done still builds the full object."
)
PREVIEW_FULL_TIP = "Show the same mesh or native map that Done will create."

DEFAULT_PREVIEW_MODE = PREVIEW_SIMPLE

# 2× spacing ⇒ about 8× fewer voxels for generated fields.
SIMPLE_FIELD_SPACING_SCALE = 2.0
SIMPLE_GRID_STRIDE = 2
SIMPLE_MESH_QUALITY = 1
SIMPLE_ARROW_QUALITY = 0


def normalize_preview_mode(mode, default=DEFAULT_PREVIEW_MODE) -> str:
    text = str(mode or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "none": PREVIEW_OFF,
        "no": PREVIEW_OFF,
        "off": PREVIEW_OFF,
        "false": PREVIEW_OFF,
        "draft": PREVIEW_SIMPLE,
        "fast": PREVIEW_SIMPLE,
        "cheap": PREVIEW_SIMPLE,
        "on": PREVIEW_FULL,
        "true": PREVIEW_FULL,
        "live": PREVIEW_FULL,
    }
    text = aliases.get(text, text)
    if text in PREVIEW_MODES:
        return text
    return default


def preview_is_on(mode) -> bool:
    return normalize_preview_mode(mode) != PREVIEW_OFF


def preview_is_simple(mode) -> bool:
    return normalize_preview_mode(mode) == PREVIEW_SIMPLE


def preview_is_full(mode) -> bool:
    return normalize_preview_mode(mode) == PREVIEW_FULL


def preview_can_promote(mode) -> bool:
    """Only a full-fidelity preview may be renamed into the committed object."""
    return preview_is_full(mode)


def preview_mesh_quality(quality, mode, *, minimum=1) -> int:
    if preview_is_simple(mode):
        return int(SIMPLE_MESH_QUALITY)
    return max(int(minimum), int(quality))


def preview_arrow_quality(quality, mode) -> int:
    if preview_is_simple(mode):
        return int(SIMPLE_ARROW_QUALITY)
    return int(quality)


def preview_wireframe(wireframe, mode=None) -> bool:
    """Respect the wireframe toggle; simple mode saves cost via quality / field coarsening."""
    del mode
    return bool(wireframe)


def preview_domain(domain, mode):
    if domain is None or not preview_is_simple(mode):
        return domain
    scale = float(SIMPLE_FIELD_SPACING_SCALE)
    spacing = float(getattr(domain, "spacing", 0.0) or 0.0) * scale
    with_spacing = getattr(domain, "with_spacing", None)
    if callable(with_spacing):
        return with_spacing(spacing)
    return domain


def preview_iso_kind(kind, mode) -> str:
    """Filled IsoSurface becomes IsoMesh in simple mode; Volume stays a volume."""
    name = str(kind or "IsoSurface")
    if preview_is_simple(mode) and name == "IsoSurface":
        return "IsoMesh"
    return name


def stride_sample_grid(grid, stride=SIMPLE_GRID_STRIDE):
    """Every ``stride``-th voxel; origin unchanged, step sizes scaled."""
    if grid is None:
        return None
    step = max(1, int(stride))
    if step <= 1:
        return grid
    import numpy as np

    from ...volumetric.GridData import GridData

    counts = np.asarray(getattr(grid, "step_counts", (0, 0, 0)), dtype=int).reshape(3)
    shape = tuple(int(c) + 1 for c in counts)
    values = np.asarray(getattr(grid, "values", ()), dtype=float)
    if values.size != int(np.prod(shape)):
        return grid
    sampled = values.reshape(shape)[::step, ::step, ::step]
    new_counts = np.asarray(sampled.shape, dtype=int) - 1
    new_counts = np.maximum(new_counts, 0)
    new_steps = np.asarray(grid.step_sizes, dtype=float).reshape(3) * float(step)
    origin = tuple(float(v) for v in np.asarray(grid.origin).reshape(3))
    copied = GridData(
        sampled.reshape(-1),
        step_sizes=tuple(float(v) for v in new_steps),
        step_counts=tuple(int(v) for v in new_counts),
        origin=origin,
        name=getattr(grid, "name", None),
    )
    affine = getattr(grid, "A_to", None)
    if affine is not None:
        copied.A_to = np.array(affine, copy=True)
    return copied


def preview_grid(grid, mode):
    if grid is None or not preview_is_simple(mode):
        return grid
    return stride_sample_grid(grid, SIMPLE_GRID_STRIDE)


def read_preview_mode(obj, default=DEFAULT_PREVIEW_MODE) -> str:
    """Preview fidelity stored on a session object (meshes, fields, visuals)."""
    if obj is None:
        return normalize_preview_mode(None, default=default)
    return normalize_preview_mode(getattr(obj, "preview_mode", None), default=default)


def stamp_preview_mode(obj, mode) -> None:
    if obj is None:
        return
    try:
        obj.preview_mode = normalize_preview_mode(mode)
    except Exception:
        pass


def preview_mode_choices():
    return (
        (PREVIEW_OFF, PREVIEW_OFF_LABEL, PREVIEW_OFF_TIP),
        (PREVIEW_SIMPLE, PREVIEW_SIMPLE_LABEL, PREVIEW_SIMPLE_TIP),
        (PREVIEW_FULL, PREVIEW_FULL_LABEL, PREVIEW_FULL_TIP),
    )


def preview_mode_icon_kind(mode) -> str:
    """Action-icon key for preview radio glyphs (no Qt)."""
    key = normalize_preview_mode(mode)
    return {
        PREVIEW_OFF: "preview_off",
        PREVIEW_SIMPLE: "preview_simple",
        PREVIEW_FULL: "preview_full",
    }.get(key, "preview_simple")


class PreviewModeRadios:
    """No / Simple / Full radios for Appearance and field visual pages."""

    def __init__(self, on_changed=None, initial=DEFAULT_PREVIEW_MODE):
        from ..pick import qt_modules
        from ..widgets.type_icons import apply_choice_icon

        QtCore, QtGui, QtWidgets = qt_modules()
        wrap = QtWidgets.QWidget()
        wrap.setObjectName("pmvPreviewMode")
        row = QtWidgets.QHBoxLayout(wrap)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        group = QtWidgets.QButtonGroup(wrap)
        group.setExclusive(True)
        self._buttons = {}
        start = normalize_preview_mode(initial)
        for value, label, tip in preview_mode_choices():
            radio = QtWidgets.QRadioButton(label)
            radio.setToolTip(tip)
            radio.setObjectName("pmvPreviewMode_%s" % value)
            apply_choice_icon(radio, preview_mode_icon_kind(value), QtGui, QtCore, QtWidgets)
            group.addButton(radio)
            row.addWidget(radio)
            self._buttons[value] = radio
        row.addStretch(1)
        self._buttons[start].setChecked(True)
        group.buttonToggled.connect(lambda *_: self._emit(on_changed))
        self.widget = wrap
        self._group = group

    def mode(self) -> str:
        for value, btn in self._buttons.items():
            if btn.isChecked():
                return value
        return DEFAULT_PREVIEW_MODE

    def set_mode(self, mode) -> None:
        key = normalize_preview_mode(mode)
        btn = self._buttons.get(key)
        if btn is not None:
            btn.setChecked(True)

    def buttons(self):
        return self._buttons

    def tooltips(self):
        return [(self._buttons[value], tip, label) for value, label, tip in preview_mode_choices()]

    def connect_changed(self, callback):
        if callback is None:
            return
        self._group.buttonToggled.connect(lambda *_: callback(self.mode()))

    def _emit(self, callback):
        if callback is not None:
            callback(self.mode())
