"""Prompt when a new custom colormap is highly similar to one already saved."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ...util.colormap_spec import (
    CHOICE_CANCEL,
    CHOICE_KEEP_NEW,
    CHOICE_USE_EXISTING,
    ColormapDefinition,
    SimilarColormapChoice,
    SimilarColormapMatch,
    closest_similar_custom_colormap,
    custom_preset_definition,
    is_custom_preset_name,
    ramp_rgba,
)
from ..pick import overlay_exec, qt_modules
from ..tooltips import apply_required_tooltips
from ..widgets.theme import apply_primary_button_style, apply_wizard_page_style, muted_label_css
from .colormap_dialog import _pixel, _ramp_image

USE_EXISTING_TIP = "Discard this copy and use the similar colormap that already exists."
KEEP_NEW_TIP = "Save this as a new custom colormap anyway."
_CONTEXT = "SimilarColormapDialog"
_HIGHLIGHT = (255, 236, 179)


def resolve_similar_custom_colormap(
    parent,
    defn: ColormapDefinition,
    exclude_name: Optional[str] = None,
) -> SimilarColormapChoice:
    """Ask whether to reuse a highly similar custom colormap.

    Returns ``keep_new`` when nothing similar exists or Qt is unavailable.
    """
    match = closest_similar_custom_colormap(defn, exclude_name=exclude_name)
    if match is None:
        return SimilarColormapChoice(CHOICE_KEEP_NEW)
    _, _, QtWidgets = qt_modules()
    if QtWidgets is None or not hasattr(QtWidgets, "QDialog"):
        return SimilarColormapChoice(CHOICE_KEEP_NEW)
    return prompt_similar_colormap(parent, defn, match)


@dataclass
class SimilarApply:
    cancelled: bool = False
    definition: Optional[ColormapDefinition] = None
    name: Optional[str] = None


def persist_custom_colormap(parent, defn: ColormapDefinition, name: str):
    """Write a custom preset, or reuse a similar existing one.

    Returns the preset name to keep, or ``None`` if the user cancelled.
    """
    name = str(name or defn.preset or "").strip()
    if not name:
        return None
    choice = resolve_similar_custom_colormap(parent, defn, exclude_name=name)
    if choice.action == CHOICE_CANCEL:
        return None
    if choice.action == CHOICE_USE_EXISTING and choice.name:
        return choice.name
    from ...util.colormap_spec import save_custom_preset

    save_custom_preset(name, defn)
    return name


def maybe_reuse_similar_colormap(parent, defn: ColormapDefinition, selected_name: Optional[str] = None) -> SimilarApply:
    """When picking a named map, offer an existing custom duplicate instead.

    Selecting an already-saved custom name is left alone. Builtins (and other
    new ramps) are compared to saved customs.
    """
    selected = str(selected_name or defn.preset or "").strip()
    if selected and is_custom_preset_name(selected):
        return SimilarApply(definition=defn, name=selected)
    choice = resolve_similar_custom_colormap(parent, defn, exclude_name=selected or None)
    if choice.action == CHOICE_CANCEL:
        return SimilarApply(cancelled=True)
    if choice.action == CHOICE_USE_EXISTING and choice.name:
        existing = custom_preset_definition(choice.name)
        if existing is not None:
            return SimilarApply(definition=existing, name=choice.name)
    return SimilarApply(definition=defn, name=selected or None)


def prompt_similar_colormap(parent, new_defn: ColormapDefinition, match: SimilarColormapMatch) -> SimilarColormapChoice:
    dialog = build_similar_colormap_dialog(parent, new_defn, match)
    if dialog is None:
        return SimilarColormapChoice(CHOICE_KEEP_NEW)
    result = overlay_exec(dialog, parent)
    if int(result or 0) == 1:
        return SimilarColormapChoice(CHOICE_USE_EXISTING, match.name)
    if int(result or 0) == 2:
        return SimilarColormapChoice(CHOICE_KEEP_NEW)
    return SimilarColormapChoice(CHOICE_CANCEL)


def build_similar_colormap_dialog(parent, new_defn: ColormapDefinition, match: SimilarColormapMatch):
    QtCore, QtGui, QtWidgets = qt_modules()
    if QtWidgets is None or not hasattr(QtWidgets, "QDialog"):
        return None
    dialog = QtWidgets.QDialog(parent)
    dialog.setObjectName("pmvSimilarColormapDialog")
    dialog.setWindowTitle("Similar colormap")
    dialog.setModal(True)
    apply_wizard_page_style(dialog)
    dialog.resize(560, 420)

    root = QtWidgets.QVBoxLayout(dialog)
    intro = QtWidgets.QLabel(
        "A highly similar custom colormap already exists. Differences in color "
        "and alpha stops are highlighted below."
    )
    intro.setWordWrap(True)
    intro.setStyleSheet(muted_label_css())
    root.addWidget(intro)

    ramps = QtWidgets.QHBoxLayout()
    ramps.addWidget(_named_ramp(QtWidgets, QtGui, "New colormap", new_defn), stretch=1)
    ramps.addWidget(_named_ramp(QtWidgets, QtGui, "Existing: %s" % match.name, match.definition), stretch=1)
    root.addLayout(ramps)

    diff_label = QtWidgets.QLabel("Difference (brighter = larger color/alpha change)")
    diff_label.setStyleSheet(muted_label_css())
    root.addWidget(diff_label)
    diff = QtWidgets.QLabel()
    diff.setObjectName("pmvSimilarColormapDiff")
    pix = _diff_ramp_image(QtGui, new_defn, match.definition, 480, 12)
    if pix is not None:
        diff.setPixmap(pix)
        diff.setScaledContents(True)
        diff.setMinimumHeight(12)
    root.addWidget(diff)

    table = _stop_table(QtCore, QtGui, QtWidgets, match.similarity.stop_diffs)
    table.setObjectName("pmvSimilarColormapStops")
    root.addWidget(table, stretch=1)

    buttons = QtWidgets.QHBoxLayout()
    buttons.addStretch(1)
    keep = QtWidgets.QPushButton("Keep new colormap")
    keep.setObjectName("pmvSimilarKeepNew")
    keep.setAutoDefault(False)
    existing = QtWidgets.QPushButton("Use %s" % match.name)
    existing.setAutoDefault(False)
    existing.setDefault(True)
    apply_primary_button_style(existing, "pmvSimilarUseExisting")
    keep.clicked.connect(lambda *_: dialog.done(2))
    existing.clicked.connect(lambda *_: dialog.done(1))
    apply_required_tooltips(
        [
            (keep, KEEP_NEW_TIP, "Keep new colormap"),
            (existing, USE_EXISTING_TIP, "Use existing colormap"),
        ],
        context=_CONTEXT,
    )
    buttons.addWidget(keep)
    buttons.addWidget(existing)
    root.addLayout(buttons)
    return dialog


def _named_ramp(QtWidgets, QtGui, title, defn):
    box = QtWidgets.QWidget()
    col = QtWidgets.QVBoxLayout(box)
    col.setContentsMargins(0, 0, 0, 0)
    col.setSpacing(4)
    label = QtWidgets.QLabel(title)
    col.addWidget(label)
    ramp = QtWidgets.QLabel()
    pix = _ramp_image(QtGui, defn, 240, 18) if QtGui is not None else None
    if pix is not None:
        ramp.setPixmap(pix)
        ramp.setScaledContents(True)
        ramp.setMinimumHeight(18)
    col.addWidget(ramp)
    return box


def _diff_ramp_image(QtGui, new_defn, existing_defn, width, height):
    if QtGui is None or not hasattr(QtGui, "QImage"):
        return None
    width = max(2, int(width))
    height = max(2, int(height))
    try:
        left = ramp_rgba(new_defn, n=width)
        right = ramp_rgba(existing_defn, n=width)
        delta = abs(left - right)
        heat = delta[:, :3].max(axis=1) * 0.75 + delta[:, 3] * 0.25
    except Exception:
        return None
    image = QtGui.QImage(width, 1, getattr(QtGui.QImage, "Format_RGB32", 4))
    for i, value in enumerate(heat):
        t = max(0.0, min(1.0, float(value) / 0.12))
        r = int(round(255 * t))
        g = int(round(40 * t))
        b = int(round(20 * t))
        image.setPixel(i, 0, _pixel(QtGui, (r / 255.0, g / 255.0, b / 255.0, 1.0)))
    pix = QtGui.QPixmap.fromImage(image)
    return pix.scaled(width, height)


def _swatch(QtWidgets, QtGui, rgba):
    label = QtWidgets.QLabel()
    label.setFixedSize(18, 14)
    if rgba is None:
        label.setText("—")
        return label
    r, g, b, a = [max(0, min(255, int(round(float(c) * 255.0)))) for c in rgba]
    label.setStyleSheet(
        "background: rgba(%d,%d,%d,%d); border: 1px solid #888;" % (r, g, b, a)
    )
    return label


def _stop_table(QtCore, QtGui, QtWidgets, diffs):
    table = QtWidgets.QTableWidget(0, 5)
    table.setHorizontalHeaderLabels(("Position", "New", "α new", "Existing", "α existing"))
    table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
    table.verticalHeader().setVisible(False)
    table.setRowCount(len(diffs))
    for row, item in enumerate(diffs):
        pos = item.position_new if item.position_new is not None else item.position_existing
        pos_text = "—" if pos is None else "%.3f" % pos
        if item.kind == "extra_new":
            pos_text = "%.3f (new)" % (item.position_new or 0.0)
        elif item.kind == "extra_existing":
            pos_text = "%.3f (existing)" % (item.position_existing or 0.0)
        table.setItem(row, 0, QtWidgets.QTableWidgetItem(pos_text))
        table.setCellWidget(row, 1, _swatch(QtWidgets, QtGui, item.rgba_new))
        table.setItem(
            row,
            2,
            QtWidgets.QTableWidgetItem("—" if item.rgba_new is None else "%.2f" % item.rgba_new[3]),
        )
        table.setCellWidget(row, 3, _swatch(QtWidgets, QtGui, item.rgba_existing))
        table.setItem(
            row,
            4,
            QtWidgets.QTableWidgetItem(
                "—" if item.rgba_existing is None else "%.2f" % item.rgba_existing[3]
            ),
        )
        if item.highlight:
            for col in (0, 2, 4):
                cell = table.item(row, col)
                if cell is not None and QtGui is not None:
                    cell.setBackground(QtGui.QColor(*_HIGHLIGHT))
    header = table.horizontalHeader()
    header.setStretchLastSection(True)
    table.setMinimumHeight(120)
    return table
