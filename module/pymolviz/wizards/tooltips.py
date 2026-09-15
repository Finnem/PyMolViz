"""Hover help for wizard settings, with a DEBUG check when tooltips are missing.

Call ``apply_required_tooltips`` when building a page, then
``warn_missing_setting_tooltips`` on the page root so future controls are
caught on first build. Logger name: ``pymolviz.wizards``.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from .pick import qt_modules

LOGGER = logging.getLogger("pymolviz.wizards")

HOOK_TO_SELECTION_TIP = (
    "When a new point already has an atom identity (selection, or a camera "
    "snap hit), keep it attached so the visual follows if that atom moves. "
    "Per-point Attach in the table overrides this for existing points."
)
ANCHOR_TIP = (
    "When checked, this point stays attached to its atom and follows if the atom moves. "
    "When unchecked, the position stays fixed at the current xyz."
)
SNAP_TO_ATOM_TIP = (
    "When adding from camera center, if a visible atom is within 2 Å of the "
    "camera-center marker, snap the point onto that atom. Does not apply to "
    "selection placement."
)
UPDATE_TO_CAMERA_TIP = (
    "Move the selected table row(s) to the current camera/screen center "
    "(uses Snap new points to atoms and Keep new points attached to atoms)."
)
UPDATE_TO_SELECTION_TIP = (
    "Move the selected table row(s) onto the current PyMOL selection. "
    "One selected atom updates every chosen row; several atoms are applied "
    "to the chosen rows in order."
)
EMPTY_PYMOL_SELECTION_MSG = (
    "No atoms in the current PyMOL selection.\n"
    "Pick atoms first (they go into sele), then try again."
)
FLIP_CLIP_TIP = (
    "Swap which side of this plane is kept. The eye jumps to the new keep side."
)
DELETE_CLIP_TIP = "Remove this clipping plane and recut the mesh."
EDIT_CLIP_TIP = (
    "Set this plane's position (origin) and orientation (keep-side normal) numerically."
)
CLIP_ORIGIN_TIP = (
    "A point on the plane in Ångströms. The Drag widget stays centered on the "
    "visible rectangle through this origin."
)
CLIP_NORMAL_TIP = (
    "Direction pointing toward the kept half-space. Values are unit-normalized "
    "when applied (for example 0, 0, 2 becomes 0, 0, 1)."
)

_SETTING_CLASS_NAMES = frozenset({
    "QCheckBox",
    "QSpinBox",
    "QDoubleSpinBox",
    "QComboBox",
    "QLineEdit",
    "QSlider",
    "QPushButton",
})
_SKIP_ANCESTOR_NAMES = frozenset({
    "QTableWidget",
    "QTableView",
    "QAbstractItemView",
    "QMenu",
})

TooltipItem = Union[object, Tuple[object, str], Tuple[object, str, str]]


def _set_tooltip(widget, text: str) -> None:
    setter = getattr(widget, "setToolTip", None)
    if callable(setter):
        setter(str(text))
        return
    inner = getattr(widget, "widget", None)
    if callable(inner) and not hasattr(inner, "setToolTip"):
        try:
            inner = inner()
        except Exception:
            inner = None
    if inner is not None and inner is not widget:
        _set_tooltip(inner, text)


def _read_tooltip(widget) -> str:
    getter = getattr(widget, "toolTip", None)
    if callable(getter):
        try:
            return str(getter() or "").strip()
        except Exception:
            return ""
    inner = getattr(widget, "widget", None)
    if callable(inner) and not hasattr(inner, "toolTip"):
        try:
            inner = inner()
        except Exception:
            inner = None
    if inner is not None and inner is not widget:
        return _read_tooltip(inner)
    return ""


def _describe(widget, name: Optional[str] = None) -> str:
    if name:
        return name
    for attr in ("objectName", "text", "placeholderText"):
        getter = getattr(widget, attr, None)
        if not callable(getter):
            continue
        try:
            value = getter()
        except Exception:
            value = ""
        if value:
            return str(value)
    return type(widget).__name__


def _as_items(widgets: Union[TooltipItem, Sequence[TooltipItem]]) -> List[Tuple[object, Optional[str]]]:
    if widgets is None:
        return []
    if not isinstance(widgets, (list, tuple)) or (
        isinstance(widgets, tuple)
        and widgets
        and not isinstance(widgets[0], (list, tuple))
        and hasattr(widgets[0], "setToolTip")
        and len(widgets) in (2, 3)
        and isinstance(widgets[1], str)
    ):
        widgets = [widgets]
    items = []
    for item in widgets:
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            widget = item[0]
            name = item[2] if len(item) > 2 else None
            if widget is not None:
                items.append((widget, name))
            continue
        items.append((item, None))
    return items


def require_tooltips(
    widgets: Union[TooltipItem, Sequence[TooltipItem]],
    *,
    context: str = "",
) -> list:
    """DEBUG-warn if any registered setting widget has an empty tooltip.

    Accepts a widget, a sequence of widgets, or ``(widget, tooltip[, name])``
    pairs (tooltip text is ignored here; use ``apply_required_tooltips`` to set).
    Does not crash.
    """
    missing = []
    prefix = ("%s: " % context) if context else ""
    for widget, name in _as_items(widgets):
        if _read_tooltip(widget):
            continue
        missing.append(widget)
        LOGGER.debug("Missing tooltip on %s%s", prefix, _describe(widget, name))
    return missing


def apply_required_tooltips(
    items: Iterable[TooltipItem],
    *,
    context: str = "",
) -> list:
    """Set tooltips on widgets, then DEBUG-warn if any text is still empty.

    Each item is ``(widget, tooltip)`` or ``(widget, tooltip, name)``.
    """
    registered = []
    for item in items:
        if item is None:
            continue
        if not isinstance(item, (list, tuple)):
            registered.append(item)
            continue
        widget = item[0]
        text = item[1] if len(item) > 1 else ""
        name = item[2] if len(item) > 2 else None
        if widget is None:
            continue
        if text:
            _set_tooltip(widget, text)
        registered.append((widget, "", name) if name else widget)
    return require_tooltips(registered, context=context)


def _qt_parent(widget):
    parent = getattr(widget, "parent", None)
    if not callable(parent):
        return None
    try:
        return parent()
    except Exception:
        return None


def tooltip_from_child_chain(start, stop_at=None) -> str:
    """First non-empty tooltip walking from *start* up to *stop_at* (inclusive)."""
    current = start
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = _read_tooltip(current)
        if text:
            return text
        if current is stop_at:
            break
        current = _qt_parent(current)
    return ""


def install_item_widget_tooltips(view):
    """Show child-widget tooltips inside a ``QListWidget`` / item view.

    ``QAbstractItemView`` consumes ``ToolTip`` events for the item, so hover
    text on row buttons never appears unless the viewport is filtered.
    Returns the filter (keep a reference) or ``None``.
    """
    QtCore, _, QtWidgets = qt_modules()
    if QtCore is None or QtWidgets is None or view is None:
        return None
    viewport_fn = getattr(view, "viewport", None)
    viewport = viewport_fn() if callable(viewport_fn) else None
    if viewport is None:
        return None

    def _event_pos(event):
        pos = event.pos() if hasattr(event, "pos") else None
        if pos is None and hasattr(event, "position"):
            pos = event.position()
        if pos is not None and hasattr(pos, "toPoint"):
            pos = pos.toPoint()
        return pos

    class _ItemWidgetTooltipFilter(QtCore.QObject):
        def eventFilter(self, obj, event):
            if event.type() != QtCore.QEvent.ToolTip:
                return False
            pos = _event_pos(event)
            if pos is None:
                return False
            item_at = getattr(view, "itemAt", None)
            widget_at = getattr(view, "itemWidget", None)
            if not callable(item_at) or not callable(widget_at):
                return False
            item = item_at(pos)
            widget = widget_at(item) if item is not None else None
            if widget is None:
                return False
            mapper = getattr(widget, "mapFrom", None)
            local = mapper(viewport, pos) if callable(mapper) else pos
            child_at = getattr(widget, "childAt", None)
            child = child_at(local) if callable(child_at) else None
            tip = tooltip_from_child_chain(child or widget, stop_at=widget)
            if not tip:
                return False
            global_pos = event.globalPos() if hasattr(event, "globalPos") else None
            QtWidgets.QToolTip.showText(global_pos, tip, child or widget)
            return True

    filt = _ItemWidgetTooltipFilter(view)
    viewport.installEventFilter(filt)
    setter = getattr(view, "setMouseTracking", None)
    if callable(setter):
        setter(True)
    vp_track = getattr(viewport, "setMouseTracking", None)
    if callable(vp_track):
        vp_track(True)
    return filt


def _inside_skipped_ancestor(widget) -> bool:
    current = widget
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in _SKIP_ANCESTOR_NAMES:
            return True
        current = _qt_parent(current)
    return False


def iter_setting_widgets(root) -> list:
    """Known option widgets under *root* (checkboxes, spins, combos, edits, buttons)."""
    _, _, QtWidgets = qt_modules()
    if QtWidgets is None or root is None:
        return []
    types = (
        QtWidgets.QCheckBox,
        QtWidgets.QSpinBox,
        QtWidgets.QDoubleSpinBox,
        QtWidgets.QComboBox,
        QtWidgets.QLineEdit,
        QtWidgets.QSlider,
        QtWidgets.QPushButton,
    )
    found = []
    finder = getattr(root, "findChildren", None)
    if callable(finder):
        for typ in types:
            try:
                found.extend(finder(typ))
            except Exception:
                continue
    if type(root).__name__ in _SETTING_CLASS_NAMES:
        found.insert(0, root)
    out = []
    seen = set()
    for widget in found:
        ident = id(widget)
        if ident in seen or _inside_skipped_ancestor(widget):
            continue
        seen.add(ident)
        out.append(widget)
    return out


def warn_missing_setting_tooltips(root, *, context: str = "") -> list:
    """Walk known setting widgets under *root* and DEBUG-warn if a tooltip is missing."""
    return require_tooltips(iter_setting_widgets(root), context=context)
