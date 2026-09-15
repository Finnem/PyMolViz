"""Page titles that show where the builder sits in the Visuals window."""

from __future__ import annotations

from html import escape
from typing import Sequence

from .theme import INK, MUTED, PAGE, ROW, rgb_css

SEPARATOR = " › "

CRUMB_VISUALS = "Visuals"
CRUMB_ADD_OBJECT = "Add Visual"
CRUMB_SPHERES = "Spheres"
CRUMB_BOXES = "Boxes"
CRUMB_SURFACE = "Surface"
CRUMB_ARROWS = "Arrows"
CRUMB_FIELDS = "Fields"
CRUMB_ADD_FIELD = "Add Field"
CRUMB_VOLUME = "Volume"
CRUMB_ISOVOLUME = "IsoVolume"
CRUMB_ISOSURFACE = "IsoSurface"
CRUMB_ISOMESH = "IsoMesh"
CRUMB_IMPLICIT = "Implicit Surface"
CRUMB_FROM_SELECTION = "From Selection"

BACK_TIP = "Return to the previous page."
BACK_BUTTON_CSS = (
    "QPushButton#pmvBackButton { font-size: 13px; font-weight: 600; color: %s;"
    " padding: 2px 12px; min-height: 18px; min-width: 72px; border-radius: 6px;"
    " border: 1px solid %s; background: %s; }"
    "QPushButton#pmvBackButton:hover { background: %s; }"
    "QPushButton#pmvBackButton:pressed { padding-top: 3px; padding-bottom: 1px; }"
)


def breadcrumb_text(parts: Sequence[str]) -> str:
    return SEPARATOR.join(part for part in parts if part)


def breadcrumb_html(parts: Sequence[str]) -> str:
    """Rich-text crumbs: ancestors muted, current leaf bold."""
    crumbs = [part for part in parts if part]
    if not crumbs:
        return ""
    *head, leaf = crumbs
    sep = ' <span style="color: %s;">›</span> ' % rgb_css(MUTED)
    bits = [
        '<span style="font-weight:400; color: %s;">%s</span>' % (rgb_css(MUTED), escape(part))
        for part in head
    ]
    bits.append("<b>%s</b>" % escape(leaf))
    return sep.join(bits)


def create_crumbs(*leaf: str) -> tuple:
    return (CRUMB_VISUALS, CRUMB_ADD_OBJECT) + tuple(leaf)


def edit_crumbs(*leaf: str) -> tuple:
    return (CRUMB_VISUALS,) + tuple(leaf)


def create_field_crumbs(*leaf: str) -> tuple:
    return (CRUMB_FIELDS, CRUMB_ADD_FIELD) + tuple(leaf)


def create_field_visual_crumbs(*leaf: str) -> tuple:
    return (CRUMB_FIELDS, CRUMB_ADD_OBJECT) + tuple(leaf)


def edit_field_crumbs(*leaf: str) -> tuple:
    return (CRUMB_FIELDS,) + tuple(leaf)


def make_page_header(QtWidgets, on_back, parts: Sequence[str], back_tooltip: str = BACK_TIP):
    """Back button plus centered breadcrumb. Returns ``(layout, back, title)``."""
    from ..pick import qt_modules

    QtCore, _, _ = qt_modules()
    layout = QtWidgets.QHBoxLayout()
    layout.setContentsMargins(0, 0, 0, 4)
    back = QtWidgets.QPushButton("← Back")
    back.setObjectName("pmvBackButton")
    back.setFlat(False)
    back.setAutoDefault(False)
    back.setDefault(False)
    back.setToolTip(back_tooltip)
    back.setStyleSheet(
        BACK_BUTTON_CSS % (rgb_css(INK), rgb_css(MUTED), rgb_css(PAGE), rgb_css(ROW))
    )
    back.setMinimumHeight(18)
    title = QtWidgets.QLabel()
    title.setStyleSheet("font-size: 16px; color: %s;" % rgb_css(INK))
    title.setText(breadcrumb_html(parts))
    if QtCore is not None:
        rich = getattr(QtCore.Qt, "RichText", None)
        if rich is not None:
            title.setTextFormat(rich)
        center = getattr(QtCore.Qt, "AlignCenter", None)
        if center is not None:
            title.setAlignment(center)
    mirror = QtWidgets.QWidget()
    layout.addWidget(back, 0)
    layout.addStretch(1)
    layout.addWidget(title, 0)
    layout.addStretch(1)
    layout.addWidget(mirror, 0)
    if on_back is not None:
        back.clicked.connect(on_back)
        width = max(int(back.sizeHint().width()), 72)
        mirror.setFixedWidth(width)
        mirror.setMinimumWidth(width)
    else:
        back.hide()
        mirror.hide()
    return layout, back, title


def set_breadcrumb(label, parts: Sequence[str]) -> None:
    if label is None:
        return
    label.setText(breadcrumb_html(parts))
