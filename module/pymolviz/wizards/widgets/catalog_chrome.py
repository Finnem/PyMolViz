"""Fields catalog look: table chrome, kind pills, row icons.

Shared tokens and QSS live in ``theme``. This module keeps Fields-specific
helpers (field rail, add-visual dashed control, selected field-row fill).
"""

from __future__ import annotations

from .theme import (
    BORDER,
    CHEVRON,
    DASH,
    HEADER,
    HEADER_INK,
    ICON_BTN_EDGE,
    ICON_BTN_HOVER,
    INK,
    MUTED,
    PAGE,
    PRIMARY,
    PRIMARY_HOVER,
    PRIMARY_PRESSED,
    RAIL,
    ROW,
    SELECTED,
    apply_catalog_table_style,
    apply_dashed_button_style,
    apply_primary_button_style,
    apply_section_style,
    cell_band_css,
    dashed_button_css,
    primary_button_css,
    rgb_css,
    row_icon_css,
)

# Aliases so existing Fields tests keep importing catalog names.
CATALOG_PAGE = PAGE
CATALOG_HEADER = HEADER
CATALOG_HEADER_INK = HEADER_INK
CATALOG_BORDER = BORDER
CATALOG_ROW = ROW
CATALOG_SELECTED = SELECTED
CATALOG_MUTED = MUTED
CATALOG_INK = INK
CATALOG_PRIMARY = PRIMARY
CATALOG_PRIMARY_HOVER = PRIMARY_HOVER
CATALOG_PRIMARY_PRESSED = PRIMARY_PRESSED
CATALOG_RAIL = RAIL
CATALOG_CHEVRON = CHEVRON
CATALOG_DASH = DASH
CATALOG_ICON_BTN_EDGE = ICON_BTN_EDGE
CATALOG_ICON_BTN_HOVER = ICON_BTN_HOVER

ADD_FIELD_STYLE = primary_button_css("pmvAddField")
ADD_VISUAL_STYLE = dashed_button_css("pmvAddVisual")
ROW_ICON_STYLE = row_icon_css("pmvRowIcon")

_FOOTER_STYLE = "QWidget#pmvLibraryFooter { background: %s; border: none; }" % rgb_css(ROW)

LIBRARY_ROW_MIN_HEIGHT = 38


def library_row_fill(kind, selected=False):
    """Row band RGB. Only the current field row gets the light-blue selection."""
    from ..catalog import KIND_FIELD

    if selected and kind == KIND_FIELD:
        return SELECTED
    return ROW


def apply_fields_catalog_section_style(frame) -> None:
    apply_section_style(frame)


def apply_fields_table_style(table) -> None:
    apply_catalog_table_style(table, "pmvFieldsTable", "pmvFieldsHeader")


def apply_add_field_button_style(button) -> None:
    apply_primary_button_style(button, "pmvAddField")


def apply_add_visual_button_style(button) -> None:
    apply_dashed_button_style(button, "pmvAddVisual")


def library_footer_css() -> str:
    return _FOOTER_STYLE


def kind_badge_css(fill, ink) -> str:
    return (
        "QLabel#pmvKindBadge { background: %s; color: %s; border: none;"
        " border-radius: 8px; padding: 2px 8px; font-size: 11px; font-weight: 600; }"
        % (rgb_css(fill), rgb_css(ink))
    )


def make_kind_badge(QtWidgets, text, fill, ink):
    label = QtWidgets.QLabel(str(text or ""))
    label.setObjectName("pmvKindBadge")
    label.setStyleSheet(kind_badge_css(fill, ink))
    expanding = getattr(QtWidgets.QSizePolicy, "Fixed", None)
    preferred = getattr(QtWidgets.QSizePolicy, "Preferred", None)
    if expanding is not None and preferred is not None:
        label.setSizePolicy(preferred, expanding)
    return label


def make_lib_cell(QtWidgets, band, min_height=LIBRARY_ROW_MIN_HEIGHT):
    wrap = QtWidgets.QWidget()
    wrap.setObjectName("pmvLibCell")
    wrap.setAutoFillBackground(True)
    wrap.setStyleSheet(band)
    wrap.setMinimumHeight(int(min_height))
    expanding = getattr(QtWidgets.QSizePolicy, "Expanding", None)
    if expanding is not None:
        wrap.setSizePolicy(expanding, expanding)
    return wrap


def make_kind_cell(QtCore, QtWidgets, type_label, band, min_height=LIBRARY_ROW_MIN_HEIGHT):
    from ..catalog import kind_badge_rgb

    wrap = make_lib_cell(QtWidgets, band, min_height=min_height)
    layout = QtWidgets.QHBoxLayout(wrap)
    layout.setContentsMargins(6, 6, 8, 6)
    layout.setSpacing(0)
    text = str(type_label or "")
    if not text:
        return wrap
    fill, ink = kind_badge_rgb(text)
    badge = make_kind_badge(QtWidgets, text, fill, ink)
    flag = getattr(getattr(QtCore, "Qt", None), "WA_TransparentForMouseEvents", None)
    if flag is not None:
        badge.setAttribute(flag, True)
    layout.addWidget(badge, 0)
    layout.addStretch(1)
    return wrap


def make_row_icon_button(
    QtWidgets,
    text,
    tip,
    name,
    on_click,
    context,
    icon=None,
):
    from ..pick import qt_modules
    from ..tooltips import apply_required_tooltips
    from .type_icons import action_icon_pixmap

    QtCore, QtGui, _ = qt_modules()
    btn = QtWidgets.QPushButton(text)
    btn.setObjectName("pmvRowIcon")
    btn.setStyleSheet(ROW_ICON_STYLE)
    btn.setAutoDefault(False)
    btn.setDefault(False)
    if QtCore is not None:
        no_focus = getattr(QtCore.Qt, "NoFocus", None)
        if no_focus is not None:
            btn.setFocusPolicy(no_focus)
        hand = getattr(QtCore.Qt, "PointingHandCursor", None)
        if hand is not None:
            btn.setCursor(hand)
    if icon and QtGui is not None:
        pix = action_icon_pixmap(icon, QtGui, QtCore, QtWidgets, size=14)
        if pix is not None:
            btn.setIcon(QtGui.QIcon(pix))
            btn.setIconSize(pix.size())
            btn.setText("")
        elif not text:
            btn.setText("⌫")
    btn.clicked.connect(on_click)
    apply_required_tooltips(
        [(btn, tip, name)],
        context=context,
    )
    return btn


def make_field_rail(QtWidgets, QtCore=None):
    rail = QtWidgets.QWidget()
    rail.setObjectName("pmvFieldRail")
    rail.setFixedWidth(4)
    rail.setStyleSheet(
        "QWidget#pmvFieldRail { background: %s; border: none; }" % rgb_css(RAIL)
    )
    expanding = getattr(QtWidgets.QSizePolicy, "Expanding", None)
    fixed = getattr(QtWidgets.QSizePolicy, "Fixed", None)
    if expanding is not None and fixed is not None:
        rail.setSizePolicy(fixed, expanding)
    if QtCore is not None:
        flag = getattr(QtCore.Qt, "WA_TransparentForMouseEvents", None)
        if flag is not None:
            rail.setAttribute(flag, True)
    return rail
