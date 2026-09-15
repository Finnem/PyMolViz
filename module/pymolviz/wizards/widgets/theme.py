"""Shared wizard look: Fields catalog tokens on a light card.

Screenshot palette (cool-grey page, blue primary, pastel kind pills).
Visuals, Fields, builders, and dialogs all use these tokens so one-off hex
does not drift. Mix helpers remain for warning banners.
"""

from __future__ import annotations

# Sampled from the Fields catalog mock (light card on cool grey).
PAGE = (245, 249, 251)
HEADER = (205, 226, 240)
HEADER_INK = (28, 36, 48)
BORDER = (196, 210, 220)
ROW = (255, 255, 255)
SELECTED = (219, 238, 249)
DETAIL = (237, 247, 252)
MUTED = (110, 122, 134)
INK = (25, 28, 35)
PRIMARY = (42, 130, 236)
PRIMARY_HOVER = (32, 116, 224)
PRIMARY_PRESSED = (24, 100, 210)
RAIL = PRIMARY
CHEVRON = (140, 152, 164)
DASH = (188, 200, 210)
ICON_BTN_EDGE = (220, 228, 234)
ICON_BTN_HOVER = (236, 242, 248)
SECONDARY_PRESSED = (232, 240, 248)
TREE_LINE = (176, 188, 198)
WHITE = (255, 255, 255)

BANNER_INFO = "info"
BANNER_WARNING = "warning"
EDITABLE_FIELD = "pmvEditableField"
WARNING_ACCENT = (214, 148, 48)

PAGE_MARGINS = (12, 10, 12, 10)
PAGE_SPACING = 10


def mix_rgb(a, b, t):
    """Blend two 3-tuples of 0–255 (or 0–1) channels. ``t=1`` is *b*."""
    t = max(0.0, min(1.0, float(t)))
    aa = _as_8bit(a)
    bb = _as_8bit(b)
    return tuple(int(round((1.0 - t) * x + t * y)) for x, y in zip(aa, bb))


def _as_8bit(rgb):
    vals = [float(x) for x in list(rgb)[:3]]
    if vals and max(vals) <= 1.0:
        vals = [v * 255.0 for v in vals]
    return tuple(max(0, min(255, int(round(v)))) for v in vals)


def rgb_css(rgb, alpha=1.0) -> str:
    r, g, b = _as_8bit(rgb)
    a = max(0.0, min(1.0, float(alpha)))
    if a >= 0.999:
        return "rgb(%d, %d, %d)" % (r, g, b)
    return "rgba(%d, %d, %d, %.3f)" % (r, g, b, a)


def qcolor(QtGui, rgb, alpha=255):
    """``QColor`` from a theme RGB tuple. Transparent paints stay local."""
    if QtGui is None:
        return None
    r, g, b = _as_8bit(rgb)
    return QtGui.QColor(r, g, b, max(0, min(255, int(alpha))))


def _qcolor_rgb(color):
    return (int(color.red()), int(color.green()), int(color.blue()))


def palette_rgbs(widget):
    pal = widget.palette()
    window = pal.color(pal.Window)
    base = pal.color(pal.Base)
    highlight = pal.color(pal.Highlight)
    text = pal.color(pal.WindowText)
    mid = pal.color(pal.Mid)
    return {
        "window": _qcolor_rgb(window),
        "base": _qcolor_rgb(base),
        "highlight": _qcolor_rgb(highlight),
        "text": _qcolor_rgb(text),
        "mid": _qcolor_rgb(mid),
    }


def _button_selector(object_name, fallback="QPushButton"):
    if object_name:
        return "QPushButton#%s" % object_name
    return fallback


def editable_fill(_colors=None):
    """Input well: white card, matching the Fields catalog."""
    return tuple(ROW)


def editable_edge(_colors=None):
    return tuple(BORDER)


def banner_fill(_colors=None, kind=BANNER_INFO):
    if str(kind) == BANNER_WARNING:
        return mix_rgb(PAGE, WARNING_ACCENT, 0.32)
    return tuple(SELECTED)


def banner_edge(_colors=None, kind=BANNER_INFO):
    if str(kind) == BANNER_WARNING:
        return mix_rgb(BORDER, WARNING_ACCENT, 0.55)
    return tuple(BORDER)


def banner_ink(_colors=None, kind=BANNER_INFO):
    if str(kind) == BANNER_WARNING:
        return mix_rgb(INK, WARNING_ACCENT, 0.42)
    return tuple(HEADER_INK)


def muted_label_css() -> str:
    return "color: %s;" % rgb_css(MUTED)


def empty_title_css() -> str:
    return "font-size: 14px; font-weight: 600; color: %s;" % rgb_css(INK)


def page_heading_css() -> str:
    return "font-size: 16px; color: %s;" % rgb_css(INK)


def cell_band_css(fill) -> str:
    return "QWidget#pmvLibCell { background: %s; }" % rgb_css(fill)


def selected_row_fill(selected=False):
    return SELECTED if selected else ROW


def primary_button_css(object_name="pmvPrimary") -> str:
    sel = _button_selector(object_name)
    return (
        "%s { background: %s; color: white; border: none;"
        " border-radius: 8px; padding: 10px 14px; font-weight: 600; font-size: 13px;"
        " min-height: 36px; }"
        "%s:hover { background: %s; }"
        "%s:pressed { background: %s; padding-top: 11px; padding-bottom: 9px; }"
        "%s:disabled { background: %s; color: %s; }"
        % (
            sel,
            rgb_css(PRIMARY),
            sel,
            rgb_css(PRIMARY_HOVER),
            sel,
            rgb_css(PRIMARY_PRESSED),
            sel,
            rgb_css(DASH),
            rgb_css(ROW),
        )
    )


def compact_primary_button_css(object_name="pmvCommit") -> str:
    sel = _button_selector(object_name)
    return (
        "%s { background: %s; color: white; border: none;"
        " border-radius: 6px; padding: 2px 12px; font-weight: 600; font-size: 12px;"
        " min-height: 18px; max-height: 18px; }"
        "%s:hover { background: %s; }"
        "%s:pressed { background: %s; }"
        "%s:disabled { background: %s; color: %s; }"
        % (
            sel,
            rgb_css(PRIMARY),
            sel,
            rgb_css(PRIMARY_HOVER),
            sel,
            rgb_css(PRIMARY_PRESSED),
            sel,
            rgb_css(DASH),
            rgb_css(ROW),
        )
    )


def dashed_button_css(object_name="pmvAddVisual") -> str:
    sel = _button_selector(object_name)
    return (
        "%s { text-align: center; padding: 6px 12px;"
        " font-weight: 600; color: %s; background: %s;"
        " border: 1px dashed %s; border-radius: 6px; min-height: 28px; }"
        "%s:hover { background: %s; border: 1px dashed %s; }"
        "%s:pressed { background: %s; }"
        % (
            sel,
            rgb_css(MUTED),
            rgb_css(ROW),
            rgb_css(DASH),
            sel,
            rgb_css(SELECTED),
            rgb_css(PRIMARY),
            sel,
            rgb_css(SECONDARY_PRESSED),
        )
    )


def secondary_button_css(object_name="pmvSecondary") -> str:
    sel = _button_selector(object_name)
    return (
        "%s { background: %s; color: %s; border: 1px solid %s;"
        " border-radius: 6px; padding: 6px 12px; font-weight: 600; }"
        "%s:hover { background: %s; border: 1px solid %s; }"
        "%s:pressed { background: %s; }"
        % (
            sel,
            rgb_css(ROW),
            rgb_css(INK),
            rgb_css(BORDER),
            sel,
            rgb_css(SELECTED),
            rgb_css(PRIMARY),
            sel,
            rgb_css(SECONDARY_PRESSED),
        )
    )


def apply_shrinking_combo(combo, QtWidgets, *, min_chars=8) -> None:
    """Keep a combo filling its row without sizing the window to the longest item."""
    if combo is None or QtWidgets is None:
        return
    expanding = getattr(QtWidgets.QSizePolicy, "Expanding", None)
    fixed = getattr(QtWidgets.QSizePolicy, "Fixed", None)
    if expanding is not None and fixed is not None:
        combo.setSizePolicy(expanding, fixed)
    adjust = getattr(QtWidgets.QComboBox, "AdjustToMinimumContentsLengthWithIcon", None)
    if adjust is not None:
        combo.setSizeAdjustPolicy(adjust)
    setter = getattr(combo, "setMinimumContentsLength", None)
    if callable(setter):
        setter(int(min_chars))


def apply_compact_combo(combo, QtWidgets, *, min_chars=8, min_width=72) -> None:
    apply_shrinking_combo(combo, QtWidgets, min_chars=min_chars)


def row_icon_css(object_name="pmvRowIcon") -> str:
    sel = _button_selector(object_name)
    return (
        "%s { background: %s; border: 1px solid %s;"
        " border-radius: 6px; padding: 2px; min-width: 28px; max-width: 32px;"
        " min-height: 28px; max-height: 32px; color: %s; font-weight: 700; }"
        "%s:hover { background: %s; }"
        % (
            sel,
            rgb_css(ROW),
            rgb_css(ICON_BTN_EDGE),
            rgb_css(MUTED),
            sel,
            rgb_css(ICON_BTN_HOVER),
        )
    )


def catalog_table_css(table_id="pmvCatalogTable", header_id="pmvCatalogHeader") -> str:
    table = "QTableWidget#%s" % table_id
    header = "QHeaderView#%s" % header_id
    return (
        "%s { background: %s; border: none;"
        " gridline-color: transparent; outline: none;"
        " selection-background-color: %s; selection-color: %s; }"
        "%s::item { padding: 0px; border: none; }"
        "%s::item:selected { background: %s; color: %s; }"
        "%s::section { background: %s; color: %s;"
        " border: none; border-bottom: 1px solid %s; padding: 7px 8px;"
        " font-weight: 600; }"
        "QTableCornerButton::section { background: %s; border: none; }"
        % (
            table,
            rgb_css(ROW),
            rgb_css(SELECTED),
            rgb_css(INK),
            table,
            table,
            rgb_css(SELECTED),
            rgb_css(INK),
            header,
            rgb_css(ROW),
            rgb_css(MUTED),
            rgb_css(BORDER),
            rgb_css(ROW),
        )
    )


def section_css() -> str:
    return (
        "QFrame#pmvSection { background: %s; border: 1px solid %s;"
        " border-radius: 8px; }"
        "QFrame#pmvSectionHeader { background: %s; border: none;"
        " border-top-left-radius: 7px; border-top-right-radius: 7px; }"
        "QFrame#pmvSection[collapsed=\"true\"] QFrame#pmvSectionHeader {"
        " border-radius: 7px; }"
        "QLabel#pmvSectionTitle { font-weight: 700; color: %s; font-size: 14px; }"
        "QLabel#pmvSectionSummary { color: %s; }"
        "QWidget#pmvSectionBody { background: transparent; border: none; }"
        % (
            rgb_css(ROW),
            rgb_css(BORDER),
            rgb_css(HEADER),
            rgb_css(HEADER_INK),
            rgb_css(MUTED),
        )
    )


def type_card_css() -> str:
    return (
        "QPushButton#pmvTypeCard { text-align: left; padding: 0px;"
        " border: 1px solid %s; border-radius: 6px;"
        " background: %s; }"
        "QPushButton#pmvTypeCard:hover { background: %s;"
        " border: 1px solid %s; }"
        "QPushButton#pmvTypeCard:pressed { background: %s; }"
        % (
            rgb_css(BORDER),
            rgb_css(ROW),
            rgb_css(SELECTED),
            rgb_css(PRIMARY),
            rgb_css(SECONDARY_PRESSED),
        )
    )


def type_card_title_css() -> str:
    return (
        "font-weight: 600; font-size: 13px; color: %s;"
        " background: transparent; border: none;" % rgb_css(INK)
    )


def type_card_subtitle_css() -> str:
    return "color: %s; background: transparent; border: none;" % rgb_css(MUTED)


def action_bar_css() -> str:
    return "#pmvBuilderActionBar { border-top: 1px solid %s; background: %s; }" % (
        rgb_css(BORDER),
        rgb_css(PAGE),
    )


def swatch_button_css(fill_css, extra="") -> str:
    return (
        "QPushButton { background: %s; border: 1px solid %s;"
        " border-radius: 3px; padding: 0px;%s }"
        "QPushButton:hover { border: 1px solid %s; }"
        % (fill_css, rgb_css(BORDER), extra, rgb_css(INK))
    )


def _control_css() -> str:
    return (
        "QWidget#pmvWizardPage QRadioButton {"
        " color: %s; spacing: 6px; }"
        "QWidget#pmvWizardPage QRadioButton::indicator {"
        " width: 14px; height: 14px; border: 1px solid %s;"
        " border-radius: 8px; background: %s; }"
        "QWidget#pmvWizardPage QRadioButton::indicator:hover {"
        " border: 1px solid %s; }"
        "QWidget#pmvWizardPage QRadioButton::indicator:checked {"
        " background: %s; border: 1px solid %s; }"
        % (
            rgb_css(INK),
            rgb_css(BORDER),
            rgb_css(ROW),
            rgb_css(PRIMARY),
            rgb_css(PRIMARY),
            rgb_css(PRIMARY),
        )
    )


def wizard_page_css() -> str:
    fill = rgb_css(editable_fill())
    edge = rgb_css(editable_edge())
    return (
        "QWidget#pmvWizardPage { background: %s; }"
        "QWidget#pmvWizardPage QLineEdit { padding: 4px 8px; border-radius: 4px; }"
        "QWidget#pmvWizardPage QComboBox {"
        " padding: 3px 6px; border-radius: 4px;"
        " background: %s; border: 1px solid %s; }"
        "QWidget#pmvWizardPage QComboBox:hover,"
        "QWidget#pmvWizardPage QComboBox:focus {"
        " border: 1px solid %s; }"
        "QWidget#pmvWizardPage QComboBox::drop-down {"
        " subcontrol-origin: padding; subcontrol-position: top right;"
        " width: 18px; border: none; }"
        "QWidget#pmvWizardPage QComboBox QAbstractItemView {"
        " background: %s; border: 1px solid %s;"
        " selection-background-color: %s; }"
        "QWidget#pmvWizardPage QSpinBox, QWidget#pmvWizardPage QDoubleSpinBox {"
        " padding: 3px 6px; border-radius: 4px;"
        " background: %s; border: 1px solid %s; }"
        "QWidget#pmvWizardPage QSpinBox:hover, QWidget#pmvWizardPage QDoubleSpinBox:hover,"
        "QWidget#pmvWizardPage QSpinBox:focus, QWidget#pmvWizardPage QDoubleSpinBox:focus {"
        " border: 1px solid %s; }"
        "QWidget#pmvWizardPage QLineEdit#pmvObjectName,"
        "QWidget#pmvWizardPage QComboBox#%s {"
        " background: %s; border: 1px solid %s; border-radius: 4px; }"
        "QWidget#pmvWizardPage QLineEdit#pmvObjectName { padding: 3px 8px; }"
        "QWidget#pmvWizardPage QComboBox#%s { padding: 3px 8px 3px 6px; }"
        "%s"
        % (
            rgb_css(PAGE),
            fill,
            edge,
            rgb_css(PRIMARY),
            fill,
            edge,
            rgb_css(SELECTED),
            fill,
            edge,
            rgb_css(PRIMARY),
            EDITABLE_FIELD,
            fill,
            edge,
            EDITABLE_FIELD,
            _control_css(),
        )
    )


def apply_wizard_page_style(widget) -> None:
    """Page background, inputs, and radio chrome. Boolean toggles paint as Switch."""
    if widget is None:
        return
    widget.setObjectName("pmvWizardPage")
    widget.setStyleSheet(wizard_page_css())


def apply_primary_button_style(button, object_name="pmvPrimary") -> None:
    if button is None:
        return
    button.setObjectName(object_name)
    button.setStyleSheet(primary_button_css(object_name))


def mark_primary_button(button) -> None:
    apply_primary_button_style(button, "pmvPrimary")


def apply_secondary_button_style(button, object_name="pmvSecondary") -> None:
    if button is None:
        return
    button.setObjectName(object_name)
    button.setStyleSheet(secondary_button_css(object_name))


def apply_dashed_button_style(button, object_name="pmvAddVisual") -> None:
    if button is None:
        return
    button.setObjectName(object_name)
    button.setStyleSheet(dashed_button_css(object_name))


def apply_type_card_style(button) -> None:
    if button is None:
        return
    button.setObjectName("pmvTypeCard")
    button.setStyleSheet(type_card_css())


def apply_catalog_table_style(
    table, table_id="pmvCatalogTable", header_id="pmvCatalogHeader"
) -> None:
    if table is None:
        return
    table.setObjectName(table_id)
    header = table.horizontalHeader()
    if header is not None:
        header.setObjectName(header_id)
    table.setStyleSheet(catalog_table_css(table_id, header_id))


def style_info_banner(label, kind=BANNER_INFO) -> None:
    if label is None:
        return
    kind = BANNER_WARNING if str(kind) == BANNER_WARNING else BANNER_INFO
    name = "pmvWarningBanner" if kind == BANNER_WARNING else "pmvInfoBanner"
    label.setObjectName(name)
    label.setWordWrap(True)
    label.setStyleSheet(
        "QLabel#%s { background: %s; border: 1px solid %s;"
        " border-radius: 4px; padding: 8px 10px; color: %s; }"
        % (
            name,
            rgb_css(banner_fill(kind=kind)),
            rgb_css(banner_edge(kind=kind)),
            rgb_css(banner_ink(kind=kind)),
        )
    )


def apply_page_layout(layout) -> None:
    if layout is None:
        return
    layout.setContentsMargins(*PAGE_MARGINS)
    layout.setSpacing(PAGE_SPACING)


def apply_section_style(frame) -> None:
    if frame is None:
        return
    frame.setStyleSheet(section_css())
