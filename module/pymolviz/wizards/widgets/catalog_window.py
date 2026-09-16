"""Shared catalog window shell: empty library state and type-picker pages."""

from __future__ import annotations

from typing import Callable, Collection, Optional, Sequence, Tuple

from ..pick import configure_tool_window, qt_modules
from ..tooltips import apply_required_tooltips
from .breadcrumb import make_page_header
from .scrolling import configure_resizable_window, make_scrolling_body
from .theme import (
    apply_page_layout,
    apply_type_card_style,
    apply_wizard_page_style,
    empty_title_css,
    muted_label_css,
    type_card_subtitle_css,
    type_card_title_css,
)
from .type_icons import type_icon_pixmap


def transparent_for_mouse(QtCore, widget) -> None:
    flag = getattr(QtCore.Qt, "WA_TransparentForMouseEvents", None)
    if flag is not None:
        widget.setAttribute(flag, True)


def build_empty_library_page(
    QtCore,
    QtWidgets,
    *,
    empty_title: str,
    empty_hint: str,
    add_button_text: str,
    add_tip: str,
    on_add,
    tooltip_context: str,
    style_add_button: Callable,
):
    page = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(page)
    layout.addStretch(1)

    title = QtWidgets.QLabel(empty_title)
    title.setAlignment(QtCore.Qt.AlignCenter)
    title.setStyleSheet(empty_title_css())
    hint = QtWidgets.QLabel(empty_hint)
    hint.setAlignment(QtCore.Qt.AlignCenter)
    hint.setWordWrap(True)
    hint.setStyleSheet(muted_label_css())

    btn = QtWidgets.QPushButton(add_button_text)
    btn.setAutoDefault(False)
    btn.setDefault(False)
    btn.setMinimumWidth(180)
    style_add_button(btn)
    hand = getattr(QtCore.Qt, "PointingHandCursor", None)
    if hand is not None:
        btn.setCursor(hand)
    btn.clicked.connect(on_add)
    apply_required_tooltips(
        [(btn, add_tip, add_button_text)],
        context=tooltip_context,
    )
    btn_row = QtWidgets.QHBoxLayout()
    btn_row.addStretch(1)
    btn_row.addWidget(btn)
    btn_row.addStretch(1)

    layout.addWidget(title)
    layout.addWidget(hint)
    layout.addSpacing(12)
    layout.addLayout(btn_row)
    layout.addStretch(1)
    return page


def build_type_picker_page(
    QtCore,
    QtWidgets,
    *,
    title_parts: Sequence[str],
    back_tip: str,
    subtitle: str,
    entries: Sequence[Tuple[str, str, str, str]],
    on_pick,
    on_back,
    tooltip_context: str,
    icon_rgb_fn,
    disabled_kinds: Optional[Collection[str]] = None,
):
    page = QtWidgets.QWidget()
    apply_wizard_page_style(page)
    layout = QtWidgets.QVBoxLayout(page)
    apply_page_layout(layout)

    header, back, _title = make_page_header(
        QtWidgets,
        on_back,
        title_parts,
        back_tip,
    )
    hint = QtWidgets.QLabel(subtitle)
    hint.setWordWrap(True)
    hint.setStyleSheet(muted_label_css())
    layout.addLayout(header)
    layout.addWidget(hint)

    scroll, body = make_scrolling_body(page)
    disabled = frozenset(disabled_kinds or ())
    for name, kind, text, icon_key in entries:
        body.addWidget(
            build_type_picker_card(
                QtCore,
                QtWidgets,
                name,
                kind,
                text,
                icon_key,
                on_pick,
                tooltip_context=tooltip_context,
                icon_rgb_fn=icon_rgb_fn,
                disabled=kind in disabled,
            )
        )
    body.addStretch(1)
    layout.addWidget(scroll, stretch=1)
    apply_required_tooltips(
        [(back, back_tip, "Back")],
        context=tooltip_context,
    )
    return page


def build_type_picker_card(
    QtCore,
    QtWidgets,
    name,
    kind,
    hint,
    icon_key,
    on_pick,
    *,
    tooltip_context: str,
    icon_rgb_fn,
    disabled: bool = False,
):
    _, QtGui, _ = qt_modules()
    btn = QtWidgets.QPushButton()
    btn.setAutoDefault(False)
    btn.setDefault(False)
    apply_type_card_style(btn)
    expanding = getattr(QtWidgets.QSizePolicy, "Expanding", None)
    preferred = getattr(QtWidgets.QSizePolicy, "Preferred", None)
    if expanding is not None and preferred is not None:
        btn.setSizePolicy(expanding, preferred)
    hand = getattr(QtCore.Qt, "PointingHandCursor", None)
    if hand is not None:
        btn.setCursor(hand)
    btn.setMinimumHeight(56)
    if disabled:
        btn.setEnabled(False)

    inner = QtWidgets.QHBoxLayout(btn)
    inner.setContentsMargins(12, 10, 12, 10)
    inner.setSpacing(12)

    glyph = QtWidgets.QLabel()
    pix = type_icon_pixmap(
        icon_key,
        QtGui,
        QtCore,
        QtWidgets,
        color=icon_rgb_fn(icon_key),
    )
    if pix is not None:
        glyph.setPixmap(pix)
    glyph.setFixedSize(32, 32)
    transparent_for_mouse(QtCore, glyph)

    text = QtWidgets.QVBoxLayout()
    text.setSpacing(2)
    title = QtWidgets.QLabel(name)
    title.setStyleSheet(type_card_title_css())
    subtitle = QtWidgets.QLabel(hint)
    subtitle.setWordWrap(True)
    subtitle.setStyleSheet(type_card_subtitle_css())
    transparent_for_mouse(QtCore, title)
    transparent_for_mouse(QtCore, subtitle)
    text.addWidget(title)
    text.addWidget(subtitle)

    align = getattr(QtCore.Qt, "AlignVCenter", None)
    if align is not None:
        inner.addWidget(glyph, 0, align)
    else:
        inner.addWidget(glyph)
    inner.addLayout(text, 1)

    btn.clicked.connect(lambda _checked=False, k=kind, n=name: on_pick(n, k))
    apply_required_tooltips(
        [(btn, hint, name)],
        context=tooltip_context,
    )
    return btn


def open_catalog_dialog(
    QtCore,
    QtWidgets,
    *,
    window_title: str,
    width: int,
    height: int,
    build_root_pages,
    on_destroyed,
):
    """Create a non-modal catalog dialog with a stacked root. Returns (window, stack)."""
    window = QtWidgets.QDialog()
    window.setWindowTitle(window_title)
    window.setModal(False)
    configure_tool_window(window)
    window.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
    window.resize(width, height)
    configure_resizable_window(window)

    root = QtWidgets.QVBoxLayout(window)
    apply_page_layout(root)
    apply_wizard_page_style(window)
    stack = QtWidgets.QStackedWidget()
    build_root_pages(stack)
    root.addWidget(stack, stretch=1)
    window.destroyed.connect(on_destroyed)
    return window, stack
