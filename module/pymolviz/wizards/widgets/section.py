"""Titled content sections used by Visuals menus.

Replaces ``QGroupBox`` chrome with a full-width tinted header bar and a
padded body, matching the Geometry / Appearance / Points / Modifiers look.
"""

from __future__ import annotations

from typing import Optional

from ..pick import qt_modules
from .theme import apply_section_style


class Section:
    """Outer frame, header bar, and body layout for one menu block."""

    def __init__(
        self,
        title: str,
        parent=None,
        *,
        collapsible: bool = False,
        form: bool = False,
        expanded: bool = True,
        expanding: bool = False,
    ):
        QtCore, _, QtWidgets = qt_modules()
        self._title_text = str(title)
        self._collapsible = bool(collapsible)
        self._expanded = True if not collapsible else bool(expanded)
        self._on_toggled = None

        frame = QtWidgets.QFrame(parent)
        frame.setObjectName("pmvSection")
        if expanding:
            pol = getattr(QtWidgets.QSizePolicy, "Expanding", None)
            if pol is not None:
                frame.setSizePolicy(pol, pol)
        outer = QtWidgets.QVBoxLayout(frame)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QtWidgets.QFrame()
        header.setObjectName("pmvSectionHeader")
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(12, 7, 12, 7)
        header_layout.setSpacing(8)

        self.title_label = QtWidgets.QLabel()
        self.title_label.setObjectName("pmvSectionTitle")
        self.summary_label = QtWidgets.QLabel()
        self.summary_label.setObjectName("pmvSectionSummary")
        transparent = getattr(QtCore.Qt, "WA_TransparentForMouseEvents", None)
        if transparent is not None:
            self.title_label.setAttribute(transparent, True)
            self.summary_label.setAttribute(transparent, True)
        header_layout.addWidget(self.title_label, 0)
        header_layout.addStretch(1)
        header_layout.addWidget(self.summary_label, 0)
        outer.addWidget(header)

        body = QtWidgets.QWidget()
        body.setObjectName("pmvSectionBody")
        if expanding:
            pol = getattr(QtWidgets.QSizePolicy, "Expanding", None)
            if pol is not None:
                body.setSizePolicy(pol, pol)
        if form:
            layout = QtWidgets.QFormLayout(body)
            layout.setContentsMargins(12, 10, 12, 12)
            layout.setHorizontalSpacing(12)
            layout.setVerticalSpacing(8)
        else:
            layout = QtWidgets.QVBoxLayout(body)
            layout.setContentsMargins(12, 10, 12, 12)
            layout.setSpacing(8)
        outer.addWidget(body, stretch=1 if expanding else 0)

        self.frame = frame
        self.header = header
        self.body = body
        self.layout = layout
        apply_section_style(frame)
        self._sync_chrome()

        if self._collapsible:
            hand = getattr(QtCore.Qt, "PointingHandCursor", None)
            if hand is not None:
                header.setCursor(hand)
            owner = self

            class _HeaderClick(QtCore.QObject):
                def eventFilter(inner, obj, event):
                    press = getattr(QtCore.QEvent, "MouseButtonPress", None)
                    if press is not None and event.type() == press:
                        owner.toggle()
                        return True
                    return False

            filt = _HeaderClick(header)
            header.installEventFilter(filt)
            self._header_filter = filt

    @property
    def widget(self):
        return self.frame

    def set_summary(self, text: Optional[str]) -> None:
        if self.summary_label is None:
            return
        self.summary_label.setText("" if text is None else str(text))

    def set_expanded(self, expanded: bool) -> None:
        want = True if not self._collapsible else bool(expanded)
        if want == self._expanded:
            self._sync_chrome()
            return
        self._expanded = want
        self._sync_chrome()
        if self._on_toggled is not None:
            self._on_toggled(self._expanded)

    def is_expanded(self) -> bool:
        return bool(self._expanded)

    def toggle(self) -> None:
        if not self._collapsible:
            return
        self.set_expanded(not self._expanded)

    def set_toggled_callback(self, callback) -> None:
        self._on_toggled = callback

    def _sync_chrome(self) -> None:
        if self._collapsible:
            arrow = "▾" if self._expanded else "▸"
            self.title_label.setText("%s  %s" % (arrow, self._title_text))
        else:
            self.title_label.setText(self._title_text)
        self.body.setVisible(bool(self._expanded))
        collapsed = self._collapsible and not self._expanded
        self.frame.setProperty("collapsed", collapsed)
        style = self.frame.style()
        if style is not None:
            style.unpolish(self.frame)
            style.polish(self.frame)
        self.header.update()


def make_section(
    title: str,
    parent=None,
    *,
    collapsible: bool = False,
    form: bool = False,
    expanded: bool = True,
    expanding: bool = False,
) -> Section:
    """Build a titled section. ``form=True`` uses a ``QFormLayout`` body."""
    return Section(
        title,
        parent,
        collapsible=collapsible,
        form=form,
        expanded=expanded,
        expanding=expanding,
    )
