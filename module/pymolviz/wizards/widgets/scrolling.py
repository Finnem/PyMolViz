"""Vertical grow-or-scroll body used by Visuals menus.

The window can shrink below the editor's preferred height. Extra height goes
to expanding lists; when even the compact minimum no longer fits, the body
becomes a scroll area so header and action bar stay on screen.
"""

from __future__ import annotations

from typing import Tuple

from ..pick import qt_modules, qt_widget_alive
from .theme import PAGE_SPACING

WINDOW_MIN_WIDTH = 560
WINDOW_MIN_HEIGHT = 340
WINDOW_DEFAULT_WIDTH = 1280
WINDOW_DEFAULT_HEIGHT = 720
SCROLL_BODY_MIN_HEIGHT = 120
EXPANDING_LIST_MIN_HEIGHT = 96


def content_needs_vertical_scroll(viewport_height, content_min_height) -> bool:
    """True when the viewport is shorter than the content's compact minimum."""
    return int(viewport_height) < int(content_min_height)


def scroll_inner_min_height(content_min_height, viewport_height) -> int:
    """Inner-widget min height: lock to content when scrolling, else 0 to expand."""
    content_min = max(int(content_min_height), 0)
    viewport = max(int(viewport_height), 0)
    if content_needs_vertical_scroll(viewport, content_min):
        return content_min
    return 0


def viewport_right_inset(scroll_width, viewport_width) -> int:
    """Width of the vertical scrollbar gutter (0 when bars overlay or are hidden)."""
    return max(int(scroll_width) - int(viewport_width), 0)


def configure_resizable_window(window, width=WINDOW_MIN_WIDTH, height=WINDOW_MIN_HEIGHT) -> None:
    """Allow the Visuals dialog to shrink; show a size grip when Qt provides one."""
    if window is None:
        return
    try:
        window.setMinimumSize(int(width), int(height))
    except Exception:
        pass
    grip = getattr(window, "setSizeGripEnabled", None)
    if callable(grip):
        try:
            grip(True)
        except Exception:
            pass


def apply_expanding_list_policy(widget, QtWidgets, *, min_height=EXPANDING_LIST_MIN_HEIGHT) -> None:
    if widget is None or QtWidgets is None:
        return
    expanding = getattr(QtWidgets.QSizePolicy, "Expanding", None)
    if expanding is not None:
        widget.setSizePolicy(expanding, expanding)
    widget.setMinimumHeight(int(min_height))


def make_scrolling_body(parent=None) -> Tuple[object, object]:
    """Return ``(scroll_area, inner_layout)`` for menu content below a header."""
    QtCore, _, QtWidgets = qt_modules()
    scroll = QtWidgets.QScrollArea(parent)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(getattr(QtWidgets.QFrame, "NoFrame", 0))
    always_off = getattr(getattr(QtCore, "Qt", None), "ScrollBarAlwaysOff", None)
    as_needed = getattr(getattr(QtCore, "Qt", None), "ScrollBarAsNeeded", None)
    if always_off is not None:
        scroll.setHorizontalScrollBarPolicy(always_off)
    if as_needed is not None:
        scroll.setVerticalScrollBarPolicy(as_needed)
    scroll.setMinimumHeight(SCROLL_BODY_MIN_HEIGHT)
    apply_expanding_list_policy(scroll, QtWidgets, min_height=SCROLL_BODY_MIN_HEIGHT)
    adjust = getattr(
        getattr(QtWidgets, "QAbstractScrollArea", None), "AdjustIgnored", None,
    )
    if adjust is not None:
        scroll.setSizeAdjustPolicy(adjust)

    inner = QtWidgets.QWidget()
    inner.setObjectName("pmvScrollBody")
    layout = QtWidgets.QVBoxLayout(inner)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(PAGE_SPACING)
    apply_expanding_list_policy(inner, QtWidgets, min_height=0)
    scroll.setWidget(inner)
    _install_expand_or_scroll(scroll, inner, QtCore)
    return scroll, layout


def _install_expand_or_scroll(scroll, inner, QtCore) -> None:
    if QtCore is None or not hasattr(QtCore, "QObject"):
        return
    state = {"syncing": False}

    def sync():
        if state["syncing"]:
            return
        if not qt_widget_alive(scroll) or not qt_widget_alive(inner):
            return
        state["syncing"] = True
        try:
            body = inner.layout()
            if body is not None:
                content_min = int(body.minimumSize().height())
            else:
                content_min = int(inner.minimumSizeHint().height())
            viewport = scroll.viewport()
            vp_h = int(viewport.height()) if viewport is not None else 0
            inner.setMinimumHeight(scroll_inner_min_height(content_min, vp_h))
            _sync_gutter_peers(scroll)
        except RuntimeError:
            pass
        finally:
            state["syncing"] = False

    class _Filter(QtCore.QObject):
        def eventFilter(self, obj, event):
            etype = event.type()
            resize = getattr(QtCore.QEvent, "Resize", None)
            show = getattr(QtCore.QEvent, "Show", None)
            layout_request = getattr(QtCore.QEvent, "LayoutRequest", None)
            if etype in (resize, show, layout_request):
                sync()
            return False

    filt = _Filter(scroll)
    scroll.installEventFilter(filt)
    inner.installEventFilter(filt)
    viewport = getattr(scroll, "viewport", None)
    if callable(viewport):
        vp = viewport()
        if vp is not None:
            vp.installEventFilter(filt)
    try:
        bar = scroll.verticalScrollBar()
        if bar is not None:
            bar.rangeChanged.connect(lambda *_args: sync())
    except Exception:
        pass
    scroll._pmv_expand_filter = filt
    scroll._pmv_expand_sync = sync
    scroll._pmv_gutter_peers = []
    sync()


def _sync_gutter_peers(scroll) -> None:
    peers = getattr(scroll, "_pmv_gutter_peers", None)
    if not peers:
        return
    viewport = scroll.viewport() if callable(getattr(scroll, "viewport", None)) else None
    vp_w = int(viewport.width()) if viewport is not None else int(scroll.width())
    inset = viewport_right_inset(int(scroll.width()), vp_w)
    for widget in list(peers):
        if not qt_widget_alive(widget):
            continue
        try:
            widget.setContentsMargins(0, 0, inset, 0)
        except RuntimeError:
            pass


def bind_width_to_scroll_viewport(widget, scroll) -> None:
    """Keep ``widget`` as wide as the scroll viewport, not the scrollbar gutter."""
    if widget is None or scroll is None:
        return
    peers = getattr(scroll, "_pmv_gutter_peers", None)
    if peers is None:
        scroll._pmv_gutter_peers = []
        peers = scroll._pmv_gutter_peers
    if widget not in peers:
        peers.append(widget)
    _sync_gutter_peers(scroll)
    sync = getattr(scroll, "_pmv_expand_sync", None)
    if callable(sync):
        sync()
