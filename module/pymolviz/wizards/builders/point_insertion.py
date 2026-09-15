"""Live Add-from source widget for the shared points table."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

from ..pick import qt_modules, qt_widget_alive
from ..tooltips import HOOK_TO_SELECTION_TIP, SNAP_TO_ATOM_TIP
from ..widgets.switch import make_switch
from ..widgets.theme import (
    BANNER_INFO,
    BANNER_WARNING,
    mark_primary_button,
    style_info_banner,
)
from ..widgets.type_icons import apply_source_icon, source_icon_kind, source_icon_pixmap
from .points import (
    ADD_POINT_LABEL,
    INSERT_SOURCE_CAMERA,
    INSERT_SOURCE_FRESH,
    INSERT_SOURCE_SELECTION,
    INSERTION_FRESH_WAITING,
    INSERTION_NOTHING_SELECTED,
    VisualPoint,
    insertion_add_label,
    insertion_can_add,
    insertion_preview_fingerprint,
    insertion_preview_text,
    insertion_selection_count,
    resolve_insertion_points,
)
from .zoom_selection import ZOOM_TO_SELECTION_TIP

SOURCE_SELECTION = INSERT_SOURCE_SELECTION
SOURCE_CAMERA = INSERT_SOURCE_CAMERA
SOURCE_FRESH = INSERT_SOURCE_FRESH

SNAP_LABEL = "Snap to atoms"
HOOK_LABEL = "Anchor to atoms"
ZOOM_LABEL = "Zoom to new points"
SHOW_COORDS_LABEL = "Show coordinates"
EXPORT_SEL_LABEL = "Create PyMOL selection"
SOURCE_SELECTION_LABEL = "Current selection"
SOURCE_CAMERA_LABEL = "Camera"
SOURCE_FRESH_LABEL = "Fresh selection"
ADD_FROM_LABEL = "Add from"

CANCEL_FRESH_TIP = "Cancel waiting for a new PyMOL selection."
FRESH_SELECTION_TIP = (
    "Click Add, then select atoms in PyMOL. The current selection is ignored."
)
SOURCE_TIP = "Where new points come from. Manual placement is not in this editor."
CURRENT_SELECTION_TIP = (
    "Use the atoms already selected in PyMOL when you click Add."
)
CAMERA_SOURCE_TIP = "Place at the current view center. Always available."
ADD_POINT_HEADER_LABEL = ADD_POINT_LABEL
ADD_POINT_TIP = (
    "Insert points from the chosen source. Current selection uses atoms "
    "selected now. Fresh selection waits for a new pick after Add. "
    "Camera is always available. Snap only applies to camera placement."
)
PREVIEW_TIP = (
    "Live preview of what Add will insert. The actual click re-reads PyMOL, "
    "so this label is never used as the inserted coordinates."
)
SHOW_COORDS_TIP = (
    "Show X/Y/Z columns. When a point is anchored to an atom, coordinates are "
    "usually an implementation detail until you edit them."
)
EXPORT_SEL_TIP = (
    "Create a PyMOL selection from enabled points and show their labels. "
    "Labels hide again when this editor closes."
)


class AddFromSourceRadios:
    """Current selection / Camera / Fresh selection radios."""

    def __init__(self, on_changed=None):
        QtCore, QtGui, QtWidgets = qt_modules()
        wrap = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(wrap)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(QtWidgets.QLabel(ADD_FROM_LABEL))
        group = QtWidgets.QButtonGroup(wrap)
        group.setExclusive(True)
        self._buttons = {}
        choices = (
            (INSERT_SOURCE_SELECTION, SOURCE_SELECTION_LABEL, CURRENT_SELECTION_TIP),
            (INSERT_SOURCE_CAMERA, SOURCE_CAMERA_LABEL, CAMERA_SOURCE_TIP),
            (INSERT_SOURCE_FRESH, SOURCE_FRESH_LABEL, FRESH_SELECTION_TIP),
        )
        for value, label, tip in choices:
            radio = QtWidgets.QRadioButton(label)
            radio.setToolTip(tip)
            apply_source_icon(radio, value, QtGui, QtCore, QtWidgets)
            group.addButton(radio)
            row.addWidget(radio)
            self._buttons[value] = radio
        self._buttons[INSERT_SOURCE_SELECTION].setChecked(True)
        group.buttonToggled.connect(lambda *_: self._emit(on_changed))
        self.widget = wrap
        self._group = group

    def source(self) -> str:
        for value, btn in self._buttons.items():
            if btn.isChecked():
                return value
        return INSERT_SOURCE_SELECTION

    def buttons(self):
        return self._buttons

    def connect_changed(self, callback):
        if callback is None:
            return
        self._group.buttonToggled.connect(lambda *_: callback(self.source()))

    def _emit(self, on_changed):
        if on_changed is not None:
            on_changed(self.source())


def make_add_from_source_toggle(on_changed=None):
    """Current / Camera / Fresh radios with source icons."""
    radios = AddFromSourceRadios(on_changed=on_changed)
    return radios.widget, radios


def insertion_banner_kind(text: str) -> str:
    if str(text) == INSERTION_NOTHING_SELECTED:
        return BANNER_WARNING
    return BANNER_INFO


class PointInsertionWidget:
    """Add-from radios, live preview, placement flags, and compact Add."""

    def __init__(
        self,
        parent,
        cmd,
        context: str,
        *,
        on_add: Callable,
        on_export: Callable[[], None],
        on_show_coords: Optional[Callable[[bool], None]] = None,
        get_existing: Optional[Callable[[], Sequence[VisualPoint]]] = None,
        hide_add: bool = False,
        hide_coords: bool = False,
        on_can_add_changed: Optional[Callable[[bool], None]] = None,
        on_wait_changed: Optional[Callable[[bool], None]] = None,
    ):
        self.cmd = cmd
        self._context = context
        self._on_add = on_add
        self._get_existing = get_existing or (lambda: ())
        self._hide_add = bool(hide_add)
        self._hide_coords = bool(hide_coords)
        self._on_can_add_changed = on_can_add_changed
        self._on_wait_changed = on_wait_changed
        self._waiting_fresh = False
        self._timer = None
        self._poll_page = None
        self._focus_filter = None
        self._last_fingerprint = None
        self._last_preview_ui = None
        self._source = None
        self._source_radios = None
        self._source_icon = None
        self._preview = None
        self._add_btn = None
        self.snap = None
        self.hook = None
        self.zoom = None
        self.show_coords = None
        self.export_sel = None
        self._widget = None
        self._build(parent, on_export, on_show_coords)

    @property
    def widget(self):
        return self._widget

    def source(self) -> str:
        radios = getattr(self, "_source_radios", None)
        if radios is not None:
            return radios.source()
        combo = getattr(self, "_source", None)
        if combo is None:
            return INSERT_SOURCE_SELECTION
        data = combo.currentData()
        return str(data) if data else INSERT_SOURCE_SELECTION

    def snap_checked(self) -> bool:
        return bool(self.snap is not None and self.snap.isChecked())

    def hook_checked(self) -> bool:
        return bool(self.hook is not None and self.hook.isChecked())

    def zoom_checked(self) -> bool:
        return bool(self.zoom is not None and self.zoom.isChecked())

    @property
    def snap_checkbox(self):
        return self.snap

    @property
    def hook_checkbox(self):
        return self.hook

    @property
    def zoom_checkbox(self):
        return self.zoom

    def stop_timer(self) -> None:
        self.stop_preview_timer()

    def _clicked_add(self) -> None:
        if getattr(self, "_waiting_fresh", False):
            self._set_waiting_fresh(False)
            self.refresh_preview()
            return
        if self.source() == INSERT_SOURCE_FRESH:
            self._clear_pymol_selection()
            self._set_waiting_fresh(True)
            self.refresh_preview()
            return
        pts = self.resolve_points(existing=self._get_existing())
        if not pts:
            return
        self._on_add(pts)

    def _set_waiting_fresh(self, waiting: bool) -> None:
        self._waiting_fresh = bool(waiting)
        callback = getattr(self, "_on_wait_changed", None)
        if callback is not None:
            callback(self._waiting_fresh)

    def _clear_pymol_selection(self) -> None:
        try:
            self.cmd.select("sele", "none")
        except Exception:
            pass
        try:
            self.cmd.unpick()
        except Exception:
            pass

    def can_add(self) -> bool:
        if getattr(self, "_waiting_fresh", False):
            return True
        return insertion_can_add(self.cmd, self.source())

    def resolve_points(self, existing: Sequence[VisualPoint] = ()) -> List[VisualPoint]:
        source = self.source()
        if source == INSERT_SOURCE_FRESH:
            source = INSERT_SOURCE_SELECTION
        snap = self.snap_checked() if source == INSERT_SOURCE_CAMERA else False
        hook = self.hook_checked()
        return resolve_insertion_points(
            self.cmd, source, existing=existing, snap=snap, hook=hook,
        )

    def _host_is_visible(self) -> bool:
        host = self._poll_page if self._poll_page is not None else self._widget
        if host is None:
            return True
        is_vis = getattr(host, "isVisible", None)
        if not callable(is_vis):
            return True
        if not qt_widget_alive(host):
            return False
        try:
            return bool(is_vis())
        except RuntimeError:
            return False

    def _current_fingerprint(self):
        source = self.source()
        if getattr(self, "_waiting_fresh", False):
            source = INSERT_SOURCE_SELECTION
        return insertion_preview_fingerprint(
            self.cmd, source, snap=self.snap_checked(),
        )

    def refresh_preview(self) -> None:
        """Update preview copy and Add enabled state. Never schedules a CGO remesh."""
        source = self.source()
        if source != INSERT_SOURCE_FRESH and getattr(self, "_waiting_fresh", False):
            self._set_waiting_fresh(False)
        if self.snap is not None:
            self.snap.setEnabled(source == INSERT_SOURCE_CAMERA)
        if self.hook is not None:
            self.hook.setEnabled(True)
        fingerprint = self._current_fingerprint()
        waiting = getattr(self, "_waiting_fresh", False)
        if waiting:
            text = INSERTION_FRESH_WAITING
        else:
            text = insertion_preview_text(
                self.cmd, source, snap=self.snap_checked(),
            )
        enabled = self.can_add()
        kind = insertion_banner_kind(text)
        ui = (text, bool(enabled), kind, str(source), bool(waiting))
        self._last_fingerprint = fingerprint
        if ui == getattr(self, "_last_preview_ui", None):
            return
        self._last_preview_ui = ui
        if self._preview is not None:
            self._preview.setText(text)
            if hasattr(self._preview, "palette"):
                style_info_banner(self._preview, kind=kind)
        if self._add_btn is not None:
            if waiting:
                self._add_btn.setText("Cancel pick")
            else:
                self._add_btn.setText(ADD_POINT_HEADER_LABEL)
            set_tip = getattr(self._add_btn, "setToolTip", None)
            if callable(set_tip):
                set_tip(CANCEL_FRESH_TIP if waiting else ADD_POINT_TIP)
            self._add_btn.setEnabled(enabled)
        callback = getattr(self, "_on_can_add_changed", None)
        if callback is not None:
            callback(bool(enabled))
        self._sync_source_icon(source)

    def _sync_source_icon(self, source: str) -> None:
        icon = getattr(self, "_source_icon", None)
        if icon is None:
            return
        set_pix = getattr(icon, "setPixmap", None)
        if not callable(set_pix):
            return
        QtCore, QtGui, QtWidgets = qt_modules()
        pix = source_icon_pixmap(source, QtGui, QtCore, QtWidgets)
        if pix is None:
            return
        set_pix(pix)
        set_tip = getattr(icon, "setToolTip", None)
        if callable(set_tip):
            set_tip(source_icon_kind(source))

    def _poll_preview(self) -> None:
        page = self._poll_page
        if page is not None and not qt_widget_alive(page):
            self.stop_preview_timer()
            return
        if not self._host_is_visible():
            self.stop_preview_timer()
            return
        fingerprint = self._current_fingerprint()
        if getattr(self, "_waiting_fresh", False):
            count = int(fingerprint[1]) if fingerprint is not None else 0
            if count > 0:
                pts = self.resolve_points(existing=self._get_existing())
                self._set_waiting_fresh(False)
                if pts:
                    self._on_add(pts)
                self.refresh_preview()
                return
        if fingerprint == getattr(self, "_last_fingerprint", None):
            return
        self.refresh_preview()

    def _ensure_preview_timer(self) -> None:
        timer = self._timer
        if timer is None:
            return
        parent = self._poll_page if self._poll_page is not None else self._widget
        if parent is not None and not qt_widget_alive(parent):
            return
        if not self._host_is_visible():
            return
        try:
            if not timer.isActive():
                timer.start()
        except RuntimeError:
            pass

    def _install_focus_refresh(self, page, QtCore) -> None:
        owner = self

        class _FocusFilter(QtCore.QObject):
            def eventFilter(self, obj, event):
                etype = event.type()
                if etype == QtCore.QEvent.FocusIn:
                    owner.refresh_preview()
                elif etype == QtCore.QEvent.Show:
                    owner._ensure_preview_timer()
                    owner.refresh_preview()
                elif etype == QtCore.QEvent.Hide:
                    owner.stop_preview_timer()
                return False

        filt = _FocusFilter(page)
        page.installEventFilter(filt)
        self._focus_filter = filt

    def start_preview_timer(self, page) -> None:
        QtCore, _, _ = qt_modules()
        if QtCore is None or not hasattr(QtCore, "QTimer"):
            self.refresh_preview()
            return
        self._poll_page = page
        if page is not None and self._focus_filter is None:
            self._install_focus_refresh(page, QtCore)
        if self._timer is None:
            parent = page if page is not None else self._widget
            if parent is None:
                self.refresh_preview()
                return
            timer = QtCore.QTimer(parent)
            timer.setInterval(250)
            timer.timeout.connect(self._poll_preview)
            self._timer = timer
        self._ensure_preview_timer()
        self.refresh_preview()

    def stop_preview_timer(self) -> None:
        timer = self._timer
        if timer is None:
            return
        try:
            timer.stop()
        except RuntimeError:
            pass

    def tooltips(self) -> Sequence[Tuple[object, str]]:
        radios = getattr(self, "_source_radios", None)
        radio_tips = ()
        if radios is not None:
            buttons = radios.buttons()
            radio_tips = (
                (buttons[INSERT_SOURCE_SELECTION], CURRENT_SELECTION_TIP, SOURCE_SELECTION_LABEL),
                (buttons[INSERT_SOURCE_CAMERA], CAMERA_SOURCE_TIP, SOURCE_CAMERA_LABEL),
                (buttons[INSERT_SOURCE_FRESH], FRESH_SELECTION_TIP, SOURCE_FRESH_LABEL),
            )
        elif self._source is not None:
            radio_tips = ((self._source, SOURCE_TIP, "Add from"),)
        return radio_tips + (
            (self._preview, PREVIEW_TIP, "Insertion preview"),
            (self._add_btn, ADD_POINT_TIP, ADD_POINT_HEADER_LABEL),
            (self.snap, SNAP_TO_ATOM_TIP),
            (self.hook, HOOK_TO_SELECTION_TIP),
            (self.zoom, ZOOM_TO_SELECTION_TIP),
            (self.show_coords, SHOW_COORDS_TIP),
            (self.export_sel, EXPORT_SEL_TIP),
        )

    def _build(self, parent, on_export, on_show_coords):
        QtCore, _, QtWidgets = qt_modules()
        box = QtWidgets.QWidget(parent)
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        source_row = QtWidgets.QHBoxLayout()
        source_row.setSpacing(8)
        self._add_from, self._source_radios = make_add_from_source_toggle(
            on_changed=lambda *_: self.refresh_preview(),
        )
        source_row.addWidget(self._add_from)
        self._add_btn = QtWidgets.QPushButton(ADD_POINT_HEADER_LABEL)
        self._add_btn.setAutoDefault(False)
        self._add_btn.setDefault(False)
        self._add_btn.clicked.connect(lambda *_args: self._clicked_add())
        mark_primary_button(self._add_btn)
        source_row.addWidget(self._add_btn)
        if self._hide_add:
            self._add_btn.hide()
        source_row.addStretch(1)
        layout.addLayout(source_row)

        self._preview = QtWidgets.QLabel(INSERTION_NOTHING_SELECTED)
        self._preview.setWordWrap(True)
        style_info_banner(
            self._preview, kind=insertion_banner_kind(INSERTION_NOTHING_SELECTED),
        )
        layout.addWidget(self._preview)

        flags = QtWidgets.QGridLayout()
        flags.setContentsMargins(0, 0, 0, 0)
        flags.setHorizontalSpacing(16)
        flags.setVerticalSpacing(6)
        self.snap = make_switch(SNAP_LABEL, icon="snap")
        self.snap.setChecked(True)
        self.snap.toggled.connect(lambda *_: self.refresh_preview())
        self.hook = make_switch(HOOK_LABEL, icon="anchor")
        self.hook.setChecked(True)
        self.zoom = make_switch(ZOOM_LABEL, icon="zoom")
        self.show_coords = make_switch(SHOW_COORDS_LABEL)
        self.show_coords.setChecked(False)
        if on_show_coords is not None:
            self.show_coords.toggled.connect(on_show_coords)
        flags.addWidget(self.snap, 0, 0)
        flags.addWidget(self.hook, 0, 1)
        flags.addWidget(self.zoom, 1, 0)
        flags.addWidget(self.show_coords, 1, 1)
        if self._hide_coords:
            self.show_coords.hide()
        flags.setColumnStretch(0, 1)
        flags.setColumnStretch(1, 1)
        layout.addLayout(flags)

        extra = QtWidgets.QHBoxLayout()
        extra.addStretch(1)
        self.export_sel = QtWidgets.QPushButton(EXPORT_SEL_LABEL)
        self.export_sel.clicked.connect(lambda *_args: on_export())
        extra.addWidget(self.export_sel)
        layout.addLayout(extra)

        self._widget = box
        self.refresh_preview()
