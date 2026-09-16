"""Live Add-from source widget for the shared points table."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

from ..pick import idle_selection_poll_ok, qt_modules, qt_widget_alive
from ..tooltips import HOOK_TO_SELECTION_TIP, SNAP_TO_ATOM_TIP
from ..widgets.switch import make_switch
from ..widgets.theme import (
    BANNER_INFO,
    BANNER_WARNING,
    add_bar_summary_css,
    add_bar_switch_css,
    apply_add_bar_button_style,
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
    INSERTION_SELECTION_EMPTY,
    VisualPoint,
    insertion_add_label,
    insertion_can_add,
    insertion_preview_fingerprint,
    insertion_preview_text,
    insertion_selection_caption,
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
SOURCE_FRESH_LABEL = "Add Clicked Atoms"
ADD_FROM_LABEL = "Add from"
CLICKED_ATOMS_LABEL = "Add Clicked Atoms"
ADD_CAMERA_LABEL = "Add Camera Center"
ADD_SELECTION_LABEL = ADD_POINT_LABEL
ADD_BAR_CONTROL_HEIGHT = 40
ADD_BAR_CAPTION_MIN = 20
SELECTION_CAPTION_TIP = (
    "Residue identity of the atoms currently selected in PyMOL, "
    "for example 42 GLY (14 atoms)."
)

CANCEL_FRESH_TIP = "Stop adding atoms as they are clicked in PyMOL."
FRESH_SELECTION_TIP = (
    "When on, each new PyMOL atom selection is added automatically. "
    "The current selection is cleared when you turn this on."
)
SOURCE_TIP = "Where new points come from. Manual placement is not in this editor."
CURRENT_SELECTION_TIP = (
    "Add a point for each atom currently selected in PyMOL."
)
CAMERA_SOURCE_TIP = (
    "Add a point at the current view center. Snap to atoms applies here."
)
ADD_POINT_HEADER_LABEL = ADD_SELECTION_LABEL
ADD_POINT_TIP = (
    "Add Camera Center places a point at the view. Add Current Selection "
    "uses atoms selected now. Add Clicked Atoms keeps adding each new pick."
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


class InsertionAddBar:
    """Add Clicked Atoms switch plus Camera / Current Selection buttons."""

    def __init__(
        self,
        on_clicked_atoms=None,
        on_camera=None,
        on_selection=None,
    ):
        QtCore, QtGui, QtWidgets = qt_modules()
        wrap = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(wrap)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        expanding = getattr(QtWidgets.QSizePolicy, "Expanding", None)
        fixed = getattr(QtWidgets.QSizePolicy, "Fixed", None)
        minimum = getattr(QtWidgets.QSizePolicy, "Minimum", None)
        align_vcenter = getattr(getattr(QtCore, "Qt", None), "AlignVCenter", None)
        align_left = getattr(getattr(QtCore, "Qt", None), "AlignLeft", None)
        align_top = getattr(getattr(QtCore, "Qt", None), "AlignTop", None)

        self.clicked = make_switch(CLICKED_ATOMS_LABEL)
        self.clicked.setChecked(False)
        self.clicked.setToolTip(FRESH_SELECTION_TIP)
        font = self.clicked.font()
        if hasattr(font, "setPointSize"):
            font.setPointSize(13)
            self.clicked.setFont(font)
        if on_clicked_atoms is not None:
            self.clicked.toggled.connect(lambda checked: on_clicked_atoms(bool(checked)))
        clicked_wrap = QtWidgets.QFrame()
        clicked_wrap.setObjectName("pmvAddBarSwitch")
        clicked_wrap.setStyleSheet(add_bar_switch_css())
        clicked_layout = QtWidgets.QHBoxLayout(clicked_wrap)
        clicked_layout.setContentsMargins(12, 0, 12, 0)
        clicked_layout.addWidget(self.clicked)
        if align_vcenter is not None:
            clicked_layout.setAlignment(self.clicked, align_vcenter)
        self._lock_control(clicked_wrap, expanding, fixed)
        grid.addWidget(clicked_wrap, 0, 0)

        self.camera = QtWidgets.QPushButton(ADD_CAMERA_LABEL)
        self.camera.setAutoDefault(False)
        self.camera.setDefault(False)
        self.camera.setToolTip(CAMERA_SOURCE_TIP)
        apply_source_icon(
            self.camera, INSERT_SOURCE_CAMERA, QtGui, QtCore, QtWidgets, size=18,
        )
        apply_add_bar_button_style(self.camera)
        self._lock_control(self.camera, expanding, fixed)
        if on_camera is not None:
            self.camera.clicked.connect(lambda *_: on_camera())
        grid.addWidget(self.camera, 0, 1)

        self.selection = QtWidgets.QPushButton(ADD_SELECTION_LABEL)
        self.selection.setAutoDefault(False)
        self.selection.setDefault(False)
        self.selection.setToolTip(CURRENT_SELECTION_TIP)
        apply_source_icon(
            self.selection, INSERT_SOURCE_SELECTION, QtGui, QtCore, QtWidgets, size=18,
        )
        apply_add_bar_button_style(self.selection)
        self._lock_control(self.selection, expanding, fixed)
        if on_selection is not None:
            self.selection.clicked.connect(lambda *_: on_selection())
        grid.addWidget(self.selection, 0, 2)

        self.selection_summary = QtWidgets.QLabel(INSERTION_SELECTION_EMPTY)
        self.selection_summary.setObjectName("pmvAddBarSummary")
        self.selection_summary.setWordWrap(True)
        self.selection_summary.setStyleSheet(add_bar_summary_css())
        self.selection_summary.setToolTip(SELECTION_CAPTION_TIP)
        self.selection_summary.setMinimumHeight(ADD_BAR_CAPTION_MIN)
        if align_left is not None and align_top is not None:
            self.selection_summary.setAlignment(align_left | align_top)
        if expanding is not None and minimum is not None:
            self.selection_summary.setSizePolicy(expanding, minimum)
        grid.addWidget(self.selection_summary, 1, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        grid.setRowMinimumHeight(0, ADD_BAR_CONTROL_HEIGHT)
        grid.setRowMinimumHeight(1, ADD_BAR_CAPTION_MIN)
        wrap.setMinimumHeight(ADD_BAR_CONTROL_HEIGHT + ADD_BAR_CAPTION_MIN + 4)
        if expanding is not None and minimum is not None:
            wrap.setSizePolicy(expanding, minimum)
        self.widget = wrap

    def _lock_control(self, widget, expanding, fixed) -> None:
        if expanding is not None and fixed is not None:
            widget.setSizePolicy(expanding, fixed)
        widget.setFixedHeight(ADD_BAR_CONTROL_HEIGHT)

    def clicked_atoms_checked(self) -> bool:
        return bool(self.clicked is not None and self.clicked.isChecked())

    def set_clicked_atoms(self, checked: bool, notify: bool = True) -> None:
        box = self.clicked
        if box is None:
            return
        want = bool(checked)
        if bool(box.isChecked()) == want:
            return
        blocker = getattr(box, "blockSignals", None)
        if not notify and callable(blocker):
            blocker(True)
        try:
            box.setChecked(want)
        finally:
            if not notify and callable(blocker):
                blocker(False)

    def set_selection_enabled(self, enabled: bool) -> None:
        if self.selection is not None:
            self.selection.setEnabled(bool(enabled))

    def set_camera_enabled(self, enabled: bool) -> None:
        if self.camera is not None:
            self.camera.setEnabled(bool(enabled))

    def set_selection_label(self, text: str) -> None:
        if self.selection is not None:
            self.selection.setText(str(text))

    def set_selection_caption(self, text: str) -> None:
        label = getattr(self, "selection_summary", None)
        if label is not None:
            label.setText(str(text))

    def sync_from_cmd(self, cmd) -> None:
        self.set_selection_caption(insertion_selection_caption(cmd))
        self.set_selection_enabled(
            insertion_can_add(cmd, INSERT_SOURCE_SELECTION) if cmd is not None else False
        )

    def tooltips(self) -> Sequence[Tuple[object, str]]:
        return (
            (self.clicked, FRESH_SELECTION_TIP, CLICKED_ATOMS_LABEL),
            (self.camera, CAMERA_SOURCE_TIP, ADD_CAMERA_LABEL),
            (self.selection, CURRENT_SELECTION_TIP, ADD_SELECTION_LABEL),
            (self.selection_summary, SELECTION_CAPTION_TIP, "Current selection"),
        )


def make_add_from_source_toggle(on_changed=None):
    """Clicked-atoms switch plus camera/selection add buttons."""
    bar = InsertionAddBar(on_clicked_atoms=on_changed)
    return bar.widget, bar


def insertion_banner_kind(text: str) -> str:
    if str(text) == INSERTION_NOTHING_SELECTED:
        return BANNER_WARNING
    return BANNER_INFO


class PointInsertionWidget:
    """Clicked-atoms toggle, camera/selection add, live preview, and placement flags."""

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
        self._add_bar = None
        self._source = None
        self._source_radios = None
        self._source_icon = None
        self._preview = None
        self._add_btn = None
        self._camera_btn = None
        self._selection_btn = None
        self._selection_summary = None
        self.clicked = None
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
        if self.clicked_atoms_checked():
            return INSERT_SOURCE_FRESH
        return INSERT_SOURCE_SELECTION

    def clicked_atoms_checked(self) -> bool:
        bar = getattr(self, "_add_bar", None)
        if bar is not None:
            return bar.clicked_atoms_checked()
        box = getattr(self, "clicked", None)
        return bool(box is not None and box.isChecked())

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
        self._add_selection()

    def _add_camera(self) -> None:
        pts = self.resolve_points(
            existing=self._get_existing(), source=INSERT_SOURCE_CAMERA,
        )
        if not pts:
            return
        self._on_add(pts)

    def _add_selection(self) -> None:
        if getattr(self, "_waiting_fresh", False):
            return
        pts = self.resolve_points(
            existing=self._get_existing(), source=INSERT_SOURCE_SELECTION,
        )
        if not pts:
            return
        self._on_add(pts)

    def _on_clicked_atoms_toggled(self, checked: bool) -> None:
        from ..last_click import use_atom_selection_mode

        use_atom_selection_mode(self.cmd, bool(checked))
        if bool(checked):
            self._clear_pymol_selection()
            self._set_waiting_fresh(True)
        else:
            self._set_waiting_fresh(False)
        self.refresh_preview()

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
        return insertion_can_add(self.cmd, INSERT_SOURCE_SELECTION)

    def resolve_points(self, existing: Sequence[VisualPoint] = (), source=None) -> List[VisualPoint]:
        kind = str(source or INSERT_SOURCE_SELECTION)
        if kind == INSERT_SOURCE_FRESH:
            kind = INSERT_SOURCE_SELECTION
        snap = self.snap_checked() if kind == INSERT_SOURCE_CAMERA else False
        hook = self.hook_checked()
        return resolve_insertion_points(
            self.cmd, kind, existing=existing, snap=snap, hook=hook,
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
        return insertion_preview_fingerprint(
            self.cmd, INSERT_SOURCE_SELECTION, snap=self.snap_checked(),
        )

    def refresh_preview(self) -> None:
        """Update preview copy and Add enabled state. Never schedules a CGO remesh."""
        waiting = getattr(self, "_waiting_fresh", False) or self.clicked_atoms_checked()
        if self.snap is not None:
            self.snap.setEnabled(True)
        if self.hook is not None:
            self.hook.setEnabled(True)
        fingerprint = self._current_fingerprint()
        if waiting:
            text = INSERTION_FRESH_WAITING
        else:
            text = insertion_preview_text(
                self.cmd, INSERT_SOURCE_SELECTION, snap=False,
            )
        enabled = self.can_add()
        kind = insertion_banner_kind(text)
        caption = insertion_selection_caption(self.cmd)
        ui = (text, bool(enabled), kind, bool(waiting), caption)
        self._last_fingerprint = fingerprint
        if ui == getattr(self, "_last_preview_ui", None):
            return
        self._last_preview_ui = ui
        if self._preview is not None:
            self._preview.setText(text)
            if hasattr(self._preview, "palette"):
                style_info_banner(self._preview, kind=kind)
        self._set_selection_enabled(enabled)
        self._set_selection_caption(caption)
        if self._add_btn is not None:
            set_tip = getattr(self._add_btn, "setToolTip", None)
            if callable(set_tip):
                set_tip(CURRENT_SELECTION_TIP)
            self._add_btn.setEnabled(enabled)
        callback = getattr(self, "_on_can_add_changed", None)
        if callback is not None:
            callback(bool(enabled))
        self._sync_source_icon(INSERT_SOURCE_SELECTION if not waiting else INSERT_SOURCE_FRESH)

    def _set_selection_enabled(self, enabled: bool) -> None:
        bar = getattr(self, "_add_bar", None)
        if bar is not None:
            bar.set_selection_enabled(enabled)
        btn = getattr(self, "_selection_btn", None)
        if btn is not None:
            btn.setEnabled(bool(enabled))

    def _set_selection_caption(self, text: str) -> None:
        bar = getattr(self, "_add_bar", None)
        if bar is not None:
            bar.set_selection_caption(text)
        label = getattr(self, "_selection_summary", None)
        setter = getattr(label, "setText", None) if label is not None else None
        if callable(setter):
            setter(str(text))

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
        if not idle_selection_poll_ok(page):
            return
        fingerprint = self._current_fingerprint()
        if getattr(self, "_waiting_fresh", False):
            count = int(fingerprint[1]) if fingerprint is not None else 0
            if count > 0:
                pts = self.resolve_points(
                    existing=self._get_existing(), source=INSERT_SOURCE_SELECTION,
                )
                if pts:
                    self._on_add(pts)
                self._clear_pymol_selection()
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
        bar = getattr(self, "_add_bar", None)
        add_tips = bar.tooltips() if bar is not None else ()
        return add_tips + (
            (self._preview, PREVIEW_TIP, "Insertion preview"),
            (self.snap, SNAP_TO_ATOM_TIP),
            (self.hook, HOOK_TO_SELECTION_TIP),
            (self.zoom, ZOOM_TO_SELECTION_TIP),
            (self.show_coords, SHOW_COORDS_TIP),
            (self.export_sel, EXPORT_SEL_TIP),
        )

    def attach_add_to_section(self, section) -> None:
        """Move the add bar onto the Points section title strip."""
        bar = getattr(self, "_add_bar", None)
        if bar is None or getattr(self, "_hide_add", False):
            return
        if section is None or not hasattr(section, "add_header_widget"):
            return
        section.add_header_widget(bar.widget)

    def _build(self, parent, on_export, on_show_coords):
        QtCore, _, QtWidgets = qt_modules()
        box = QtWidgets.QWidget(parent)
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._add_bar = InsertionAddBar(
            on_clicked_atoms=self._on_clicked_atoms_toggled,
            on_camera=self._add_camera,
            on_selection=self._add_selection,
        )
        self.clicked = self._add_bar.clicked
        self._camera_btn = self._add_bar.camera
        self._selection_btn = self._add_bar.selection
        self._add_btn = self._selection_btn
        self._selection_summary = self._add_bar.selection_summary
        layout.addWidget(self._add_bar.widget)
        if self._hide_add:
            self._add_bar.widget.hide()

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
