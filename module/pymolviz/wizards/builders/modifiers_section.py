"""Collapsed Modifiers section: clip planes (and future ops)."""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

from ..pick import qt_modules
from ..tooltips import SHOW_CLIP_VISUALS_TIP
from ..widgets.section import make_section
from ..widgets.switch import make_switch
from .clip_modifier import ADD_CLIP_TIP, ClipModifierController


class ModifiersSection:
    """▸ Modifiers header with a Clip plane list, collapsed by default."""

    def __init__(
        self,
        parent,
        cmd,
        context: str,
        preview=None,
        *,
        get_span_points: Optional[Callable] = None,
        on_changed: Optional[Callable[[], None]] = None,
        page=None,
    ):
        if isinstance(cmd, ClipModifierController):
            self._controller = cmd
        else:
            self._controller = ClipModifierController(
                cmd=cmd,
                page=page,
                context=context,
                preview=preview,
                span_points=get_span_points or (lambda: None),
                on_changed=on_changed or (lambda: None),
            )
        self._context = context
        self._header = None
        self._body = None
        self._add_btn = None
        self._show_gizmos = None
        self._widget = None
        self._section = None
        self._build(parent)

    @property
    def widget(self):
        return self._widget

    @property
    def controller(self) -> ClipModifierController:
        return self._controller

    @property
    def clip(self) -> ClipModifierController:
        return self._controller

    def tooltips(self) -> Sequence[Tuple[object, str]]:
        return (
            (self._add_btn, ADD_CLIP_TIP, "Add clip plane"),
            (self._show_gizmos, SHOW_CLIP_VISUALS_TIP, "Show clip visuals"),
            (self._header, "Optional operations applied after geometry is built.", "Modifiers"),
        )

    def set_expanded(self, expanded: bool) -> None:
        if self._section is not None:
            self._section.set_expanded(bool(expanded))
        self._sync_header()

    def refresh_header(self) -> None:
        self._sync_header()

    def refresh_summary(self) -> None:
        self.refresh_header()

    def apply_gizmo_state(self, state) -> None:
        from ...util.clip_gizmo import normalize_clip_gizmo_state

        shown = normalize_clip_gizmo_state(state)["shown"]
        if self._show_gizmos is not None:
            self._show_gizmos.blockSignals(True)
            self._show_gizmos.setChecked(shown)
            self._show_gizmos.blockSignals(False)
        self._controller.apply_gizmo_state(state)

    def gizmo_state(self):
        return self._controller.gizmo_state()

    def cleanup(self) -> None:
        self._controller.reset()
        if self._show_gizmos is not None:
            self._show_gizmos.blockSignals(True)
            self._show_gizmos.setChecked(True)
            self._show_gizmos.blockSignals(False)

    def _build(self, parent):
        _, _, QtWidgets = qt_modules()
        section = make_section(
            "Modifiers", collapsible=True, expanded=False,
        )
        section.header.setToolTip("Optional operations applied after geometry is built.")
        self._section = section
        self._header = section.header
        self._body = section.body
        self._add_btn = QtWidgets.QPushButton("Add clip plane")
        self._add_btn.clicked.connect(self._controller.add_plane)
        section.layout.addWidget(self._add_btn)
        self._show_gizmos = make_switch("Show clip visuals")
        self._show_gizmos.setChecked(self._controller.gizmos_shown())
        self._show_gizmos.toggled.connect(self._controller.set_gizmos_shown)
        section.layout.addWidget(self._show_gizmos)
        clip_list = QtWidgets.QListWidget()
        clip_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        clip_list.setMinimumHeight(72)
        clip_list.setMouseTracking(True)
        section.layout.addWidget(clip_list)
        self._widget = section.widget
        self._controller.attach_list(clip_list)
        self._sync_header()

    def _sync_header(self) -> None:
        n = self._controller.modifier_count()
        if n <= 0:
            count = "None"
        elif n == 1:
            count = "1 modifier"
        else:
            count = "%d modifiers" % n
        if self._section is not None:
            self._section.set_summary(count)
