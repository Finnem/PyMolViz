"""Shared editor chrome: breadcrumb, action bar, preview lifetime, commit."""

from __future__ import annotations

from typing import Callable, Optional

from ..pick import DeferredCallback, overlay_get_save_file_name, overlay_information, qt_modules
from ..tooltips import EMPTY_PYMOL_SELECTION_MSG, apply_required_tooltips, warn_missing_setting_tooltips
from ..widgets.action_bar import BuilderActionBar
from ..widgets.name_section import BuilderNameSection
from ..widgets.scrolling import bind_width_to_scroll_viewport, make_scrolling_body
from ..widgets.theme import apply_page_layout, apply_wizard_page_style
from ..widgets.breadcrumb import (
    BACK_TIP,
    create_crumbs,
    edit_crumbs,
    make_page_header,
    set_breadcrumb,
)
from .object_names import unused_object_name
from .preview import persist_live_preview


class BuilderPage:
    """Common Visuals-editor shell. Subclasses supply options and preview."""

    DEFAULT_NAME = "pmv_object"
    CRUMB_LEAF = "Object"
    CONTEXT = "BuilderPage"

    def __init__(
        self,
        cmd_,
        on_back: Callable[[], None],
        on_create: Optional[Callable[[], None]] = None,
        parent=None,
    ):
        self.cmd = cmd_
        self._on_back = on_back
        self._on_create = on_create
        self._deferred = DeferredCallback()
        self._page = None
        self._preview = None
        self._object_name = None
        self._create_btn = None
        self._action_bar = None
        self._name_section = None
        self._title = None
        self._editing_id = None
        self._loaded_name = None
        self._suspend_preview = False
        self._outer_layout = None
        self._init_editor()
        self._build(parent)

    @property
    def widget(self):
        return self._page

    def _init_editor(self):
        """Set preview and mesh-specific fields before the Qt tree is built."""

    def cleanup_preview(self):
        self._deferred.cancel()
        self._cleanup_ephemeral()
        if self._preview is not None:
            self._preview.cleanup()

    def _cleanup_ephemeral(self):
        """Drop UI-only PyMOL overlays (labels, gizmos, pick state)."""

    def _go_back(self):
        self.cleanup_preview()
        self._on_back()

    def _create_breadcrumb(self):
        return create_crumbs(self.CRUMB_LEAF)

    def _edit_breadcrumb(self):
        return edit_crumbs(self.CRUMB_LEAF)

    def _apply_create_chrome(self):
        self._editing_id = None
        self._loaded_name = None
        if self._object_name is not None:
            self._object_name.setText(unused_object_name(self.DEFAULT_NAME, self.cmd))
        if self._action_bar is not None:
            self._action_bar.set_editing(False)
        set_breadcrumb(self._title, self._create_breadcrumb())

    def _apply_edit_chrome(self, name: str):
        self._loaded_name = name
        if self._object_name is not None:
            self._object_name.setText(name)
        if self._action_bar is not None:
            self._action_bar.set_editing(True)
        set_breadcrumb(self._title, self._edit_breadcrumb())

    def _typed_name(self) -> str:
        if self._object_name is None:
            return self.DEFAULT_NAME
        return self._object_name.text().strip() or self.DEFAULT_NAME

    def _schedule_preview(self):
        if self._suspend_preview:
            return
        self._deferred.schedule(self._refresh_preview, page=self._page)

    def _refresh_preview(self):
        raise NotImplementedError

    def _can_commit(self) -> bool:
        return False

    def _collection(self, name: str):
        raise NotImplementedError

    def _retarget(self, collection) -> bool:
        return False

    def _prepare_persist(self, name: str):
        """Optional last preview push before promoting the live object."""

    def _sync_commit_enabled(self):
        if self._action_bar is not None:
            self._action_bar.set_commit_enabled(self._can_commit())

    def _create_cgo(self):
        if not self._can_commit():
            return
        name = unused_object_name(self._typed_name(), self.cmd, keep=self._loaded_name)
        self._prepare_persist(name)
        persist_live_preview(
            self.cmd,
            self._preview,
            name,
            obj_id=self._editing_id,
            retarget=self._retarget,
            fallback=lambda: self._collection(name),
        )
        if self._on_create is not None:
            self._on_create()

    def _export_cgo(self):
        if not self._can_commit():
            return
        _, _, QtWidgets = qt_modules()
        name = self._typed_name()
        path, _ = overlay_get_save_file_name(
            self._page,
            "Export CGO script",
            "%s.py" % name,
            "Python (*.py)",
        )
        if not path:
            return
        self._collection(name).write(path)

    def _warn_empty_pymol_selection(self, title):
        _, _, QtWidgets = qt_modules()
        if QtWidgets is not None:
            overlay_information(
                self._page, title, EMPTY_PYMOL_SELECTION_MSG,
            )

    def _require_qt(self):
        QtCore, QtGui, QtWidgets = qt_modules()
        if QtWidgets is None:
            raise RuntimeError("PyMOL Qt UI required")
        return QtCore, QtGui, QtWidgets

    def _mount_shell(self, parent, QtWidgets):
        page = QtWidgets.QWidget(parent)
        self._page = page
        apply_wizard_page_style(page)
        outer = QtWidgets.QVBoxLayout(page)
        apply_page_layout(outer)
        header, back, self._title = make_page_header(
            QtWidgets, self._go_back, self._create_breadcrumb(), BACK_TIP,
        )
        outer.addLayout(header)
        self._name_section = BuilderNameSection(
            page,
            name=unused_object_name(self.DEFAULT_NAME, self.cmd),
            context=self.CONTEXT,
        )
        self._object_name = self._name_section.name_edit
        outer.addWidget(self._name_section.widget)
        scroll, body = make_scrolling_body(page)
        bind_width_to_scroll_viewport(self._name_section.widget, scroll)
        outer.addWidget(scroll, stretch=1)
        self._outer_layout = outer
        return page, body, back

    def _mount_action_bar(self, page, root):
        self._action_bar = BuilderActionBar(
            page,
            on_commit=self._create_cgo,
            on_export=self._export_cgo,
            context=self.CONTEXT,
        )
        self._create_btn = self._action_bar.done_btn
        host = self._outer_layout if self._outer_layout is not None else root
        host.addWidget(self._action_bar.widget)

    def _finish_build(self, page, back, extra_tips):
        apply_required_tooltips(
            [(back, BACK_TIP)] + list(extra_tips or []),
            context=self.CONTEXT,
        )
        self._page = page
        self._after_build()
        warn_missing_setting_tooltips(page, context=self.CONTEXT)

    def _after_build(self):
        self._sync_commit_enabled()

    def _build(self, parent):
        raise NotImplementedError
