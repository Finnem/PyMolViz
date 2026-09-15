"""Persistent bottom bar: commit and overflow export."""

from __future__ import annotations

from typing import Callable

from ..tooltips import apply_required_tooltips
from .theme import (
    action_bar_css,
    apply_secondary_button_style,
    compact_primary_button_css,
    mark_primary_button,
)

DONE_LABEL = "Done"
UPDATE_IN_PYMOL_LABEL = "Update in PyMOL"
MORE_LABEL = "More"
EXPORT_SCRIPT_LABEL = "Export…"

DONE_TIP = (
    "Keep this visual in the PyMOL session under the name at the top. "
    "Edits already preview live in the viewer."
)
UPDATE_IN_PYMOL_TIP = (
    "Apply the live preview to this named object in the PyMOL session."
)
MORE_TIP = "Secondary actions, including exporting a PyMolViz pack or Python script."
EXPORT_SCRIPT_TIP = (
    "Write a PyMolViz pack (.pmv) or a Python script that rebuilds this visual."
)


def qt_modules_or_raise():
    from ..pick import qt_modules

    QtCore, QtGui, QtWidgets = qt_modules()
    if QtWidgets is None:
        raise RuntimeError("PyMOL Qt UI required")
    return QtCore, QtGui, QtWidgets


class BuilderActionBar:
    """Primary commit and overflow export, pinned to the bottom of the editor."""

    def __init__(
        self,
        parent,
        on_commit: Callable[[], None],
        on_export: Callable[[], None],
        *,
        context: str = "BuilderActionBar",
    ):
        _, _, QtWidgets = qt_modules_or_raise()
        frame = QtWidgets.QFrame(parent)
        frame.setObjectName("pmvBuilderActionBar")
        frame.setStyleSheet(action_bar_css())
        layout = QtWidgets.QHBoxLayout(frame)
        layout.setContentsMargins(8, 4, 8, 2)
        layout.setSpacing(8)

        done = QtWidgets.QPushButton(DONE_LABEL)
        done.setDefault(True)
        done.setAutoDefault(True)
        mark_primary_button(done)
        done.setObjectName("pmvCommit")
        done.setStyleSheet(compact_primary_button_css("pmvCommit"))
        done.setMaximumHeight(18)
        done.clicked.connect(lambda *_args: on_commit())

        more = QtWidgets.QPushButton(MORE_LABEL)
        more.setAutoDefault(False)
        more.setDefault(False)
        apply_secondary_button_style(more)
        menu = QtWidgets.QMenu(more)
        export_act = menu.addAction(EXPORT_SCRIPT_LABEL)
        if export_act is not None:
            export_act.setToolTip(EXPORT_SCRIPT_TIP)
            export_act.triggered.connect(lambda *_args: on_export())
        more.setMenu(menu)

        layout.addStretch(1)
        layout.addWidget(done)
        layout.addWidget(more)

        apply_required_tooltips(
            [
                (done, DONE_TIP, DONE_LABEL),
                (more, MORE_TIP, MORE_LABEL),
            ],
            context=context,
        )

        self.widget = frame
        self.done_btn = done
        self.more_btn = more
        self.export_action = export_act
        self._editing = False

    def set_editing(self, editing: bool) -> None:
        self._editing = bool(editing)
        if self._editing:
            self.done_btn.setText(UPDATE_IN_PYMOL_LABEL)
            self.done_btn.setToolTip(UPDATE_IN_PYMOL_TIP)
            return
        self.done_btn.setText(DONE_LABEL)
        self.done_btn.setToolTip(DONE_TIP)

    def set_commit_enabled(self, enabled: bool) -> None:
        self.done_btn.setEnabled(bool(enabled))
