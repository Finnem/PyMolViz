"""Object-name row at the top of Visuals editors."""

from __future__ import annotations

from ..tooltips import apply_required_tooltips
from .theme import INK, rgb_css

NAME_LABEL = "Name:"
NAME_PLACEHOLDER = "Object name"
NAME_TIP = "Name of this visual in the PyMOL object list."

_LABEL_STYLE = "QLabel#pmvObjectNameLabel { font-weight: 600; color: %s; }" % rgb_css(INK)


def qt_modules_or_raise():
    from ..pick import qt_modules

    QtCore, QtGui, QtWidgets = qt_modules()
    if QtWidgets is None:
        raise RuntimeError("PyMOL Qt UI required")
    return QtCore, QtGui, QtWidgets


class BuilderNameSection:
    """Inline ``Name:`` label and emphasized object-name field, no section box."""

    def __init__(
        self,
        parent,
        *,
        name: str = "",
        context: str = "BuilderNameSection",
    ):
        QtCore, _, QtWidgets = qt_modules_or_raise()
        row = QtWidgets.QWidget(parent)
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        label = QtWidgets.QLabel(NAME_LABEL)
        label.setObjectName("pmvObjectNameLabel")
        label.setStyleSheet(_LABEL_STYLE)
        name_edit = QtWidgets.QLineEdit()
        name_edit.setObjectName("pmvObjectName")
        name_edit.setPlaceholderText(NAME_PLACEHOLDER)
        if name:
            name_edit.setText(name)
        align = getattr(getattr(QtCore, "Qt", None), "AlignVCenter", None)
        if align is not None:
            layout.addWidget(label, 0, align)
            layout.addWidget(name_edit, 1, align)
        else:
            layout.addWidget(label, 0)
            layout.addWidget(name_edit, 1)

        apply_required_tooltips(
            [(name_edit, NAME_TIP, NAME_PLACEHOLDER)],
            context=context,
        )
        self.widget = row
        self.name_edit = name_edit
        self.label = label
