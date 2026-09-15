"""Tree-branch gutter for nested library rows (no QWidget subclass at import)."""

from __future__ import annotations

from ..catalog import NEST_ELL, NEST_INDENT_PX, NEST_LINE_RGB, nest_connector
from ..pick import qt_modules


def make_nest_branch(row, parent=None, width=None):
    """Vertical stem plus a tee/L connector for one child row."""
    QtCore, QtGui, QtWidgets = qt_modules()
    if QtWidgets is None:
        return None
    branch = QtWidgets.QWidget(parent)
    connector = nest_connector(row)
    branch._pmv_last_child = connector == NEST_ELL
    span = int(width if width is not None else max(NEST_INDENT_PX, 18))
    branch.setFixedWidth(max(span, 18))
    expanding = getattr(QtWidgets.QSizePolicy, "Expanding", None)
    fixed = getattr(QtWidgets.QSizePolicy, "Fixed", None)
    if expanding is not None and fixed is not None:
        branch.setSizePolicy(fixed, expanding)
    transparent = getattr(QtCore.Qt, "WA_TransparentForMouseEvents", None)
    if transparent is not None:
        branch.setAttribute(transparent, True)
    branch.setObjectName("pmvNestBranch")
    branch.setStyleSheet("background: transparent;")
    branch.paintEvent = lambda event, widget=branch: _paint_nest_branch(
        widget, event, QtCore, QtGui,
    )
    return branch


def _paint_nest_branch(widget, _event, QtCore, QtGui):
    painter = QtGui.QPainter(widget)
    try:
        color = QtGui.QColor(*NEST_LINE_RGB)
        pen = QtGui.QPen(color)
        pen.setWidth(1)
        cap = getattr(getattr(QtCore, "Qt", None), "FlatCap", None)
        if cap is not None:
            pen.setCapStyle(cap)
        dash = getattr(getattr(QtCore, "Qt", None), "CustomDashLine", None)
        if dash is None:
            dash = getattr(getattr(QtCore, "Qt", None), "DashLine", None)
        if dash is not None:
            pen.setStyle(dash)
        try:
            pen.setDashPattern([1.6, 2.2])
        except Exception:
            pass
        painter.setPen(pen)
        rect = widget.rect()
        stem_x = max(int(rect.left()) + 10, int(rect.center().x()) - 4)
        mid_y = int(rect.center().y())
        top = int(rect.top())
        bottom = mid_y if getattr(widget, "_pmv_last_child", False) else int(rect.bottom())
        painter.drawLine(stem_x, top, stem_x, bottom)
        painter.drawLine(stem_x, mid_y, int(rect.right()) - 2, mid_y)
    finally:
        painter.end()
