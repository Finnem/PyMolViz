"""Zoom the PyMOL camera to points selected in a builder table."""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence

from ..pick import qt_modules
from .points import VisualPoint

TMP_PREFIX = "_pmv_zoom_tmp"
TMP_SELE = "_pmv_zoom_sel"

ZOOM_TO_SELECTION_TIP = (
    "When on, selecting table rows or double-clicking a row zooms the camera "
    "to frame those points. Turning it on also zooms the current selection "
    "immediately."
)


def _resolve_xyz(cmd_, pt: VisualPoint):
    try:
        from ...runtime.context import ResolveContext

        context = ResolveContext(cmd_)
        return pt.resolve(context)
    except Exception:
        return pt.xyz()


def zoom_to_visual_points(cmd_, points: Sequence[VisualPoint], animate: int = 0) -> None:
    """Frame the camera on one or more visual points via temporary pseudoatoms."""
    if not points:
        return
    xyz_list = [_resolve_xyz(cmd_, pt) for pt in points]
    _purge_tmp(cmd_)
    names = []
    try:
        for i, xyz in enumerate(xyz_list):
            name = "%s_%d" % (TMP_PREFIX, i)
            names.append(name)
            cmd_.pseudoatom(name, pos=[float(xyz[0]), float(xyz[1]), float(xyz[2])])
        sele_expr = " or ".join('object "%s"' % n for n in names)
        try:
            cmd_.select(TMP_SELE, sele_expr)
        except Exception:
            return
        try:
            cmd_.zoom(TMP_SELE, animate=animate, buffer=3)
        except TypeError:
            try:
                cmd_.zoom(TMP_SELE, buffer=3)
            except TypeError:
                cmd_.zoom(TMP_SELE)
    finally:
        _purge_tmp(cmd_, names)
        try:
            cmd_.select(TMP_SELE, "none")
        except Exception:
            pass


def _purge_tmp(cmd_, names: Iterable[str] = ()):
    for name in names:
        try:
            cmd_.delete(name)
        except Exception:
            pass
    try:
        for obj in cmd_.get_names("objects"):
            if str(obj).startswith(TMP_PREFIX):
                try:
                    cmd_.delete(obj)
                except Exception:
                    pass
    except Exception:
        pass


def points_from_rows(
    all_points: Sequence[VisualPoint],
    rows: Sequence[int],
) -> List[VisualPoint]:
    return [all_points[r] for r in rows if 0 <= r < len(all_points)]


def points_from_pair_rows(pairs, rows: Sequence[int]) -> List[VisualPoint]:
    out = []
    for row in rows:
        if 0 <= row < len(pairs):
            out.append(pairs[row].start)
            out.append(pairs[row].end)
    return out


def _selected_rows(table) -> List[int]:
    sm = table.selectionModel()
    if sm is not None:
        return sorted(index.row() for index in sm.selectedRows())
    return sorted({index.row() for index in table.selectedIndexes()})


def wire_zoom_to_selection(
    table,
    toggle,
    cmd_,
    point_supplier: Callable[[Optional[List[int]]], List[VisualPoint]],
) -> None:
    """Connect table selection, double-click, and toggle to camera zoom."""

    def run_zoom(rows: Optional[List[int]] = None):
        if not toggle.isChecked():
            return
        if rows is None:
            rows = _selected_rows(table)
        if not rows:
            return
        points = point_supplier(rows)
        if points:
            zoom_to_visual_points(cmd_, points)

    def schedule_zoom(rows: Optional[List[int]] = None):
        QtCore, _, _ = qt_modules()
        if QtCore is None:
            run_zoom(rows)
            return
        QtCore.QTimer.singleShot(0, lambda: run_zoom(rows))

    toggle.toggled.connect(lambda checked: schedule_zoom() if checked else None)

    sm = table.selectionModel()
    if sm is not None:
        sm.selectionChanged.connect(lambda *_args: schedule_zoom())
    else:
        table.itemSelectionChanged.connect(lambda: schedule_zoom())

    table.cellDoubleClicked.connect(lambda row, _col: schedule_zoom([row]))
