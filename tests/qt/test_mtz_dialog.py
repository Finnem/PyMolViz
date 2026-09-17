"""Qt: MTZ map picker lists columns and loads checked amplitude/phase pairs."""

from __future__ import annotations

import pytest

from pymolviz.util.io import mtz_inventory


class _FakeMtzCol:
    def __init__(self, label, ctype):
        self.label = label
        self.type = ctype


class _FakeMtz:
    def __init__(self, columns):
        self.columns = [_FakeMtzCol(label, ctype) for label, ctype in columns]
        self.title = "refine"
        self.nreflections = 12
        self.spacegroup = type("SG", (), {"hm": "P 1"})()


def _qt_app():
    from pymolviz.wizards.pick import qt_modules

    QtCore, _, QtWidgets = qt_modules()
    if QtWidgets is None or not hasattr(QtWidgets, "QDialog"):
        pytest.skip("Qt widgets unavailable")
    try:
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        return QtCore, QtWidgets, app
    except Exception:
        pytest.skip("QApplication not usable")


@pytest.mark.qt
def test_mtz_import_dialog_loads_checked_maps():
    from pymolviz.wizards.builders.mtz_dialog import show_mtz_import_dialog
    from pymolviz.wizards.pick import qt_widget_alive

    QtCore, QtWidgets, _app = _qt_app()
    parent = QtWidgets.QWidget()
    inventory = mtz_inventory(_FakeMtz([
        ("H", "H"), ("K", "H"), ("L", "H"),
        ("FWT", "F"), ("PHWT", "P"),
        ("DELFWT", "F"), ("PHDELWT", "P"),
        ("FOBS", "F"), ("SIGFOBS", "Q"), ("PHIB", "P"),
    ]))
    inventory["path"] = "/tmp/1abc.mtz"
    picked = []
    dialog = show_mtz_import_dialog(parent, inventory, on_load=picked.append)
    assert dialog is not None
    maps = dialog.findChild(QtWidgets.QTableWidget, "pmvMtzMapTable")
    cols = dialog.findChild(QtWidgets.QTableWidget, "pmvMtzColumnTable")
    assert maps is not None
    assert maps.rowCount() >= 2
    assert cols is not None
    assert cols.rowCount() >= 8
    check_off = getattr(QtCore.Qt, "Unchecked", 0)
    for row in range(maps.rowCount()):
        if row != 0:
            maps.item(row, 0).setCheckState(check_off)
    box = dialog.findChild(QtWidgets.QDialogButtonBox)
    load = None
    for btn in box.buttons():
        if btn.text().replace("&", "").startswith("Load"):
            load = btn
            break
    assert load is not None
    load.click()
    assert len(picked) == 1
    assert [row["factor"] for row in picked[0]] == ["FWT"]
    if qt_widget_alive(dialog):
        dialog.close()
    parent.close()
