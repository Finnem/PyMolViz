"""Optional Qt wizard tests (run with ``pytest -m qt``)."""

from __future__ import annotations

import logging
from unittest import mock

import pytest

from pymolviz.wizards.pick import DeferredCallback, qt_widget_alive
from pymolviz.wizards.tooltips import (
    apply_required_tooltips,
    require_tooltips,
)


class _FakeWidget:
    def __init__(self, class_name="QPushButton", tooltip="", *, alive=True):
        self.__class__.__name__ = class_name
        self._tooltip = tooltip
        self._alive = alive

    def setToolTip(self, text):
        self._tooltip = str(text)

    def toolTip(self):
        return self._tooltip

    def objectName(self):
        if not self._alive:
            raise RuntimeError("wrapped C/C++ object has been deleted")
        return self.__class__.__name__


def _qt_alive_via_object_name(widget) -> bool:
    """Exercise the objectName() fallback path used when shiboken/sip are absent."""
    if widget is None:
        return False
    try:
        widget.objectName()
        return True
    except RuntimeError:
        return False


@pytest.mark.qt
def test_apply_required_tooltips_leaves_no_gaps():
    widgets = [
        _FakeWidget(class_name="QPushButton", tooltip=""),
        _FakeWidget(class_name="QSpinBox", tooltip=""),
    ]
    missing = apply_required_tooltips(
        [(widgets[0], "Create CGO"), (widgets[1], "Quality level")],
        context="TestPage",
    )
    assert missing == []


@pytest.mark.qt
def test_require_tooltips_reports_gap(caplog):
    widget = _FakeWidget(class_name="QPushButton", tooltip="")
    caplog.set_level(logging.DEBUG, logger="pymolviz.wizards")
    missing = require_tooltips([widget], context="TestPage")
    assert len(missing) == 1
    assert any("missing tooltip" in rec.message.lower() for rec in caplog.records)


@pytest.mark.qt
def test_qt_widget_alive_detects_deleted_widget():
    alive = _FakeWidget()
    dead = _FakeWidget(alive=False)
    with mock.patch.dict("sys.modules", {"shiboken6": None, "shiboken2": None, "sip": None}):
        assert _qt_alive_via_object_name(alive) is True
        assert _qt_alive_via_object_name(dead) is False
        assert qt_widget_alive(None) is False


@pytest.mark.qt
def test_deferred_callback_skips_after_page_destroyed():
    calls = []
    pending = []

    class _FakeTimer:
        @staticmethod
        def singleShot(_delay, callback):
            pending.append(callback)

    class _FakeQtCore:
        QTimer = _FakeTimer

    page = _FakeWidget(alive=True)

    import pymolviz.wizards.pick as pick_mod

    original = pick_mod.qt_modules
    pick_mod.qt_modules = lambda: (_FakeQtCore, None, None)
    try:
        deferred = DeferredCallback()
        with mock.patch.object(pick_mod, "qt_widget_alive", side_effect=_qt_alive_via_object_name):
            deferred.schedule(lambda: calls.append("run"), page=page)
            pending.pop()()
            assert calls == ["run"]

            page._alive = False
            calls.clear()
            deferred.schedule(lambda: calls.append("late"), page=page)
            assert pending == []
            assert calls == []
    finally:
        pick_mod.qt_modules = original


@pytest.mark.qt
def test_deferred_callback_cancel_prevents_stale_run():
    calls = []
    pending = []

    class _FakeTimer:
        @staticmethod
        def singleShot(_delay, callback):
            pending.append(callback)

    class _FakeQtCore:
        QTimer = _FakeTimer

    import pymolviz.wizards.pick as pick_mod

    original = pick_mod.qt_modules
    pick_mod.qt_modules = lambda: (_FakeQtCore, None, None)
    try:
        deferred = DeferredCallback()
        deferred.schedule(lambda: calls.append("stale"))
        deferred.cancel()
        deferred.schedule(lambda: calls.append("fresh"))
        for callback in pending:
            callback()
        assert calls == ["fresh"]
    finally:
        pick_mod.qt_modules = original


@pytest.mark.qt
def test_new_section_tooltip_copy_is_applied():
    from pymolviz.wizards.builders.clip_modifier import ADD_CLIP_TIP
    from pymolviz.wizards.builders.point_insertion import (
        ADD_POINT_HEADER_LABEL,
        ADD_POINT_TIP,
        HOOK_LABEL,
        SNAP_LABEL,
        ZOOM_LABEL,
        insertion_add_label,
    )
    from pymolviz.wizards.builders.zoom_selection import ZOOM_TO_SELECTION_TIP
    from pymolviz.wizards.tooltips import HOOK_TO_SELECTION_TIP, SNAP_TO_ATOM_TIP

    widgets = [
        _FakeWidget(class_name="QComboBox", tooltip=""),
        _FakeWidget(class_name="QPushButton", tooltip=""),
        _FakeWidget(class_name="QPushButton", tooltip=""),
        _FakeWidget(class_name="QCheckBox", tooltip=""),
        _FakeWidget(class_name="QCheckBox", tooltip=""),
        _FakeWidget(class_name="QCheckBox", tooltip=""),
        _FakeWidget(class_name="QCheckBox", tooltip=""),
        _FakeWidget(class_name="QCheckBox", tooltip=""),
    ]
    missing = apply_required_tooltips(
        [
            (widgets[0], "How color is applied across points: one swatch, per-row swatches, or a 3D field."),
            (widgets[1], ADD_POINT_TIP, ADD_POINT_HEADER_LABEL),
            (widgets[2], ADD_CLIP_TIP, "Add clip plane"),
            (widgets[3], SNAP_TO_ATOM_TIP),
            (widgets[4], HOOK_TO_SELECTION_TIP),
            (widgets[5], ZOOM_TO_SELECTION_TIP),
        ],
        context="SharedEditorSections",
    )
    assert missing == []
    assert "Add Camera" in ADD_POINT_TIP
    assert insertion_add_label(14) == ADD_POINT_HEADER_LABEL
    assert insertion_add_label(1, "sele") == ADD_POINT_HEADER_LABEL
    assert SNAP_LABEL == "Snap to atoms"
    assert HOOK_LABEL == "Anchor to atoms"
    assert ZOOM_LABEL == "Zoom to new points"
    assert "clip" in ADD_CLIP_TIP.lower()
    assert "camera" in ADD_POINT_TIP.lower()


@pytest.mark.qt
def test_arrow_endpoint_action_tooltips():
    from pymolviz.wizards.builders.arrow_list import (
        ATTACH_LABEL,
        CAMERA_ENDPOINT_TIP,
        PICK_END_TIP,
        PICK_START_TIP,
    )

    pick = _FakeWidget(class_name="QPushButton", tooltip="")
    cam = _FakeWidget(class_name="QPushButton", tooltip="")
    attach = _FakeWidget(class_name="QCheckBox", tooltip="")
    missing = apply_required_tooltips(
        [
            (pick, PICK_START_TIP, "Pick atom"),
            (cam, CAMERA_ENDPOINT_TIP, "Use camera center"),
            (attach, "Keep the start linked to this atom so the arrow follows if the atom moves.", ATTACH_LABEL),
        ],
        context="ArrowPairEditor",
    )
    assert missing == []
    assert "pick" in pick.toolTip().lower()
    assert "camera" in cam.toolTip().lower()
    assert "atom" in attach.toolTip().lower()
    assert "endpoint" in PICK_END_TIP.lower()


@pytest.mark.qt
def test_field_library_layout_copy_and_tooltips():
    from pymolviz.wizards.catalog import NEST_ELL, nest_connector
    from pymolviz.wizards.field_visuals import (
        ADD_FIELD_BUTTON,
        ADD_FIELD_TIP,
        ADD_VISUAL_BUTTON,
        ADD_VISUAL_TIP,
        FIELD_ADD_VISUAL_SPAN,
        FIELD_COLUMNS,
        SYMMETRIZE_BUTTON,
        SYMMETRIZE_TIP,
    )
    from pymolviz.wizards.widgets.nest import make_nest_branch

    edit = _FakeWidget(class_name="QPushButton", tooltip="")
    delete = _FakeWidget(class_name="QPushButton", tooltip="")
    add_field = _FakeWidget(class_name="QPushButton", tooltip="")
    add_visual = _FakeWidget(class_name="QPushButton", tooltip="")
    symmetrize = _FakeWidget(class_name="QPushButton", tooltip="")
    missing = apply_required_tooltips(
        [
            (edit, "Open the builder for this visual.", "Edit"),
            (delete, "Remove this field or visual from the session.", "Delete"),
            (add_field, ADD_FIELD_TIP, ADD_FIELD_BUTTON),
            (add_visual, ADD_VISUAL_TIP, ADD_VISUAL_BUTTON),
            (symmetrize, SYMMETRIZE_TIP, SYMMETRIZE_BUTTON),
        ],
        context="FieldVisualsWindow",
    )
    assert missing == []
    assert FIELD_COLUMNS[0] == "Name"
    assert FIELD_COLUMNS[1] == "Kind"
    assert "Geometry" in FIELD_COLUMNS[2]
    assert FIELD_COLUMNS[4] == "Edit"
    assert FIELD_COLUMNS[5] == "Delete"
    assert "lattice" in SYMMETRIZE_TIP.lower() or "crystal" in SYMMETRIZE_TIP.lower()
    assert SYMMETRIZE_BUTTON == "Symmetrize"
    assert FIELD_ADD_VISUAL_SPAN == 3
    assert nest_connector({"depth": 1, "last_child": True}) == NEST_ELL
    from pymolviz.wizards.catalog import NEST_LINE_RGB, kind_badge_rgb
    from pymolviz.wizards.widgets.catalog_chrome import ADD_FIELD_STYLE, ADD_VISUAL_STYLE
    from pymolviz.wizards.widgets.sticky_add import ADD_BUTTON_STYLE
    from pymolviz.wizards.widgets.theme import (
        HEADER,
        PRIMARY,
        SELECTED,
        action_bar_css,
        catalog_table_css,
        section_css,
        type_card_css,
        wizard_page_css,
    )

    assert "dashed" in ADD_VISUAL_STYLE
    assert "42, 130, 236" in ADD_FIELD_STYLE
    assert "42, 130, 236" in ADD_BUTTON_STYLE
    assert PRIMARY == (42, 130, 236)
    assert HEADER == (205, 226, 240)
    assert SELECTED == (219, 238, 249)
    assert "205, 226, 240" in section_css()
    assert "pmvSectionActions" in section_css()
    assert "219, 238, 249" in catalog_table_css()
    assert "245, 249, 251" in wizard_page_css()
    combo = wizard_page_css().split("QComboBox {", 1)[1].split("}", 1)[0]
    assert "border: 1px solid" in combo
    assert "42, 130, 236" in type_card_css()
    assert "196, 210, 220" in action_bar_css()
    fill, _ink = kind_badge_rgb("IsoSurface")
    assert fill[0] > fill[2]
    assert NEST_LINE_RGB[2] - NEST_LINE_RGB[0] < 30
    import pymolviz.wizards.widgets.nest as nest_mod

    original = nest_mod.qt_modules
    nest_mod.qt_modules = lambda: (None, None, None)
    try:
        assert make_nest_branch({"depth": 1, "last_child": True}) is None
    finally:
        nest_mod.qt_modules = original

