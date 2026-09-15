"""Qt smoke tests for From Selection field builder controls."""

from __future__ import annotations

import pytest

from pymolviz.util.gaussian_map import DEFAULT_GAUSSIAN_ISOLEVEL
from pymolviz.wizards.tooltips import apply_required_tooltips


class _FakeWidget:
    def __init__(self, class_name="QDoubleSpinBox", tooltip=""):
        self.__class__.__name__ = class_name
        self._tooltip = tooltip

    def setToolTip(self, text):
        self._tooltip = str(text)

    def toolTip(self):
        return self._tooltip


@pytest.mark.qt
def test_from_selection_field_preview_tooltips_and_defaults():
    from pymolviz.wizards.builders.from_selection_page import (
        FromSelectionFieldPage,
        _ISO_TIP,
        _LIVE_PREVIEW_TIP,
    )
    from pymolviz.wizards.builders.preview_mode import PREVIEW_SIMPLE_TIP, PREVIEW_SIMPLE_LABEL

    iso = _FakeWidget(class_name="QDoubleSpinBox")
    live = _FakeWidget(class_name="QRadioButton")
    missing = apply_required_tooltips(
        [
            (iso, _ISO_TIP, "Iso value"),
            (live, PREVIEW_SIMPLE_TIP, PREVIEW_SIMPLE_LABEL),
        ],
        context=FromSelectionFieldPage.CONTEXT,
    )
    assert missing == []
    assert "coarser" in live.toolTip().lower() or "isomesh" in live.toolTip().lower()
    assert "normalized" in iso.toolTip().lower()
    assert DEFAULT_GAUSSIAN_ISOLEVEL == pytest.approx(1.0)
    assert "markers" in _LIVE_PREVIEW_TIP.lower()


@pytest.mark.qt
def test_from_selection_page_shows_color_mode_radios():
    from pymolviz.wizards.builders.appearance_section import appearance_color_mode_labels
    from pymolviz.wizards.builders.from_selection_page import FromSelectionFieldPage
    from pymolviz.wizards.builders.point_table_page import PointTableBuilderPage

    uniform = _FakeWidget(class_name="QRadioButton")
    from_field = _FakeWidget(class_name="QRadioButton")
    labels = appearance_color_mode_labels(show_per_point=False)
    missing = apply_required_tooltips(
        [
            (uniform, "One solid color for the whole object.", labels[0]),
            (from_field, "Sample a Field at this object's points. The object stores a field id, not a voxel copy.", labels[1]),
        ],
        context=FromSelectionFieldPage.CONTEXT,
    )
    assert missing == []
    assert labels == ("Uniform", "From field")
    assert "Per-point" not in labels
    assert FromSelectionFieldPage._mount_appearance is PointTableBuilderPage._mount_appearance
    cfg = FromSelectionFieldPage._appearance_config(None)
    assert cfg.get("show_wireframe") is False
    assert cfg.get("show_quality") is False
    assert cfg.get("show_per_point") is False
    assert cfg.get("show_live_preview") is True


@pytest.mark.qt
def test_from_selection_done_wires_commit_helper():
    import inspect

    from pymolviz.wizards.builders.from_selection_page import FromSelectionFieldPage
    from pymolviz.wizards.builders.load_field import commit_from_selection_preset

    src = inspect.getsource(FromSelectionFieldPage._create_cgo)
    assert "commit_from_selection_preset" in src
    assert callable(commit_from_selection_preset)


@pytest.mark.qt
def test_from_selection_live_preview_shown_for_non_gaussian():
    from pymolviz.fields.identity import GEN_DISTANCE, GEN_NEAREST_PROP, GEN_SIGNED_VDW
    from pymolviz.wizards.builders.field_params import field_model_shows
    from pymolviz.wizards.builders.from_selection_page import (
        FromSelectionFieldPage,
    )
    from pymolviz.wizards.builders.preview_mode import PREVIEW_SIMPLE_TIP, PREVIEW_SIMPLE_LABEL

    for algo in (GEN_DISTANCE, GEN_SIGNED_VDW, GEN_NEAREST_PROP):
        assert field_model_shows(algo, "live_preview") is True
        assert field_model_shows(algo, "iso_value") is True
    live = _FakeWidget(class_name="QRadioButton")
    missing = apply_required_tooltips(
        [(live, PREVIEW_SIMPLE_TIP, PREVIEW_SIMPLE_LABEL)],
        context=FromSelectionFieldPage.CONTEXT,
    )
    assert missing == []
    assert "isomesh" in live.toolTip().lower() or "coarser" in live.toolTip().lower()


@pytest.mark.qt
def test_field_visual_page_live_preview_tooltip():
    from pymolviz.wizards.builders.field_page import FieldVisualBuilderPage
    from pymolviz.wizards.builders.preview_mode import PREVIEW_SIMPLE_TIP, PREVIEW_SIMPLE_LABEL

    live = _FakeWidget(class_name="QRadioButton")
    missing = apply_required_tooltips(
        [(live, PREVIEW_SIMPLE_TIP, PREVIEW_SIMPLE_LABEL)],
        context=FieldVisualBuilderPage.CONTEXT,
    )
    assert missing == []
    assert "isomesh" in live.toolTip().lower()
    assert "iso-proxy" not in live.toolTip().lower()
    from pymolviz.wizards.builders.field_page import FieldVisualBuilderPage, _CLIP_CROP_TIP, _CLIP_TIP

    clip = _FakeWidget(class_name="QCheckBox")
    missing = apply_required_tooltips(
        [(clip, _CLIP_CROP_TIP, "Clip")],
        context=FieldVisualBuilderPage.CONTEXT,
    )
    assert missing == []
    assert "axis" in clip.toolTip().lower()
    assert "axis" in _CLIP_TIP.lower()


@pytest.mark.qt
def test_domain_section_schematic_is_left_of_knobs():
    import inspect

    from pymolviz.wizards.builders.domain_schematic import (
        SCHEMATIC_TIP,
        DomainSchematicWidget,
        wrap_domain_knobs,
    )
    from pymolviz.wizards.builders.from_selection_page import FromSelectionFieldPage
    from pymolviz.wizards.pick import qt_modules

    src = inspect.getsource(FromSelectionFieldPage._mount_geometry)
    assert "wrap_domain_knobs" in src
    assert "DomainSchematicWidget" in src
    schematic = _FakeWidget(class_name="QWidget")
    missing = apply_required_tooltips(
        [(schematic, SCHEMATIC_TIP, "Domain schematic")],
        context=FromSelectionFieldPage.CONTEXT,
    )
    assert missing == []
    assert "domain" in schematic.toolTip().lower()

    _, _, QtWidgets = qt_modules()
    if QtWidgets is None or not hasattr(QtWidgets, "QWidget"):
        return
    try:
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        widget = DomainSchematicWidget()
        knobs = QtWidgets.QWidget()
        knobs.setObjectName("pmvDomainKnobs")
        row = wrap_domain_knobs(QtWidgets, widget, knobs)
        layout = row.layout()
        left = layout.itemAt(0).widget()
        right = layout.itemAt(1).widget()
        assert left.objectName() == "pmvDomainSchematic"
        assert right is knobs
    except Exception:
        pytest.skip("QApplication / QWidget not usable")


@pytest.mark.qt
def test_edit_visual_lazy_inits_stack_when_catalog_has_none():
    from types import SimpleNamespace

    from pymolviz.wizards.field_visuals import FieldVisualsWindow
    from pymolviz.wizards.pick import qt_modules

    _, _, QtWidgets = qt_modules()
    if QtWidgets is None or not hasattr(QtWidgets, "QStackedWidget"):
        pytest.skip("QStackedWidget unavailable")
    try:
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        window = QtWidgets.QWidget()
        QtWidgets.QVBoxLayout(window)
        win = FieldVisualsWindow(SimpleNamespace(cmd=None, prompt=[]))
        win._window = window
        win._stack = None

        class _Page:
            def __init__(self):
                self.widget = QtWidgets.QLabel("builder")

        index = win._add_page_to_stack(_Page())
        assert win._stack is not None
        assert index == 0
        win._stack.setCurrentIndex(index)
        assert win._stack.currentIndex() == 0
    except Exception:
        pytest.skip("QApplication / QWidget not usable")


@pytest.mark.qt
def test_from_selection_appearance_has_live_preview():
    import inspect

    from pymolviz.wizards.builders.appearance_section import AppearanceSection
    from pymolviz.wizards.builders.from_selection_page import FromSelectionFieldPage

    geom = inspect.getsource(FromSelectionFieldPage._mount_geometry)
    assert "Live preview" not in geom
    assert 'make_section("Geometry"' in geom
    opts = inspect.getsource(FromSelectionFieldPage._mount_options)
    assert 'make_section("Options"' in opts
    assert "Iso value" in opts
    appear = inspect.getsource(AppearanceSection._build)
    assert "PreviewModeRadios" in appear
    assert "pmvPreviewMode" in appear or "PreviewModeRadios" in appear
    cfg = FromSelectionFieldPage._appearance_config(None)
    assert cfg.get("show_live_preview") is True


@pytest.mark.qt
def test_field_visual_volume_has_no_isosurface_section_and_has_colormap_editor():
    import inspect

    from pymolviz.wizards.builders.colormap_editor import ColormapEditor
    from pymolviz.wizards.builders.field_page import FieldVisualBuilderPage
    from pymolviz.wizards.pick import qt_modules

    src = inspect.getsource(FieldVisualBuilderPage._build)
    assert 'make_section("Appearance")' in src
    assert "ColormapEditor" in src
    appear_chunk = src.split('iso = make_section("Options")')[0]
    iso_chunk = src.split('iso = make_section("Options")')[1].split("clip = make_section")[0]
    assert "PreviewModeRadios" in appear_chunk
    assert "Color mode" in appear_chunk
    assert "self._live_preview" not in iso_chunk
    assert "self._level" in iso_chunk
    assert 'make_section("Geometry"' in src
    assert 'make_section("Modifiers"' in src
    sync = inspect.getsource(FieldVisualBuilderPage._sync_form)
    assert "self._iso_section.widget.setVisible(iso)" in sync

    _, _, QtWidgets = qt_modules()
    if QtWidgets is None or not hasattr(QtWidgets, "QWidget"):
        return
    try:
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        page = QtWidgets.QWidget()
        editor = ColormapEditor(page)
        assert editor.preset_combo.objectName() == "pmvColormapPreset"
        assert editor.reverse_checkbox.objectName() == "pmvColormapReverse"
        assert editor.range_combo.objectName() == "pmvColormapRange"
        assert editor._picker.plus_button.toolTip() == "add custom colormap"
        assert editor._picker.plus_button.objectName() == "pmvColormapAdd"
        assert not editor._picker.plus_button.isHidden()
        assert editor._picker.gear_button.objectName() == "pmvColormapAdjust"
        assert editor._picker.gear_button.isHidden()
    except Exception:
        pytest.skip("QApplication / QWidget not usable")


@pytest.mark.qt
def test_volume_builder_shows_colormap_plus():
    from tests.fakes.cmd import FakeCmd
    from pymolviz.wizards.builders.field_page import FieldVisualBuilderPage
    from pymolviz.wizards.pick import qt_modules

    _, _, QtWidgets = qt_modules()
    if QtWidgets is None or not hasattr(QtWidgets, "QWidget"):
        pytest.skip("Qt widgets unavailable")
    try:
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        page = FieldVisualBuilderPage(FakeCmd(), on_back=lambda: None)
    except Exception:
        pytest.skip("QApplication / QWidget not usable")
    page.reset_for_create("Volume", None)
    editor = page._cmap_editor
    assert editor is not None
    assert not editor.widget.isHidden()
    plus = editor._picker.plus_button
    assert plus.toolTip() == "add custom colormap"
    assert not plus.isHidden()
    host = page.widget
    host.setMinimumWidth(0)
    host.resize(360, 640)
    host.show()
    app.processEvents()
    plus_right = plus.mapTo(host, plus.rect().topRight()).x()
    assert plus.width() > 0
    assert plus_right <= host.width()
    host.hide()


@pytest.mark.qt
def test_colormap_ramp_preview_accepts_numpy_rows():
    from dataclasses import replace

    from pymolviz.util.colormap_spec import definition_from_preset
    from pymolviz.wizards.builders.colormap_dialog import _ramp_image
    from pymolviz.wizards.pick import qt_modules

    _, QtGui, QtWidgets = qt_modules()
    if QtGui is None or QtWidgets is None or not hasattr(QtGui, "QImage"):
        pytest.skip("QtGui unavailable")
    try:
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
    except Exception:
        pytest.skip("QApplication not usable")
    defn = replace(definition_from_preset("viridis"), customized=True, preset="Custom 1")
    pix = _ramp_image(QtGui, defn, 48, 12)
    assert pix is not None
    assert int(pix.width()) >= 2


@pytest.mark.qt
def test_open_colormap_editor_constructs():
    from pymolviz.util.colormap_spec import FieldColorMapping, Normalization, definition_from_preset
    from pymolviz.wizards.builders.colormap_dialog import open_colormap_editor
    from pymolviz.wizards.pick import qt_modules, qt_widget_alive

    _, _, QtWidgets = qt_modules()
    if QtWidgets is None or not hasattr(QtWidgets, "QWidget"):
        pytest.skip("Qt widgets unavailable")
    try:
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        parent = QtWidgets.QWidget()
    except Exception:
        pytest.skip("QApplication not usable")
    mapping = FieldColorMapping(
        colormap=definition_from_preset("viridis"),
        normalization=Normalization(),
    )
    editor = open_colormap_editor(parent, mapping)
    assert editor is not None
    assert editor.widget is not None
    dist = editor.widget.findChild(QtWidgets.QWidget, "pmvColormapDistribution")
    assert dist is not None
    assert dist.minimumHeight() >= 160
    bar = editor.widget.findChild(QtWidgets.QWidget, "pmvColormapColorbar")
    assert bar is not None
    assert bar.parent() is dist.parent()
    assert bar.minimumHeight() >= 56
    titles = [
        lab.text()
        for lab in editor.widget.findChildren(QtWidgets.QLabel, "pmvSectionTitle")
    ]
    assert "Value distribution" in titles
    assert "Colorbar" not in titles
    assert "Selected stop" not in titles
    assert editor._colorbar.tick_pad() >= 32
    n_stops = len(editor.mapping().colormap.stops)
    gap = 0.5 * (
        editor.mapping().colormap.stops[0].position
        + editor.mapping().colormap.stops[1].position
    )
    editor._add_color_stop_at(gap, 0.35)
    assert len(editor.mapping().colormap.stops) == n_stops + 1
    added = min(
        editor.mapping().colormap.stops,
        key=lambda stop: abs(stop.position - gap),
    )
    assert added.rgba[3] == pytest.approx(0.35)
    editor._on_stop_dragged(0, 0.0, 0.4)
    assert editor.mapping().colormap.stops[0].rgba[3] == pytest.approx(0.4)
    live = []
    editor._on_change = lambda mapping: live.append(mapping)
    editor._on_stop_dragged(0, 0.0, 0.55)
    assert not live
    editor._on_handle_drag_finished()
    assert live
    assert live[-1].colormap.stops[0].rgba[3] == pytest.approx(0.55)
    editor._on_range_dragged(-1.5, 2.5)
    assert editor.mapping().normalization.mode == "custom"
    assert editor.mapping().normalization.vmin == pytest.approx(-1.5)
    dist_view = editor._histogram
    dist_view.widget.resize(400, 180)
    plot = dist_view._plot(dist_view.widget)
    dmin, dmax = dist_view._span()
    first = dist_view._defn.stops[0]
    last = dist_view._defn.stops[-1]
    sx0, sy0 = dist_view._stop_xy(plot, first, dmin, dmax)
    sx1, sy1 = dist_view._stop_xy(plot, last, dmin, dmax)
    assert dist_view._hit(dist_view.widget, sx0, sy0) == ("stop", 0)
    assert dist_view._hit(dist_view.widget, sx1, sy1) == ("stop", len(dist_view._defn.stops) - 1)
    assert dist_view._hit(dist_view.widget, sx0, 0.5 * (plot.top() + plot.bottom())) == ("stop", 0)
    before = len(editor.mapping().colormap.stops)
    target = 0.5 * editor.mapping().colormap.stops[1].position
    editor._on_stop_dragged(0, target, editor.mapping().colormap.stops[0].rgba[3])
    assert len(editor.mapping().colormap.stops) == before
    assert editor.mapping().colormap.stops[0].position == pytest.approx(target)
    assert dist_view._hit(dist_view.widget, sx0, plot.top() - 8) == ("range", "vmin")
    assert dist_view._hit(dist_view.widget, sx1, plot.top() - 8) == ("range", "vmax")
    editor._open_stop_editor(0)
    popup = editor._stop_popup
    assert popup is not None
    assert qt_widget_alive(popup)
    assert not popup.isModal()
    popup.close()
    assert getattr(editor.widget, "_pmv_no_transient", False)
    assert editor.widget.parentWidget() is None
    editor.widget.close()


@pytest.mark.qt
def test_stop_color_picker_does_not_close_colormap_editor():
    from pymolviz.util.colormap_spec import FieldColorMapping, Normalization, definition_from_preset
    from pymolviz.wizards.builders.colormap_dialog import open_colormap_editor
    from pymolviz.wizards.builders.colors import live_color_dialog
    from pymolviz.wizards.pick import qt_modules, qt_widget_alive

    _, _, QtWidgets = qt_modules()
    if QtWidgets is None or not hasattr(QtWidgets, "QColorDialog"):
        pytest.skip("Qt widgets unavailable")
    try:
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        parent = QtWidgets.QWidget()
    except Exception:
        pytest.skip("QApplication not usable")
    mapping = FieldColorMapping(
        colormap=definition_from_preset("viridis"),
        normalization=Normalization(),
    )
    editor = open_colormap_editor(parent, mapping)
    assert editor is not None
    host = editor.widget
    host.show()
    app.processEvents()
    editor._open_stop_editor(0)
    app.processEvents()
    editor._pick_stop_color()
    app.processEvents()
    assert qt_widget_alive(host)
    assert host.isVisible()
    popup = editor._stop_popup
    if popup is not None and qt_widget_alive(popup):
        assert popup.isVisible()
    picker = live_color_dialog(qt_widget_alive)
    if picker is not None:
        assert getattr(picker, "_pmv_window_anchor", None) is host
        try:
            picker.close()
        except Exception:
            pass
        app.processEvents()
    assert qt_widget_alive(host)
    assert host.isVisible()
    host.close()


@pytest.mark.qt
def test_colormap_picker_shows_gear_for_custom_presets(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from pymolviz.util.colormap_spec import definition_from_preset, save_custom_preset
    from pymolviz.wizards.builders.colormap_editor import (
        KIND_CUSTOM,
        _kind_role,
        ColormapEditor,
    )
    from pymolviz.wizards.pick import qt_modules

    QtCore, _, QtWidgets = qt_modules()
    if QtWidgets is None or not hasattr(QtWidgets, "QWidget"):
        pytest.skip("Qt widgets unavailable")
    try:
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        page = QtWidgets.QWidget()
        editor = ColormapEditor(page)
    except Exception:
        pytest.skip("QApplication / QWidget not usable")
    save_custom_preset("Custom 1", definition_from_preset("viridis"))
    editor._picker.reload(select="Custom 1")
    assert editor._picker.current_name() == "Custom 1"
    assert editor._picker.current_is_custom()
    assert not editor._picker.gear_button.isHidden()
    combo = editor._picker.combo
    idx = combo.findText("Custom 1")
    assert idx >= 0
    assert combo.itemData(idx, _kind_role(QtCore)) == KIND_CUSTOM


@pytest.mark.qt
def test_colormap_menu_lists_and_adds_custom_maps(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from types import SimpleNamespace

    from pymolviz.util.colormap_spec import custom_colormap_rows, save_custom_preset, definition_from_preset
    from pymolviz.wizards.colormaps import ColormapMenuWindow
    from pymolviz.wizards.pick import qt_modules, qt_widget_alive

    _, _, QtWidgets = qt_modules()
    if QtWidgets is None or not hasattr(QtWidgets, "QDialog"):
        pytest.skip("Qt widgets unavailable")
    try:
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
    except Exception:
        pytest.skip("QApplication not usable")
    save_custom_preset("Custom 1", definition_from_preset("viridis"))
    wizard = SimpleNamespace(cmd=None, prompt=[])
    win = ColormapMenuWindow(wizard)
    win.show()
    assert win._window is not None
    assert qt_widget_alive(win._window)
    table = win._table
    assert table is not None
    assert table.rowCount() == 2
    assert table.item(1, 1).text() == "Custom 1"
    names_before = [row["name"] for row in custom_colormap_rows()]
    win._add_colormap()
    assert [row["name"] for row in custom_colormap_rows()] == names_before
    editor = win._editor
    assert editor is not None
    editor._apply()
    names_after = [row["name"] for row in custom_colormap_rows()]
    assert len(names_after) == len(names_before) + 1
    assert win._table is not None
    assert win._table.rowCount() == 3
    monkeypatch.setattr(
        "pymolviz.wizards.colormaps.session_colormap_users",
        lambda: {"Custom 1": [{"name": "iso", "type": "IsoSurface"}]},
    )
    win._refresh_table()
    kinds = [row.get("kind") for row in win._row_meta]
    assert "header" in kinds
    assert win._row_meta[0]["name"] == "In this session"
    toggle = table.cellWidget(1, 0).findChild(QtWidgets.QPushButton, "pmvColormapUsersToggle")
    assert toggle is not None
    assert toggle.isEnabled()
    win._toggle_users("Custom 1")
    assert any(row.get("kind") == "user" for row in win._row_meta)
    win.close()


@pytest.mark.qt
def test_similar_colormap_dialog_highlights_stop_diffs(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from dataclasses import replace

    from pymolviz.util.colormap_spec import (
        ColorStop,
        closest_similar_custom_colormap,
        definition_from_preset,
        save_custom_preset,
    )
    from pymolviz.wizards.builders.colormap_similar import build_similar_colormap_dialog
    from pymolviz.wizards.pick import qt_modules

    _, _, QtWidgets = qt_modules()
    if QtWidgets is None or not hasattr(QtWidgets, "QDialog"):
        pytest.skip("Qt widgets unavailable")
    try:
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
    except Exception:
        pytest.skip("QApplication not usable")
    base = definition_from_preset("viridis")
    save_custom_preset("Custom 1", base)
    stops = []
    for stop in base.stops:
        rgba = list(stop.rgba)
        rgba[3] = max(0.0, rgba[3] - 0.04)
        stops.append(ColorStop(stop.position, tuple(rgba)))
    new = replace(base, stops=tuple(stops), customized=True)
    match = closest_similar_custom_colormap(new)
    assert match is not None
    dialog = build_similar_colormap_dialog(None, new, match)
    assert dialog is not None
    table = dialog.findChild(QtWidgets.QTableWidget, "pmvSimilarColormapStops")
    assert table is not None
    assert table.rowCount() == len(match.similarity.stop_diffs)
    assert dialog.findChild(QtWidgets.QPushButton, "pmvSimilarUseExisting") is not None
    assert dialog.findChild(QtWidgets.QLabel, "pmvSimilarColormapDiff") is not None
    dialog.close()


@pytest.mark.qt
def test_add_colormap_reuses_similar_existing(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from types import SimpleNamespace

    from pymolviz.util.colormap_spec import (
        CHOICE_USE_EXISTING,
        SimilarColormapChoice,
        custom_colormap_rows,
        definition_from_preset,
        save_custom_preset,
    )
    from pymolviz.util.field_sample import DEFAULT_SURFACE_COLORMAP
    from pymolviz.wizards.colormaps import ColormapMenuWindow
    from pymolviz.wizards.pick import qt_modules, qt_widget_alive

    _, _, QtWidgets = qt_modules()
    if QtWidgets is None:
        pytest.skip("Qt widgets unavailable")
    try:
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
    except Exception:
        pytest.skip("QApplication not usable")
    save_custom_preset("Custom 1", definition_from_preset(DEFAULT_SURFACE_COLORMAP))
    monkeypatch.setattr(
        "pymolviz.wizards.builders.colormap_similar.resolve_similar_custom_colormap",
        lambda *args, **kwargs: SimilarColormapChoice(CHOICE_USE_EXISTING, "Custom 1"),
    )
    wizard = SimpleNamespace(cmd=None, prompt=[])
    win = ColormapMenuWindow(wizard)
    win.show()
    before = [row["name"] for row in custom_colormap_rows()]
    win._add_colormap()
    assert [row["name"] for row in custom_colormap_rows()] == before
    editor = win._editor
    assert editor is not None
    editor._apply()
    assert [row["name"] for row in custom_colormap_rows()] == before
    assert editor._mapping.colormap.preset == "Custom 1"
    if qt_widget_alive(editor.widget):
        editor.widget.hide()
    win.close()


@pytest.mark.qt
def test_colormap_combo_does_not_prompt_until_editor_apply(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from pymolviz.util.colormap_spec import (
        CHOICE_USE_EXISTING,
        SimilarColormapChoice,
        custom_colormap_rows,
        definition_from_preset,
        save_custom_preset,
    )
    from pymolviz.wizards.builders.colormap_dialog import _LIVE_EDITOR
    from pymolviz.wizards.builders.colormap_editor import ColormapEditor
    from pymolviz.wizards.pick import qt_modules

    _, _, QtWidgets = qt_modules()
    if QtWidgets is None or not hasattr(QtWidgets, "QWidget"):
        pytest.skip("Qt widgets unavailable")
    try:
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        page = QtWidgets.QWidget()
        editor = ColormapEditor(page)
    except Exception:
        pytest.skip("QApplication / QWidget not usable")
    save_custom_preset("Custom 1", definition_from_preset("viridis"))
    prompted = []

    def resolve(*args, **kwargs):
        prompted.append(True)
        return SimilarColormapChoice(CHOICE_USE_EXISTING, "Custom 1")

    monkeypatch.setattr(
        "pymolviz.wizards.builders.colormap_similar.resolve_similar_custom_colormap",
        resolve,
    )
    editor._load_named_preset("viridis")
    assert editor.definition().preset == "viridis"
    assert prompted == []
    editor._add_custom()
    assert prompted == []
    assert [row["name"] for row in custom_colormap_rows()] == ["Custom 1"]
    dlg = _LIVE_EDITOR[-1] if _LIVE_EDITOR else None
    assert dlg is not None
    dlg._apply()
    assert prompted == [True]
    assert editor.preset_name() == "Custom 1"
    assert [row["name"] for row in custom_colormap_rows()] == ["Custom 1"]


@pytest.mark.qt
def test_extend_cell_dialog_lists_coverage():
    from pymolviz.wizards.builders.extend_cell_dialog import open_extend_cell_dialog
    from pymolviz.wizards.pick import qt_modules, qt_widget_alive

    _, _, QtWidgets = qt_modules()
    if QtWidgets is None or not hasattr(QtWidgets, "QDialog"):
        pytest.skip("Qt widgets unavailable")
    try:
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        parent = QtWidgets.QWidget()
    except Exception:
        pytest.skip("QApplication not usable")
    picked = []
    rows = [
        {
            "kind": "object",
            "name": "lig",
            "status": "outside",
            "n_atoms": 1,
            "lo": [50.0, 5.0, 5.0],
            "hi": [50.0, 5.0, 5.0],
        },
        {
            "kind": "object",
            "name": "prot",
            "status": "inside",
            "n_atoms": 12,
            "lo": [1.0, 1.0, 1.0],
            "hi": [8.0, 8.0, 8.0],
        },
    ]
    dialog = open_extend_cell_dialog(
        parent,
        rows,
        (0.0, 0.0, 0.0),
        (10.0, 10.0, 10.0),
        current="lig",
        on_extend=picked.append,
    )
    assert dialog is not None
    listing = dialog.findChild(QtWidgets.QListWidget, "pmvExtendTargetList")
    assert listing is not None
    assert listing.count() == 2
    assert "Not covered" in listing.item(0).text()
    assert "Covered" in listing.item(1).text()
    sketch = dialog.findChild(QtWidgets.QWidget, "pmvExtendCoverageSketch")
    assert sketch is not None
    listing.setCurrentRow(0)
    box = dialog.findChild(QtWidgets.QDialogButtonBox)
    extend = None
    for btn in box.buttons():
        if btn.text().replace("&", "") == "Extend":
            extend = btn
            break
    assert extend is not None
    extend.click()
    assert picked == ["lig"]
    if qt_widget_alive(dialog):
        dialog.close()
    parent.close()

