"""Copy and column layout for builder pages (no Qt required)."""

from pymolviz.wizards.builders.anchor_table import (
    ANCHOR_COL,
    COLOR_COL,
    COORDINATE_COLS,
    ENABLED_COL,
    NAME_COL,
    POINT_ANCHOR_COL,
    POINT_COLOR_COL,
    POINT_ENABLED_COL,
    POINT_NAME_COL,
    POINT_SOURCE_COL,
    POINT_X_COL,
    POINT_Y_COL,
    POINT_Z_COL,
    SOURCE_COL,
    SURFACE_RADIUS_COL,
    point_columns,
    surface_point_columns,
)
from pymolviz.wizards.builders.point_insertion import (
    ADD_POINT_HEADER_LABEL,
    EXPORT_SEL_LABEL,
    HOOK_LABEL,
    SHOW_COORDS_LABEL,
    SNAP_LABEL,
    ZOOM_LABEL,
)
from pymolviz.wizards.builders.points import INSERTION_NOTHING_SELECTED
from pymolviz.wizards.builders.surface_params import (
    ALGORITHM_CONTROL_VISIBILITY,
    algorithm_shows_probe,
)
from pymolviz.wizards.tooltips import HOOK_TO_SELECTION_TIP, SNAP_TO_ATOM_TIP
from pymolviz.wizards.widgets.action_bar import (
    DONE_LABEL,
    DONE_TIP,
    EXPORT_SCRIPT_LABEL,
    EXPORT_SCRIPT_TIP,
    UPDATE_IN_PYMOL_LABEL,
)
from pymolviz.wizards.widgets.name_section import (
    NAME_LABEL,
    NAME_PLACEHOLDER,
    NAME_TIP,
)
from pymolviz.wizards.widgets.breadcrumb import (
    CRUMB_ADD_OBJECT,
    CRUMB_SPHERES,
    CRUMB_VISUALS,
    breadcrumb_html,
    breadcrumb_text,
    create_crumbs,
    edit_crumbs,
)


def test_point_list_headers_match_arrow_family():
    from pymolviz.wizards.builders.spatial_item_list import (
        ARROW_LIST_COLUMNS,
        POINT_LIST_COLUMNS,
    )

    point_titles = [title for title, _width in POINT_LIST_COLUMNS if title]
    arrow_titles = [title for title, _width in ARROW_LIST_COLUMNS if title]
    assert POINT_LIST_COLUMNS[0][0] == ""
    assert point_titles[0] == "#"
    assert point_titles[1] == "Enabled"
    assert "Label" in point_titles
    assert "Source / Status" in point_titles
    assert "X (Å)" in point_titles
    assert "Color" in point_titles
    assert "Delete" in point_titles
    assert arrow_titles[0] == "#"
    assert "Start point" in arrow_titles
    assert "End point" in arrow_titles
    assert "Source / Status" in arrow_titles
    assert "Color" in arrow_titles
    assert "Delete" in arrow_titles


def test_point_table_headers_separate_color_from_label():
    cols = point_columns()
    assert cols[POINT_ENABLED_COL] == ENABLED_COL == "Enabled"
    assert cols[POINT_ANCHOR_COL] == ANCHOR_COL == "Anchor"
    assert cols[POINT_COLOR_COL] == COLOR_COL == "Color"
    assert cols[POINT_NAME_COL] == NAME_COL == "Atom / Label"
    assert cols[POINT_SOURCE_COL] == SOURCE_COL == "Source"
    assert cols[POINT_X_COL] == "X"
    assert cols[POINT_Y_COL] == "Y"
    assert cols[POINT_Z_COL] == "Z"
    assert COORDINATE_COLS == (POINT_X_COL, POINT_Y_COL, POINT_Z_COL)


def test_surface_table_keeps_radius_after_color_column():
    cols = surface_point_columns()
    assert cols[SURFACE_RADIUS_COL] == "R (Å)"
    assert cols[POINT_NAME_COL] == "Atom / Label"


def test_point_option_labels_distinguish_snap_from_attach():
    import inspect
    from pymolviz.wizards.builders.point_insertion import PointInsertionWidget

    assert SNAP_LABEL == "Snap to atoms"
    assert HOOK_LABEL == "Anchor to atoms"
    assert ZOOM_LABEL == "Zoom to new points"
    assert SNAP_LABEL != HOOK_LABEL
    assert EXPORT_SEL_LABEL == "Create PyMOL selection"
    assert SHOW_COORDS_LABEL == "Show coordinates"
    flags = inspect.getsource(PointInsertionWidget._build)
    assert 'icon="snap"' in flags
    assert 'icon="anchor"' in flags
    assert 'icon="zoom"' in flags


def test_snap_and_attach_tooltips_explain_placement_vs_follow():
    assert "placement" in SNAP_TO_ATOM_TIP.lower()
    assert "anchored" in HOOK_TO_SELECTION_TIP.lower()
    assert "follow" in HOOK_TO_SELECTION_TIP.lower()


def test_action_bar_commit_labels_are_outcome_oriented():
    assert DONE_LABEL == "Done"
    assert UPDATE_IN_PYMOL_LABEL == "Update in PyMOL"
    assert "CGO" not in DONE_LABEL
    assert "CGO" not in UPDATE_IN_PYMOL_LABEL
    assert "export" in EXPORT_SCRIPT_LABEL.lower()
    assert "script" in EXPORT_SCRIPT_TIP.lower()
    assert ".pmv" in EXPORT_SCRIPT_TIP.lower()
    assert "top" in DONE_TIP.lower()
    import inspect
    from pymolviz.wizards.widgets.action_bar import BuilderActionBar
    assert "compact_primary_button_css" in inspect.getsource(BuilderActionBar.__init__)
    assert "setMaximumHeight(18)" in inspect.getsource(BuilderActionBar.__init__)


def test_object_name_is_an_inline_label_and_field():
    assert NAME_LABEL == "Name:"
    assert NAME_PLACEHOLDER == "Object name"
    assert "object list" in NAME_TIP.lower()


def test_breadcrumb_create_and_edit_paths():
    assert breadcrumb_text(create_crumbs(CRUMB_SPHERES)) == (
        "%s › %s › %s" % (CRUMB_VISUALS, CRUMB_ADD_OBJECT, CRUMB_SPHERES)
    )
    assert breadcrumb_text(edit_crumbs(CRUMB_SPHERES)) == (
        "%s › %s" % (CRUMB_VISUALS, CRUMB_SPHERES)
    )
    assert CRUMB_ADD_OBJECT == "Add Visual"


def test_breadcrumb_html_bolds_only_the_current_leaf():
    html = breadcrumb_html(create_crumbs(CRUMB_SPHERES))
    assert "<b>Spheres</b>" in html
    assert "<b>Visuals</b>" not in html
    assert "<b>Add Visual</b>" not in html
    assert "Visuals" in html
    assert "Add Visual" in html
    assert "color:gray" not in html.replace(" ", "")
    assert "110, 122, 134" in html
    edit = breadcrumb_html(edit_crumbs(CRUMB_SPHERES))
    assert "<b>Spheres</b>" in edit
    assert "Visuals" in edit
    assert "Add Visual" not in edit


def test_mix_rgb_blends_channels_and_accepts_unit_floats():
    from pymolviz.wizards.widgets.theme import mix_rgb, rgb_css

    assert mix_rgb((0, 0, 0), (255, 0, 0), 0.0) == (0, 0, 0)
    assert mix_rgb((0, 0, 0), (255, 0, 0), 1.0) == (255, 0, 0)
    assert mix_rgb((0, 0, 0), (10, 0, 0), 0.5) == (5, 0, 0)
    assert mix_rgb((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1.0) == (255, 255, 255)
    assert rgb_css((255, 128, 0)) == "rgb(255, 128, 0)"
    assert rgb_css((0, 0, 0), alpha=0.5).startswith("rgba(0, 0, 0,")


def test_editable_fields_and_warning_banner_follow_scheme():
    from pymolviz.wizards.widgets.theme import (
        BANNER_INFO,
        BANNER_WARNING,
        BORDER,
        EDITABLE_FIELD,
        HEADER,
        PAGE,
        PRIMARY,
        ROW,
        SELECTED,
        DETAIL,
        banner_fill,
        catalog_table_css,
        editable_edge,
        editable_fill,
        primary_button_css,
        compact_primary_button_css,
        section_css,
        swatch_button_css,
        type_card_css,
        wizard_page_css,
    )

    colors = {
        "window": (236, 236, 240),
        "base": (255, 255, 255),
        "highlight": (40, 90, 180),
        "text": (24, 24, 28),
        "mid": (160, 160, 168),
    }
    fill = editable_fill(colors)
    assert fill == ROW == colors["base"]
    edge = editable_edge(colors)
    assert edge == BORDER
    assert edge != colors["base"]
    assert edge != colors["mid"]
    info = banner_fill(colors, BANNER_INFO)
    warn = banner_fill(colors, BANNER_WARNING)
    assert info == SELECTED
    assert warn != info
    assert warn[0] > info[0]
    assert EDITABLE_FIELD == "pmvEditableField"
    assert PRIMARY == (42, 130, 236)
    assert HEADER == (205, 226, 240)
    assert PAGE == (245, 249, 251)
    assert SELECTED == (219, 238, 249)
    assert DETAIL == (237, 247, 252)
    assert DETAIL[0] > SELECTED[0]
    assert DETAIL != ROW
    css = wizard_page_css()
    spin = css.split("QDoubleSpinBox {", 1)[1].split("}", 1)[0]
    assert "border: 1px solid" in spin
    assert "padding: 0px" in swatch_button_css("rgb(1, 2, 3)")
    assert "min-height: 18px" in compact_primary_button_css()
    assert "max-height: 18px" in compact_primary_button_css()
    assert "205, 226, 240" in section_css()
    assert "219, 238, 249" in catalog_table_css()
    assert "245, 249, 251" in wizard_page_css()
    css = wizard_page_css()
    combo = css.split("QComboBox {", 1)[1].split("}", 1)[0]
    assert "border: 1px solid" in combo
    assert "%d, %d, %d" % BORDER in combo
    assert "QComboBox QAbstractItemView" in css
    assert "42, 130, 236" in type_card_css()


def test_insertion_banner_kind_warns_when_nothing_selected():
    from pymolviz.wizards.builders.point_insertion import insertion_banner_kind
    from pymolviz.wizards.builders.points import INSERTION_NOTHING_SELECTED
    from pymolviz.wizards.widgets.theme import BANNER_INFO, BANNER_WARNING
    from pymolviz.wizards.widgets.type_icons import source_icon_kind, source_icon_pixmap

    assert insertion_banner_kind(INSERTION_NOTHING_SELECTED) == BANNER_WARNING
    assert insertion_banner_kind("GLY 42 (3 atoms)") == BANNER_INFO
    assert insertion_banner_kind("Camera center  (1.00, 2.00, 3.00)") == BANNER_INFO
    assert source_icon_kind("selection") == "selection"
    assert source_icon_kind("camera") == "camera"
    assert source_icon_pixmap("selection", None, None) is None
    from pymolviz.wizards.widgets.type_icons import apply_source_icon, action_icon_pixmap
    assert apply_source_icon(None, "selection", None, None) is None
    assert action_icon_pixmap("forward", None, None) is None
    assert action_icon_pixmap("snap", None, None) is None
    assert action_icon_pixmap("zoom", None, None) is None
    assert action_icon_pixmap("anchor", None, None) is None


def test_visuals_window_scrolls_when_shorter_than_content():
    from pymolviz.wizards.widgets.scrolling import (
        EXPANDING_LIST_MIN_HEIGHT,
        SCROLL_BODY_MIN_HEIGHT,
        WINDOW_DEFAULT_WIDTH,
        WINDOW_MIN_HEIGHT,
        WINDOW_MIN_WIDTH,
        configure_resizable_window,
        content_needs_vertical_scroll,
        scroll_inner_min_height,
        viewport_right_inset,
    )

    assert WINDOW_MIN_HEIGHT < 720
    assert WINDOW_MIN_WIDTH <= WINDOW_DEFAULT_WIDTH
    assert WINDOW_DEFAULT_WIDTH > 640
    assert SCROLL_BODY_MIN_HEIGHT < WINDOW_MIN_HEIGHT
    assert EXPANDING_LIST_MIN_HEIGHT <= SCROLL_BODY_MIN_HEIGHT
    assert content_needs_vertical_scroll(399, 400) is True
    assert content_needs_vertical_scroll(400, 400) is False
    assert content_needs_vertical_scroll(500, 400) is False
    assert scroll_inner_min_height(400, 399) == 400
    assert scroll_inner_min_height(400, 400) == 0
    assert scroll_inner_min_height(400, 720) == 0
    assert viewport_right_inset(400, 400) == 0
    assert viewport_right_inset(400, 384) == 16
    assert viewport_right_inset(400, 500) == 0

    class _FakeWindow:
        def __init__(self):
            self.min_size = None
            self.grip = None

        def setMinimumSize(self, width, height):
            self.min_size = (int(width), int(height))

        def setSizeGripEnabled(self, enabled):
            self.grip = bool(enabled)

    window = _FakeWindow()
    configure_resizable_window(window)
    assert window.min_size == (WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
    assert window.grip is True


def test_builder_pages_share_shell_and_point_table_bases():
    from pymolviz.wizards.builders.arrow_page import ArrowBuilderPage
    from pymolviz.wizards.builders.base import BuilderPage
    from pymolviz.wizards.builders.box_page import BoxBuilderPage
    from pymolviz.wizards.builders.point_table_page import PointTableBuilderPage
    from pymolviz.wizards.builders.sphere_page import SphereBuilderPage
    from pymolviz.wizards.builders.surface_page import SurfaceBuilderPage

    assert issubclass(PointTableBuilderPage, BuilderPage)
    assert issubclass(SphereBuilderPage, PointTableBuilderPage)
    assert issubclass(BoxBuilderPage, PointTableBuilderPage)
    assert issubclass(SurfaceBuilderPage, PointTableBuilderPage)
    assert issubclass(ArrowBuilderPage, BuilderPage)
    assert not issubclass(ArrowBuilderPage, PointTableBuilderPage)
    assert SphereBuilderPage.DEFAULT_NAME == "pmv_spheres"
    assert BoxBuilderPage.DEFAULT_NAME == "pmv_boxes"
    assert SurfaceBuilderPage.DEFAULT_NAME == "pmv_surface"
    assert ArrowBuilderPage.DEFAULT_NAME == "pmv_arrows"
    assert SphereBuilderPage.CRUMB_LEAF == CRUMB_SPHERES


def test_ascii_float_locale_hides_grouping():
    from pymolviz.wizards.widgets.ascii_locale import apply_ascii_float_locale

    class FakeLocale:
        C = "C"
        OmitGroupSeparator = 1

        def __init__(self, _name):
            self.opts = 0

        def numberOptions(self):
            return 0

        def setNumberOptions(self, opts):
            self.opts = opts

    class FakeQtCore:
        QLocale = FakeLocale

    class FakeSpin:
        def __init__(self):
            self.locale = None
            self.grouped = None

        def setLocale(self, locale):
            self.locale = locale

        def setGroupSeparatorShown(self, shown):
            self.grouped = shown

        def setKeyboardTracking(self, tracking):
            self.tracking = tracking

    spin = FakeSpin()
    apply_ascii_float_locale(spin, FakeQtCore)
    assert spin.locale is not None
    assert spin.locale.opts == FakeLocale.OmitGroupSeparator
    assert spin.grouped is False
    assert spin.tracking is False


def test_solid_color_clears_field_id():
    from pymolviz.wizards.builders.colors import ColorChoice
    from pymolviz.wizards.builders.points import VisualPoint, apply_global_color

    pts = [VisualPoint("a", "manual", 0, 0, 0, field_id="map1")]
    apply_global_color(pts, ColorChoice(rgba=pts[0].rgba()))
    assert pts[0].field_id is None


def test_with_color_clears_field_metadata():
    from pymolviz.wizards.builders.points import VisualPoint

    pt = VisualPoint(
        "a", "manual", 0, 0, 0,
        color=(0.5, 0.6, 0.7),
        field_id="map1",
        field_colormap="viridis",
        field_clims=(0.0, 1.0),
    )
    solid = pt.with_color(pt.rgba())
    assert solid.field_id is None
    assert solid.field_colormap is None
    assert solid.field_clims is None
    assert solid.color == (0.5, 0.6, 0.7)


def test_infer_color_mode_uniform_per_point_and_field():
    from pymolviz.wizards.builders.points import VisualPoint, infer_color_mode
    from pymolviz.wizards.builders.surface_params import (
        COLOR_MODE_FIELD,
        COLOR_MODE_PER_POINT,
        COLOR_MODE_UNIFORM,
    )

    uniform = [
        VisualPoint("a", "manual", 0, 0, 0, color=(1.0, 0.0, 0.0)),
        VisualPoint("b", "manual", 1, 0, 0, color=(1.0, 0.0, 0.0)),
    ]
    assert infer_color_mode(uniform) == COLOR_MODE_UNIFORM
    per = [uniform[0], VisualPoint("c", "manual", 2, 0, 0, color=(0.0, 1.0, 0.0))]
    assert infer_color_mode(per) == COLOR_MODE_PER_POINT
    field = [VisualPoint("a", "manual", 0, 0, 0, field_id="map1")]
    assert infer_color_mode(field) == COLOR_MODE_FIELD


def test_surface_algorithm_visibility_matrix():
    assert algorithm_shows_probe("GAUSS") is False
    assert algorithm_shows_probe("MC") is True
    assert algorithm_shows_probe("SASA") is True
    assert ALGORITHM_CONTROL_VISIBILITY["GAUSS"]["probe"] is False


def test_point_insertion_preview_and_resolve(fake_cmd):
    from pymolviz.wizards.builders.point_insertion import (
        SOURCE_SELECTION,
        insertion_add_label,
        insertion_can_add,
        insertion_preview_text,
        insertion_selection_count,
    )
    from tests.fakes.cmd import FakeAtom

    fake_cmd.add_atom(FakeAtom(
        "prot", 1, 1.0, 2.0, 3.0, name="CA", chain="A", resn="GLY", resi="42",
    ))
    fake_cmd.select("sele", 'object "prot" and id 1')
    text = insertion_preview_text(fake_cmd, SOURCE_SELECTION)
    assert "CA" in text
    assert "GLY 42" in text
    assert "1.00" in text and "2.00" in text and "3.00" in text
    assert insertion_can_add(fake_cmd, SOURCE_SELECTION) is True
    assert insertion_add_label(insertion_selection_count(fake_cmd)) == ADD_POINT_HEADER_LABEL
    from pymolviz.wizards.builders.points import selection_points
    pts = selection_points(fake_cmd, (), hook_to_selection=True)
    assert len(pts) == 1


def test_insertion_ignores_stale_pk1_and_named_selections(fake_cmd):
    """Clearing sele must show empty even if pk1 or saved selections remain."""
    from pymolviz.wizards.builders.points import (
        INSERT_SOURCE_CAMERA,
        INSERT_SOURCE_SELECTION,
        insertion_can_add,
        insertion_preview_text,
        insertion_selection_count,
        resolve_insertion_points,
    )
    from tests.fakes.cmd import FakeAtom

    fake_cmd.add_atom(FakeAtom(
        "prot", 1, 1.0, 2.0, 3.0, name="CA", chain="A", resn="GLY", resi="42",
    ))
    fake_cmd.select("pk1", 'object "prot" and id 1')
    fake_cmd.select("saved", 'object "prot" and id 1')
    fake_cmd.select("sele", "none")

    assert insertion_selection_count(fake_cmd) == 0
    assert insertion_preview_text(fake_cmd, INSERT_SOURCE_SELECTION) == INSERTION_NOTHING_SELECTED
    assert insertion_can_add(fake_cmd, INSERT_SOURCE_SELECTION) is False
    assert resolve_insertion_points(
        fake_cmd, INSERT_SOURCE_SELECTION, existing=(), snap=False, hook=True,
    ) == []
    assert insertion_can_add(fake_cmd, INSERT_SOURCE_CAMERA) is True


def test_insertion_preview_tracks_selection_count(fake_cmd):
    from pymolviz.wizards.builders.points import (
        INSERT_SOURCE_SELECTION,
        insertion_add_label,
        insertion_preview_text,
        insertion_selection_count,
    )
    from tests.fakes.cmd import FakeAtom

    assert insertion_preview_text(fake_cmd, INSERT_SOURCE_SELECTION) == INSERTION_NOTHING_SELECTED
    assert insertion_selection_count(fake_cmd) == 0
    assert insertion_add_label(0) == ADD_POINT_HEADER_LABEL

    for atom_id in (1, 2, 3):
        fake_cmd.add_atom(FakeAtom(
            "prot", atom_id, float(atom_id), 0.0, 0.0,
            name="CA", chain="A", resn="GLY", resi="42",
        ))
    fake_cmd.select("sele", 'object "prot"')
    assert insertion_selection_count(fake_cmd) == 3
    assert insertion_add_label(insertion_selection_count(fake_cmd)) == ADD_POINT_HEADER_LABEL
    text = insertion_preview_text(fake_cmd, INSERT_SOURCE_SELECTION)
    assert text == "GLY 42 (3 atoms)"


def test_insertion_preview_multi_residue_summary(fake_cmd):
    from pymolviz.wizards.builders.points import (
        INSERT_SOURCE_SELECTION,
        insertion_preview_text,
        insertion_selection_count,
    )
    from tests.fakes.cmd import FakeAtom

    fake_cmd.add_atom(FakeAtom(
        "prot", 1, 1.0, 2.0, 3.0, name="CA", chain="A", resn="GLY", resi="42",
    ))
    fake_cmd.add_atom(FakeAtom(
        "prot", 2, 4.0, 5.0, 6.0, name="CB", chain="A", resn="ALA", resi="43",
    ))
    fake_cmd.select("sele", 'object "prot"')
    assert insertion_selection_count(fake_cmd) == 2
    text = insertion_preview_text(fake_cmd, INSERT_SOURCE_SELECTION)
    assert text.startswith("2 atoms selected")
    assert "CA · A · GLY 42" in text


def test_insertion_treats_disabled_sele_as_nothing_selected(fake_cmd):
    """disable sele must not keep Add N points — count_atoms('sele') stays N."""
    from pymolviz.wizards.builders.points import (
        INSERT_SOURCE_SELECTION,
        insertion_add_label,
        insertion_can_add,
        insertion_preview_fingerprint,
        insertion_preview_text,
        insertion_selection_count,
    )
    from tests.fakes.cmd import FakeAtom

    for atom_id in range(1, 23):
        fake_cmd.add_atom(FakeAtom(
            "prot", atom_id, float(atom_id), 0.0, 0.0,
            name="CA", chain="A", resn="GLY", resi=str(atom_id),
        ))
    fake_cmd.select("sele", 'object "prot"')

    assert fake_cmd.count_atoms("sele") == 22
    assert "sele" in fake_cmd.get_names("selections")
    assert "sele" in fake_cmd.get_names("selections", enabled_only=1)
    assert insertion_selection_count(fake_cmd) == 22
    assert insertion_can_add(fake_cmd, INSERT_SOURCE_SELECTION) is True
    assert insertion_add_label(insertion_selection_count(fake_cmd)) == ADD_POINT_HEADER_LABEL
    live = insertion_preview_fingerprint(fake_cmd, INSERT_SOURCE_SELECTION)
    assert live[1] == 22
    assert live[3] is True

    fake_cmd.disable("sele")
    # Real PyMOL: selection object remains; count_atoms is unchanged.
    assert fake_cmd.count_atoms("sele") == 22
    assert "sele" in fake_cmd.get_names("selections")
    assert "sele" not in fake_cmd.get_names("selections", enabled_only=1)
    assert insertion_selection_count(fake_cmd) == 0
    assert insertion_preview_text(fake_cmd, INSERT_SOURCE_SELECTION) == INSERTION_NOTHING_SELECTED
    assert insertion_can_add(fake_cmd, INSERT_SOURCE_SELECTION) is False
    assert insertion_add_label(insertion_selection_count(fake_cmd)) == ADD_POINT_HEADER_LABEL
    disabled = insertion_preview_fingerprint(fake_cmd, INSERT_SOURCE_SELECTION)
    assert disabled != live
    assert disabled == (INSERT_SOURCE_SELECTION, 0, None, False)

    fake_cmd.enable("sele")
    assert insertion_selection_count(fake_cmd) == 22
    assert insertion_can_add(fake_cmd, INSERT_SOURCE_SELECTION) is True
    assert insertion_add_label(insertion_selection_count(fake_cmd)) == ADD_POINT_HEADER_LABEL
    assert insertion_preview_fingerprint(fake_cmd, INSERT_SOURCE_SELECTION) == live


def test_insertion_preview_fingerprint_skips_if_unchanged(fake_cmd):
    from pymolviz.wizards.builders.points import (
        INSERT_SOURCE_SELECTION,
        insertion_preview_fingerprint,
    )
    from tests.fakes.cmd import FakeAtom

    empty = insertion_preview_fingerprint(fake_cmd, INSERT_SOURCE_SELECTION)
    assert empty == insertion_preview_fingerprint(fake_cmd, INSERT_SOURCE_SELECTION)
    assert empty[0] == INSERT_SOURCE_SELECTION
    assert empty[1] == 0
    assert empty[2] is None
    assert empty[3] is False

    fake_cmd.add_atom(FakeAtom("prot", 1, 1.0, 2.0, 3.0, name="CA"))
    fake_cmd.add_atom(FakeAtom("prot", 2, 4.0, 5.0, 6.0, name="CB"))
    fake_cmd.select("sele", 'object "prot" and id 1')
    one = insertion_preview_fingerprint(fake_cmd, INSERT_SOURCE_SELECTION)
    assert one[1] == 1
    assert one[2] == ("prot", 1)
    assert one[3] is True
    assert one == insertion_preview_fingerprint(fake_cmd, INSERT_SOURCE_SELECTION)

    fake_cmd.select("sele", 'object "prot"')
    two = insertion_preview_fingerprint(fake_cmd, INSERT_SOURCE_SELECTION)
    assert two[1] == 2
    assert two != one
    fake_cmd.select("sele", 'object "prot" and id 2')
    swapped = insertion_preview_fingerprint(fake_cmd, INSERT_SOURCE_SELECTION)
    assert swapped[1] == 1
    assert swapped[2] == ("prot", 2)
    assert swapped != one


def test_insertion_poll_skips_refresh_when_fingerprint_unchanged(fake_cmd):
    from pymolviz.wizards.builders.point_insertion import PointInsertionWidget
    from tests.fakes.cmd import FakeAtom

    class _FakeBtn:
        def __init__(self):
            self.text = ""
            self.enabled = True
            self.writes = 0

        def setText(self, text):
            self.writes += 1
            self.text = str(text)

        def setEnabled(self, enabled):
            self.enabled = bool(enabled)

    class _FakeLabel:
        def __init__(self):
            self.text = ""
            self.writes = 0

        def setText(self, text):
            self.writes += 1
            self.text = str(text)

    class _FakeCombo:
        def currentData(self):
            return "selection"

    class _FakeCheck:
        def isChecked(self):
            return True

        def setEnabled(self, *_args):
            pass

    widget = PointInsertionWidget.__new__(PointInsertionWidget)
    widget.cmd = fake_cmd
    widget._get_existing = lambda: ()
    widget._timer = None
    widget._poll_page = None
    widget._widget = object()
    widget._source = _FakeCombo()
    widget._preview = _FakeLabel()
    widget._add_btn = _FakeBtn()
    widget.snap = _FakeCheck()
    widget.hook = _FakeCheck()

    widget.refresh_preview()
    writes = widget._preview.writes
    widget._poll_preview()
    widget._poll_preview()
    assert widget._preview.writes == writes

    fake_cmd.add_atom(FakeAtom("prot", 1, 1.0, 2.0, 3.0, name="CA"))
    fake_cmd.select("sele", 'object "prot"')
    widget._poll_preview()
    assert widget._preview.writes == writes + 1
    assert widget._add_btn.text == ADD_POINT_HEADER_LABEL


def test_insertion_poll_clears_preview_when_sele_disabled(fake_cmd):
    from pymolviz.wizards.builders.point_insertion import PointInsertionWidget
    from tests.fakes.cmd import FakeAtom

    class _FakeBtn:
        def __init__(self):
            self.text = ""
            self.enabled = True
            self.writes = 0

        def setText(self, text):
            self.writes += 1
            self.text = str(text)

        def setEnabled(self, enabled):
            self.enabled = bool(enabled)

    class _FakeLabel:
        def __init__(self):
            self.text = ""
            self.writes = 0

        def setText(self, text):
            self.writes += 1
            self.text = str(text)

    class _FakeCombo:
        def currentData(self):
            return "selection"

    class _FakeCheck:
        def isChecked(self):
            return True

        def setEnabled(self, *_args):
            pass

    for atom_id in range(1, 23):
        fake_cmd.add_atom(FakeAtom(
            "prot", atom_id, float(atom_id), 0.0, 0.0, name="CA",
        ))
    fake_cmd.select("sele", 'object "prot"')

    widget = PointInsertionWidget.__new__(PointInsertionWidget)
    widget.cmd = fake_cmd
    widget._get_existing = lambda: ()
    widget._timer = None
    widget._poll_page = None
    widget._widget = object()
    widget._source = _FakeCombo()
    widget._preview = _FakeLabel()
    widget._add_btn = _FakeBtn()
    widget.snap = _FakeCheck()
    widget.hook = _FakeCheck()

    widget.refresh_preview()
    assert widget._add_btn.text == ADD_POINT_HEADER_LABEL
    assert widget._add_btn.enabled is True
    writes = widget._preview.writes

    fake_cmd.disable("sele")
    assert fake_cmd.count_atoms("sele") == 22
    widget._poll_preview()
    assert widget._preview.writes == writes + 1
    assert widget._preview.text == INSERTION_NOTHING_SELECTED
    assert widget._add_btn.text == ADD_POINT_HEADER_LABEL
    assert widget._add_btn.enabled is False

    fake_cmd.enable("sele")
    widget._poll_preview()
    assert widget._add_btn.text == ADD_POINT_HEADER_LABEL
    assert widget._add_btn.enabled is True


def test_point_insertion_widget_camera_source_enabled_without_selection(fake_cmd):
    from pymolviz.wizards.builders.point_insertion import (
        INSERT_SOURCE_CAMERA,
        PointInsertionWidget,
    )

    class _FakeBtn:
        def __init__(self):
            self.text = ""
            self.enabled = True

        def setText(self, text):
            self.text = str(text)

        def setEnabled(self, enabled):
            self.enabled = bool(enabled)

    class _FakeLabel:
        def __init__(self):
            self.text = ""

        def setText(self, text):
            self.text = str(text)

    class _FakeCombo:
        def __init__(self, data):
            self._data = data

        def currentData(self):
            return self._data

    class _FakeCheck:
        def isChecked(self):
            return True

        def setEnabled(self, *_args):
            pass

    fake_cmd.set_view([
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, -50.0,
        1.0, 2.0, 3.0,
        2.0, 200.0, 0.0,
    ])

    widget = PointInsertionWidget.__new__(PointInsertionWidget)
    widget.cmd = fake_cmd
    widget._get_existing = lambda: ()
    widget._timer = None
    widget._poll_page = None
    widget._source = _FakeCombo(INSERT_SOURCE_CAMERA)
    widget._preview = _FakeLabel()
    widget._add_btn = _FakeBtn()
    widget.snap = _FakeCheck()
    widget.hook = _FakeCheck()

    widget.refresh_preview()
    assert widget._add_btn.enabled is True
    assert widget._add_btn.text == ADD_POINT_HEADER_LABEL
    assert widget._preview.text.startswith("Camera center")


def test_fresh_selection_waits_for_atoms_after_add(fake_cmd):
    from pymolviz.wizards.builders.point_insertion import (
        INSERT_SOURCE_FRESH,
        INSERTION_FRESH_WAITING,
        PointInsertionWidget,
    )
    from pymolviz.wizards.builders.points import (
        INSERTION_FRESH_HINT,
        insertion_can_add,
        insertion_preview_text,
    )
    from tests.fakes.cmd import FakeAtom

    assert insertion_can_add(fake_cmd, INSERT_SOURCE_FRESH) is True
    assert insertion_preview_text(fake_cmd, INSERT_SOURCE_FRESH) == INSERTION_FRESH_HINT

    class _FakeBtn:
        def __init__(self):
            self.text = ""
            self.enabled = True
            self.tip = ""

        def setText(self, text):
            self.text = str(text)

        def setEnabled(self, enabled):
            self.enabled = bool(enabled)

        def setToolTip(self, tip):
            self.tip = str(tip)

    class _FakeLabel:
        def __init__(self):
            self.text = ""

        def setText(self, text):
            self.text = str(text)

    class _FakeCombo:
        def currentData(self):
            return INSERT_SOURCE_FRESH

    class _FakeCheck:
        def isChecked(self):
            return True

        def setEnabled(self, *_args):
            pass

    added = []
    widget = PointInsertionWidget.__new__(PointInsertionWidget)
    widget.cmd = fake_cmd
    widget._get_existing = lambda: ()
    widget._on_add = lambda pts: added.append(pts)
    widget._on_wait_changed = None
    widget._on_can_add_changed = None
    widget._timer = None
    widget._poll_page = None
    widget._widget = object()
    widget._waiting_fresh = False
    widget._source = _FakeCombo()
    widget._preview = _FakeLabel()
    widget._add_btn = _FakeBtn()
    widget.snap = _FakeCheck()
    widget.hook = _FakeCheck()

    fake_cmd.add_atom(FakeAtom("prot", 1, 1.0, 2.0, 3.0, name="CA"))
    fake_cmd.select("sele", 'object "prot"')
    widget._clicked_add()
    assert added == []
    assert widget._waiting_fresh is True
    assert widget._preview.text == INSERTION_FRESH_WAITING
    assert fake_cmd.count_atoms("sele") == 0

    fake_cmd.select("sele", 'object "prot"')
    widget._poll_preview()
    assert len(added) == 1
    assert widget._waiting_fresh is False



def test_point_insertion_widget_refresh_updates_button(fake_cmd):
    from pymolviz.wizards.builders.point_insertion import PointInsertionWidget
    from tests.fakes.cmd import FakeAtom

    calls = []

    class _FakeBtn:
        def __init__(self):
            self.text = ""
            self.enabled = True

        def setText(self, text):
            self.text = str(text)

        def setEnabled(self, enabled):
            self.enabled = bool(enabled)

        def setAutoDefault(self, *_args):
            pass

        def setDefault(self, *_args):
            pass

        def clicked(self):
            return _FakeSignal()

    class _FakeLabel:
        def __init__(self):
            self.text = ""

        def setText(self, text):
            self.text = str(text)

        def setWordWrap(self, *_args):
            pass

        def setStyleSheet(self, *_args):
            pass

    class _FakeCombo:
        def __init__(self):
            self._data = "selection"

        def currentData(self):
            return self._data

        def currentIndexChanged(self):
            return _FakeSignal()

    class _FakeCheck:
        def __init__(self, checked=True):
            self._checked = checked

        def isChecked(self):
            return self._checked

        def setEnabled(self, *_args):
            pass

        def setChecked(self, checked):
            self._checked = bool(checked)

        def toggled(self):
            return _FakeSignal()

    class _FakeSignal:
        def connect(self, *_args):
            pass

    widget = PointInsertionWidget.__new__(PointInsertionWidget)
    widget.cmd = fake_cmd
    widget._context = "Test"
    widget._on_add = lambda pts: calls.append(pts)
    widget._get_existing = lambda: ()
    widget._timer = None
    widget._poll_page = None
    widget._focus_filter = None
    widget._source = _FakeCombo()
    widget._preview = _FakeLabel()
    widget._add_btn = _FakeBtn()
    widget.snap = _FakeCheck(True)
    widget.hook = _FakeCheck(True)
    widget.zoom = _FakeCheck(False)
    widget.show_coords = None
    widget.export_sel = None
    widget._widget = object()

    widget.refresh_preview()
    assert widget._preview.text == INSERTION_NOTHING_SELECTED
    assert widget._add_btn.text == ADD_POINT_HEADER_LABEL
    assert widget._add_btn.enabled is False

    fake_cmd.add_atom(FakeAtom("prot", 1, 1.0, 2.0, 3.0, name="CA", chain="A", resn="GLY", resi="42"))
    fake_cmd.add_atom(FakeAtom("prot", 2, 4.0, 5.0, 6.0, name="CB", chain="A", resn="ALA", resi="43"))
    fake_cmd.select("sele", 'object "prot"')
    widget.refresh_preview()
    assert widget._add_btn.text == ADD_POINT_HEADER_LABEL
    assert widget._add_btn.enabled is True
    assert "2 atoms selected" in widget._preview.text


def test_insertion_add_label_is_count_independent():
    from pymolviz.wizards.builders.point_insertion import insertion_add_label

    assert ADD_POINT_HEADER_LABEL == "Add Point(s)"
    assert INSERTION_NOTHING_SELECTED == (
        "Nothing selected. Select atoms, or use Fresh selection to pick after Add."
    )
    assert insertion_add_label(0) == ADD_POINT_HEADER_LABEL
    assert insertion_add_label(1) == ADD_POINT_HEADER_LABEL
    assert insertion_add_label(14) == ADD_POINT_HEADER_LABEL


def test_picker_color_choice_solid_and_field():
    from pymolviz.wizards.builders.colors import picker_color_choice

    solid = picker_color_choice((1.0, 0.0, 0.0, 1.0))
    assert solid.field_id is None
    assert solid.rgba[:3] == (1.0, 0.0, 0.0)
    field = picker_color_choice(
        (1.0, 0.0, 0.0, 0.5),
        field_id="pymol_map:density",
        colormap="viridis",
        clims=(0.0, 1.0),
    )
    assert field.field_id == "pymol_map:density"
    assert field.colormap == "viridis"
    assert field.clims == (0.0, 1.0)
    assert field.clim_mode == "custom"
    assert field.rgba[3] == 0.5


def test_appearance_color_action_labels():
    from pymolviz.wizards.builders.appearance_section import (
        appearance_color_action_labels,
    )

    labels = appearance_color_action_labels()
    assert labels == ("Color All", "Color Selection", "Reset Colors")


def test_appearance_color_mode_labels_and_apply():
    from pymolviz.wizards.builders.appearance_section import (
        appearance_clim_mode_labels,
        appearance_color_mode_labels,
        apply_color_mode_to_points,
    )
    from pymolviz.wizards.builders.points import VisualPoint, infer_color_mode
    from pymolviz.wizards.builders.surface_params import (
        COLOR_MODE_FIELD,
        COLOR_MODE_PER_POINT,
        COLOR_MODE_UNIFORM,
    )

    assert appearance_color_mode_labels() == ("Uniform", "Per-point", "From field")
    assert appearance_color_mode_labels(show_per_point=False) == ("Uniform", "From field")
    assert appearance_clim_mode_labels() == ("Auto", "Custom", "Symmetric", "Percentile")
    pts = [
        VisualPoint("a", "manual", 0, 0, 0, color=(1.0, 0.0, 0.0), field_id="map1"),
        VisualPoint("b", "manual", 1, 0, 0, color=(0.0, 1.0, 0.0)),
    ]
    apply_color_mode_to_points(pts, COLOR_MODE_UNIFORM)
    assert infer_color_mode(pts) == COLOR_MODE_UNIFORM
    assert pts[0].field_id is None
    assert pts[1].color[:3] == pts[0].color[:3]
    apply_color_mode_to_points(pts, COLOR_MODE_FIELD, field_id="map1", colormap="viridis")
    assert infer_color_mode(pts) == COLOR_MODE_FIELD
    assert pts[0].field_id == "map1"
    assert pts[1].field_colormap == "viridis"
    apply_color_mode_to_points(pts, COLOR_MODE_PER_POINT)
    assert pts[0].field_id is None
    assert infer_color_mode(pts) in (COLOR_MODE_PER_POINT, COLOR_MODE_UNIFORM)
    spec = {
        "preset": "custom",
        "customized": True,
        "stops": [
            {"position": 0.0, "rgba": [0.0, 0.0, 1.0, 1.0]},
            {"position": 1.0, "rgba": [1.0, 0.0, 0.0, 1.0]},
        ],
    }
    apply_color_mode_to_points(
        pts, COLOR_MODE_FIELD, field_id="map1", colormap="custom", colormap_spec=spec,
    )
    assert pts[0].field_colormap_spec["customized"] is True
    assert pts[0].color_choice().colormap_spec["preset"] == "custom"


def test_colormap_editor_dialog_labels_without_qt():
    from pymolviz.wizards.builders.colormap_dialog import (
        EDIT_LABEL,
        EXPORT_COLORBAR_LABEL,
        colormap_editor_action_labels,
        colormap_editor_title,
        colormap_full_range_labels,
    )
    from pymolviz.wizards.builders.colormap_editor import colormap_range_mode_labels

    assert colormap_editor_title() == "Edit Colormap"
    assert colormap_editor_action_labels()[0] == "add custom colormap"
    assert colormap_editor_action_labels()[1:] == ("Save As...", "Edit...", "Export Colorbar...")
    assert colormap_full_range_labels()[0] == "Auto"
    assert colormap_full_range_labels()[2].startswith("Symmetric")
    assert colormap_range_mode_labels() == ("Auto", "Custom", "Symmetric", "Percentile")
    assert EDIT_LABEL == "Edit..."
    assert EXPORT_COLORBAR_LABEL == "Export Colorbar..."
    from pymolviz.wizards.builders.colormap_editor import (
        ADD_CUSTOM_COLORMAP_TIP,
        ADJUST_CUSTOM_COLORMAP_TIP,
        colormap_add_custom_tip,
        colormap_adjust_custom_tip,
    )

    assert ADD_CUSTOM_COLORMAP_TIP == "add custom colormap"
    assert colormap_add_custom_tip() == "add custom colormap"
    assert colormap_adjust_custom_tip() == ADJUST_CUSTOM_COLORMAP_TIP


def test_colormap_dialog_as_rgba_accepts_numpy_row():
    import numpy as np

    from pymolviz.wizards.builders.colormap_dialog import _as_rgba

    assert _as_rgba(None) == (0.0, 0.0, 0.0, 1.0)
    assert _as_rgba((0.2, 0.4, 0.6, 0.8)) == (0.2, 0.4, 0.6, 0.8)
    row = np.array([0.1, 0.2, 0.3, 1.0])
    assert _as_rgba(row) == (0.1, 0.2, 0.3, 1.0)
    rgb = np.array([0.5, 0.0, 1.0])
    assert _as_rgba(rgb) == (0.5, 0.0, 1.0, 1.0)


def test_field_model_hides_irrelevant_params():
    from pymolviz.fields.identity import GEN_DISTANCE, GEN_GAUSSIAN, GEN_NEAREST_PROP
    from pymolviz.wizards.builders.field_params import field_model_shows

    assert field_model_shows(GEN_GAUSSIAN, "quality") is True
    assert field_model_shows(GEN_GAUSSIAN, "iso_value") is True
    assert field_model_shows(GEN_GAUSSIAN, "live_preview") is True
    assert field_model_shows(GEN_GAUSSIAN, "property") is False
    assert field_model_shows(GEN_DISTANCE, "quality") is False
    assert field_model_shows(GEN_DISTANCE, "iso_value") is True
    assert field_model_shows(GEN_DISTANCE, "live_preview") is True
    assert field_model_shows(GEN_NEAREST_PROP, "property") is True
    assert field_model_shows(GEN_NEAREST_PROP, "live_preview") is True


def test_color_mode_hides_field_controls_except_from_field():
    from pymolviz.wizards.builders.surface_params import (
        COLOR_MODE_FIELD,
        COLOR_MODE_PER_POINT,
        COLOR_MODE_UNIFORM,
        color_mode_shows,
    )

    assert color_mode_shows(COLOR_MODE_UNIFORM, "field") is False
    assert color_mode_shows(COLOR_MODE_PER_POINT, "colormap") is False
    assert color_mode_shows(COLOR_MODE_FIELD, "field") is True
    assert color_mode_shows(COLOR_MODE_FIELD, "colormap") is True
    assert color_mode_shows(COLOR_MODE_FIELD, "clim") is True


class _FakeWidget:
    def isWidgetType(self):
        return True

    def parentWidget(self):
        return None


class _FakeLayout:
    def __init__(self, owner=None):
        self._owner = owner

    def isWidgetType(self):
        return False

    def parentWidget(self):
        return self._owner

    def addWidget(self, *_args, **_kwargs):
        return None


def test_dialog_parent_returns_widget_not_layout():
    from pymolviz.wizards.builders.colors import _dialog_parent

    page = _FakeWidget()
    layout = _FakeLayout(page)
    assert _dialog_parent(page) is page
    assert _dialog_parent(layout) is page
    assert _dialog_parent(None) is None
    assert _dialog_parent(_FakeLayout(None)) is None
    assert _dialog_parent(object()) is None


def test_stamp_new_points_always_uses_distinct_palette():
    from pymolviz.wizards.builders.appearance_section import AppearanceSection
    from pymolviz.wizards.builders.points import VisualPoint

    section = AppearanceSection.__new__(AppearanceSection)
    section._points = [
        VisualPoint("a", "manual", 0, 0, 0, color=(1.0, 0.0, 0.0)),
        VisualPoint("b", "manual", 1, 0, 0, color=(1.0, 0.0, 0.0)),
    ]
    stamped = section.stamp_new_points([
        VisualPoint("c", "manual", 2, 0, 0, color=(1.0, 0.0, 0.0)),
        VisualPoint("d", "manual", 3, 0, 0, color=(1.0, 0.0, 0.0)),
    ])
    assert stamped[0].color != (1.0, 0.0, 0.0) or stamped[1].color != stamped[0].color
    assert stamped[0].field_id is None
    assert stamped[0].color != stamped[1].color


def test_pick_rgb_keeps_opacity_dict_separate_from_section_chrome():
    import inspect

    from pymolviz.wizards.builders import colors

    src = inspect.getsource(colors.pick_rgb)
    assert "opacity = make_section" not in src
    assert "opacity_section = make_section" in src
    assert 'opacity["value"]' in src
    assert "recent_colors[0]" not in src
    assert "picker_start_rgba" in src


def test_picker_opens_on_object_color_not_last_custom():
    from pymolviz.wizards.builders.colors import picker_start_rgba

    object_rgba = (0.2, 0.4, 0.8, 0.5)
    recent = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0)]
    assert picker_start_rgba(object_rgba, recent) == object_rgba


def test_field_picker_mount_does_not_emit():
    import inspect

    from pymolviz.wizards.builders import colors

    src = inspect.getsource(colors._mount_field_picker)
    assert "_sync_state(emit=False)" in src


def test_color_choice_signature_skips_same_rgba():
    from pymolviz.wizards.builders.colors import ColorChoice, color_choice_signature

    a = ColorChoice(rgba=(0.2, 0.4, 0.8, 1.0))
    b = ColorChoice(rgba=(0.2, 0.4, 0.8, 1.0))
    c = ColorChoice(rgba=(1.0, 0.0, 0.0, 1.0))
    assert color_choice_signature(a) == color_choice_signature(b)
    assert color_choice_signature(a) != color_choice_signature(c)


def test_picker_ok_is_not_overwritten_by_later_cancel():
    from pymolviz.wizards.builders.colors import ColorChoice, make_picker_finish

    results = []
    remembered = []
    last = {"value": ColorChoice(rgba=(1.0, 0.0, 0.0, 1.0))}
    finish = make_picker_finish(
        accepted=1,
        compose=lambda: ColorChoice(rgba=(0.0, 1.0, 0.0, 1.0)),
        last=last,
        remember=lambda rgba: remembered.append(rgba),
        on_done=results.append,
    )
    finish(1)
    finish(0)
    finish(0)
    assert len(results) == 1
    assert results[0].rgba[1] == 1.0
    assert len(remembered) == 1


def test_bind_color_pick_result_blocks_restore_after_ok():
    from pymolviz.wizards.builders.colors import bind_color_pick_result

    applied = []
    restored = []
    done = bind_color_pick_result(
        lambda choice: applied.append(choice),
        lambda: restored.append(True),
    )
    done("green")
    done(None)
    assert applied == ["green"]
    assert restored == []
    done(None)
    assert restored == []


def test_live_color_dialog_is_a_singleton():
    from pymolviz.wizards.builders.colors import (
        forget_color_dialog,
        live_color_dialog,
        remember_color_dialog,
    )

    holder = {"widget": None}
    remember_color_dialog("dlg", holder=holder)
    assert live_color_dialog(lambda _w: True, holder=holder) == "dlg"
    dead = {"widget": "dlg"}
    assert live_color_dialog(lambda _w: False, holder=dead) is None
    assert dead["widget"] is None
    forget_color_dialog("dlg", holder=holder)
    assert live_color_dialog(lambda _w: True, holder=holder) is None


def test_color_dialog_stack_keeps_parent_when_nested():
    from pymolviz.wizards.builders.colors import (
        _COLOR_DIALOG_STACK,
        forget_color_dialog,
        live_color_dialog,
        remember_color_dialog,
    )

    before = list(_COLOR_DIALOG_STACK)
    try:
        _COLOR_DIALOG_STACK[:] = []
        remember_color_dialog("sphere")
        remember_color_dialog("stop")
        assert live_color_dialog(lambda _w: True) == "stop"
        forget_color_dialog("stop")
        assert live_color_dialog(lambda _w: True) == "sphere"
    finally:
        _COLOR_DIALOG_STACK[:] = before


def test_pick_rgb_reuses_open_dialog_and_commits_on_color_selected():
    import inspect

    from pymolviz.wizards.builders import colors

    src = inspect.getsource(colors.pick_rgb)
    assert "live_color_dialog" in src
    assert "colorSelected.connect" in src
    assert "configure_tool_window" not in src
    assert "_keep_front" in src
    assert "allow_field" in src
    assert "if allow_field:" in src
    assert "_mount_field_picker" in src
    assert "_pmv_raise_last" in src
    assert "window_anchor" in src
    assert "window_anchor is None" in src


def test_colormap_stop_color_picker_is_solid_only():
    import inspect

    from pymolviz.wizards.builders import colormap_dialog

    src = inspect.getsource(colormap_dialog.ColormapEditorDialog._pick_stop_color)
    assert "allow_field=False" in src
    special = inspect.getsource(colormap_dialog.ColormapEditorDialog._pick_special)
    assert "allow_field=False" in special


def test_colormap_preset_reload_skips_deleted_combo(monkeypatch):
    from pymolviz.wizards.builders.colormap_editor import ColormapPresetPicker

    picker = ColormapPresetPicker.__new__(ColormapPresetPicker)
    picker._combo = object()
    picker._default_name = "RdYlBu_r"
    monkeypatch.setattr(
        "pymolviz.wizards.builders.colormap_editor.qt_widget_alive",
        lambda _w: False,
    )
    picker.reload(select="viridis")


def test_colormap_editor_set_mapping_skips_deleted_combo(monkeypatch):
    from pymolviz.util.colormap_spec import FieldColorMapping, Normalization, definition_from_preset
    from pymolviz.wizards.builders.colormap_editor import ColormapEditor, ColormapPresetPicker

    picker = ColormapPresetPicker.__new__(ColormapPresetPicker)
    picker._combo = object()
    picker._default_name = "RdYlBu_r"
    picker._syncing = False
    picker._gear = object()
    editor = ColormapEditor.__new__(ColormapEditor)
    editor._picker = picker
    editor._default_name = "RdYlBu_r"
    editor._definition = definition_from_preset("viridis")
    editor._norm_extra = Normalization()
    editor._reverse = object()
    editor._range = object()
    editor._lo = object()
    editor._hi = object()
    editor._strip = object()
    editor._widget = object()
    editor._syncing = False
    monkeypatch.setattr(
        "pymolviz.wizards.builders.colormap_editor.qt_widget_alive",
        lambda _w: False,
    )
    editor.set_mapping(
        FieldColorMapping(
            colormap=definition_from_preset("plasma"),
            normalization=Normalization(),
        )
    )
    picker.set_current("plasma")


def test_picker_close_should_accept_only_while_visible():
    from pymolviz.wizards.builders.colors import picker_close_should_accept

    assert picker_close_should_accept(visible=True, already_accepted=False) is True
    assert picker_close_should_accept(visible=False, already_accepted=False) is False
    assert picker_close_should_accept(visible=True, already_accepted=True) is False


def test_apply_global_color_rows_is_color_selection_helper():
    from pymolviz.wizards.builders.colors import ColorChoice
    from pymolviz.wizards.builders.points import VisualPoint, apply_global_color

    pts = [
        VisualPoint("a", "manual", 0, 0, 0, color=(1.0, 0.0, 0.0)),
        VisualPoint("b", "manual", 1, 0, 0, color=(0.0, 1.0, 0.0)),
    ]
    apply_global_color(pts, ColorChoice(rgba=(0.0, 0.0, 1.0, 1.0)), rows=[1])
    assert pts[0].color[0] == 1.0
    assert pts[1].color[2] == 1.0


def test_sphere_box_arrow_appearance_flags():
    from pymolviz.wizards.builders.arrow_page import ArrowBuilderPage
    from pymolviz.wizards.builders.box_page import BoxBuilderPage
    from pymolviz.wizards.builders.from_selection_page import FromSelectionFieldPage
    from pymolviz.wizards.builders.sphere_page import SphereBuilderPage
    from pymolviz.wizards.builders.surface_page import SurfaceBuilderPage

    sphere = SphereBuilderPage._appearance_config(SphereBuilderPage)
    assert sphere["show_wireframe"] is True
    assert sphere["show_quality"] is True
    assert sphere["quality_range"] == (1, 5)
    assert sphere.get("show_per_point", True) is True
    box = BoxBuilderPage._appearance_config(BoxBuilderPage)
    assert box["show_quality"] is False
    surface = SurfaceBuilderPage._appearance_config(SurfaceBuilderPage)
    assert surface["show_quality"] is True
    from_sel = FromSelectionFieldPage._appearance_config(FromSelectionFieldPage)
    assert from_sel.get("show_per_point") is False
    assert hasattr(ArrowBuilderPage, "_build")


def test_object_visual_editors_put_selection_left_of_options():
    import inspect

    from pymolviz.wizards.builders.arrow_page import ArrowBuilderPage
    from pymolviz.wizards.builders.point_table_page import PointTableBuilderPage
    from pymolviz.wizards.widgets.scrolling import WINDOW_DEFAULT_WIDTH

    points_src = inspect.getsource(PointTableBuilderPage._build)
    assert "_mount_editor_columns" in points_src
    assert points_src.index("_mount_points_section") < points_src.index("_mount_geometry")
    assert points_src.index("_mount_geometry") < points_src.index("_mount_options")
    assert points_src.index("_mount_options") < points_src.index("_mount_appearance")
    assert points_src.index("_mount_appearance") < points_src.index("_mount_modifiers")
    assert "right.addStretch" in points_src

    arrow_src = inspect.getsource(ArrowBuilderPage._build)
    assert "_mount_editor_columns" in arrow_src
    assert "left.addWidget(arrows.widget" in arrow_src
    assert "right.addWidget(geom.widget)" in arrow_src
    assert 'make_section("Options")' in arrow_src
    geom_chunk = arrow_src.split("self._appearance = AppearanceSection")[0]
    assert "ArrowTypeControl" not in geom_chunk.split("opts = make_section")[0]
    assert "ArrowTypeControl" in arrow_src.split("opts = make_section")[1]
    assert "right.addWidget(self._appearance.widget)" in arrow_src
    assert "right.addWidget(self._modifiers.widget)" in arrow_src
    assert WINDOW_DEFAULT_WIDTH == 1280


def test_arrow_page_has_clicked_atom_toggle():
    import inspect

    from pymolviz.wizards.builders.arrow_page import (
        CLICKED_ATOM_LABEL,
        CLICKED_ATOM_TIP,
        SELECTION_AVERAGE_LABEL,
        SELECTION_CENTER_TIP,
        ArrowBuilderPage,
    )

    src = inspect.getsource(ArrowBuilderPage._build)
    assert "_clicked_atom_box" in src
    assert "CLICKED_ATOM_LABEL" in src
    assert "SELECTION_AVERAGE_LABEL" in src
    assert "QRadioButton" in src
    assert "clicked" in CLICKED_ATOM_TIP.lower()
    assert "average" in SELECTION_AVERAGE_LABEL.lower()
    assert "average" in SELECTION_CENTER_TIP.lower()
    add = inspect.getsource(ArrowBuilderPage.add_arrow)
    assert "INSERT_SOURCE_CAMERA" in add
    assert "INSERT_SOURCE_FRESH" in add
    assert "camera_center_point" in add
    assert "_begin_incomplete" in add
    assert "Select start in PyMOL" in add


def test_arrow_row_uses_global_width_and_labeled_attach():
    import inspect

    from pymolviz.wizards.builders import spatial_point_editor as point_editor_mod
    from pymolviz.wizards.builders.arrow_list import (
        ATTACH_LABEL,
        CAMERA_CENTER_LABEL,
        CAMERA_ENDPOINT_TIP,
        PICK_ATOM_LABEL,
        PICK_START_TIP,
        UPDATE_POSITION_LABEL,
        XYZ_LABEL,
        ArrowPairEditor,
    )
    from pymolviz.wizards.builders.point_list import PointListEditor
    from pymolviz.wizards.builders.point_table_page import PointTableBuilderPage
    from pymolviz.wizards.builders.spatial_point_editor import SpatialPointEditor

    details = inspect.getsource(ArrowPairEditor._details)
    assert "set_arrow_title" not in details
    assert "Start → End" not in details
    assert "action_icon_pixmap" in details
    assert "forward" in details
    assert "_endpoint_card" in details
    card = inspect.getsource(ArrowPairEditor._endpoint_card)
    assert "SpatialPointEditor" in card
    assert "edit_endpoint_color" in card
    row = inspect.getsource(ArrowPairEditor._make_block)
    assert "end_color" in row
    assert "edit_color" in row
    assert ATTACH_LABEL == "Anchor to atom"
    assert UPDATE_POSITION_LABEL == "Update position"
    assert XYZ_LABEL.startswith("XYZ")
    assert PICK_ATOM_LABEL == "Pick atom"
    assert CAMERA_CENTER_LABEL == "Use camera center"
    editor = inspect.getsource(SpatialPointEditor.__init__)
    assert "QDoubleSpinBox" in editor
    assert "apply_ascii_float_locale" in editor
    from pymolviz.wizards.builders.spatial_item_list import _apply_row_swatch, make_spatial_detail
    assert "padding: 0px" in inspect.getsource(_apply_row_swatch)
    assert "DETAIL" in inspect.getsource(make_spatial_detail)
    assert "apply_source_icon" in editor
    assert "name_row" not in editor
    assert "source_color" in editor
    assert "CURRENT_REF_LABEL" not in editor
    assert "SOURCE_STATUS_LABEL" in editor
    assert "COLOR_LABEL" in editor
    assert 'icon="anchor"' in editor
    from pymolviz.wizards.widgets.log_slider import LogSegmentRadiusWidget
    assert "sliderReleased" in inspect.getsource(LogSegmentRadiusWidget.connect_changed)
    pick = inspect.getsource(PointTableBuilderPage.pick_point_atom)
    assert "_clear_pymol_selection" in pick
    assert "_start_atom_pick_timer" in pick
    poll = inspect.getsource(PointTableBuilderPage._poll_atom_pick)
    assert "take_single_selection_point" in poll
    assert "MULTI_CLICKED" in poll
    list_src = inspect.getsource(PointListEditor._make_block)
    assert "picking=pick_index == index" in list_src
    action = inspect.getsource(point_editor_mod._action_button)
    assert "setFlat(False)" in action
    assert "apply_secondary_button_style" in action
    assert "selection" in PICK_START_TIP.lower()
    assert "camera" in CAMERA_ENDPOINT_TIP.lower()


def test_arrow_add_button_is_layout_footer_not_overlay():
    import inspect

    from pymolviz.wizards.builders.arrow_list import ArrowPairEditor
    from pymolviz.wizards.builders.spatial_item_list import SpatialItemList

    src = inspect.getsource(ArrowPairEditor.__init__)
    assert "StickyAddOverlay" not in src
    assert 'add_text="+ Add arrow"' in src
    assert "make_add_from_source_toggle" in src
    assert "set_add_source_icon" not in src
    attach = inspect.getsource(ArrowPairEditor.attach_add_to_section)
    assert "add_header_widget(self._add_from)" in attach
    from pymolviz.wizards.builders.point_insertion import AddFromSourceRadios, SOURCE_FRESH_LABEL

    radio_src = inspect.getsource(AddFromSourceRadios.__init__)
    assert "QRadioButton" in radio_src
    assert "INSERT_SOURCE_FRESH" in radio_src
    assert SOURCE_FRESH_LABEL == "Fresh selection"
    list_src = inspect.getsource(SpatialItemList.__init__)
    assert "QPushButton(add_text)" in list_src
    assert "setSizePolicy" in list_src
    assert "make_spatial_header" in list_src


def test_point_list_reuses_spatial_point_editor():
    import inspect

    from pymolviz.wizards.builders.point_list import PointListEditor
    from pymolviz.wizards.builders.point_table_page import PointTableBuilderPage

    src = inspect.getsource(PointListEditor._make_block)
    assert "SpatialPointEditor" in src
    assert src.count("SpatialPointEditor") == 1
    mount = inspect.getsource(PointTableBuilderPage._mount_points_section)
    assert "PointListEditor" in mount
    assert "QTableWidget" not in mount
    assert 'add_text="+ Add point"' in inspect.getsource(PointListEditor.__init__)
    assert "set_add_waiting" in inspect.getsource(PointListEditor)
    assert "on_wait_changed" in inspect.getsource(PointTableBuilderPage._mount_points_section)


def test_spatial_row_chevron_and_click_toggles_expand():
    import inspect

    from pymolviz.wizards.builders.arrow_page import ArrowBuilderPage
    from pymolviz.wizards.builders.point_table_page import PointTableBuilderPage
    from pymolviz.wizards.builders.spatial_item_list import (
        CHEVRON_CLOSED,
        CHEVRON_OPEN,
        make_spatial_item_row,
    )

    assert CHEVRON_CLOSED == "▸"
    assert CHEVRON_OPEN == "▾"
    row_src = inspect.getsource(make_spatial_item_row)
    assert "CHEVRON_OPEN if selected else CHEVRON_CLOSED" in row_src
    select_pt = inspect.getsource(PointTableBuilderPage.select_point)
    assert "self._selected_index = None" in select_pt
    select_arr = inspect.getsource(ArrowBuilderPage.select_arrow)
    assert "self._selected_id = None" in select_arr

