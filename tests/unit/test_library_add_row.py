"""Layout helper for the sticky Add Object row and unique object names."""

from pymolviz.wizards.add_visual import add_object_overlay_rect, objects_need_sticky_add
from pymolviz.wizards.builders.object_names import unused_object_name


def test_add_object_stays_inline_when_rows_fit():
    assert objects_need_sticky_add(0, 100, 20) is False
    assert objects_need_sticky_add(4, 100, 20) is False


def test_add_object_sticks_when_objects_plus_add_overflow():
    assert objects_need_sticky_add(5, 100, 20) is True
    assert objects_need_sticky_add(10, 80, 24) is True


def test_add_object_taller_button_can_force_sticky():
    assert objects_need_sticky_add(4, 100, 20, add_height=30) is True
    assert objects_need_sticky_add(3, 100, 20, add_height=30) is False


def _assert_inside_table(rect, table_w, table_h):
    x, y, w, h, _margin = rect
    assert x >= 0
    assert y >= 0
    assert w >= 1
    assert h >= 1
    assert x + w <= table_w
    assert y + h <= table_h


def test_overlay_stays_inside_table_when_short():
    rect = add_object_overlay_rect(
        table_width=200,
        table_height=40,
        viewport_x=0,
        viewport_y=24,
        viewport_height=16,
        n_objects=0,
        row_height=20,
        add_height=30,
        header_height=24,
    )
    _assert_inside_table(rect, 200, 40)
    _x, y, _w, h, margin = rect
    assert y == 40 - h
    assert margin < 30


def test_overlay_fits_when_table_is_shorter_than_button():
    rect = add_object_overlay_rect(
        table_width=200,
        table_height=22,
        viewport_x=0,
        viewport_y=20,
        viewport_height=2,
        n_objects=3,
        row_height=20,
        add_height=30,
        header_height=20,
    )
    _assert_inside_table(rect, 200, 22)
    _x, y, _w, h, _margin = rect
    assert h == 22
    assert y == 0


def test_overlay_pins_to_bottom_when_content_overflows():
    rect = add_object_overlay_rect(
        table_width=200,
        table_height=120,
        viewport_x=0,
        viewport_y=24,
        viewport_height=96,
        n_objects=10,
        row_height=20,
        add_height=30,
        header_height=24,
    )
    _assert_inside_table(rect, 200, 120)
    _x, y, _w, h, margin = rect
    assert y == 120 - h
    assert margin > 0


def test_overlay_inline_when_empty_and_tall():
    rect = add_object_overlay_rect(
        table_width=200,
        table_height=400,
        viewport_x=0,
        viewport_y=24,
        viewport_height=376,
        n_objects=0,
        row_height=20,
        add_height=30,
        header_height=24,
    )
    _assert_inside_table(rect, 200, 400)
    _x, y, _w, h, margin = rect
    assert y == 24
    assert h == 30
    assert margin == 0


def test_add_object_not_sticky_until_viewport_is_known():
    assert objects_need_sticky_add(20, 0, 20) is False
    assert objects_need_sticky_add(20, 100, 0) is False


def test_unused_object_name_keeps_base_when_free():
    assert unused_object_name("pmv_spheres", taken=()) == "pmv_spheres"
    assert unused_object_name("pmv_spheres", taken=("pmv_boxes",)) == "pmv_spheres"


def test_unused_object_name_counts_up():
    taken = {"pmv_spheres", "pmv_spheres_1"}
    assert unused_object_name("pmv_spheres", taken=taken) == "pmv_spheres_2"


def test_unused_object_name_keep_allows_current():
    taken = {"pmv_spheres", "pmv_spheres_1"}
    assert unused_object_name("pmv_spheres", taken=taken, keep="pmv_spheres") == "pmv_spheres"


def test_add_object_types_are_plural_without_lines():
    from pymolviz.wizards.add_visual import (
        ADD_VISUAL_BUTTON,
        ADD_VISUAL_LABEL,
        MESH_TYPES,
    )

    names = [entry[0] for entry in MESH_TYPES]
    assert names == ["Spheres", "Boxes", "Surface", "Arrows"]
    assert "Lines" not in names
    kind_by_name = {entry[0]: entry[1] for entry in MESH_TYPES}
    assert kind_by_name["Spheres"] == "Sphere"
    assert kind_by_name["Boxes"] == "Box"
    assert kind_by_name["Surface"] == "Surface"
    assert kind_by_name["Arrows"] == "Arrows"
    icons = [entry[3] for entry in MESH_TYPES]
    assert icons == ["sphere", "cube", "surface", "arrow"]
    hints = {entry[0]: entry[2] for entry in MESH_TYPES}
    assert "solid or wireframe" in hints["Spheres"].lower()
    assert "line" in hints["Arrows"].lower()
    assert "dash" in hints["Arrows"].lower()
    assert ADD_VISUAL_LABEL == "Add Visual"
    assert ADD_VISUAL_BUTTON == "+ Add Visual"


def test_type_icon_pixmap_skips_without_qt():
    from pymolviz.wizards.widgets.type_icons import type_icon_pixmap

    assert type_icon_pixmap("sphere", None, None) is None
    from pymolviz.wizards.widgets.type_icons import action_icon_pixmap, source_icon_pixmap

    assert source_icon_pixmap("camera", None, None) is None
    assert action_icon_pixmap("trash", None, None) is None


def test_library_empty_state_copy():
    from pymolviz.wizards.add_visual import (
        EMPTY_LIBRARY_HINT,
        EMPTY_LIBRARY_TITLE,
        library_shows_empty_state,
    )

    assert library_shows_empty_state(0) is True
    assert library_shows_empty_state(1) is False
    assert EMPTY_LIBRARY_TITLE == "No visual objects yet"
    assert "spheres" in EMPTY_LIBRARY_HINT.lower()
    assert "boxes" in EMPTY_LIBRARY_HINT.lower()
    assert "surfaces" in EMPTY_LIBRARY_HINT.lower()
    assert "arrows" in EMPTY_LIBRARY_HINT.lower()


def test_library_list_columns_are_actionable():
    from pymolviz.wizards.add_visual import OBJECT_COLUMNS

    assert OBJECT_COLUMNS == ("Name", "Type", "Visible", "Actions")


def test_field_visuals_library_copy_and_types():
    from pymolviz.wizards.field_visuals import (
        ADD_FIELD_BUTTON,
        ADD_FIELD_LABEL,
        ADD_VISUAL_BUTTON,
        EMPTY_LIBRARY_HINT,
        EMPTY_LIBRARY_TITLE,
        FIELD_ADD_VISUAL_SPAN,
        FIELD_COLUMNS,
        FIELD_SOURCES,
        FIELD_VISUAL_TYPES,
        LIBRARY_ROW_MIN_HEIGHT,
        library_shows_empty_state,
    )

    assert library_shows_empty_state(0) is True
    assert library_shows_empty_state(1) is False
    assert EMPTY_LIBRARY_TITLE == "No fields yet"
    assert "pymol" in EMPTY_LIBRARY_HINT.lower()
    assert FIELD_COLUMNS == (
        "Name",
        "Kind",
        "Used by / Geometry",
        "Visible",
        "Edit",
        "Delete",
    )
    assert FIELD_ADD_VISUAL_SPAN == 3
    assert LIBRARY_ROW_MIN_HEIGHT >= 32
    assert ADD_FIELD_LABEL == "Add Field"
    assert ADD_FIELD_BUTTON == "+ Add Field"
    assert ADD_VISUAL_BUTTON == "+ Add Visual"
    from pymolviz.wizards.widgets.catalog_chrome import (
        ADD_FIELD_STYLE,
        ADD_VISUAL_STYLE,
        CATALOG_PRIMARY,
        CATALOG_SELECTED,
        library_row_fill,
    )
    from pymolviz.wizards.catalog import KIND_FIELD, KIND_VISUAL, NEST_LINE_RGB

    assert CATALOG_PRIMARY == (42, 130, 236)
    assert "42, 130, 236" in ADD_FIELD_STYLE
    assert "dashed" in ADD_VISUAL_STYLE
    assert library_row_fill(KIND_FIELD, selected=True) == CATALOG_SELECTED
    assert library_row_fill(KIND_FIELD, selected=False) == (255, 255, 255)
    assert library_row_fill(KIND_VISUAL, selected=True) == (255, 255, 255)
    assert abs(NEST_LINE_RGB[0] - NEST_LINE_RGB[1]) < 20
    assert NEST_LINE_RGB[2] < 220
    sources = [entry[1] for entry in FIELD_SOURCES]
    assert sources == ["from_selection", "map", "xyz", "orca", "mtz", "derived"]
    from pymolviz.wizards.field_visuals import DISABLED_FIELD_SOURCES
    assert "derived" in DISABLED_FIELD_SOURCES
    kinds = [entry[1] for entry in FIELD_VISUAL_TYPES]
    assert kinds == ["Volume", "IsoVolume", "IsoSurface", "IsoMesh"]
    icons = [entry[3] for entry in FIELD_VISUAL_TYPES]
    assert "volume" in icons
    assert "isomesh" in icons


def test_field_breadcrumb_paths():
    from pymolviz.wizards.widgets.breadcrumb import (
        CRUMB_ADD_FIELD,
        CRUMB_ADD_OBJECT,
        CRUMB_FIELDS,
        CRUMB_FROM_SELECTION,
        CRUMB_VOLUME,
        breadcrumb_text,
        create_field_crumbs,
        create_field_visual_crumbs,
        edit_field_crumbs,
    )

    assert CRUMB_FIELDS == "Fields"
    assert CRUMB_ADD_FIELD == "Add Field"
    assert CRUMB_FROM_SELECTION == "From Selection"
    assert breadcrumb_text(create_field_crumbs(CRUMB_FROM_SELECTION)) == (
        "%s › %s › %s" % (CRUMB_FIELDS, CRUMB_ADD_FIELD, CRUMB_FROM_SELECTION)
    )
    assert breadcrumb_text(create_field_visual_crumbs(CRUMB_VOLUME)) == (
        "%s › %s › %s" % (CRUMB_FIELDS, CRUMB_ADD_OBJECT, CRUMB_VOLUME)
    )
    assert breadcrumb_text(edit_field_crumbs("density", CRUMB_VOLUME)) == (
        "%s › %s › %s" % (CRUMB_FIELDS, "density", CRUMB_VOLUME)
    )
