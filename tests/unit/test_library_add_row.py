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
    from pymolviz.wizards.add_visual import MESH_TYPES

    names = [entry[0] for entry in MESH_TYPES]
    assert names == ["Spheres", "Boxes", "Surface", "Arrows"]
    assert "Lines" not in names
    kind_by_name = {entry[0]: entry[1] for entry in MESH_TYPES}
    assert kind_by_name["Spheres"] == "Sphere"
    assert kind_by_name["Boxes"] == "Box"
    assert kind_by_name["Surface"] == "Surface"
    assert kind_by_name["Arrows"] == "Arrows"
