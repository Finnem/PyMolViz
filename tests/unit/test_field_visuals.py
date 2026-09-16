"""Field library grouping and volumetric visual helpers (no Qt)."""

from __future__ import annotations

import numpy as np
import pytest

from pymolviz.util.field_sample import PYMOL_MAP_ID_PREFIX, _NATIVE_GRIDS, grid_from_pymol_map
from pymolviz.volumetric.GridData import GridData
from pymolviz.wizards.builders.field_visual import (
    clip_map_name,
    default_iso_level,
    default_visual_name,
    load_geometry_map,
    make_field_visual,
)
from pymolviz.wizards.builders.load_field import field_kind_from_path, default_field_name
from pymolviz.wizards.catalog import (
    KIND_ADD_VISUAL,
    KIND_FIELD,
    KIND_VISUAL,
    NEST_ELL,
    NEST_NONE,
    NEST_TEE,
    add_visual_row_id,
    field_library_detail_text,
    field_library_field_count,
    field_library_rows,
    field_row_shows_edit,
    field_row_shows_symmetrize,
    is_field,
    nest_connector,
    object_rows,
    parse_add_visual_row_id,
    row_indent_px,
)


def _tiny_grid(name="density", obj_id="grid1"):
    values = np.arange(8, dtype=float)
    grid = GridData(
        values,
        step_sizes=(1.0, 1.0, 1.0),
        step_counts=(1, 1, 1),
        origin=(0.0, 0.0, 0.0),
        name=name,
    )
    grid.id = obj_id
    return grid


def _map_cmd(values, name="density"):
    from tests.fakes.cmd import FakeCmd

    cmd = FakeCmd()
    cmd.objects[name] = []
    cmd.object_types[name] = "object:map"
    cmd.volume_fields[name] = values
    cmd.extents[name] = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    return cmd


def test_native_map_is_a_field_with_add_visual_child():
    _NATIVE_GRIDS.clear()
    values = np.zeros((2, 2, 2), dtype=float)
    cmd = _map_cmd(values)
    rows = field_library_rows([], cmd=cmd)
    assert field_library_field_count(rows) == 1
    assert [row["kind"] for row in rows] == [KIND_FIELD, KIND_ADD_VISUAL]
    assert rows[0]["name"] == "density"
    assert rows[0]["type"] == "Map"
    assert rows[0]["native"] is True
    assert rows[0]["id"] == PYMOL_MAP_ID_PREFIX + "density"
    assert rows[0]["used_by"] == "0 visuals"
    assert field_library_detail_text(rows[0]) == "0 visuals"
    assert rows[1]["id"] == add_visual_row_id(rows[0]["id"])
    assert parse_add_visual_row_id(rows[1]["id"]) == rows[0]["id"]
    assert row_indent_px(rows[0]) == 0
    assert row_indent_px(rows[1]) > 0


def test_volume_nests_under_its_grid():
    grid = _tiny_grid()
    volume = make_field_visual("Volume", grid, "density_volume", obj_id="vol1")
    rows = field_library_rows([grid, volume], cmd=None)
    kinds = [row["kind"] for row in rows]
    assert kinds == [KIND_FIELD, KIND_VISUAL, KIND_ADD_VISUAL]
    assert rows[0]["id"] == "grid1"
    assert rows[0]["type"] == "Grid"
    assert rows[1]["id"] == "vol1"
    assert rows[1]["name"] == "density_volume"
    assert rows[1]["type"] == "Volume"
    assert rows[1]["field_id"] == "grid1"
    assert rows[1]["editor"] == "Volume"
    assert rows[1]["geometry_field"] == "density"
    assert rows[0]["used_by"] == "1 visual"
    assert field_library_detail_text(rows[0]) == "1 visual"
    assert field_library_detail_text(rows[1]) == "Geometry: density"
    assert field_library_detail_text(rows[2]) == ""
    assert field_row_shows_edit(rows[0]) is False
    assert field_row_shows_edit(rows[1]) is True
    assert field_row_shows_symmetrize(rows[0]) is True
    assert field_row_shows_symmetrize(rows[1]) is False
    assert nest_connector(rows[0]) == NEST_NONE
    assert nest_connector(rows[1]) == NEST_TEE
    assert nest_connector(rows[2]) == NEST_ELL
    assert rows[2]["field_id"] == "grid1"
    assert object_rows([grid, volume]) == []


def test_volume_accepts_custom_preset_name(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from pymolviz.util.colormap_spec import definition_from_preset, save_custom_preset

    save_custom_preset("Custom 2", definition_from_preset("viridis"))
    grid = _tiny_grid()
    volume = make_field_visual("Volume", grid, "density_volume", colormap="Custom 2")
    assert volume is not None
    assert volume.colormap is not None
    color = np.asarray(volume.colormap.get_color(volume.clims[0]), dtype=float).reshape(-1)
    assert color.size >= 3
    assert np.all(np.isfinite(color[:3]))


def test_volume_range_clims_sample_colormap_interior():
    from pymolviz.util.colormap_spec import ColorStop, ColormapDefinition

    spec = ColormapDefinition(
        preset="custom",
        customized=True,
        stops=(
            ColorStop(0.0, (0.0, 0.0, 1.0, 1.0)),
            ColorStop(0.5, (1.0, 0.0, 0.0, 1.0)),
            ColorStop(1.0, (0.0, 1.0, 0.0, 1.0)),
        ),
    ).to_dict()
    grid = _tiny_grid()
    volume = make_field_visual(
        "Volume",
        grid,
        "density_volume",
        colormap="custom",
        colormap_spec=spec,
        clims=(0.0, 7.0),
    )
    assert len(volume.clims) > 2
    mid = 0.5 * (float(np.min(volume.clims)) + float(np.max(volume.clims)))
    rgb = np.asarray(volume.colormap.get_color(mid), dtype=float).reshape(-1)[:3]
    two_knot_lerp = np.array([0.0, 0.5, 0.5])
    assert np.linalg.norm(rgb - np.array([1.0, 0.0, 0.0])) < np.linalg.norm(rgb - two_knot_lerp)


def test_volume_load_updates_ramp_on_existing_object():
    from tests.fakes.cmd import FakeCmd
    from pymolviz.util.colormap_spec import ColorStop, ColormapDefinition

    grid = _tiny_grid()
    cmd = FakeCmd()
    cmd.load_brick(grid, grid.name)
    red = ColormapDefinition(
        preset="custom",
        customized=True,
        stops=(
            ColorStop(0.0, (1.0, 0.0, 0.0, 1.0)),
            ColorStop(1.0, (1.0, 0.0, 0.0, 1.0)),
        ),
    ).to_dict()
    blue = ColormapDefinition(
        preset="custom",
        customized=True,
        stops=(
            ColorStop(0.0, (0.0, 0.0, 1.0, 1.0)),
            ColorStop(1.0, (0.0, 0.0, 1.0, 1.0)),
        ),
    ).to_dict()
    first = make_field_visual(
        "Volume", grid, "vol", colormap="custom", colormap_spec=red, clims=(0.0, 7.0),
    )
    first.load(cmd)
    ramp_red = list(cmd.objects["vol_volume_color_ramp"])
    second = make_field_visual(
        "Volume", grid, "vol", colormap="custom", colormap_spec=blue, clims=(0.0, 7.0),
    )
    second.load(cmd)
    ramp_blue = list(cmd.objects["vol_volume_color_ramp"])
    assert ramp_red != ramp_blue
    assert cmd.object_types.get("vol") == "object:volume"
    assert cmd.objects["vol"]["ramp"] == "vol_volume_color_ramp"


def test_isosurface_load_passes_side_isomesh_omits():
    from tests.fakes.cmd import FakeCmd
    from pymolviz.volumetric.IsoMesh import IsoMesh
    from pymolviz.volumetric.IsoSurface import IsoSurface

    grid = _tiny_grid()
    cmd = FakeCmd()
    cmd.load_brick(grid, grid.name)
    iso_kwargs = []
    mesh_kwargs = []
    orig_iso = cmd.isosurface
    orig_mesh = cmd.isomesh

    def capture_iso(*args, **kwargs):
        iso_kwargs.append(dict(kwargs))
        return orig_iso(*args, **kwargs)

    def capture_mesh(*args, **kwargs):
        mesh_kwargs.append(dict(kwargs))
        return orig_mesh(*args, **kwargs)

    cmd.isosurface = capture_iso
    cmd.isomesh = capture_mesh

    surf = make_field_visual("IsoSurface", grid, "iso", level=1.0, side=-1)
    surf.load(cmd)
    wire = make_field_visual("IsoMesh", grid, "mesh", level=0.5, side=-1)
    wire.load(cmd)

    assert iso_kwargs[-1].get("side") == -1
    assert "side" not in mesh_kwargs[-1]
    assert "side = -1" in IsoSurface(grid, 1.0, name="s", side=-1)._script_string()
    assert "side" not in IsoMesh(grid, 1.0, name="m", side=-1)._script_string()


def test_volume_clip_reloads_cropped_map_not_the_field():
    from tests.fakes.cmd import FakeCmd

    values = np.arange(27, dtype=float)
    grid = GridData(
        values,
        step_sizes=(1.0, 1.0, 1.0),
        step_counts=(2, 2, 2),
        origin=(0.0, 0.0, 0.0),
        name="density",
    )
    cmd = FakeCmd()
    cmd.load_brick(grid, "density")
    full = cmd.extents["density"]
    volume = make_field_visual(
        "Volume",
        grid,
        "vol",
        clip_aabb=[[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]],
        clims=(0.0, 26.0),
    )
    volume.load(cmd)
    clip_name = clip_map_name(grid)
    assert clip_name == "density_clip"
    assert clip_name in cmd.objects
    assert cmd.objects["vol"]["map"] == clip_name
    assert cmd.extents["density"] == full
    volume.clip_aabb = [[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]]
    volume.load(cmd)
    assert cmd.objects["vol"]["map"] == clip_name
    assert "density" in cmd.objects


def test_load_geometry_map_without_clip_keeps_source_map():
    from tests.fakes.cmd import FakeCmd

    grid = _tiny_grid()
    cmd = FakeCmd()
    name, rebuilt = load_geometry_map(cmd, grid, None)
    assert name == grid.name
    assert rebuilt is False
    assert name in cmd.objects


def test_visual_detail_text_separates_geometry_and_color():
    grid = _tiny_grid()
    volume = make_field_visual("Volume", grid, "density_volume", obj_id="vol1")
    volume.color_field_id = "tint"
    rows = field_library_rows([grid, volume], cmd=None)
    visual = [row for row in rows if row["kind"] == KIND_VISUAL][0]
    text = field_library_detail_text(visual)
    assert text.startswith("Geometry:")
    assert "Color:" in text
    assert "tint" in text
    assert field_row_shows_edit({"kind": KIND_FIELD, "editor": None}) is False
    assert field_row_shows_edit({"kind": KIND_FIELD, "editor": "FromSelection"}) is True
    assert nest_connector(None) == NEST_NONE


def test_from_selection_field_row_is_editable():
    from pymolviz.fields import Domain, Field, GEN_GAUSSIAN
    from pymolviz.wizards.builders.field_visual import field_visual_options
    from pymolviz.wizards.catalog import field_row, field_source_editor_kind

    field = Field(
        name="blob",
        obj_id="fld1",
        generator={
            "type": GEN_GAUSSIAN,
            "atoms": [{"xyz": [0.0, 0.0, 0.0], "elem": "C"}],
            "quality": 3,
            "resolution": 2.0,
        },
        domain=Domain(),
    )
    row = field_row(field)
    assert field_source_editor_kind(field) == "FromSelection"
    assert row["editor"] == "FromSelection"
    assert field_row_shows_edit(row) is True
    iso = make_field_visual("IsoSurface", _tiny_grid(), "blob_iso", level=1.25, obj_id="iso_opts", color=(0.1, 0.2, 0.3))
    opts = field_visual_options(iso)
    assert opts["kind"] == "IsoSurface"
    assert opts["level"] == pytest.approx(1.25)
    assert opts["color"][0] == pytest.approx(0.1)


def test_kind_badge_rgb_matches_catalog_hues():
    from pymolviz.wizards.catalog import kind_badge_rgb, visual_icon_kind, visual_icon_rgb

    dist_fill, dist_ink = kind_badge_rgb("Distance")
    gauss_fill, gauss_ink = kind_badge_rgb("Gaussian")
    map_fill, map_ink = kind_badge_rgb("Map")
    iso_fill, iso_ink = kind_badge_rgb("IsoSurface")
    mesh_fill, mesh_ink = kind_badge_rgb("IsoMesh")
    assert dist_fill[1] > dist_fill[0]
    assert dist_ink[1] > dist_ink[0]
    assert gauss_fill[2] > gauss_fill[0]
    assert map_fill[2] > map_fill[0]
    assert iso_fill[0] > iso_fill[2]
    assert iso_fill[1] > iso_fill[2]
    assert mesh_fill[0] > mesh_fill[2]
    assert mesh_fill[1] > 200
    unknown_fill, _ink = kind_badge_rgb("NotAType")
    assert unknown_fill == kind_badge_rgb("")[0]
    assert visual_icon_kind("IsoSurface") == "surface"
    assert visual_icon_kind("IsoMesh") == "isomesh"
    assert visual_icon_rgb("IsoSurface")[0] > visual_icon_rgb("IsoSurface")[2]
    assert visual_icon_rgb("IsoMesh")[1] > visual_icon_rgb("IsoMesh")[0]
    assert visual_icon_kind("Distance") is None
    sphere_fill, _sphere_ink = kind_badge_rgb("Spheres")
    arrows_fill, _arrows_ink = kind_badge_rgb("Arrows")
    assert sphere_fill[2] > sphere_fill[0]
    assert arrows_fill[1] > arrows_fill[0]
    from pymolviz.wizards.catalog import type_card_icon_rgb

    assert type_card_icon_rgb("volume") == visual_icon_rgb("Volume")
    assert type_card_icon_rgb("surface") == visual_icon_rgb("IsoSurface")


def test_isosurface_nests_under_native_map():
    _NATIVE_GRIDS.clear()
    values = np.arange(8, dtype=float).reshape(2, 2, 2)
    cmd = _map_cmd(values)
    grid = grid_from_pymol_map("density", cmd=cmd)
    surf = make_field_visual("IsoSurface", grid, "density_iso", level=1.0, obj_id="iso1")
    rows = field_library_rows([surf], cmd=cmd)
    assert field_library_field_count(rows) == 1
    assert rows[0]["native"] is True
    visuals = [row for row in rows if row["kind"] == KIND_VISUAL]
    assert len(visuals) == 1
    assert visuals[0]["id"] == "iso1"
    assert visuals[0]["type"] == "IsoSurface"
    assert is_field(surf) is True


def test_orphan_visual_still_gets_a_field_parent():
    grid = _tiny_grid(name="orphan_grid", obj_id="orphan_grid")
    mesh = make_field_visual("IsoMesh", grid, "orphan_mesh", level=0.5, obj_id="mesh1")
    rows = field_library_rows([mesh], cmd=None)
    assert [row["kind"] for row in rows] == [KIND_FIELD, KIND_VISUAL, KIND_ADD_VISUAL]
    assert rows[0]["name"] == "orphan_grid"
    assert rows[1]["id"] == "mesh1"


def test_field_kind_from_path_and_default_name():
    assert field_kind_from_path("/tmp/map.ccp4") == "map"
    assert field_kind_from_path("density.xyz") == "xyz"
    assert field_kind_from_path("refl.mtz") == "mtz"
    assert field_kind_from_path("cube.txt") == "orca"
    assert default_field_name("/data/foo bar.ccp4") == "foo_bar"


def test_persist_field_visual_records_session(monkeypatch):
    from pymolviz.runtime import session as session_mod
    from pymolviz.wizards.builders.field_visual import persist_field_visual

    session_mod.clear()
    grid = _tiny_grid()
    grid.is_loaded = True
    visual = make_field_visual("IsoSurface", grid, "iso", level=1.0, obj_id="iso_persist")
    monkeypatch.setattr(visual, "load", lambda: None)
    persist_field_visual(None, visual)
    assert session_mod.get("iso_persist") is visual
    session_mod.clear()


def test_persist_volume_loads_map_when_native_wrapper_stale():
    from tests.fakes.cmd import FakeCmd
    from pymolviz.runtime import session as session_mod
    from pymolviz.util.field_sample import PYMOL_MAP_ID_PREFIX
    from pymolviz.wizards.builders.field_visual import persist_field_visual

    session_mod.clear()
    grid = _tiny_grid()
    grid.id = PYMOL_MAP_ID_PREFIX + "density"
    grid.is_loaded = True
    cmd = FakeCmd()
    visual = make_field_visual("Volume", grid, "density_volume", obj_id="vol_persist")
    persist_field_visual(cmd, visual)
    names = [str(n) for n in cmd.get_names("objects")]
    assert "density" in names
    assert "density_volume" in names
    assert cmd.object_types.get("density_volume") == "object:volume"
    assert session_mod.get("vol_persist") is visual
    from pymolviz.runtime.runtime import get_runtime

    binding = get_runtime(cmd).bindings.get("vol_persist")
    assert binding is not None
    assert binding.pymol_name == "density_volume"
    session_mod.clear()
    grid = _tiny_grid()
    assert default_visual_name("Volume", grid) == "density_volume"
    assert default_iso_level(grid) == np.mean(np.arange(8, dtype=float))
    iso = make_field_visual("IsoSurface", grid, "iso")
    assert iso.level == default_iso_level(grid)
    vol = make_field_visual("IsoVolume", grid, "ivol")
    assert type(vol).__name__ == "IsoVolume"
    assert vol.grid_data is grid


def test_persist_second_field_visual_gets_numbered_name():
    from tests.fakes.cmd import FakeCmd
    from pymolviz.runtime import session as session_mod
    from pymolviz.wizards.builders.field_visual import persist_field_visual

    session_mod.clear()
    grid = _tiny_grid()
    grid.is_loaded = True
    cmd = FakeCmd()
    first = make_field_visual("IsoSurface", grid, "density_isosurface", level=1.0, obj_id="iso1")
    persist_field_visual(cmd, first)
    second = make_field_visual("IsoSurface", grid, "density_isosurface", level=1.5, obj_id="iso2")
    persist_field_visual(cmd, second)
    assert getattr(first, "_name", None) == "density_isosurface"
    assert getattr(second, "_name", None) == "density_isosurface_1"
    names = [str(n) for n in cmd.get_names("objects")]
    assert "density_isosurface" in names
    assert "density_isosurface_1" in names
    session_mod.clear()


def test_load_field_file_map_uses_pymol_object(tmp_path):
    _NATIVE_GRIDS.clear()
    from pymolviz.wizards.builders.load_field import load_field_file

    cmd = _map_cmd(np.zeros((2, 2, 2), dtype=float), name="kept")
    path = tmp_path / "density.ccp4"
    path.write_bytes(b"map")
    cmd.volume_fields["density"] = np.zeros((2, 2, 2), dtype=float)
    cmd.extents["density"] = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    grid = load_field_file(cmd, str(path), kind="map", name="density")
    assert grid is not None
    assert grid.id == PYMOL_MAP_ID_PREFIX + "density"
    assert "density" in cmd.objects
    assert cmd.object_types["density"] == "object:map"


def test_load_field_file_xyz(tmp_path):
    from pymolviz.wizards.builders.load_field import load_field_file
    from tests.fakes.cmd import FakeCmd

    path = tmp_path / "tiny.xyz"
    lines = []
    values = []
    n = 0
    for x in (0.0, 1.0):
        for y in (0.0, 1.0):
            for z in (0.0, 1.0):
                val = float(n)
                values.append(val)
                lines.append("%.1f %.1f %.1f %.1f" % (x, y, z, val))
                n += 1
    path.write_text("\n".join(lines) + "\n")
    cmd = FakeCmd()
    grid = load_field_file(cmd, str(path), kind="xyz", name="tiny")
    assert grid is not None
    assert str(grid._name) == "tiny"
    assert len(np.asarray(grid.values).reshape(-1)) == 8


def test_grid_from_implicit_atoms_registers_normalized_field():
    from pymolviz.runtime import session as session_mod
    from pymolviz.wizards.builders.load_field import (
        centers_and_elements_from_points,
        grid_from_implicit_atoms,
    )
    from pymolviz.wizards.builders.points import AtomRef, VisualPoint
    from pymolviz.points import FixedPoint
    from tests.fakes.cmd import FakeCmd

    session_mod.clear()
    carbon = VisualPoint(
        "C1", "manual", 0.0, 0.0, 0.0,
        point_source=FixedPoint((0.0, 0.0, 0.0)),
        atom_ref=AtomRef("m", 1, elem="C"),
    )
    oxygen = VisualPoint(
        "O1", "manual", 1.2, 0.0, 0.0,
        point_source=FixedPoint((1.2, 0.0, 0.0)),
        atom_ref=AtomRef("m", 2, elem="O"),
    )
    disabled = VisualPoint(
        "skip", "manual", 9.0, 9.0, 9.0,
        enabled=False,
        point_source=FixedPoint((9.0, 9.0, 9.0)),
    )
    centers, elements = centers_and_elements_from_points([carbon, oxygen, disabled])
    assert elements == ["C", "O"]
    assert len(centers) == 2
    cmd = FakeCmd()
    grid = grid_from_implicit_atoms(
        cmd, centers, elements, "blob", quality=1,
    )
    assert grid is not None
    brick = getattr(grid, "grid_data", grid)
    assert str(getattr(grid, "_name", None) or brick._name) == "blob"
    values = np.asarray(grid.values, dtype=float)
    assert values.size == int(np.prod(np.asarray(brick.step_counts) + 1))
    assert float(values.mean()) == pytest.approx(0.0, abs=1e-5)
    assert session_mod.get(grid.id) is grid
    session_mod.clear()


class _FakeBuilderPage:
    def __init__(self, *a, **k):
        self.widget = object()
        self.loaded = None
        self.created = None

    def load_object(self, obj):
        self.loaded = obj

    def reset_for_create(self, kind, field):
        self.created = (kind, field)

    def cleanup_preview(self):
        pass


class _FakeStack:
    def __init__(self):
        self.widgets = []
        self.current = None

    def addWidget(self, widget):
        if widget is None:
            raise AttributeError("addWidget on None")
        self.widgets.append(widget)
        return len(self.widgets) - 1

    def setCurrentIndex(self, index):
        self.current = index


def test_edit_visual_does_not_call_addwidget_on_none_stack(monkeypatch):
    from types import SimpleNamespace

    from pymolviz.wizards import field_visuals as fv_mod
    from tests.fakes.cmd import FakeCmd

    monkeypatch.setattr(fv_mod, "FieldVisualBuilderPage", _FakeBuilderPage)
    wizard = SimpleNamespace(cmd=FakeCmd(), prompt=[])
    obj = SimpleNamespace(id="iso1", name="iso", _name="iso")

    catalog_only = fv_mod.FieldVisualsWindow(wizard)
    catalog_only._window = SimpleNamespace()
    catalog_only._stack = None
    catalog_only._open_visual_editor(obj)
    assert catalog_only._builder_page is not None
    assert catalog_only._builder_page.loaded is obj
    assert catalog_only._stack is None
    assert catalog_only._builder_stack_index is None

    with_stack = fv_mod.FieldVisualsWindow(wizard)
    with_stack._window = SimpleNamespace()
    with_stack._stack = _FakeStack()
    with_stack._open_visual_editor(obj)
    assert with_stack._builder_stack_index == 0
    assert with_stack._stack.widgets[0] is with_stack._builder_page.widget
    assert with_stack._stack.current == 0
    assert with_stack._builder_page.loaded is obj


def test_confirm_delete_uses_overlay_question():
    import inspect

    from pymolviz.wizards.add_visual import AddVisualWindow
    from pymolviz.wizards.field_visuals import FieldVisualsWindow

    field_src = inspect.getsource(FieldVisualsWindow._confirm_delete)
    visual_src = inspect.getsource(AddVisualWindow._confirm_delete)
    delete_src = inspect.getsource(FieldVisualsWindow._on_delete_row)
    assert "overlay_question" in field_src
    assert "overlay_question" in visual_src
    assert "QMessageBox.question" not in field_src
    assert "QMessageBox.question" not in visual_src
    assert "Delete this field and its dependents?" in delete_src

    from pymolviz.wizards.builders.field_page import FieldVisualBuilderPage
    from pymolviz.wizards.builders.from_selection_page import FromSelectionFieldPage
    from pymolviz.wizards.builders.surface_page import SurfaceBuilderPage

    assert "overlay_question" in inspect.getsource(FieldVisualBuilderPage._confirm_heavy_iso)
    assert "overlay_question" in inspect.getsource(FromSelectionFieldPage._confirm_heavy_map)
    assert "overlay_question" in inspect.getsource(SurfaceBuilderPage._confirm_heavy_surface)

