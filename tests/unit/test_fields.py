"""Field identity, generators, wrapping, and visual helpers (no Qt, no pymolviz.wizard)."""

from __future__ import annotations

import numpy as np
import pytest

from pymolviz.fields import (
    GEN_DERIVED,
    GEN_DISTANCE,
    GEN_GAUSSIAN,
    GEN_IMPORTED,
    GEN_NEAREST_PROP,
    GEN_SIGNED_VDW,
    KIND_CATEGORICAL,
    KIND_SCALAR,
    Domain,
    Field,
    as_field,
    canonical_field_spec,
    clim_range,
    crop_grid_to_aabb,
    dependents_of_field,
    ensure_brick,
    field_identity_hash,
    intern_field,
    isosurface_mesh_from_grid,
    normalize_isovalues,
    register_generated_field,
    specs_equal,
)
from pymolviz.fields.domain import BOUNDS_AROUND_SELECTION, BOUNDS_CUSTOM_BOX
from pymolviz.volumetric.GridData import GridData
from pymolviz.wizards.builders.load_field import (
    atom_records_from_points,
    centers_and_elements_from_points,
)
from pymolviz.wizards.builders.points import AtomRef, VisualPoint
from pymolviz.points import FixedPoint


def _gaussian_field(padding=0.0, spacing=1.0, name="blob"):
    atoms = [
        {"xyz": [0.0, 0.0, 0.0], "elem": "C"},
        {"xyz": [1.2, 0.0, 0.0], "elem": "O"},
    ]
    return Field(
        name=name,
        generator={
            "type": GEN_GAUSSIAN,
            "atoms": atoms,
            "quality": 1,
            "resolution": 2.0,
            "b_floor": 20.0,
        },
        domain=Domain(
            bounds_mode=BOUNDS_AROUND_SELECTION,
            padding=padding,
            spacing=spacing,
        ),
    )


def test_equivalent_gaussian_specs_match_and_exclude_display_name():
    a = _gaussian_field(name="one")
    b = _gaussian_field(name="two")
    spec_a = canonical_field_spec(a)
    spec_b = canonical_field_spec(b)
    assert specs_equal(spec_a, spec_b)
    assert field_identity_hash(spec_a) == field_identity_hash(spec_b)
    assert spec_a.get("name") is None
    assert a.name != b.name


def test_different_padding_is_a_different_field():
    a = _gaussian_field(padding=0.0)
    b = _gaussian_field(padding=2.0)
    assert not specs_equal(canonical_field_spec(a), canonical_field_spec(b))
    assert field_identity_hash(canonical_field_spec(a)) != field_identity_hash(
        canonical_field_spec(b)
    )


def test_field_does_not_render_cgo():
    from pymolviz.runtime.renderer import cgo_tokens, renders_cgo, resolved_cgo_tokens

    field = _gaussian_field()

    def boom(*_a, **_k):
        raise AssertionError("Field must not produce CGO")

    field._create_CGO_list = boom
    assert field.renders_cgo is False
    assert field.is_visual is False
    assert renders_cgo(field) is False
    assert resolved_cgo_tokens(field, None) == []
    assert cgo_tokens(field, None) == []


def test_imported_vs_generated_identity():
    generated = _gaussian_field()
    values = np.arange(8, dtype=float)
    grid = GridData(
        values,
        step_sizes=(1.0, 1.0, 1.0),
        step_counts=(1, 1, 1),
        origin=(0.0, 0.0, 0.0),
        name="imported",
    )
    imported = as_field(grid)
    assert imported.generator["type"] == GEN_IMPORTED
    assert not specs_equal(canonical_field_spec(generated), canonical_field_spec(imported))


def test_atom_order_does_not_change_identity():
    atoms_a = [
        {"xyz": [0.0, 0.0, 0.0], "elem": "C", "atom_id": 1},
        {"xyz": [1.0, 0.0, 0.0], "elem": "O", "atom_id": 2},
    ]
    atoms_b = list(reversed(atoms_a))
    a = Field(generator={"type": GEN_DISTANCE, "atoms": atoms_a}, domain=Domain(spacing=0.5))
    b = Field(generator={"type": GEN_DISTANCE, "atoms": atoms_b}, domain=Domain(spacing=0.5))
    assert specs_equal(canonical_field_spec(a), canonical_field_spec(b))


def test_centers_and_elements_from_points_skips_disabled():
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
    records = atom_records_from_points([carbon, oxygen, disabled])
    assert len(records) == 2
    assert {row["elem"] for row in records} == {"C", "O"}


def test_intern_field_reuses_equivalent_gaussian(monkeypatch):
    from pymolviz.runtime import session as session_mod

    session_mod.clear()
    first = register_generated_field(_gaussian_field(name="a"))
    second = intern_field(_gaussian_field(name="b"))
    assert second is first
    assert session_mod.get(first.id) is first
    session_mod.clear()


def test_ensure_brick_gaussian_shape_not_golden_dump():
    field = _gaussian_field(spacing=1.0)
    grid = ensure_brick(field)
    assert grid is not None
    values = np.asarray(grid.values, dtype=float)
    assert values.size == int(np.prod(np.asarray(grid.step_counts) + 1))
    assert float(values.mean()) == pytest.approx(0.0, abs=1e-4)
    assert field.grid_data is grid


def test_distance_and_signed_vdw_generators():
    atoms = [
        {"xyz": [0.0, 0.0, 0.0], "elem": "C", "vdw": 1.7},
        {"xyz": [2.0, 0.0, 0.0], "elem": "O", "vdw": 1.52},
    ]
    domain = Domain(bounds_mode=BOUNDS_AROUND_SELECTION, padding=1.0, spacing=1.0)
    dist = Field(generator={"type": GEN_DISTANCE, "atoms": atoms}, domain=domain, units="Å")
    vdw = Field(generator={"type": GEN_SIGNED_VDW, "atoms": atoms}, domain=domain, units="Å")
    dgrid = ensure_brick(dist)
    sgrid = ensure_brick(vdw)
    assert dgrid is not None and sgrid is not None
    dvals = np.asarray(dgrid.values, dtype=float)
    svals = np.asarray(sgrid.values, dtype=float)
    assert dvals.size == svals.size
    assert float(np.min(dvals)) == pytest.approx(0.0, abs=0.6)
    assert float(np.min(svals)) < 0.0
    assert dist.units == "Å"


def test_nearest_atom_property_categorical_kind():
    atoms = [
        {"xyz": [0.0, 0.0, 0.0], "elem": "C", "chain": "A", "b_factor": 10.0},
        {"xyz": [2.0, 0.0, 0.0], "elem": "O", "chain": "B", "b_factor": 40.0},
    ]
    domain = Domain(padding=1.0, spacing=1.0)
    cat = Field(
        generator={"type": GEN_NEAREST_PROP, "atoms": atoms, "property": "elem", "color_snapshot": True},
        domain=domain,
    )
    grid = ensure_brick(cat)
    assert grid is not None
    assert cat.kind == KIND_CATEGORICAL
    assert cat.categories is not None
    assert set(cat.categories) >= {"C", "O"}
    bf = Field(
        generator={"type": GEN_NEAREST_PROP, "atoms": atoms, "property": "b_factor", "color_snapshot": True},
        domain=domain,
    )
    bgrid = ensure_brick(bf)
    assert bf.kind == KIND_SCALAR
    assert float(np.max(np.asarray(bgrid.values))) >= 10.0


def test_derived_generator_is_schema_only():
    field = Field(
        generator={"type": GEN_DERIVED, "op": "add", "source_a": "a", "source_b": "b"},
        domain=Domain(),
    )
    spec = canonical_field_spec(field)
    assert spec["generator"]["type"] == GEN_DERIVED
    assert spec["generator"]["op"] == "add"
    assert ensure_brick(field) is None


def test_wrap_griddata_as_field_keeps_id():
    grid = GridData(
        np.arange(8, dtype=float),
        step_sizes=(1.0, 1.0, 1.0),
        step_counts=(1, 1, 1),
        origin=(0.0, 0.0, 0.0),
        name="density",
    )
    grid.id = "grid-wrap"
    field = as_field(grid)
    assert type(field).__name__ == "Field"
    assert field.id == "grid-wrap"
    assert field.grid_data is grid
    assert as_field(grid) is field


def test_isovalues_and_clim_range_shapes():
    entries = normalize_isovalues(None, default_level=1.5, default_side=-1)
    assert len(entries) == 1
    assert entries[0]["value"] == pytest.approx(1.5)
    assert entries[0]["side"] == -1
    both = normalize_isovalues(
        [{"value": 1.0, "side": 1}, {"value": 1.0, "side": -1}]
    )
    assert [row["side"] for row in both] == [1, -1]
    values = np.array([-2.0, 5.0])
    assert clim_range("auto", values) is None
    assert clim_range("symmetric", values) == [-5.0, 5.0]
    assert clim_range("custom", values, custom=(1.0, 0.0)) == [0.0, 1.0]


def test_crop_grid_to_aabb_keeps_step():
    grid = GridData(
        np.arange(27, dtype=float),
        step_sizes=(1.0, 1.0, 1.0),
        step_counts=(2, 2, 2),
        origin=(0.0, 0.0, 0.0),
        name="box",
    )
    cropped = crop_grid_to_aabb(grid, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert cropped is not None
    assert list(np.asarray(cropped.step_sizes, dtype=float)) == pytest.approx([1.0, 1.0, 1.0])
    assert int(np.prod(np.asarray(cropped.step_counts) + 1)) == len(cropped.values)


def test_isosurface_mesh_from_tiny_grid():
    values = np.zeros((3, 3, 3), dtype=float)
    values[1, 1, 1] = 1.0
    grid = GridData(
        values.reshape(-1),
        step_sizes=(1.0, 1.0, 1.0),
        step_counts=(2, 2, 2),
        origin=(0.0, 0.0, 0.0),
        name="iso",
    )
    vertices, normals, faces = isosurface_mesh_from_grid(grid, 0.5)
    assert faces.shape[0] >= 1
    assert vertices.shape[0] >= 3
    assert normals.shape[0] == vertices.shape[0]


def test_dependents_of_field_lists_visual_and_mesh():
    from pymolviz.meshes.Sphere import Sphere
    from pymolviz.wizards.builders.field_visual import make_field_visual

    field = _gaussian_field()
    grid = ensure_brick(field)
    visual = make_field_visual("IsoSurface", field, "iso", level=0.0, obj_id="iso1")
    sphere = Sphere(FixedPoint((0.0, 0.0, 0.0)), 0.4, bypass_colormap=True, obj_id="sph1")
    sphere.field_id = str(field.id)
    deps = dependents_of_field(field, [field, visual, sphere])
    kinds = {type(obj).__name__ for obj in deps}
    assert "IsoSurface" in kinds
    assert "Sphere" in kinds


def test_field_visual_binds_geometry_color_isovalues_and_clip():
    from pymolviz.fields.isovalues import isovalues_for_side
    from pymolviz.wizards.builders.field_visual import make_field_visual

    field = _gaussian_field()
    ensure_brick(field)
    entries = isovalues_for_side(0.25, "both")
    visual = make_field_visual(
        "IsoSurface",
        field,
        "iso",
        isovalues=entries,
        clip_aabb=[[-1.0, -1.0, -1.0], [2.0, 2.0, 2.0]],
        geometry_field_id=field.id,
        color_field_id=field.id,
        obj_id="iso_bind",
    )
    assert visual.geometry_field_id == str(field.id)
    assert visual.color_field_id == str(field.id)
    assert len(visual.isovalues) == 2
    assert {row["side"] for row in visual.isovalues} == {1, -1}
    assert visual.clip_aabb is not None


def test_field_from_selection_presets_and_reuse():
    from pymolviz.runtime import session as session_mod
    from pymolviz.wizards.builders.load_field import field_from_selection
    from pymolviz.wizards.builders.points import AtomRef, VisualPoint
    from tests.fakes.cmd import FakeCmd

    session_mod.clear()
    pts = [
        VisualPoint(
            "C1", "manual", 0.0, 0.0, 0.0,
            point_source=FixedPoint((0.0, 0.0, 0.0)),
            atom_ref=AtomRef("m", 1, elem="C"),
        ),
        VisualPoint(
            "O1", "manual", 1.5, 0.0, 0.0,
            point_source=FixedPoint((1.5, 0.0, 0.0)),
            atom_ref=AtomRef("m", 2, elem="O"),
        ),
    ]
    cmd = FakeCmd()
    domain = Domain(padding=1.0, spacing=1.0)
    dist = field_from_selection(
        cmd, pts, "dist", algorithm=GEN_DISTANCE, domain=domain,
    )
    again = field_from_selection(
        cmd, pts, "dist2", algorithm=GEN_DISTANCE, domain=domain,
    )
    assert dist is not None
    assert again is dist
    vdw = field_from_selection(
        cmd, pts, "vdw", algorithm=GEN_SIGNED_VDW, domain=domain,
    )
    assert vdw is not None
    assert vdw is not dist
    brick = ensure_brick(vdw)
    assert brick is not None
    session_mod.clear()


def test_convert_isosurface_stamps_provenance():
    from pymolviz.fields.convert import convert_isosurface_to_surface
    from pymolviz.wizards.builders.field_visual import make_field_visual

    values = np.zeros((3, 3, 3), dtype=float)
    values[1, 1, 1] = 1.0
    grid = GridData(
        values.reshape(-1),
        step_sizes=(1.0, 1.0, 1.0),
        step_counts=(2, 2, 2),
        origin=(0.0, 0.0, 0.0),
        name="iso",
    )
    field = as_field(grid)
    visual = make_field_visual("IsoSurface", field, "iso", level=0.5, obj_id="iso_conv")
    surface = convert_isosurface_to_surface(visual, name="baked")
    assert surface.created_from["field_id"] == str(field.id)
    assert surface.created_from["isovalue"] == pytest.approx(0.5)
    assert np.asarray(surface.vertices).shape[0] >= 3


def test_apply_point_color_keeps_field_id():
    from pymolviz.meshes.Sphere import Sphere
    from pymolviz.util.field_sample import _NATIVE_GRIDS
    from pymolviz.wizards.builders.preview import apply_point_color_to_mesh
    from pymolviz.wizards.builders.points import VisualPoint

    values = np.linspace(0.0, 1.0, 8)
    grid = GridData(
        values,
        step_sizes=(1.0, 1.0, 1.0),
        step_counts=(1, 1, 1),
        origin=(0.0, 0.0, 0.0),
        name="density",
    )
    _NATIVE_GRIDS[str(grid.id)] = grid
    sphere = Sphere(FixedPoint((0.5, 0.5, 0.5)), 0.4, bypass_colormap=True, frequency=2)
    pt = VisualPoint(
        "a", "manual", 0.5, 0.5, 0.5,
        field_id=str(grid.id),
        field_colormap="viridis",
        point_source=FixedPoint((0.5, 0.5, 0.5)),
    )
    apply_point_color_to_mesh(sphere, pt)
    assert sphere.field_id == str(grid.id)
    _NATIVE_GRIDS.pop(str(grid.id), None)


def test_delete_field_removes_dependents():
    from pymolviz.runtime import session as session_mod
    from pymolviz.wizards.builders.field_visual import (
        delete_field_and_visuals,
        make_field_visual,
    )

    session_mod.clear()
    field = register_generated_field(_gaussian_field(name="delme"))
    visual = make_field_visual("IsoMesh", field, "mesh", level=0.0, obj_id="mesh_del")
    visual.load = lambda: None
    session_mod.add(visual)
    delete_field_and_visuals(None, field)
    assert session_mod.get(str(field.id)) is None
    assert session_mod.get("mesh_del") is None
    session_mod.clear()


def _colored_points(c0, c1):
    return [
        VisualPoint(
            "C1", "manual", 0.0, 0.0, 0.0,
            color=c0,
            point_source=FixedPoint((0.0, 0.0, 0.0)),
            atom_ref=AtomRef("m", 1, elem="C"),
        ),
        VisualPoint(
            "O1", "manual", 1.5, 0.0, 0.0,
            color=c1,
            point_source=FixedPoint((1.5, 0.0, 0.0)),
            atom_ref=AtomRef("m", 2, elem="O"),
        ),
    ]


def test_preview_fields_with_register_false_do_not_intern():
    from pymolviz.runtime import session as session_mod
    from pymolviz.wizards.builders.load_field import field_from_selection
    from tests.fakes.cmd import FakeCmd

    session_mod.clear()
    cmd = FakeCmd()
    pts = _colored_points((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    domain = Domain(padding=1.0, spacing=1.0)
    geom = field_from_selection(
        cmd, pts, "preview_geom", algorithm=GEN_DISTANCE, domain=domain, register=False,
    )
    assert geom is not None
    assert session_mod.get(str(geom.id)) is None
    session_mod.clear()


def _stub_preset_visual_loads(monkeypatch, cmd=None):
    from pymolviz.fields.field import Field
    from pymolviz.volumetric.GridData import GridData
    from pymolviz.volumetric.IsoSurface import IsoSurface

    field_loads = []
    visual_loads = []

    monkeypatch.setattr(Field, "load", lambda self: field_loads.append(self))
    monkeypatch.setattr(GridData, "load", lambda self: setattr(self, "is_loaded", True))

    def iso_load(self):
        visual_loads.append(self)
        if cmd is not None:
            map_name = getattr(getattr(self, "grid_data", None), "name", None) or "map"
            cmd.isosurface(self.name, map_name, getattr(self, "level", 0.0))
        self.is_loaded = True

    monkeypatch.setattr(IsoSurface, "load", iso_load)
    return field_loads, visual_loads


def test_from_selection_commit_creates_native_isosurface_uniform(monkeypatch):
    from pymolviz.runtime import session as session_mod
    from pymolviz.volumetric.ColorRamp import ColorRamp
    from pymolviz.wizards.builders.load_field import (
        commit_from_selection_preset,
        default_color_field_id,
    )
    from pymolviz.wizards.catalog import KIND_FIELD, KIND_VISUAL, field_library_rows
    from tests.fakes.cmd import FakeCmd

    session_mod.clear()
    cmd = FakeCmd()
    field_loads, visual_loads = _stub_preset_visual_loads(monkeypatch, cmd)
    pts = _colored_points((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    domain = Domain(padding=1.0, spacing=1.0)
    geom, color, visual = commit_from_selection_preset(
        cmd, pts, "blob", algorithm=GEN_DISTANCE, domain=domain,
    )
    assert geom is not None and color is None and visual is not None
    assert type(visual).__name__ == "IsoSurface"
    assert visual.geometry_field_id == str(geom.id)
    assert visual.color_field_id is None
    assert default_color_field_id(geom) is None
    assert visual.level == pytest.approx(0.0)
    assert field_loads == []
    assert visual_loads == [visual]
    assert visual.name in cmd.objects
    assert cmd.object_types.get(visual.name) == "object:isosurface"
    assert not isinstance(visual.color, ColorRamp)
    assert tuple(float(c) for c in visual.color[:3]) == pytest.approx((1.0, 0.0, 0.0))
    assert session_mod.get(str(visual.id)) is visual
    rows = field_library_rows(session_mod.all_objects(), cmd=cmd)
    geom_row = next(row for row in rows if row["kind"] == KIND_FIELD and row["id"] == str(geom.id))
    assert geom_row["used_by"] == "1 visual"
    assert any(row["kind"] == KIND_VISUAL and row["id"] == str(visual.id) for row in rows)
    session_mod.clear()


def test_from_selection_commit_from_field_binds_scalar_color(monkeypatch):
    from pymolviz.runtime import session as session_mod
    from pymolviz.volumetric.ColorRamp import ColorRamp
    from pymolviz.wizards.builders.load_field import (
        commit_from_selection_preset,
        default_color_field_id,
        field_from_selection,
    )
    from pymolviz.wizards.builders.surface_params import COLOR_MODE_FIELD
    from tests.fakes.cmd import FakeCmd

    session_mod.clear()
    cmd = FakeCmd()
    _stub_preset_visual_loads(monkeypatch, cmd)
    pts = _colored_points((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    domain = Domain(padding=1.0, spacing=1.0)
    tint = field_from_selection(
        cmd, pts, "tint", algorithm=GEN_DISTANCE, domain=domain,
    )
    geom, color, visual = commit_from_selection_preset(
        cmd, pts, "blob", algorithm=GEN_GAUSSIAN, domain=domain,
        quality=1, iso_level=0.5,
        color_mode=COLOR_MODE_FIELD,
        color_field_id=str(tint.id),
        colormap="viridis",
    )
    assert geom is not None and color is None and visual is not None
    assert visual.color_field_id == str(tint.id)
    assert default_color_field_id(geom) == str(tint.id)
    assert isinstance(visual.color, ColorRamp)
    session_mod.clear()


def test_from_selection_commit_ignores_legacy_per_point_mode(monkeypatch):
    from pymolviz.runtime import session as session_mod
    from pymolviz.volumetric.ColorRamp import ColorRamp
    from pymolviz.wizards.builders.load_field import (
        commit_from_selection_preset,
        default_color_field_id,
    )
    from pymolviz.wizards.builders.surface_params import COLOR_MODE_PER_POINT
    from tests.fakes.cmd import FakeCmd

    session_mod.clear()
    cmd = FakeCmd()
    _stub_preset_visual_loads(monkeypatch, cmd)
    pts = _colored_points((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    domain = Domain(padding=1.0, spacing=1.0)
    geom, color, visual = commit_from_selection_preset(
        cmd, pts, "blob", algorithm=GEN_DISTANCE, domain=domain,
        color_mode=COLOR_MODE_PER_POINT,
    )
    assert geom is not None and visual is not None
    assert color is None
    assert visual.color_field_id is None
    assert default_color_field_id(geom) is None
    assert not isinstance(visual.color, ColorRamp)
    session_mod.clear()


def test_from_selection_commit_reuses_field_and_does_not_duplicate_visual(monkeypatch):
    from pymolviz.runtime import session as session_mod
    from pymolviz.wizards.builders.load_field import commit_from_selection_preset
    from tests.fakes.cmd import FakeCmd

    session_mod.clear()
    cmd = FakeCmd()
    _stub_preset_visual_loads(monkeypatch, cmd)
    pts = _colored_points((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    domain = Domain(padding=1.0, spacing=1.0)
    geom, _color, visual = commit_from_selection_preset(
        cmd, pts, "dist", algorithm=GEN_DISTANCE, domain=domain,
    )
    again, _color2, visual2 = commit_from_selection_preset(
        cmd, pts, "dist2", algorithm=GEN_DISTANCE, domain=domain,
    )
    assert again is geom
    assert visual2 is visual
    isos = [obj for obj in session_mod.all_objects() if type(obj).__name__ == "IsoSurface"]
    assert len(isos) == 1
    session_mod.clear()


def test_from_selection_commit_gaussian_and_signed_vdw_iso(monkeypatch):
    from pymolviz.runtime import session as session_mod
    from pymolviz.util.gaussian_map import DEFAULT_GAUSSIAN_ISOLEVEL
    from pymolviz.wizards.builders.field_params import default_preset_iso_level
    from pymolviz.wizards.builders.load_field import commit_from_selection_preset
    from tests.fakes.cmd import FakeCmd

    session_mod.clear()
    cmd = FakeCmd()
    _stub_preset_visual_loads(monkeypatch, cmd)
    pts = _colored_points((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    domain = Domain(padding=2.0, spacing=1.0)
    gauss, _c, gauss_vis = commit_from_selection_preset(
        cmd, pts, "blob", algorithm=GEN_GAUSSIAN, domain=domain,
        quality=1, iso_level=1.5,
    )
    assert gauss_vis.level == pytest.approx(1.5)
    assert default_preset_iso_level(GEN_GAUSSIAN) == pytest.approx(DEFAULT_GAUSSIAN_ISOLEVEL)
    vdw, _c2, vdw_vis = commit_from_selection_preset(
        cmd, pts, "vdw", algorithm=GEN_SIGNED_VDW, domain=domain,
    )
    assert vdw is not gauss
    assert vdw_vis is not gauss_vis
    assert vdw_vis.level == pytest.approx(0.0)
    session_mod.clear()


def test_from_selection_commit_native_isosurface_not_cgo(monkeypatch):
    from pymolviz.runtime import session as session_mod
    from pymolviz.volumetric.ColorRamp import ColorRamp
    from pymolviz.volumetric.GridData import GridData
    from pymolviz.wizards.builders.field_preview import build_field_iso_preview_visual
    from pymolviz.wizards.builders.load_field import commit_from_selection_preset
    from tests.fakes.cmd import FakeCmd
    import pymol

    session_mod.clear()
    cmd = FakeCmd()
    monkeypatch.setattr(pymol, "cmd", cmd)
    monkeypatch.setattr(GridData, "load", lambda self: setattr(self, "is_loaded", True))
    pts = _colored_points((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    geom_domain = Domain(padding=2.0, spacing=0.90)
    geom, color, visual = commit_from_selection_preset(
        cmd, pts, "blob", algorithm=GEN_GAUSSIAN, domain=geom_domain,
        quality=1, iso_level=0.5,
    )
    assert geom is not None and color is None and visual is not None
    assert type(visual).__name__ == "IsoSurface"
    assert visual.color_field_id is None
    assert visual.geometry_field_id == str(geom.id)
    assert not isinstance(visual.color, ColorRamp)
    assert cmd.object_types.get(visual.name) == "object:isosurface"
    assert cmd.object_types.get(visual.name) != "object:cgo"

    session_mod.clear()
    preview = build_field_iso_preview_visual(
        cmd, pts, algorithm=GEN_GAUSSIAN, domain=geom_domain,
        quality=1, resolution=2.0, iso_level=0.5,
        name="_pmv_prev_iso",
    )
    assert preview is not None
    assert type(preview).__name__ == "IsoSurface"
    assert not isinstance(preview.color, ColorRamp)
    assert not any(type(obj).__name__ == "IsoSurface" for obj in session_mod.all_objects())
    session_mod.clear()


def test_color_ramp_discrete_slots_hold_category_rgb():
    from pymolviz.ColorMap import ColorMap
    from pymolviz.volumetric.ColorRamp import ColorRamp

    values = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=float)
    grid = GridData(
        values,
        step_sizes=(1.0, 1.0, 1.0),
        step_counts=(1, 1, 1),
        origin=(0.0, 0.0, 0.0),
        name="idx",
    )
    cmap = ColorMap([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], values_are_single_color=False)
    ramp = ColorRamp(grid, colormap=cmap, interpolate=False)
    assert ramp.interpolate is False
    ranges, colors = ramp._ramp_table()
    assert ranges[0] < 0.0 < ranges[1]
    assert ranges[2] < 1.0 < ranges[3]
    reds = np.asarray(colors[:2], dtype=float)
    blues = np.asarray(colors[2:], dtype=float)
    assert float(np.max(reds[:, 1])) < 0.2
    assert float(np.max(blues[:, 1])) < 0.2
    assert float(np.mean(reds[:, 0])) > 0.8
    assert float(np.mean(blues[:, 2])) > 0.8


def test_field_options_and_points_roundtrip_from_generator():
    from pymolviz.wizards.builders.load_field import (
        field_is_from_selection,
        field_options,
        points_from_field,
        remember_default_iso_level,
    )

    field = Field(
        name="sel",
        generator={
            "type": GEN_GAUSSIAN,
            "atoms": [
                {
                    "xyz": [1.0, 2.0, 3.0],
                    "elem": "N",
                    "object": "prot",
                    "atom_id": 12,
                    "chain": "A",
                    "resi": "10",
                    "name": "CA",
                    "b_factor": 18.5,
                }
            ],
            "quality": 4,
            "resolution": 1.5,
            "b_floor": 20.0,
        },
        domain=Domain(bounds_mode=BOUNDS_CUSTOM_BOX, padding=1.0, spacing=0.4, aabb=[[0, 0, 0], [4, 4, 4]]),
    )
    remember_default_iso_level(field, 0.8)
    assert field_is_from_selection(field) is True
    opts = field_options(field)
    assert opts["algorithm"] == GEN_GAUSSIAN
    assert opts["quality"] == 4
    assert opts["resolution"] == pytest.approx(1.5)
    assert opts["iso_level"] == pytest.approx(0.8)
    assert opts["domain"]["bounds_mode"] == BOUNDS_CUSTOM_BOX
    assert opts["domain"]["padding"] == pytest.approx(1.0)
    before = field_identity_hash(canonical_field_spec(field))
    remember_default_iso_level(field, 2.5)
    assert field_identity_hash(canonical_field_spec(field)) == before
    pts = points_from_field(field)
    assert len(pts) == 1
    assert pts[0].xyz() == pytest.approx((1.0, 2.0, 3.0))
    assert pts[0].atom_ref.model == "prot"
    assert pts[0].atom_ref.atom_id == 12
    recs = atom_records_from_points(pts)
    assert recs[0]["elem"] == "N"
    assert recs[0]["b_factor"] == pytest.approx(18.5)
