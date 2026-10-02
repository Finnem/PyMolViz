"""Shrink-to-object and double_map AABBs for field fitting."""

from __future__ import annotations

import numpy as np
import pytest

from pymolviz.fields.crystal import (
    DEFAULT_FIT_MARGIN,
    coverage_sketch,
    fit_grid_to_world_aabb,
    fit_result_aabb,
    map_world_aabb,
    scale_aabb_about_center,
)
from pymolviz.fields.field import Field


def test_fit_result_aabb_none_without_options():
    assert fit_result_aabb((0, 0, 0), (10, 10, 10), (2, 2, 2), (4, 4, 4)) is None


def test_fit_result_aabb_shrink_adds_margin():
    lo, hi = fit_result_aabb(
        (0, 0, 0), (20, 20, 20), (4, 5, 6), (6, 7, 8),
        shrink=True, margin=2.0,
    )
    assert lo == pytest.approx([2.0, 3.0, 4.0])
    assert hi == pytest.approx([8.0, 9.0, 10.0])


def test_fit_result_aabb_double_map_does_not_grow_box():
    assert fit_result_aabb((0, 0, 0), (10, 10, 10), double_map=True) is None


def test_fit_result_aabb_shrink_ignores_double_map():
    lo, hi = fit_result_aabb(
        (0, 0, 0), (40, 40, 40), (10, 10, 10), (14, 14, 14),
        shrink=True, margin=1.0, double_map=True,
    )
    assert lo == pytest.approx([9.0, 9.0, 9.0])
    assert hi == pytest.approx([15.0, 15.0, 15.0])


def test_upsample_map_double_keeps_extent_and_odd_shape():
    from pymolviz.fields.crystal_symmetrize import double_map_sample_shape, upsample_map_double
    from pymolviz.util.field_sample import grid_values_3d

    values = np.arange(5 * 4 * 3, dtype=float)
    grid = Field(
        values,
        step_sizes=(2.0, 2.0, 2.0),
        step_counts=(4, 3, 2),
        origin=(1.0, 2.0, 3.0),
        name="coarse",
    )
    dest = upsample_map_double(grid)
    assert grid_values_3d(dest).shape == double_map_sample_shape((5, 4, 3))
    lo0, hi0 = map_world_aabb(grid)
    lo1, hi1 = map_world_aabb(dest)
    assert lo0 == pytest.approx(lo1)
    assert hi0 == pytest.approx(hi1)
    assert np.asarray(dest.step_sizes, dtype=float) == pytest.approx([1.0, 1.0, 1.0])


def test_scale_aabb_about_center():
    lo, hi = scale_aabb_about_center((0, 0, 0), (4, 2, 6), 2.0)
    assert lo == pytest.approx([-2.0, -1.0, -3.0])
    assert hi == pytest.approx([6.0, 3.0, 9.0])


def test_coverage_sketch_includes_fit_rect():
    sketch = coverage_sketch(
        (0, 0, 0), (10, 10, 10),
        (2, 2, 2), (3, 3, 3),
        fit_lo=(1, 1, 1), fit_hi=(4, 4, 4),
    )
    assert sketch["map"] is not None
    assert sketch["target"] is not None
    assert sketch["fit"] is not None
    assert sketch["fit"]["w"] > 0
    assert sketch["fit"]["h"] > 0


def test_fit_grid_to_world_aabb_crops_to_margin():
    values = np.arange(11 * 11 * 11, dtype=float)
    grid = Field(
        values,
        step_sizes=(1.0, 1.0, 1.0),
        step_counts=(10, 10, 10),
        origin=(0.0, 0.0, 0.0),
        name="cell",
    )
    O = np.eye(3) * 10.0
    dest = fit_grid_to_world_aabb(
        grid, (3.0, 3.0, 3.0), (6.0, 6.0, 6.0), O, [np.eye(3)], np.zeros(3),
    )
    assert dest is not None
    lo, hi = map_world_aabb(dest)
    assert lo[0] >= 2.0
    assert hi[0] <= 7.0
    assert int(np.prod(np.asarray(dest.step_counts) + 1)) < int(np.prod(np.asarray(grid.step_counts) + 1))


def test_default_fit_margin():
    assert DEFAULT_FIT_MARGIN == pytest.approx(4.0)


def _peak_cell_field():
    """20³ brick filling a 20 Å cell, with a lone peak at world (18, 10, 10)."""
    from pymolviz.fields.crystal import attach_crystal_cell, orthogonalization_matrix

    values = np.zeros((20, 20, 20), dtype=float)
    values[18, 10, 10] = 3.0
    grid = Field(
        values.reshape(-1),
        step_sizes=(1.0, 1.0, 1.0),
        step_counts=(19, 19, 19),
        origin=(0.0, 0.0, 0.0),
        name="peak",
    )
    attach_crystal_cell(grid, orthogonalization_matrix(20.0, 20.0, 20.0), "P 21 21 2")
    return grid


def test_cell_for_grid_reads_stored_crystal_cell():
    from pymolviz.fields.crystal import cell_for_grid

    grid = _peak_cell_field()
    O, _origin, sg, brick = cell_for_grid(grid)
    assert brick is False
    assert sg == "P 21 21 2"
    assert np.diag(O) == pytest.approx([20.0, 20.0, 20.0])


def test_shrink_around_outside_cell_keeps_wrapped_peak():
    """Ligand at x=−2 is the periodic image of the peak at x=18, not cell-center density."""
    from pymolviz.fields.crystal import symmetrize_grid_to_points
    from pymolviz.fields.crystal_symmetrize import _brick_dwarfs_selection
    from pymolviz.fields.domain import grid_world_to_local
    from pymolviz.util.field_sample import grid_values_3d

    grid = _peak_cell_field()
    # Span > 5 Å so the full-cell brick is treated as dwarfing the ligand (BAN case).
    ligand = np.array([[-2.0, 10.0, 10.0], [-2.0, 16.0, 10.0]])
    lo, hi = map_world_aabb(grid)
    from pymolviz.fields.crystal import orthogonalization_matrix

    O = orthogonalization_matrix(20.0, 20.0, 20.0)
    assert _brick_dwarfs_selection(lo, hi, ligand, O) is True
    dest, info = symmetrize_grid_to_points(grid, ligand, shrink=True, margin=1.5)
    assert info.get("shrunk") is True
    assert info.get("shift") == [-1, 0, 0]
    flo, fhi = map_world_aabb(dest)
    assert flo[0] < -2.0 < fhi[0]
    local = grid_world_to_local(dest, ligand[:1]).reshape(3)
    origin = np.asarray(dest.origin, dtype=float)
    step = np.maximum(np.asarray(dest.step_sizes, dtype=float), 1e-18)
    idx = np.rint((local - origin) / step).astype(int)
    arr = grid_values_3d(dest)
    idx = np.clip(idx, 0, np.asarray(arr.shape) - 1)
    assert arr[tuple(idx)] == pytest.approx(3.0)


def test_cell_for_grid_does_not_get_symmetry_of_maps():
    from tests.fakes.cmd import FakeAtom, FakeCmd

    from pymolviz.fields.crystal import cell_for_grid

    cmd = FakeCmd()
    cmd.objects["prot"] = []
    cmd.object_types["prot"] = "object:molecule"
    cmd.add_atom(FakeAtom(model="prot", atom_id=1, x=1.0, y=2.0, z=3.0))
    cmd.objects["density"] = []
    cmd.object_types["density"] = "object:map"
    cmd.symmetries["prot"] = [20.0, 20.0, 20.0, 90.0, 90.0, 90.0, "P1"]

    def _boom(name="", state=1):
        raise AssertionError("get_symmetry must not run on maps (%r)" % name)

    def _safe(name="", state=1):
        if str(name) == "density":
            _boom(name, state)
        return [20.0, 20.0, 20.0, 90.0, 90.0, 90.0, "P1"]

    cmd.get_symmetry = _safe
    grid = Field(
        np.zeros(8, dtype=float),
        step_sizes=(1.0, 1.0, 1.0),
        step_counts=(1, 1, 1),
        origin=(0.0, 0.0, 0.0),
        name="density",
    )
    O, _origin, sg, brick_cell = cell_for_grid(
        grid, cmd=cmd, map_name="density"
    )
    assert brick_cell is False
    assert sg == "P1"
    assert O[0, 0] == pytest.approx(20.0)


def test_extend_target_rows_lists_objects_and_selections_not_maps():
    from tests.fakes.cmd import FakeAtom, FakeCmd

    from pymolviz.fields.crystal import extend_target_rows

    cmd = FakeCmd()
    cmd.objects["prot"] = []
    cmd.object_types["prot"] = "object:molecule"
    cmd.add_atom(FakeAtom(model="prot", atom_id=1, x=1.0, y=2.0, z=3.0))
    cmd.selections["ligand"] = list(cmd.atoms)
    cmd.object_types["ligand"] = "selection"
    cmd.objects["density"] = []
    cmd.object_types["density"] = "object:map"
    rows = extend_target_rows(cmd, _peak_cell_field(), skip_names=("density",))
    names = {row["name"] for row in rows}
    kinds = {row["name"]: row["kind"] for row in rows}
    assert "prot" in names
    assert kinds["prot"] == "object"
    assert "ligand" in names
    assert kinds["ligand"] == "selection"
    assert "density" not in names
    prot = next(row for row in rows if row["name"] == "prot")
    assert prot["n_atoms"] == 1
    assert prot["lo"] == pytest.approx([1.0, 2.0, 3.0])


def test_bind_grid_to_field_retargets_dependent_visuals(monkeypatch):
    from pymolviz.fields.crystal_symmetrize import bind_grid_to_field

    field = Field(
        np.zeros(8, dtype=float),
        step_sizes=(1.0, 1.0, 1.0),
        step_counts=(1, 1, 1),
        origin=(0.0, 0.0, 0.0),
        name="density",
    )
    dest = Field(
        np.ones(8, dtype=float),
        step_sizes=(1.0, 1.0, 1.0),
        step_counts=(1, 1, 1),
        origin=(2.0, 0.0, 0.0),
        name="density",
    )

    class _Visual:
        def __init__(self):
            self.id = "visual1"
            self.grid_data = field
            self.geometry_field_id = field.id

    visual = _Visual()
    monkeypatch.setattr(
        "pymolviz.runtime.session.all_objects",
        lambda: [field, visual],
    )
    monkeypatch.setattr(
        "pymolviz.runtime.session.get",
        lambda fid, **_k: field if str(fid) == str(field.id) else None,
    )
    bind_grid_to_field(field, dest)
    assert visual.grid_data is dest
