"""Lattice wrap / tile of maps onto a selection (no Qt)."""

import numpy as np
import pytest

from pymolviz.fields.crystal import (
    CrystalError,
    aabb_contains_points,
    apply_symmetrize_field,
    attach_crystal_axis_frame,
    field_supports_symmetrize,
    grid_contains_points,
    interpolate_unit_cell,
    lattice_shift_indices,
    map_corners,
    map_world_aabb,
    orthogonalization_matrix,
    selection_points,
    symmetrize_grid_to_points,
)
from pymolviz.fields.domain import grid_world_to_local
from pymolviz.fields.field import Field
from pymolviz.fields.identity import GEN_GAUSSIAN, GEN_IMPORTED
from pymolviz.util.field_sample import grid_values_3d
from pymolviz.volumetric.GridData import GridData
from tests.fakes.cmd import FakeAtom, FakeCmd


def _box_grid(origin=(0.0, 0.0, 0.0), step=1.0, counts=2, name="map"):
    counts = np.broadcast_to(np.asarray(counts, dtype=int).reshape(-1), (3,)).copy()
    shape = tuple(int(c) + 1 for c in counts)
    values = np.arange(int(np.prod(shape)), dtype=float).reshape(shape)
    return GridData(
        values.reshape(-1),
        step_sizes=(float(step), float(step), float(step)),
        step_counts=tuple(int(c) for c in counts),
        origin=origin,
        name=name,
    )


def _voxel_at(grid, xyz):
    tiled = grid_values_3d(grid)
    local = grid_world_to_local(grid, xyz).reshape(3)
    origin = np.asarray(grid.origin, dtype=float).reshape(3)
    step = np.asarray(grid.step_sizes, dtype=float).reshape(3)
    idx = np.rint((local - origin) / step).astype(int)
    return tiled[idx[0], idx[1], idx[2]]


def test_orthogonalization_gamma_120_has_expected_b_vector():
    O = orthogonalization_matrix(10.0, 10.0, 10.0, 90.0, 90.0, 120.0)
    assert O[:, 0] == pytest.approx([10.0, 0.0, 0.0])
    assert O[0, 1] == pytest.approx(-5.0)
    assert O[1, 1] == pytest.approx(10.0 * np.sin(np.deg2rad(120.0)))
    assert O[2, 2] == pytest.approx(10.0)


def test_lattice_shift_moves_map_by_integer_cell():
    grid = _box_grid(counts=(10, 10, 10), step=1.0)
    lo, hi = map_corners(grid)
    O = np.diag(hi - lo)
    points = np.array([[15.0, 5.0, 5.0]])
    shift = lattice_shift_indices(lo, hi, points, O)
    assert shift.tolist() == [1, 0, 0]
    new, info = symmetrize_grid_to_points(grid, points)
    assert info["copied"] is False
    assert info["shift"] == [1, 0, 0]
    assert np.allclose(new.origin, [10.0, 0.0, 0.0])
    nlo, nhi = map_corners(new)
    assert aabb_contains_points(nlo, nhi, points)


def test_triclinic_cell_shift_uses_lattice_not_aabb_guess():
    grid = _box_grid(counts=(10, 10, 10), step=1.0)
    cell = [10.0, 10.0, 10.0, 90.0, 90.0, 120.0, "P1"]
    O = orthogonalization_matrix(10.0, 10.0, 10.0, 90.0, 90.0, 120.0)
    lo, hi = map_corners(grid)
    center = 0.5 * (lo + hi)
    points = (center + O[:, 0]).reshape(1, 3)
    new, info = symmetrize_grid_to_points(grid, points, cell=cell)
    assert info["copied"] is False
    assert info["shift"] == [1, 0, 0]
    assert grid_contains_points(new, points)


def test_map_sits_on_protein_not_cell_wallpaper():
    grid = _box_grid(counts=(8, 8, 8), step=1.0)
    cell = [80.0, 80.0, 80.0, 90.0, 90.0, 120.0, "P1"]
    points = np.array([[40.0, 40.0, 40.0], [42.0, 41.0, 39.0]])
    new, info = symmetrize_grid_to_points(grid, points, cell=cell)
    assert info["copied"] is False
    nlo, nhi = map_world_aabb(new)
    assert nhi[0] - nlo[0] == pytest.approx(8.0)
    assert grid_contains_points(new, points)
    assert getattr(new, "_crystal_nmin", None) in (None, [])


def test_triclinic_already_on_protein_does_not_wallpaper_cell():
    grid = _box_grid(counts=(10, 10, 10), step=1.0)
    cell = [80.0, 80.0, 80.0, 90.0, 90.0, 120.0, "P1"]
    points = np.array([[5.0, 5.0, 5.0]])
    new, info = symmetrize_grid_to_points(grid, points, cell=cell)
    assert info["copied"] is False
    nlo, nhi = map_world_aabb(new)
    assert nhi[0] - nlo[0] == pytest.approx(10.0)
    assert nhi[1] - nlo[1] == pytest.approx(10.0)


def test_triclinic_tile_is_parallelepiped_not_xyz_cube():
    grid = _box_grid(counts=(10, 10, 10), step=1.0)
    cell = [10.0, 10.0, 10.0, 90.0, 90.0, 120.0, "P1"]
    O = orthogonalization_matrix(10.0, 10.0, 10.0, 90.0, 90.0, 120.0)
    points = np.array([[1.0, 1.0, 1.0], (np.array([1.0, 1.0, 1.0]) + O[:, 0])])
    new, info = symmetrize_grid_to_points(grid, points, cell=cell)
    assert np.allclose(np.asarray(new.A_to, dtype=float).reshape(4, 4), np.eye(4))
    nlo, nhi = map_world_aabb(new)
    assert nhi[0] - nlo[0] < 25.0
    assert grid_contains_points(new, points)


def test_selection_spanning_two_cells_tiles_periodic_values():
    grid = _box_grid(counts=(2, 2, 2), step=1.0)
    original = np.arange(27, dtype=float).reshape((3, 3, 3))
    points = np.array([[0.5, 1.0, 1.0], [3.5, 1.0, 1.0]])
    new, info = symmetrize_grid_to_points(grid, points)
    assert info["copied"] is True
    nlo, nhi = map_world_aabb(new)
    assert aabb_contains_points(nlo, nhi, points)
    assert _voxel_at(new, (0.0, 1.0, 1.0)) == pytest.approx(original[0, 1, 1])
    assert _voxel_at(new, (2.0, 1.0, 1.0)) == pytest.approx(original[0, 1, 1])


def test_expand_zeros_gaps_between_map_copies():
    grid = _box_grid(counts=(4, 4, 4), step=1.0)
    cell = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0, "P1"]
    points = np.array([[1.0, 1.0, 1.0], [11.0, 1.0, 1.0]])
    new, info = symmetrize_grid_to_points(grid, points, cell=cell)
    assert info["copied"] is True
    assert _voxel_at(new, (1.0, 1.0, 1.0)) == pytest.approx(_voxel_at(new, (11.0, 1.0, 1.0)))
    assert _voxel_at(new, (7.0, 1.0, 1.0)) == pytest.approx(0.0)


def test_missing_selection_raises():
    cmd = FakeCmd()
    with pytest.raises(CrystalError, match="Select some atoms"):
        selection_points(cmd)


def test_selection_falls_back_to_enabled_molecule():
    cmd = FakeCmd()
    cmd.add_atom(FakeAtom(model="prot", atom_id=1, x=15.0, y=5.0, z=5.0))
    cmd.objects["prot"] = []
    cmd.object_types["prot"] = "object:molecule"
    pts = selection_points(cmd)
    assert pts.shape == (1, 3)
    assert pts[0, 0] == pytest.approx(15.0)


def test_dummy_pymol_symmetry_tiles_with_map_period():
    cmd = FakeCmd()
    grid = _box_grid(counts=(10, 10, 10), step=1.0, name="density")
    original = np.arange(11 ** 3, dtype=float).reshape((11, 11, 11))
    points = np.array([[1.0, 5.0, 5.0], [12.0, 5.0, 5.0]])
    new, info = symmetrize_grid_to_points(grid, points, cmd=cmd, map_name="density")
    assert info["copied"] is True
    assert info["brick_cell"] is True
    nlo, nhi = map_world_aabb(new)
    assert aabb_contains_points(nlo, nhi, points)
    assert _voxel_at(new, (1.0, 5.0, 5.0)) == pytest.approx(original[1, 5, 5])
    assert _voxel_at(new, (12.0, 5.0, 5.0)) == pytest.approx(original[2, 5, 5])


def test_tile_extent_is_whole_unit_cells():
    grid = _box_grid(counts=(10, 10, 10), step=1.0)
    cell = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0, "P1"]
    points = np.array([[1.0, 5.0, 5.0], [12.0, 5.0, 5.0]])
    new, info = symmetrize_grid_to_points(grid, points, cell=cell)
    assert info["copied"] is True
    nlo, nhi = map_world_aabb(new)
    assert nlo[0] == pytest.approx(0.0)
    assert nhi[0] == pytest.approx(20.0)
    assert nlo[1] == pytest.approx(0.0)
    assert nhi[1] == pytest.approx(10.0)


def test_triclinic_tile_matches_across_cell_face():
    grid = _box_grid(counts=(10, 10, 10), step=1.0)
    cell = [10.0, 10.0, 10.0, 90.0, 90.0, 120.0, "P1"]
    O = orthogonalization_matrix(10.0, 10.0, 10.0, 90.0, 90.0, 120.0)
    points = np.array([[1.0, 1.0, 1.0], (np.array([1.0, 1.0, 1.0]) + O[:, 0])])
    new, info = symmetrize_grid_to_points(grid, points, cell=cell)
    assert grid_contains_points(new, points)
    p0 = np.array([1.0, 1.0, 1.0])
    if info["copied"]:
        assert _voxel_at(new, p0) == pytest.approx(_voxel_at(new, p0 + O[:, 0]))


def test_map_smaller_than_cell_covers_selection_without_error():
    grid = _box_grid(counts=(8, 8, 8), step=1.0)
    cell = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0, "P1"]
    points = np.array([[9.0, 4.0, 4.0], [21.0, 4.0, 4.0]])
    new, info = symmetrize_grid_to_points(grid, points, cell=cell)
    assert info["copied"] is True
    nlo, nhi = map_world_aabb(new)
    assert aabb_contains_points(nlo, nhi, points)
    assert grid_contains_points(new, points)


def test_large_tile_fits_voxel_budget():
    grid = _box_grid(counts=(100, 100, 100), step=0.1)
    cell = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0, "P1"]
    points = np.array([[1.0, 1.0, 1.0], [12.0, 12.0, 12.0]])
    new, info = symmetrize_grid_to_points(grid, points, cell=cell)
    assert info["copied"] is True
    n_vox = int(np.prod(np.asarray(new.step_counts, dtype=int) + 1))
    assert n_vox <= 192 ** 3
    assert grid_contains_points(new, points)


def test_already_inside_does_not_move():
    grid = _box_grid(counts=(10, 10, 10), step=1.0)
    new, info = symmetrize_grid_to_points(grid, np.array([[5.0, 5.0, 5.0]]))
    assert info["moved"] is False
    assert info["copied"] is False
    assert np.allclose(new.origin, [0.0, 0.0, 0.0])


def test_apply_uses_sele_and_pymol_symmetry():
    from pymolviz.runtime import session as session_mod

    session_mod.clear()
    cmd = FakeCmd()
    cmd.add_atom(FakeAtom(model="prot", atom_id=1, x=15.0, y=5.0, z=5.0))
    cmd.select("sele", "prot")
    grid = _box_grid(counts=(10, 10, 10), step=1.0, name="density")
    field = Field(name="density", generator={"type": GEN_IMPORTED}, grid_data=grid)
    cmd.symmetries["density"] = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0, "P1"]
    wrapped, new_grid, info = apply_symmetrize_field(cmd, field)
    assert info["copied"] is False
    assert info["brick_cell"] is False
    assert np.allclose(new_grid.origin, [10.0, 0.0, 0.0])
    assert wrapped.grid_data is new_grid
    session_mod.clear()


def test_symmetrize_reloads_pymol_map_extent():
    from pymolviz.runtime import session as session_mod
    from pymolviz.wizards.builders.field_visual import symmetrize_field_to_selection

    session_mod.clear()
    cmd = FakeCmd()
    cmd.add_atom(FakeAtom(model="prot", atom_id=1, x=15.0, y=5.0, z=5.0))
    cmd.select("sele", "prot")
    grid = _box_grid(counts=(10, 10, 10), step=1.0, name="density")
    field = Field(name="density", generator={"type": GEN_IMPORTED}, grid_data=grid)
    cmd.symmetries["density"] = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0, "P1"]
    _wrapped, info = symmetrize_field_to_selection(cmd, field)
    assert info["copied"] is False
    assert np.allclose(_wrapped.grid_data.origin, [10.0, 0.0, 0.0])
    assert "density" in cmd.objects
    assert cmd.symmetries["density"][0] == pytest.approx(10.0)
    session_mod.clear()


def test_symmetrize_rebinds_volume_to_moved_brick():
    from pymolviz.runtime import session as session_mod
    from pymolviz.wizards.builders.field_visual import (
        make_field_visual,
        named_grid_copy,
        symmetrize_field_to_selection,
    )

    session_mod.clear()
    cmd = FakeCmd()
    cmd.add_atom(FakeAtom(model="prot", atom_id=1, x=15.0, y=5.0, z=5.0))
    cmd.select("sele", "prot")
    grid = _box_grid(counts=(10, 10, 10), step=1.0, name="density")
    field = Field(name="density", generator={"type": GEN_IMPORTED}, grid_data=grid)
    session_mod.add(field)
    volume = make_field_visual("Volume", field, "vol", clims=(0.0, 1.0))
    volume.grid_data = named_grid_copy(grid, "_pmv_prev_geom_map")
    session_mod.add(volume)
    cmd.symmetries["density"] = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0, "P1"]
    wrapped, info = symmetrize_field_to_selection(cmd, field)
    assert info["copied"] is False
    assert volume.grid_data is wrapped.grid_data
    assert np.allclose(volume.grid_data.origin, [10.0, 0.0, 0.0])
    assert cmd.objects["vol"]["map"] == "density"
    session_mod.clear()


def test_symmetrize_volume_ttt_follows_crystal_embed():
    from pymolviz.runtime import session as session_mod
    from pymolviz.wizards.builders.field_visual import (
        make_field_visual,
        symmetrize_field_to_selection,
    )

    session_mod.clear()
    cmd = FakeCmd()
    O = orthogonalization_matrix(10.0, 10.0, 10.0, 90.0, 90.0, 120.0)
    cmd.add_atom(FakeAtom(model="prot", atom_id=1, x=1.0, y=1.0, z=1.0))
    cmd.add_atom(FakeAtom(model="prot", atom_id=2, x=float((np.array([1.0, 1.0, 1.0]) + O[:, 0])[0]), y=float((np.array([1.0, 1.0, 1.0]) + O[:, 0])[1]), z=float((np.array([1.0, 1.0, 1.0]) + O[:, 0])[2])))
    cmd.select("sele", "prot")
    grid = _box_grid(counts=(10, 10, 10), step=1.0, name="density")
    field = Field(name="density", generator={"type": GEN_IMPORTED}, grid_data=grid)
    session_mod.add(field)
    volume = make_field_visual("Volume", field, "vol", clims=(0.0, 1.0))
    session_mod.add(volume)
    cmd.symmetries["density"] = [10.0, 10.0, 10.0, 90.0, 90.0, 120.0, "P1"]
    wrapped, info = symmetrize_field_to_selection(cmd, field)
    assert grid_contains_points(wrapped.grid_data, np.array([[1.0, 1.0, 1.0], (np.array([1.0, 1.0, 1.0]) + O[:, 0])]))
    from pymolviz.fields.domain import grid_pymol_brick_params

    origin, step = grid_pymol_brick_params(wrapped.grid_data)
    identity = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    assert cmd.get_object_ttt("vol") == pytest.approx(identity)
    assert cmd.get_object_ttt("density") == pytest.approx(identity)
    assert cmd.symmetries["density"][3:7] == [90.0, 90.0, 90.0, "P1"]
    brick = cmd.objects.get("density")
    loaded_origin = getattr(brick, "origin", None)
    if loaded_origin is not None:
        assert np.asarray(loaded_origin, dtype=float).reshape(3) == pytest.approx(origin, abs=1e-4)
    session_mod.clear()


def test_symmetrize_shifts_volume_clip_with_origin():
    from pymolviz.runtime import session as session_mod
    from pymolviz.wizards.builders.field_visual import (
        make_field_visual,
        symmetrize_field_to_selection,
    )

    session_mod.clear()
    cmd = FakeCmd()
    cmd.add_atom(FakeAtom(model="prot", atom_id=1, x=15.0, y=5.0, z=5.0))
    cmd.select("sele", "prot")
    grid = _box_grid(counts=(10, 10, 10), step=1.0, name="density")
    field = Field(name="density", generator={"type": GEN_IMPORTED}, grid_data=grid)
    session_mod.add(field)
    volume = make_field_visual(
        "Volume",
        field,
        "vol",
        clims=(0.0, 1.0),
        clip_aabb=[[0.0, 0.0, 0.0], [5.0, 10.0, 10.0]],
    )
    session_mod.add(volume)
    cmd.symmetries["density"] = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0, "P1"]
    symmetrize_field_to_selection(cmd, field)
    assert volume.clip_aabb[0][0] == pytest.approx(10.0)
    assert volume.clip_aabb[1][0] == pytest.approx(15.0)
    assert cmd.objects["vol"]["map"] == "density_clip"
    session_mod.clear()


def test_symmetrize_reseeds_full_cube_clip_to_moved_brick():
    from pymolviz.runtime import session as session_mod
    from pymolviz.wizards.builders.field_visual import (
        make_field_visual,
        symmetrize_field_to_selection,
    )

    session_mod.clear()
    cmd = FakeCmd()
    cmd.add_atom(FakeAtom(model="prot", atom_id=1, x=15.0, y=5.0, z=5.0))
    cmd.select("sele", "prot")
    grid = _box_grid(counts=(10, 10, 10), step=1.0, name="density")
    field = Field(name="density", generator={"type": GEN_IMPORTED}, grid_data=grid)
    session_mod.add(field)
    volume = make_field_visual(
        "Volume",
        field,
        "vol",
        clims=(0.0, 1.0),
        clip_aabb=[[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]],
    )
    session_mod.add(volume)
    cmd.symmetries["density"] = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0, "P1"]
    symmetrize_field_to_selection(cmd, field)
    assert volume.clip_aabb[0] == pytest.approx([10.0, 0.0, 0.0])
    assert volume.clip_aabb[1] == pytest.approx([20.0, 10.0, 10.0])
    session_mod.clear()


def test_gaussian_fields_are_not_symmetrizable():
    field = Field(name="g", generator={"type": GEN_GAUSSIAN, "atoms": []})
    assert field_supports_symmetrize(field) is False
    assert field_supports_symmetrize(_box_grid()) is True


def test_crystal_axis_map_resamples_onto_pdb_coordinates():
    from pymolviz.util.field_sample import sample_grid

    n = 8
    values = np.zeros((n, n, n), dtype=float)
    values[n // 2, n // 2, n // 2] = 1.0
    grid = GridData(
        values.reshape(-1),
        step_sizes=(1.0, 1.0, 1.0),
        step_counts=(n - 1, n - 1, n - 1),
        origin=(0.0, 0.0, 0.0),
        name="map",
    )
    O = orthogonalization_matrix(20.0, 20.0, 20.0, 90.0, 90.0, 120.0)
    attach_crystal_axis_frame(grid, O)
    peak = O @ np.array([0.5, 0.5, 0.5])
    cell = [20.0, 20.0, 20.0, 90.0, 90.0, 120.0, "P1"]
    new, info = symmetrize_grid_to_points(grid, peak.reshape(1, 3), cell=cell)
    assert info.get("resampled") is True
    nlo, nhi = map_world_aabb(new)
    assert np.all((nhi - nlo) < 16.0)
    assert sample_grid(new, [peak])[0] == pytest.approx(1.0, abs=0.2)
    mate = peak + O[:, 0]
    assert interpolate_unit_cell(values, O, [peak])[0] == pytest.approx(
        interpolate_unit_cell(values, O, [mate])[0], abs=1e-6
    )


def test_extend_target_rows_mark_coverage_and_skip_maps():
    from pymolviz.fields.crystal import (
        default_extend_target_name,
        extend_target_rows,
        point_coverage_status,
    )

    cmd = FakeCmd()
    cmd.add_atom(FakeAtom("prot", 1, 5.0, 5.0, 5.0))
    cmd.add_atom(FakeAtom("lig", 1, 50.0, 5.0, 5.0, name="C1"))
    cmd.objects["prot"] = []
    cmd.objects["lig"] = []
    cmd.objects["density"] = []
    cmd.object_types["prot"] = "object:molecule"
    cmd.object_types["lig"] = "object:molecule"
    cmd.object_types["density"] = "object:map"
    cmd.select("pocket", "lig")
    grid = _box_grid(counts=(10, 10, 10), step=1.0, name="density")
    rows = extend_target_rows(cmd, grid, skip_names=("density",))
    by_name = {row["name"]: row for row in rows}
    assert "density" not in by_name
    assert by_name["prot"]["status"] == "inside"
    assert by_name["lig"]["status"] == "outside"
    assert by_name["pocket"]["kind"] == "selection"
    assert by_name["pocket"]["status"] == "outside"
    assert default_extend_target_name(rows) == "lig"
    assert point_coverage_status(grid, [[5.0, 5.0, 5.0], [50.0, 5.0, 5.0]]) == "partial"


def test_coverage_sketch_uses_axis_of_largest_offset():
    from pymolviz.fields.crystal import choose_coverage_axes, coverage_sketch

    x_off = coverage_sketch(
        (0.0, 0.0, 0.0), (10.0, 10.0, 10.0), (48.0, 4.0, 4.0), (52.0, 6.0, 6.0)
    )
    assert x_off["axes"][1] == 0
    assert x_off["axis_names"][1] == "X"
    assert x_off["target"]["y"] < x_off["map"]["y"]
    z_off = coverage_sketch(
        (0.0, 0.0, 0.0), (10.0, 10.0, 10.0), (2.0, 2.0, 40.0), (8.0, 8.0, 50.0)
    )
    assert z_off["axes"][1] == 2
    assert z_off["axis_names"] == (z_off["axis_names"][0], "Z")
    assert 2 in z_off["axes"]
    assert choose_coverage_axes((0.0, 0.0, 0.0), (10.0, 10.0, 10.0)) == (0, 1)
