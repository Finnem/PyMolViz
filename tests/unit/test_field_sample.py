"""Trilinear sampling of regular grids and colormap mapping."""

from __future__ import annotations

import numpy as np
import pytest

from pymolviz.util.field_sample import (
    rgb_from_scalars,
    sample_grid,
    sample_regular_grid,
    project_edge_midpoint,
    union_sphere_midpoint_projector,
    PYMOL_MAP_ID_PREFIX,
    grid_from_pymol_map,
    resolve_grid_from_session,
    discover_fields,
    field_choices,
    paint_mesh_by_field,
    paint_mesh_by_point_colors,
    sample_rgb_at,
    _NATIVE_GRIDS,
)
from pymolviz.volumetric.GridData import GridData


def test_sample_regular_grid_linear_interpolation():
    values = np.zeros((2, 2, 2), dtype=float)
    values[1, 0, 0] = 1.0
    values[0, 1, 0] = 10.0
    values[0, 0, 1] = 100.0
    got = sample_regular_grid(
        values, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0),
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0)],
    )
    assert got[0] == pytest.approx(0.0)
    assert got[1] == pytest.approx(1.0)
    assert got[2] == pytest.approx(0.5)
    assert got[3] == pytest.approx(5.0)


def test_sample_grid_clamps_out_of_bounds():
    values = np.arange(8, dtype=float).reshape(2, 2, 2)
    grid = GridData(
        values.reshape(-1),
        step_sizes=(1.0, 1.0, 1.0),
        step_counts=(1, 1, 1),
        origin=(0.0, 0.0, 0.0),
        name="tiny",
    )
    inside = sample_grid(grid, [(0.0, 0.0, 0.0)])
    corner = sample_grid(grid, [(1.0, 1.0, 1.0)])
    outside = sample_grid(grid, [(-5.0, -5.0, -5.0), (9.0, 9.0, 9.0)])
    assert inside[0] == pytest.approx(values[0, 0, 0])
    assert corner[0] == pytest.approx(values[1, 1, 1])
    assert outside[0] == pytest.approx(values[0, 0, 0])
    assert outside[1] == pytest.approx(values[1, 1, 1])


def test_rgb_from_scalars_spans_colormap():
    rgb, clims = rgb_from_scalars([0.0, 1.0, 2.0], "coolwarm", clims=(0.0, 2.0))
    assert rgb.shape == (3, 3)
    assert clims == (0.0, 2.0)
    assert not np.allclose(rgb[0], rgb[-1])
    unique = {tuple(np.round(row, 5)) for row in rgb}
    assert len(unique) == 3


def test_refine_mesh_for_field_splits_steep_triangles():
    from pymolviz.util.field_sample import refine_mesh_for_field

    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=float)
    faces = np.array([[0, 1, 2]], dtype=int)
    normals = np.array([[0.0, 0.0, 1.0]] * 3, dtype=float)
    values = np.array([0.0, 1.0, 0.5], dtype=float)
    v2, n2, f2, s2 = refine_mesh_for_field(
        vertices, normals, faces, values, grid=None, max_level=1, max_frac=0.2,
    )
    assert f2.shape == (4, 3)
    assert v2.shape[0] == 6
    assert n2.shape[0] == v2.shape[0]
    assert s2.shape[0] == v2.shape[0]


def test_project_edge_midpoint_lands_on_sphere():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    p, n = project_edge_midpoint(a, b, a, b)
    assert np.linalg.norm(p) == pytest.approx(1.0)
    assert p[0] == pytest.approx(p[1])
    assert p[0] > 0.6
    assert n[0] == pytest.approx(n[1])
    assert n[2] == pytest.approx(0.0)


def test_project_edge_midpoint_concave_probe_sphere():
    a = np.array([1.5, 0.0, 0.0])
    b = np.array([0.0, 1.5, 0.0])
    na = np.array([-1.0, 0.0, 0.0])
    nb = np.array([0.0, -1.0, 0.0])
    p, n = project_edge_midpoint(a, b, na, nb)
    assert np.linalg.norm(p) == pytest.approx(1.5)
    assert p[0] > 0.0 and p[1] > 0.0
    assert n[0] < 0.0 and n[1] < 0.0


def test_union_sphere_projector_snaps_to_atom_sphere():
    project = union_sphere_midpoint_projector([(0.0, 0.0, 0.0)], [2.0])
    a = np.array([2.0, 0.0, 0.0])
    b = np.array([0.0, 2.0, 0.0])
    p, n = project(a, b, a, b)
    assert np.linalg.norm(p) == pytest.approx(2.0)
    assert p[0] == pytest.approx(p[1])


def test_refine_mesh_for_field_lifts_midpoints_onto_sphere():
    from pymolviz.util.field_sample import refine_mesh_for_field

    vertices = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    faces = np.array([[0, 1, 2]], dtype=int)
    values = np.array([0.0, 1.0, 0.5], dtype=float)
    v2, n2, f2, s2 = refine_mesh_for_field(
        vertices, vertices, faces, values, grid=None, max_level=1, max_frac=0.2,
    )
    assert f2.shape[0] > 1
    for p in v2:
        assert np.linalg.norm(p) == pytest.approx(1.0, abs=1e-6)


def test_refine_mesh_for_field_keeps_flat_field():
    from pymolviz.util.field_sample import refine_mesh_for_field

    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ], dtype=float)
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    normals = np.zeros((4, 3), dtype=float)
    normals[:, 2] = 1.0
    values = np.ones(4, dtype=float)
    v2, n2, f2, s2 = refine_mesh_for_field(
        vertices, normals, faces, values, grid=None, max_level=2, max_frac=0.08,
    )
    assert f2.shape[0] == 2
    assert v2.shape[0] == 4


def test_refine_mesh_for_field_completes_steep_face_edges():
    from pymolviz.util.field_sample import refine_mesh_for_field

    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=float)
    faces = np.array([[0, 1, 2]], dtype=int)
    normals = np.array([[0.0, 0.0, 1.0]] * 3, dtype=float)
    values = np.array([0.0, 1.0, 0.0], dtype=float)
    v2, n2, f2, s2 = refine_mesh_for_field(
        vertices, normals, faces, values, grid=None, max_level=1, max_frac=0.2,
    )
    assert f2.shape == (4, 3)
    assert v2.shape[0] == 6


def test_refine_skinny_triangle_splits_longest_edges_only():
    from pymolviz.util.field_sample import refine_mesh_for_field

    vertices = np.array([
        [0.0, 0.0, 0.0],
        [8.0, 0.0, 0.0],
        [0.0, 0.4, 0.0],
    ], dtype=float)
    faces = np.array([[0, 1, 2]], dtype=int)
    normals = np.array([[0.0, 0.0, 1.0]] * 3, dtype=float)
    values = np.array([0.0, 1.0, 0.4], dtype=float)
    v2, n2, f2, s2 = refine_mesh_for_field(
        vertices, normals, faces, values, grid=None, max_level=1, max_frac=0.2,
    )
    assert f2.shape[0] == 3
    assert v2.shape[0] == 5
    short = {0, 2}
    assert any(short <= set(tri) for tri in f2)


def test_refine_mesh_for_field_splits_shared_edges_on_both_faces():
    from pymolviz.util.field_sample import refine_mesh_for_field

    vertices = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    normals = np.array([[0.0, 0.0, 1.0]] * 4, dtype=float)
    values = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
    v2, n2, f2, s2 = refine_mesh_for_field(
        vertices, normals, faces, values, grid=None, max_level=1, max_frac=0.2,
    )
    unsplit = [tuple(tri) for tri in f2 if 0 in tri and 2 in tri]
    p0 = v2[0]
    p2 = v2[2]
    chord = p2 - p0
    denom = float(np.dot(chord, chord)) + 1e-18

    def on_open_segment(point):
        t = float(np.dot(point - p0, chord) / denom)
        if t <= 1e-8 or t >= 1.0 - 1e-8:
            return False
        return float(np.linalg.norm((point - p0) - t * chord)) < 1e-6

    mids = [i for i, point in enumerate(v2) if on_open_segment(point)]
    assert mids
    assert not unsplit


def test_refine_mesh_for_field_skip_edge_keeps_frozen_edge():
    from pymolviz.util.field_sample import refine_mesh_for_field

    vertices = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
    ], dtype=float)
    faces = np.array([[0, 1, 2], [0, 3, 1]], dtype=int)
    normals = np.array([[0.0, 0.0, 1.0]] * 4, dtype=float)
    values = np.array([0.0, 1.0, 0.5, 0.5], dtype=float)

    def skip_edge(i, j):
        return {int(i), int(j)} == {0, 1}

    v2, n2, f2, s2 = refine_mesh_for_field(
        vertices, normals, faces, values, grid=None, max_level=1, max_frac=0.2,
        skip_edge=skip_edge,
    )
    p0 = v2[0]
    p1 = v2[1]
    chord = p1 - p0
    denom = float(np.dot(chord, chord)) + 1e-18

    def on_open_segment(point):
        t = float(np.dot(point - p0, chord) / denom)
        if t <= 1e-8 or t >= 1.0 - 1e-8:
            return False
        return float(np.linalg.norm((point - p0) - t * chord)) < 1e-6

    assert not any(on_open_segment(point) for point in v2)
    assert any(0 in tri and 1 in tri for tri in f2)


def test_refine_mesh_for_field_skips_edges_shorter_than_min_edge():
    from pymolviz.util.field_sample import refine_mesh_for_field

    vertices = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
    ], dtype=float)
    faces = np.array([[0, 1, 2]], dtype=int)
    normals = np.array([[0.0, 0.0, 1.0]] * 3, dtype=float)
    values = np.array([0.0, 1.0, 0.5], dtype=float)
    v2, n2, f2, s2 = refine_mesh_for_field(
        vertices, normals, faces, values, grid=None, max_level=2,
        max_frac=0.01, min_edge=1.0,
    )
    assert f2.shape[0] == 1
    assert v2.shape[0] == 3


def test_sample_grid_smooth_blurs_voxel_spike():
    values = np.zeros((5, 5, 5), dtype=float)
    values[2, 2, 2] = 1.0
    grid = GridData(
        values.reshape(-1),
        step_sizes=(1.0, 1.0, 1.0),
        step_counts=(4, 4, 4),
        origin=(0.0, 0.0, 0.0),
        name="spike",
    )
    sharp = sample_grid(grid, [(2.0, 2.0, 2.0), (3.0, 2.0, 2.0)], smooth=0.0)
    blur = sample_grid(grid, [(2.0, 2.0, 2.0), (3.0, 2.0, 2.0)], smooth=0.75)
    assert sharp[0] == pytest.approx(1.0)
    assert sharp[1] == pytest.approx(0.0)
    assert blur[0] < sharp[0]
    assert blur[1] > sharp[1]


def _map_cmd(values, name="density", origin=(0.0, 0.0, 0.0), corner=(1.0, 1.0, 1.0)):
    from tests.fakes.cmd import FakeCmd

    cmd = FakeCmd()
    cmd.objects[name] = []
    cmd.object_types[name] = "object:map"
    cmd.volume_fields[name] = values
    cmd.extents[name] = [list(origin), list(corner)]
    cmd.objects["prot"] = []
    cmd.object_types["prot"] = "object:molecule"
    return cmd


def test_grid_from_pymol_map_matches_extent_and_samples():
    _NATIVE_GRIDS.clear()
    values = np.arange(8, dtype=float).reshape(2, 2, 2)
    cmd = _map_cmd(values)
    grid = grid_from_pymol_map("density", cmd=cmd)
    assert grid is not None
    assert grid.id == PYMOL_MAP_ID_PREFIX + "density"
    assert str(grid._name) == "density"
    got = sample_grid(grid, [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)])
    assert got[0] == pytest.approx(values[0, 0, 0])
    assert got[1] == pytest.approx(values[1, 1, 1])
    assert resolve_grid_from_session(grid.id) is grid
    assert grid_from_pymol_map("prot", cmd=cmd) is None


def test_discover_fields_lists_native_maps_not_molecules():
    _NATIVE_GRIDS.clear()
    values = np.zeros((2, 2, 2), dtype=float)
    cmd = _map_cmd(values)
    found = discover_fields([], cmd=cmd)
    ids = [str(obj.id) for obj in found]
    assert ids == [PYMOL_MAP_ID_PREFIX + "density"]


def test_field_choices_uses_map_name_as_label():
    _NATIVE_GRIDS.clear()
    values = np.zeros((2, 2, 2), dtype=float)
    cmd = _map_cmd(values)
    choices = field_choices([], cmd=cmd)
    assert choices == [(PYMOL_MAP_ID_PREFIX + "density", "density")]


def test_paint_mesh_by_point_colors_blends_anchor_colors():
    from pymolviz.meshes.Mesh import Mesh

    verts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=float)
    mesh = Mesh(
        verts,
        color=(1.0, 1.0, 1.0),
        faces=[[0, 1, 2]],
        bypass_colormap=True,
    )
    centers = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    colors = [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    radii = [2.0, 2.0]
    assert paint_mesh_by_point_colors(mesh, centers, colors, radii)
    got = np.asarray(mesh.color, dtype=float).reshape(-1, 3)
    assert got.shape[0] == 3
    unique = {tuple(np.round(row, 3)) for row in got}
    assert len(unique) > 1
    assert got[0, 0] > got[1, 0]
    assert got[1, 2] > got[0, 2]
    assert paint_mesh_by_point_colors(mesh, centers, colors, radii) is False
    again = np.asarray(mesh.color, dtype=float).reshape(-1, 3)
    assert np.allclose(got, again)


def test_paint_mesh_by_field_sets_per_vertex_colors():
    _NATIVE_GRIDS.clear()
    values = np.zeros((2, 2, 2), dtype=float)
    values[1, 1, 1] = 1.0
    grid = GridData(
        values.reshape(-1),
        step_sizes=(1.0, 1.0, 1.0),
        step_counts=(1, 1, 1),
        origin=(0.0, 0.0, 0.0),
        name="grad",
    )
    _NATIVE_GRIDS[grid.id] = grid
    from pymolviz.meshes.Sphere import Sphere
    from pymolviz.points import FixedPoint

    sphere = Sphere(
        FixedPoint((0.5, 0.5, 0.5)), 0.5,
        color=(1.0, 0.0, 0.0), bypass_colormap=True, frequency=2,
    )
    assert paint_mesh_by_field(sphere, grid.id, colormap="coolwarm", refine=False)
    colors = np.asarray(sphere.color, dtype=float).reshape(-1, 3)
    assert colors.shape[0] == sphere.vertices.shape[0]
    assert sphere.field_id == str(grid.id)
    unique = {tuple(np.round(row, 4)) for row in colors}
    assert len(unique) > 1
    sampled = sample_rgb_at((1.0, 1.0, 1.0), grid.id, "coolwarm")
    assert sampled is not None
    assert len(sampled) == 3


def test_rgb_from_scalars_uses_custom_spec_stops():
    from pymolviz.util.colormap_spec import ColorStop, ColormapDefinition
    from pymolviz.util.field_sample import rgb_from_scalars

    defn = ColormapDefinition(
        preset="custom",
        stops=(
            ColorStop(0.0, (0.0, 0.0, 1.0, 1.0)),
            ColorStop(1.0, (1.0, 0.0, 0.0, 1.0)),
        ),
        customized=True,
    )
    rgb, used = rgb_from_scalars([0.0, 1.0], defn, clims=(0.0, 1.0))
    assert used == pytest.approx((0.0, 1.0))
    assert rgb[0] == pytest.approx((0.0, 0.0, 1.0), abs=0.02)
    assert rgb[1] == pytest.approx((1.0, 0.0, 0.0), abs=0.02)
    rgb2, _ = rgb_from_scalars([0.0, 1.0], defn.to_dict(), clims=(0.0, 1.0))
    assert np.allclose(rgb, rgb2)


def test_visual_point_field_choice_clears_on_solid_color():
    from pymolviz.wizards.builders.colors import ColorChoice
    from pymolviz.wizards.builders.points import VisualPoint

    pt = VisualPoint("a", "manual", 0.0, 0.0, 0.0)
    choice = ColorChoice(
        rgba=(0.2, 0.3, 0.4, 0.5),
        field_id="pymol_map:density",
        colormap="viridis",
    )
    painted = pt.with_color_choice(choice)
    assert painted.field_id == "pymol_map:density"
    assert painted.field_colormap == "viridis"
    assert painted.alpha == pytest.approx(0.5)
    solid = painted.with_color((1.0, 0.0, 0.0, 1.0))
    assert solid.field_id is None
    assert solid.color[0] == pytest.approx(1.0)
