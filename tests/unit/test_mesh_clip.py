"""Triangle-mesh clipping by infinite keep-side planes."""

from __future__ import annotations

import numpy as np
import pytest

from pymolviz.meshes.Surface import Surface
from pymolviz.util.mesh_clip import (
    apply_clip_plane_pose,
    clip_mesh_by_planes,
    clip_plane_from_view,
    clip_triangle_mesh,
    normalize_clip_planes,
    oriented_clip_normal,
)
from pymolviz.util.clip_gizmo import fit_plane_rectangle, rectangle_corners
from pymolviz.wizards.builders.points import VisualPoint
from pymolviz.wizards.builders.preview import (
    build_surface_collection,
    retarget_surface_collection,
)
from pymolviz.points import FixedPoint


def _identity_view(origin=(0.0, 0.0, 0.0)):
    view = [0.0] * 18
    view[0] = view[4] = view[8] = 1.0
    view[12], view[13], view[14] = (float(origin[0]), float(origin[1]), float(origin[2]))
    return view


def _point(name, xyz, color=(0.2, 0.6, 0.9)):
    return VisualPoint(
        name, "manual", xyz[0], xyz[1], xyz[2],
        color=color, point_source=FixedPoint(xyz),
    )


def _cube_mesh():
    """Axis-aligned cube from -1 to 1, 12 triangles, outward normals."""
    corners = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=float)
    tris = (
        (0, 2, 1, (0.0, 0.0, -1.0)), (0, 3, 2, (0.0, 0.0, -1.0)),
        (4, 5, 6, (0.0, 0.0, 1.0)), (4, 6, 7, (0.0, 0.0, 1.0)),
        (0, 1, 5, (0.0, -1.0, 0.0)), (0, 5, 4, (0.0, -1.0, 0.0)),
        (2, 3, 7, (0.0, 1.0, 0.0)), (2, 7, 6, (0.0, 1.0, 0.0)),
        (0, 4, 7, (-1.0, 0.0, 0.0)), (0, 7, 3, (-1.0, 0.0, 0.0)),
        (1, 2, 6, (1.0, 0.0, 0.0)), (1, 6, 5, (1.0, 0.0, 0.0)),
    )
    vertices = []
    normals = []
    faces = []
    for i0, i1, i2, nrm in tris:
        base = len(vertices)
        vertices.extend([corners[i0], corners[i1], corners[i2]])
        normals.extend([nrm, nrm, nrm])
        faces.append((base, base + 1, base + 2))
    return np.asarray(vertices), np.asarray(normals), np.asarray(faces, dtype=int)


def test_normalize_clip_planes_unit_normal_and_drop_zero():
    planes = normalize_clip_planes([
        {"origin": [1, 2, 3], "normal": [0, 0, 4], "scale": 6},
        {"origin": [0, 0, 0], "normal": [0, 0, 0], "scale": 2},
    ])
    assert len(planes) == 1
    assert planes[0]["origin"] == pytest.approx([1.0, 2.0, 3.0])
    assert planes[0]["normal"] == pytest.approx([0.0, 0.0, 1.0])
    assert planes[0]["scale"] == pytest.approx(6.0)


def test_apply_clip_plane_pose_normalizes_and_rejects_zero():
    plane = {"origin": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0], "scale": 5.0}
    assert apply_clip_plane_pose(plane, (1, 2, 3), (0, 0, 4))
    assert plane["origin"] == pytest.approx([1.0, 2.0, 3.0])
    assert plane["normal"] == pytest.approx([0.0, 0.0, 1.0])
    assert plane["scale"] == pytest.approx(5.0)
    assert not apply_clip_plane_pose(plane, (9, 9, 9), (0, 0, 0))
    assert plane["origin"] == pytest.approx([1.0, 2.0, 3.0])
    assert plane["normal"] == pytest.approx([0.0, 0.0, 1.0])


def test_clip_plane_from_view_keep_side_toward_camera():
    plane = clip_plane_from_view(_identity_view((0.0, 0.0, 0.0)), [(0.0, 0.0, 0.0)])
    assert plane["normal"] == pytest.approx([0.0, 0.0, 1.0])
    assert plane["origin"][2] == pytest.approx(0.0)
    assert plane["scale"] >= 4.0


def test_clip_plane_from_view_goes_through_geometry_not_camera_origin():
    plane = clip_plane_from_view(
        _identity_view((0.0, 0.0, 50.0)),
        [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)],
    )
    assert plane["origin"][2] == pytest.approx(0.0)
    assert plane["origin"][0] == pytest.approx(1.0, abs=0.05)
    assert plane["origin"][1] == pytest.approx(1.0, abs=0.05)
    assert plane["normal"] == pytest.approx([0.0, 0.0, 1.0])


def test_clip_plane_from_view_origin_is_in_plane_aabb_center():
    pts = [(10.0, 1.0, 0.0), (10.0, -1.0, 0.0), (0.0, 0.0, 0.0)]
    plane = clip_plane_from_view(_identity_view((0.0, 0.0, 40.0)), pts)
    assert plane["origin"] == pytest.approx([5.0, 0.0, 0.0], abs=1e-6)


def test_oriented_clip_normal_tilt_and_turn():
    tilted = oriented_clip_normal([0.0, 0.0, 1.0], tilt_deg=90.0, turn_deg=0.0)
    assert tilted == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)
    turned = oriented_clip_normal([0.0, 0.0, 1.0], tilt_deg=0.0, turn_deg=90.0)
    assert turned == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)


def test_fit_plane_rectangle_covers_projected_points():
    pts = np.array([
        [10.0, 1.0, 3.0],
        [-10.0, -1.0, -2.0],
        [8.0, 0.5, 0.0],
    ], dtype=float)
    center, _n, u_ext, v_ext = fit_plane_rectangle(
        (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), pts,
    )
    hu = float(np.linalg.norm(u_ext))
    hv = float(np.linalg.norm(v_ext))
    u = u_ext / hu
    v = v_ext / hv
    rel = pts - center
    assert np.all(np.abs(rel @ u) <= hu + 1e-6)
    assert np.all(np.abs(rel @ v) <= hv + 1e-6)
    assert hv > hu
    corners, _n, _ue, _ve, _c = rectangle_corners(
        (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), points=pts,
    )
    zs = [float(c[2]) for c in corners]
    assert zs == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1e-6)


def test_clip_triangle_keeps_positive_halfspace():
    verts = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]], dtype=float)
    normals = np.array([[0.0, 1.0, 0.0]] * 3, dtype=float)
    faces = np.array([[0, 1, 2]], dtype=int)
    out_v, out_n, out_f = clip_triangle_mesh(
        verts, normals, faces, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
    )
    assert out_f.shape[0] >= 1
    assert out_f.shape[1] == 3
    assert np.all(out_v[:, 2] >= -1e-6)
    assert np.any(np.abs(out_v[:, 2]) < 1e-5)
    assert out_n.shape == out_v.shape


def test_clip_discards_fully_outside_and_keeps_fully_inside():
    verts, normals, faces = _cube_mesh()
    kept_v, kept_n, kept_f = clip_triangle_mesh(
        verts, normals, faces, (0.0, 0.0, 10.0), (0.0, 0.0, 1.0),
    )
    assert kept_f.shape[0] == 0
    same_v, same_n, same_f = clip_triangle_mesh(
        verts, normals, faces, (0.0, 0.0, -10.0), (0.0, 0.0, 1.0),
    )
    assert same_f.shape[0] == faces.shape[0]
    assert same_v.shape[0] > 0
    assert same_n.shape == same_v.shape


def test_clip_cube_half_z_keeps_nonnegative():
    verts, normals, faces = _cube_mesh()
    out_v, out_n, out_f = clip_triangle_mesh(
        verts, normals, faces, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
    )
    assert out_f.shape[0] > 0
    assert out_f.shape[0] != faces.shape[0]
    assert np.all(out_v[:, 2] >= -1e-6)
    assert np.any(out_v[:, 2] > 0.5)


def test_sequential_planes_clip_to_positive_octant():
    verts, normals, faces = _cube_mesh()
    out_v, out_n, out_f = clip_mesh_by_planes(
        verts, faces, normals,
        [
            {"origin": [0, 0, 0], "normal": [0, 0, 1], "scale": 4},
            {"origin": [0, 0, 0], "normal": [1, 0, 0], "scale": 4},
            {"origin": [0, 0, 0], "normal": [0, 1, 0], "scale": 4},
        ],
    )
    assert out_f.shape[0] > 0
    assert np.all(out_v[:, 0] >= -1e-6)
    assert np.all(out_v[:, 1] >= -1e-6)
    assert np.all(out_v[:, 2] >= -1e-6)


def test_surface_clip_planes_keep_halfspace_without_remesh_source():
    mesh = Surface(
        [(0.0, 0.0, 0.0)], algorithm="GAUSS", quality=1, bypass_colormap=True,
    )
    n_faces = int(mesh.faces.shape[0])
    source_faces = int(mesh._source_faces.shape[0])
    mesh.set_clip_planes([{"origin": [0, 0, 0], "normal": [0, 0, 1], "scale": 5}])
    assert mesh.faces.shape[0] > 0
    assert mesh.faces.shape[0] < n_faces
    assert int(mesh._source_faces.shape[0]) == source_faces
    assert np.all(np.asarray(mesh.vertices)[:, 2] >= -1e-5)
    tokens = mesh._create_CGO_list()
    kinds = [t for t in tokens if isinstance(t, str)]
    assert kinds.count("VERTEX") == 3 * len(mesh.faces)
    assert kinds.count("NORMAL") == kinds.count("VERTEX")


def test_retarget_surface_applies_clip_without_new_surface():
    points = [_point("a", (0.0, 0.0, 0.0))]
    coll = build_surface_collection(
        points, 1.5, 1.4, "GAUSS", 1, False, "pmv_surface",
    )
    mesh = coll[0]
    n_faces = int(mesh.faces.shape[0])
    planes = [{"origin": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0], "scale": 5.0}]
    assert retarget_surface_collection(
        coll, points, 1.5, 1.4, "GAUSS", 1, False, clip_planes=planes,
    )
    assert coll[0] is mesh
    assert mesh.faces.shape[0] < n_faces
    assert np.all(np.asarray(mesh.vertices)[:, 2] >= -1e-5)
    assert retarget_surface_collection(
        coll, points, 1.5, 1.4, "GAUSS", 1, False, clip_planes=planes,
    )
    assert coll[0] is mesh


def test_retarget_surface_clip_repaints_per_point_colors():
    pts = [
        _point("a", (0.0, 0.0, 0.0), color=(1.0, 0.0, 0.0)),
        _point("b", (6.0, 0.0, 0.0), color=(0.0, 0.0, 1.0)),
    ]
    coll = build_surface_collection(pts, 1.5, 1.4, "GAUSS", 1, False, "pmv_surface")
    mesh = coll[0]
    planes = [{"origin": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0], "scale": 5.0}]
    assert retarget_surface_collection(
        coll, pts, 1.5, 1.4, "GAUSS", 1, False, clip_planes=planes,
    )
    colors = np.asarray(mesh.color, dtype=float).reshape(-1, 3)
    verts = np.asarray(mesh.vertices, dtype=float).reshape(-1, 3)
    assert colors.shape[0] == verts.shape[0]
    assert colors.shape[0] > 1
    assert not np.allclose(colors, colors[0])
    near_a = np.linalg.norm(verts[:, :2], axis=1) < 2.0
    near_b = np.linalg.norm(verts[:, :2] - np.array([6.0, 0.0]), axis=1) < 2.0
    if np.any(near_a):
        assert float(colors[near_a, 0].mean()) > float(colors[near_a, 2].mean())
    if np.any(near_b):
        assert float(colors[near_b, 2].mean()) > float(colors[near_b, 0].mean())


def test_sphere_honors_clip_planes():
    from pymolviz.meshes.Sphere import Sphere

    sphere = Sphere(
        FixedPoint((0.0, 0.0, 0.0)), 1.0,
        bypass_colormap=True, frequency=2,
        clip_planes=[{"origin": [0, 0, 0], "normal": [0, 0, 1], "scale": 3}],
    )
    assert sphere.faces.shape[0] > 0
    assert np.all(np.asarray(sphere.vertices)[:, 2] >= -1e-5)
    sphere.set_clip_planes([{"origin": [0, 0, 0], "normal": [0, 0, -1], "scale": 3}])
    assert np.all(np.asarray(sphere.vertices)[:, 2] <= 1e-5)


def test_box_honors_clip_planes():
    from pymolviz.meshes.CenteredBox import CenteredBox

    box = CenteredBox(
        FixedPoint((0.0, 0.0, 0.0)), (2.0, 2.0, 2.0),
        bypass_colormap=True,
        clip_planes=[{"origin": [0, 0, 0], "normal": [0, 0, 1], "scale": 3}],
    )
    assert box.faces.shape[0] > 0
    assert np.all(np.asarray(box.vertices)[:, 2] >= -1e-5)


def test_clip_segment_by_planes_shortens_or_drops():
    from pymolviz.util.mesh_clip import clip_segment_by_planes

    kept = clip_segment_by_planes(
        (0.0, 0.0, -1.0), (0.0, 0.0, 1.0),
        [{"origin": [0, 0, 0], "normal": [0, 0, 1], "scale": 3}],
    )
    assert kept is not None
    start, end = kept
    assert start[2] == pytest.approx(0.0)
    assert end[2] == pytest.approx(1.0)
    dropped = clip_segment_by_planes(
        (0.0, 0.0, -2.0), (0.0, 0.0, -1.0),
        [{"origin": [0, 0, 0], "normal": [0, 0, 1], "scale": 3}],
    )
    assert dropped is None


def test_disabled_sphere_emits_empty_cgo():
    from pymolviz.meshes.Sphere import Sphere

    sphere = Sphere(
        FixedPoint((0.0, 0.0, 0.0)), 1.0,
        bypass_colormap=True, frequency=2, enabled=False,
    )
    assert sphere._create_CGO_list() == []


def test_surface_point_enabled_omits_disabled_from_mesh_but_keeps_sources():
    from pymolviz.meshes.Surface import Surface

    mesh = Surface(
        [FixedPoint((0.0, 0.0, 0.0)), FixedPoint((4.0, 0.0, 0.0))],
        algorithm="GAUSS", quality=1, bypass_colormap=True,
        point_enabled=[True, False],
    )
    assert len(mesh.point_sources) == 2
    assert mesh.point_enabled == [True, False]
    xs = np.asarray(mesh._source_vertices, dtype=float).reshape(-1, 3)[:, 0]
    assert np.all(xs < 2.0)


def test_build_surface_collection_persists_disabled_sources():
    points = [
        _point("a", (0.0, 0.0, 0.0)),
        _point("b", (4.0, 0.0, 0.0)),
    ]
    points[1] = points[1].with_enabled(False)
    coll = build_surface_collection(
        points, 1.5, 1.4, "GAUSS", 1, False, "pmv_surface",
    )
    mesh = coll[0]
    assert len(mesh.point_sources) == 2
    assert mesh.point_enabled == [True, False]


