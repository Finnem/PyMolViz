"""SAS (rolling-ball) / ASA (accessible) solvent surfaces around point spheres."""

from __future__ import annotations

import numpy as np
import pytest

from pymolviz.meshes.Surface import Surface
from pymolviz.points import AtomPoint, FixedPoint
from pymolviz.util.solvent_surface import (
    ASA_FREQUENCY,
    DEFAULT_ATOM_RADIUS,
    DEFAULT_PROBE_RADIUS,
    _convex_cap_frequency,
    _decimate_torus_succ,
    _fill_boundary_holes,
    _reduced_surface,
    build_solvent_surface,
    element_from_atom_name,
    expanded_radii,
    normalize_algorithm,
    resolve_atom_radii,
    signed_distance,
    vdw_for_atom,
    vdw_for_element,
)
from pymolviz.wizards.builders.points import VisualPoint
from pymolviz.wizards.builders.preview import (
    build_surface_collection,
    retarget_surface_collection,
)


def _r_exp(atom_radius=DEFAULT_ATOM_RADIUS, probe_radius=DEFAULT_PROBE_RADIUS):
    return float(atom_radius) + float(probe_radius)


def _point(name, xyz, color=(0.2, 0.6, 0.9)):
    return VisualPoint(
        name, "manual", xyz[0], xyz[1], xyz[2],
        color=color, point_source=FixedPoint(xyz),
    )


def _triangle_cgo_corners(tokens):
    """Parse Mesh triangle CGO into ``(normal, vertex, color)`` per corner."""
    begin = tokens.index("BEGIN")
    assert tokens[begin + 1] == "TRIANGLES"
    end = tokens.index("END", begin)
    body = list(tokens[begin + 2:end])
    corners = []
    color = None
    normal = None
    i = 0
    while i < len(body):
        kind = body[i]
        if kind == "COLOR":
            color = np.array([float(body[i + 1]), float(body[i + 2]), float(body[i + 3])])
            i += 4
            continue
        if kind == "NORMAL":
            normal = np.array([float(body[i + 1]), float(body[i + 2]), float(body[i + 3])])
            i += 4
            continue
        if kind == "VERTEX":
            assert normal is not None
            vert = np.array([float(body[i + 1]), float(body[i + 2]), float(body[i + 3])])
            corners.append((normal, vert, color))
            i += 4
            continue
        raise AssertionError("unexpected CGO token %r" % (kind,))
    return corners


def test_normalize_algorithm():
    assert normalize_algorithm("asa") == "ASA"
    assert normalize_algorithm("SAS") == "SAS"
    assert normalize_algorithm("nope") == "SAS"


def test_empty_surface_has_no_geometry():
    verts, normals, faces = build_solvent_surface([])
    assert verts.shape == (0, 3)
    assert normals.shape == (0, 3)
    assert faces.shape == (0, 3)
    mesh = Surface([], bypass_colormap=True)
    assert mesh.vertices.shape == (0, 3)
    assert mesh.faces.shape == (0, 3)


def test_asa_single_sphere_verts_on_expanded_radius():
    center = np.array([1.0, -2.0, 0.5])
    atom_r, probe = 1.2, 1.4
    mesh = Surface(
        [center], atom_radius=atom_r, probe_radius=probe,
        algorithm="ASA", quality=1, bypass_colormap=True,
    )
    assert mesh.vertices.shape[0] > 0
    assert mesh.faces.shape[0] > 0
    radii = np.linalg.norm(mesh.vertices - center, axis=1)
    assert radii == pytest.approx(_r_exp(atom_r, probe), abs=1e-6)


def test_sas_single_sphere_verts_on_atom_radius():
    center = np.zeros(3)
    mesh = Surface(
        [center], algorithm="SAS", quality=1, bypass_colormap=True,
    )
    assert mesh.vertices.shape[0] > 0
    assert mesh.faces.shape[0] > 0
    radii = np.linalg.norm(mesh.vertices - center, axis=1)
    assert radii == pytest.approx(DEFAULT_ATOM_RADIUS, abs=1e-4)
    tokens = mesh._create_CGO_list()
    kinds = [t for t in tokens if isinstance(t, str)]
    assert kinds[:2] == ["ENABLE", "LIGHTING"]
    assert "BEGIN" in kinds
    assert "TRIANGLES" in kinds
    assert kinds[-1] == "END"


def test_sas_two_sphere_caps_use_template_icosphere():
    from pymolviz.util.geometries import geodesic_icosphere

    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    mesh = Surface(
        points, algorithm="SAS", quality=3, bypass_colormap=True,
    )
    unit, _faces, _edges = geodesic_icosphere(_convex_cap_frequency(ASA_FREQUENCY[3]))
    radius = DEFAULT_ATOM_RADIUS
    ico = unit * radius
    dist0 = np.linalg.norm(mesh.vertices - points[0], axis=1)
    cap = mesh.vertices[np.abs(dist0 - radius) < 2e-3]
    assert cap.shape[0] >= 12
    nearest = np.min(
        np.linalg.norm(cap[:, None, :] - ico[None, :, :], axis=2),
        axis=1,
    )
    assert float(np.mean(nearest < 2e-3)) > 0.25


def test_convex_cap_frequency_ladder():
    assert _convex_cap_frequency(2) == 12
    assert _convex_cap_frequency(4) == 16
    assert _convex_cap_frequency(8) == 20


def test_msms_reduced_surface_one_two_three_spheres():
    probe = DEFAULT_PROBE_RADIUS
    one = _reduced_surface(np.zeros((1, 3)), expanded_radii(DEFAULT_ATOM_RADIUS, 1, probe), probe)
    assert one["faces"] == []
    assert one["free_edges"] == []
    assert one["free_vertices"] == [0]

    two_c = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    two = _reduced_surface(two_c, expanded_radii(DEFAULT_ATOM_RADIUS, 2, probe), probe)
    assert two["faces"] == []
    assert set(two["free_edges"]) == {(0, 1)}
    assert two["free_vertices"] == []

    three_c = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, float(np.sqrt(3.0)), 0.0]])
    three = _reduced_surface(three_c, expanded_radii(DEFAULT_ATOM_RADIUS, 3, probe), probe)
    assert len(three["faces"]) == 2
    assert {frozenset((i, j, k)) for i, j, k, _q in three["faces"]} == {frozenset((0, 1, 2))}
    assert three["free_edges"] == []
    assert three["free_vertices"] == []
    assert set(three["edge_faces"].keys()) == {(0, 1), (0, 2), (1, 2)}
    assert all(len(v) == 2 for v in three["edge_faces"].values())


def test_sas_overlapping_verts_lie_between_vdw_and_accessible():
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    mesh = Surface(
        points, algorithm="SAS", quality=1, bypass_colormap=True,
    )
    sas_r = expanded_radii(DEFAULT_ATOM_RADIUS, 2, DEFAULT_PROBE_RADIUS)
    vdw_r = sas_r - DEFAULT_PROBE_RADIUS
    sdf_sas = signed_distance(mesh.vertices, points, sas_r)
    sdf_vdw = signed_distance(mesh.vertices, points, vdw_r)
    assert np.all(sdf_sas <= 0.08)
    assert np.all(sdf_vdw >= -0.08)
    assert float(np.max(sdf_vdw)) > 0.05


def _boundary_edge_count(faces):
    count = {}
    for a, b, c in np.asarray(faces, dtype=int):
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (u, v) if u < v else (v, u)
            count[key] = count.get(key, 0) + 1
    return sum(1 for n in count.values() if n == 1)


def _edge_multiplicities(faces):
    count = {}
    for a, b, c in np.asarray(faces, dtype=int):
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (u, v) if u < v else (v, u)
            count[key] = count.get(key, 0) + 1
    return count


def test_fill_boundary_holes_closes_small_triangle_not_large_gap():
    verts = np.array([
        [0.0, 0.0, 0.0], [0.12, 0.0, 0.0], [0.06, 0.10, 0.0], [0.06, 0.03, 0.11],
    ], dtype=float)
    faces = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3]], dtype=int)
    _verts, filled = _fill_boundary_holes(verts, faces)
    assert _boundary_edge_count(filled) == 0
    large = np.array([
        [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.5, 0.5, 0.5],
    ], dtype=float)
    large_faces = np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=int)
    _lv, unfilled = _fill_boundary_holes(large, large_faces)
    assert _boundary_edge_count(unfilled) == 3


def test_sas_overlapping_mesh_has_no_small_boundary_holes():
    two = Surface(
        [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
        algorithm="SAS", quality=1, bypass_colormap=True,
    )
    assert two.faces.shape[0] > 0
    assert _boundary_edge_count(two.faces) == 0
    three = Surface(
        [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, float(np.sqrt(3.0)), 0.0)],
        algorithm="SAS", quality=1, bypass_colormap=True,
    )
    assert three.faces.shape[0] > 0
    assert _boundary_edge_count(three.faces) == 0


def test_sas_overlapping_mesh_is_manifold_without_duplicate_faces():
    samples = (
        ([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)], 1),
        ([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, float(np.sqrt(3.0)), 0.0)], 1),
        ([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)], 3),
        ([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, float(np.sqrt(3.0)), 0.0)], 3),
    )
    for points, quality in samples:
        mesh = Surface(
            points, algorithm="SAS", quality=quality, bypass_colormap=True,
        )
        assert mesh.faces.shape[0] > 0
        keys = np.sort(np.asarray(mesh.faces, dtype=int), axis=1)
        assert len(np.unique(keys, axis=0)) == len(mesh.faces)
        counts = _edge_multiplicities(mesh.faces)
        assert counts
        assert all(n == 2 for n in counts.values())


def test_sas_cluster_is_watertight_without_interior_chords():
    points = np.array([
        [0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.75, 1.3, 0.0],
        [2.8, 0.4, 0.2], [3.5, 1.6, 0.0], [4.9, 1.2, 0.3],
        [5.6, 2.3, 0.1], [3.0, -0.9, 0.4],
    ])
    mesh = Surface(
        points, algorithm="SAS", quality=3, bypass_colormap=True,
        atom_radius=1.7, probe_radius=1.4, radius_mode="uniform",
    )
    assert mesh.faces.shape[0] > 200
    assert _boundary_edge_count(mesh.faces) == 0
    counts = _edge_multiplicities(mesh.faces)
    assert all(n == 2 for n in counts.values())
    vdw = np.full(len(points), 1.7)
    p0 = mesh.vertices[mesh.faces[:, 0]]
    p1 = mesh.vertices[mesh.faces[:, 1]]
    p2 = mesh.vertices[mesh.faces[:, 2]]
    cent = (p0 + p1 + p2) / 3.0
    sdf_vdw = signed_distance(cent, points, vdw)
    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    interior = (sdf_vdw < -0.15) & (area > 0.02)
    assert int(np.count_nonzero(interior)) == 0
    assert float(np.max(sdf_vdw)) > 0.05


def test_sas_triangles_are_not_extremely_skinny():
    mesh = Surface(
        [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, float(np.sqrt(3.0)), 0.0)],
        algorithm="SAS", quality=3, bypass_colormap=True,
    )
    p0 = mesh.vertices[mesh.faces[:, 0]]
    p1 = mesh.vertices[mesh.faces[:, 1]]
    p2 = mesh.vertices[mesh.faces[:, 2]]
    e0 = np.linalg.norm(p1 - p0, axis=1)
    e1 = np.linalg.norm(p2 - p1, axis=1)
    e2 = np.linalg.norm(p0 - p2, axis=1)
    longest = np.maximum(np.maximum(e0, e1), e2)
    shortest = np.minimum(np.minimum(e0, e1), e2)
    ratios = longest / np.maximum(shortest, 1e-12)
    assert float(np.median(ratios)) < 4.0
    assert float(np.percentile(ratios, 95)) < 12.0


def test_sas_contact_circle_has_no_spike_faces():
    """Cap triangles that share a torus edge should not be long Delaunay spikes."""
    points = np.array([
        [0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.75, 1.3, 0.0],
        [2.8, 0.4, 0.2], [3.5, 1.6, 0.0], [4.9, 1.2, 0.3],
        [5.6, 2.3, 0.1], [3.0, -0.9, 0.4],
    ])
    mesh = Surface(
        points, algorithm="SAS", quality=3, bypass_colormap=True,
        atom_radius=1.7, probe_radius=1.4, radius_mode="uniform",
    )
    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    p0, p1, p2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    cent = (p0 + p1 + p2) / 3.0
    sdf = signed_distance(cent, points, np.full(len(points), 1.7))
    sdf_v = signed_distance(verts, points, np.full(len(points), 1.7))
    fn = np.cross(p1 - p0, p2 - p0)
    fn = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-18)
    edge_faces = {}
    for fi, (a, b, c) in enumerate(faces):
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (u, v) if u < v else (v, u)
            edge_faces.setdefault(key, []).append(fi)
    dihs = []
    spikes = []
    for (a, b), fis in edge_faces.items():
        if len(fis) != 2:
            continue
        s0, s1 = float(sdf[fis[0]]), float(sdf[fis[1]])
        if not ((s0 < 0.025 and s1 > 0.04) or (s1 < 0.025 and s0 > 0.04)):
            continue
        c = float(np.clip(np.dot(fn[fis[0]], fn[fis[1]]), -1.0, 1.0))
        dihs.append(float(np.degrees(np.arccos(c))))
        thirds = []
        for fi in fis:
            thirds.append([idx for idx in faces[fi] if idx not in (a, b)][0])
        cap_third = thirds[0] if float(sdf_v[thirds[0]]) <= float(sdf_v[thirds[1]]) else thirds[1]
        edge = float(np.linalg.norm(verts[a] - verts[b]))
        height = float(np.linalg.norm(verts[cap_third] - 0.5 * (verts[a] + verts[b])))
        if edge < 0.08:
            continue
        spikes.append(height / max(edge, 1e-9))
    assert dihs
    assert float(np.percentile(dihs, 95)) < 22.0
    assert float(np.max(dihs)) < 28.0
    assert spikes
    assert float(np.percentile(spikes, 95)) < 4.5
    assert float(np.max(spikes)) < 6.0


def test_decimate_torus_succ_drops_clustered_uniforms_not_triples():
    """Cap and torus must share vertices: skipped rails cannot keep a successor."""
    xs = [0.0, 0.02, 0.04, 0.20, 0.40, 0.60, 0.88, 0.90]
    rails = []
    for i, x in enumerate(xs):
        row = np.array([[x, 0.0, 0.0], [x, 0.0, 1.0]], dtype=float)
        rails.append((0.0, row[0], row, i in (0, 7)))
    succ = [i + 1 for i in range(7)] + [-1]
    new_succ = _decimate_torus_succ(rails, succ, min_dist=0.08)
    edges = [(s, nxt) for s, nxt in enumerate(new_succ) if nxt >= 0]
    assert edges
    assert new_succ[1] == -1
    assert new_succ[2] == -1
    for a, b in edges:
        dist = float(np.linalg.norm(rails[a][2][0] - rails[b][2][0]))
        if rails[a][3] and rails[b][3]:
            continue
        assert dist >= 0.08 - 1e-9
    indeg = [0] * len(rails)
    for _s, nxt in edges:
        indeg[nxt] += 1
    assert max(indeg) <= 1


def test_decimate_torus_succ_keeps_theta_spaced_rails_on_tiny_circle():
    """A small contact circle must not collapse to a few huge torus facets."""
    n = 24
    rad = 0.03
    rails = []
    for i in range(n):
        theta = 2.0 * np.pi * i / float(n)
        p = np.array([rad * np.cos(theta), rad * np.sin(theta), 0.0], dtype=float)
        row = np.vstack((p, p + np.array([0.0, 0.0, 1.0])))
        rails.append((theta, p, row, False))
    succ = [(i + 1) % n for i in range(n)]
    new_succ = _decimate_torus_succ(rails, succ, min_dist=0.08)
    kept = sum(1 for nxt in new_succ if nxt >= 0)
    assert kept >= 16


def test_sas_faces_have_consistent_outward_winding():
    samples = (
        ([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)], 1, None),
        ([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, float(np.sqrt(3.0)), 0.0)], 1, None),
        ([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 1.7, 0.0)], 1, [1.70, 1.55, 1.52]),
        ([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 1.7, 0.0)], 3, [1.70, 1.55, 1.52]),
    )
    for points, quality, point_radii in samples:
        mesh = Surface(
            points, algorithm="SAS", quality=quality, bypass_colormap=True,
            point_radii=point_radii, atom_radius=1.0,
        )
        assert mesh.faces.shape[0] > 0
        pts = np.asarray(points, dtype=float)
        if point_radii is None:
            radii = expanded_radii(DEFAULT_ATOM_RADIUS, len(points), DEFAULT_PROBE_RADIUS)
        else:
            radii = np.asarray(point_radii, dtype=float) + DEFAULT_PROBE_RADIUS
        p0 = mesh.vertices[mesh.faces[:, 0]]
        p1 = mesh.vertices[mesh.faces[:, 1]]
        p2 = mesh.vertices[mesh.faces[:, 2]]
        fn = np.cross(p1 - p0, p2 - p0)
        unit = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-18)
        cent = (p0 + p1 + p2) / 3.0
        sdf_pos = signed_distance(cent + 0.05 * unit, pts, radii)
        sdf_neg = signed_distance(cent - 0.05 * unit, pts, radii)
        assert float(np.mean(sdf_pos > sdf_neg)) > 0.85


def test_sas_cgo_normals_match_triangle_winding():
    mesh = Surface(
        [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 1.7, 0.0)],
        algorithm="SAS", quality=1, bypass_colormap=True,
        point_radii=[1.70, 1.55, 1.52], atom_radius=1.0,
    )
    tokens = mesh._create_CGO_list()
    corners = _triangle_cgo_corners(tokens)
    assert len(corners) == 3 * len(mesh.faces)
    dots = []
    for t in range(len(mesh.faces)):
        verts = []
        normals = []
        for k in range(3):
            normal, vert, _color = corners[t * 3 + k]
            normals.append(normal)
            verts.append(vert)
        fn = np.cross(verts[1] - verts[0], verts[2] - verts[0])
        avg = np.mean(np.stack(normals, axis=0), axis=0)
        dots.append(float(np.dot(avg, fn)))
    assert dots
    arr = np.asarray(dots, dtype=float)
    assert float(np.mean(arr > 0.0)) > 0.85
    assert float(np.median(arr)) > 0.0


def test_sas_normals_follow_rolling_ball():
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    mesh = Surface(
        points, algorithm="SAS", quality=3, bypass_colormap=True,
    )
    sas_r = expanded_radii(DEFAULT_ATOM_RADIUS, 2, DEFAULT_PROBE_RADIUS)
    vdw_r = sas_r - DEFAULT_PROBE_RADIUS
    sdf_vdw = signed_distance(mesh.vertices, points, vdw_r)
    n = mesh.normals / np.maximum(
        np.linalg.norm(mesh.normals, axis=1, keepdims=True), 1e-18,
    )
    cap = sdf_vdw <= 0.02
    assert int(np.count_nonzero(cap)) >= 8
    owners = np.argmin(
        np.linalg.norm(mesh.vertices[:, None, :] - points[None, :, :], axis=2)
        - vdw_r.reshape(1, -1),
        axis=1,
    )
    radial = mesh.vertices - points[owners]
    radial = radial / np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-18)
    other = 1 - owners
    d_other = np.linalg.norm(mesh.vertices - points[other], axis=1) - vdw_r[other]
    far_cap = cap & (d_other > 0.75)
    assert int(np.count_nonzero(far_cap)) >= 8
    assert float(np.mean(np.sum(n[far_cap] * radial[far_cap], axis=1))) > 0.95
    valley = sdf_vdw > 0.12
    assert int(np.count_nonzero(valley)) >= 8
    origin = np.array([1.0, 0.0, 0.0])
    r_sas = float(np.sqrt(sas_r[0] * sas_r[0] - 1.0))
    delta = mesh.vertices[valley] - origin
    planar = delta.copy()
    planar[:, 0] = 0.0
    length = np.maximum(np.linalg.norm(planar, axis=1, keepdims=True), 1e-12)
    closest = origin + r_sas * (planar / length)
    toward_probe = closest - mesh.vertices[valley]
    toward_probe = toward_probe / np.maximum(
        np.linalg.norm(toward_probe, axis=1, keepdims=True), 1e-18,
    )
    assert float(np.mean(np.sum(n[valley] * toward_probe, axis=1))) > 0.85


def test_sas_cap_triangle_normals_are_smooth_not_faceted():
    mesh = Surface(
        [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
        algorithm="SAS", quality=3, bypass_colormap=True,
    )
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    sdf = signed_distance(mesh.vertices, points, np.full(2, DEFAULT_ATOM_RADIUS))
    n = mesh.normals / np.maximum(
        np.linalg.norm(mesh.normals, axis=1, keepdims=True), 1e-18,
    )
    cap_faces = []
    for a, b, c in np.asarray(mesh.faces, dtype=int):
        if max(float(sdf[a]), float(sdf[b]), float(sdf[c])) < 0.02:
            cap_faces.append((int(a), int(b), int(c)))
        if len(cap_faces) >= 12:
            break
    assert cap_faces
    for a, b, c in cap_faces:
        dots = (
            float(np.dot(n[a], n[b])),
            float(np.dot(n[b], n[c])),
            float(np.dot(n[c], n[a])),
        )
        assert min(dots) < 0.999
        assert min(dots) > 0.90


def test_sas_contact_circle_normals_do_not_crease():
    """Adjacent cap/torus vertices should not have a specular crease."""
    mesh = Surface(
        [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
        algorithm="SAS", quality=3, bypass_colormap=True,
    )
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    vdw = np.full(2, DEFAULT_ATOM_RADIUS)
    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    n = mesh.normals / np.maximum(
        np.linalg.norm(mesh.normals, axis=1, keepdims=True), 1e-18,
    )
    sdf = signed_distance(verts, points, vdw)
    angles = []
    for a, b, c in faces:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            s0, s1 = float(sdf[u]), float(sdf[v])
            lo, hi = (s0, s1) if s0 <= s1 else (s1, s0)
            if lo > 0.015 or hi < 0.03:
                continue
            cdot = float(np.clip(np.dot(n[u], n[v]), -1.0, 1.0))
            angles.append(float(np.degrees(np.arccos(cdot))))
    assert angles
    assert float(np.percentile(angles, 95)) < 10.0
    assert float(np.max(angles)) < 16.0


def test_sas_unequal_radii_vertices_lie_between_vdw_and_accessible():
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.7, 0.0]])
    point_radii = [1.70, 1.55, 1.52]
    probe = 1.4
    mesh = Surface(
        points, algorithm="SAS", quality=1, bypass_colormap=True,
        point_radii=point_radii, atom_radius=1.0, probe_radius=probe,
    )
    sas_r = np.asarray(point_radii, dtype=float) + probe
    vdw_r = np.asarray(point_radii, dtype=float)
    sdf_sas = signed_distance(mesh.vertices, points, sas_r)
    sdf_vdw = signed_distance(mesh.vertices, points, vdw_r)
    assert float(np.max(sdf_sas)) < 0.08
    assert float(np.min(sdf_vdw)) > -0.08
    assert float(np.max(sdf_vdw)) > 0.05


def test_sas_two_sphere_saddle_is_probe_torus():
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    mesh = Surface(
        points, algorithm="SAS", quality=1, bypass_colormap=True,
    )
    sas_r = expanded_radii(DEFAULT_ATOM_RADIUS, 2, DEFAULT_PROBE_RADIUS)
    vdw_r = sas_r - DEFAULT_PROBE_RADIUS
    sdf_vdw = signed_distance(mesh.vertices, points, vdw_r)
    valley = sdf_vdw > 0.05
    assert int(np.count_nonzero(valley)) >= 8
    circle_origin = np.array([1.0, 0.0, 0.0])
    r_sas = float(np.sqrt(sas_r[0] * sas_r[0] - 1.0))
    delta = mesh.vertices[valley] - circle_origin
    axial = delta[:, 0]
    radial = np.linalg.norm(delta[:, 1:3], axis=1)
    dist_circle = np.hypot(radial - r_sas, axial)
    assert float(np.median(np.abs(dist_circle - DEFAULT_PROBE_RADIUS))) < 0.12


def test_sas_three_sphere_valley_is_probe_sphere():
    points = np.array([
        [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, float(np.sqrt(3.0)), 0.0],
    ])
    probe = DEFAULT_PROBE_RADIUS
    mesh = Surface(
        points, algorithm="SAS", quality=3, bypass_colormap=True,
    )
    sas_r = expanded_radii(DEFAULT_ATOM_RADIUS, 3, probe)
    vdw_r = sas_r - probe
    sdf_vdw = signed_distance(mesh.vertices, points, vdw_r)
    valley = mesh.vertices[sdf_vdw > 0.08]
    assert valley.shape[0] >= 8
    centroid = points.mean(axis=0)
    height = float(np.sqrt(sas_r[0] * sas_r[0] - np.dot(centroid - points[0], centroid - points[0])))
    probes = (
        centroid + np.array([0.0, 0.0, height]),
        centroid + np.array([0.0, 0.0, -height]),
    )
    dist_probe = np.min(
        [np.linalg.norm(valley - q, axis=1) for q in probes],
        axis=0,
    )
    on_probe = np.abs(dist_probe - probe) < 0.18
    assert int(np.count_nonzero(on_probe)) >= 6


def test_sas_cluster_wraps_each_atom():
    points = np.array([
        [0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.75, 1.3, 0.0],
        [2.8, 0.4, 0.2], [3.5, 1.6, 0.0], [4.9, 1.2, 0.3],
        [5.6, 2.3, 0.1], [3.0, -0.9, 0.4],
    ])
    mesh = Surface(
        points, algorithm="SAS", quality=1, bypass_colormap=True,
        atom_radius=1.7, probe_radius=1.4, radius_mode="uniform",
    )
    assert mesh.faces.shape[0] > 200
    reach = [
        float(np.min(np.linalg.norm(mesh.vertices - c, axis=1)))
        for c in points
    ]
    assert max(reach) < 2.4


def test_sas_overlapping_vertices_are_uniquely_welded():
    samples = (
        ([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)], 1),
        ([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, float(np.sqrt(3.0)), 0.0)], 3),
    )
    for points, quality in samples:
        mesh = Surface(
            points, algorithm="SAS", quality=quality, bypass_colormap=True,
        )
        keys = np.round(np.asarray(mesh.vertices, dtype=float), 5)
        assert len(np.unique(keys, axis=0)) == len(mesh.vertices)


def test_asa_overlapping_spheres_meet_at_intersection():
    c0 = np.array([0.0, 0.0, 0.0])
    c1 = np.array([2.0, 0.0, 0.0])
    radius = _r_exp()
    isolated = Surface(
        [c0], algorithm="ASA", quality=1, bypass_colormap=True,
    )
    mesh = Surface(
        [c0, c1], algorithm="ASA", quality=1, bypass_colormap=True,
    )
    assert mesh.vertices.shape[0] < 2 * isolated.vertices.shape[0]
    dist0 = np.linalg.norm(mesh.vertices - c0, axis=1)
    dist1 = np.linalg.norm(mesh.vertices - c1, axis=1)
    assert np.all(dist0 >= radius - 1e-3)
    assert np.all(dist1 >= radius - 1e-3)
    circle_center = np.array([1.0, 0.0, 0.0])
    circle_radius = float(np.sqrt(radius * radius - 1.0))
    on_plane = np.abs(mesh.vertices[:, 0] - 1.0) < 0.04
    on_ring = np.abs(np.linalg.norm(mesh.vertices - circle_center, axis=1) - circle_radius) < 0.04
    assert int(np.count_nonzero(on_plane & on_ring)) >= 6
    assert mesh.faces.shape[0] > isolated.faces.shape[0] // 2


def test_asa_triple_intersection_keeps_junction():
    radius = _r_exp()
    c0 = np.zeros(3)
    c1 = np.array([2.0, 0.0, 0.0])
    c2 = np.array([1.0, np.sqrt(3.0), 0.0])
    mesh = Surface(
        [c0, c1, c2], algorithm="ASA", quality=2, bypass_colormap=True,
    )
    assert mesh.vertices.shape[0] > 0
    assert mesh.faces.shape[0] > 12
    for center in (c0, c1, c2):
        assert np.all(np.linalg.norm(mesh.vertices - center, axis=1) >= radius - 1e-3)
    centroid = (c0 + c1 + c2) / 3.0
    height = float(np.sqrt(radius * radius - np.dot(centroid - c0, centroid - c0)))
    for sign in (1.0, -1.0):
        triple = centroid + np.array([0.0, 0.0, sign * height])
        nearest = float(np.min(np.linalg.norm(mesh.vertices - triple, axis=1)))
        assert nearest < 0.35


def test_surface_cgo_is_triangle_mesh():
    mesh = Surface(
        [(0.0, 0.0, 0.0)], algorithm="ASA", quality=1, bypass_colormap=True,
    )
    tokens = mesh._create_CGO_list()
    kinds = [t for t in tokens if isinstance(t, str)]
    assert kinds[:2] == ["ENABLE", "LIGHTING"]
    assert "BEGIN" in kinds
    assert "TRIANGLES" in kinds
    assert kinds[-1] == "END"
    assert tokens.count("VERTEX") == 3 * len(mesh.faces)
    assert tokens.count("COLOR") == 1
    assert tokens.count("NORMAL") == tokens.count("VERTEX")
    for i, kind in enumerate(tokens):
        if kind == "VERTEX":
            assert tokens[i - 4] == "NORMAL"


def test_sas_wireframe_cgo_uses_unique_cones():
    from pymolviz.meshes.Mesh import _unique_undirected_edges

    mesh = Surface(
        [(0.0, 0.0, 0.0)], algorithm="SAS", quality=1,
        color=(0.2, 0.6, 0.9), bypass_colormap=True, wireframe=True,
    )
    tokens = mesh._create_CGO_list()
    kinds = [t for t in tokens if isinstance(t, str)]
    assert "CONE" in kinds
    assert "TRIANGLES" not in kinds
    n_unique = len(_unique_undirected_edges(mesh.faces))
    assert kinds.count("CONE") == n_unique
    assert n_unique < 3 * len(mesh.faces)


def test_build_surface_collection_wireframe():
    pts = [_point("a", (0.0, 0.0, 0.0))]
    coll = build_surface_collection(pts, 1.5, 1.4, "SAS", 1, True, "pmv_surface")
    tokens = coll._create_CGO_list()
    kinds = [t for t in tokens if isinstance(t, str)]
    assert "CONE" in kinds
    assert "TRIANGLES" not in kinds


def test_surface_rebuild_follows_fixed_and_atom_points():
    mesh = Surface(
        [FixedPoint((0.0, 0.0, 0.0))],
        algorithm="ASA", quality=1, bypass_colormap=True,
    )
    before = np.array(mesh.vertices, copy=True)
    mesh.point_sources = [FixedPoint((5.0, 0.0, 0.0))]
    mesh.rebuild(None)
    assert np.allclose(mesh.vertices, before + np.array([5.0, 0.0, 0.0]))

    atom = AtomPoint("prot", 1, last_xyz=(2.0, 3.0, 4.0))
    mesh.point_sources = [atom]
    mesh.rebuild(None)
    center = np.asarray(mesh.vertices).mean(axis=0)
    assert center == pytest.approx((2.0, 3.0, 4.0), abs=0.2)


def test_retarget_surface_collection_rejects_algorithm_change():
    points = [_point("a", (0.0, 0.0, 0.0))]
    coll = build_surface_collection(
        points, 1.5, 1.4, "SAS", 1, False, "pmv_surface",
    )
    assert retarget_surface_collection(coll, points, 1.5, 1.4, "SAS", 1, False)
    assert not retarget_surface_collection(coll, points, 1.5, 1.4, "ASA", 1, False)
    moved = [_point("a", (4.0, 0.0, 0.0))]
    assert not retarget_surface_collection(coll, moved, 1.5, 1.4, "SAS", 1, False)
    custom = [_point("a", (0.0, 0.0, 0.0))]
    custom[0] = custom[0].with_radius(2.0)
    assert not retarget_surface_collection(coll, custom, 1.5, 1.4, "SAS", 1, False)
    recolored = [_point("a", (0.0, 0.0, 0.0), color=(0.0, 1.0, 0.0))]
    coll[0]._create_CGO_list()
    assert coll[0]._cached_cgo is not None
    assert retarget_surface_collection(coll, recolored, 1.5, 1.4, "SAS", 1, False)
    assert np.allclose(np.asarray(coll[0].color, dtype=float).reshape(-1)[:3], (0.0, 1.0, 0.0))
    assert coll[0]._cached_cgo is None
    assert retarget_surface_collection(coll, recolored, 1.5, 1.4, "SAS", 1, True)
    assert coll[0].wireframe is True
    kinds = [t for t in coll[0]._create_CGO_list() if isinstance(t, str)]
    assert "CONE" in kinds
    assert "TRIANGLES" not in kinds


def test_protein_ca_name_is_carbon_unless_elem_is_calcium():
    assert element_from_atom_name("CA", "") == "C"
    assert element_from_atom_name("CA", "CA") == "CA"
    assert vdw_for_atom("CA", "") == pytest.approx(vdw_for_element("C"))
    assert vdw_for_atom("CA", "CA") == pytest.approx(vdw_for_element("CA"))
    protein = AtomPoint("prot", 1, name="CA", last_xyz=(0.0, 0.0, 0.0))
    calcium = AtomPoint("prot", 2, name="CA", elem="CA", last_xyz=(0.0, 0.0, 0.0))
    radii = resolve_atom_radii([protein, calcium], 2, atom_radius=1.5, radius_mode="vdw")
    assert radii[0] == pytest.approx(vdw_for_element("C"))
    assert radii[1] == pytest.approx(vdw_for_element("CA"))


def test_resolve_atom_radii_uniform_vdw_and_override():
    oxygen = AtomPoint("prot", 1, name="O", elem="O", last_xyz=(0.0, 0.0, 0.0))
    carbon = AtomPoint("prot", 2, name="CA", elem="C", last_xyz=(1.0, 0.0, 0.0))
    manual = FixedPoint((2.0, 0.0, 0.0))
    sources = [oxygen, carbon, manual]
    uniform = resolve_atom_radii(sources, 3, atom_radius=1.5, radius_mode="uniform")
    assert uniform == pytest.approx([1.5, 1.5, 1.5])
    vdw = resolve_atom_radii(sources, 3, atom_radius=1.5, radius_mode="vdw", vdw_scale=1.0)
    assert vdw[0] == pytest.approx(vdw_for_element("O"))
    assert vdw[1] == pytest.approx(vdw_for_element("C"))
    assert vdw[2] == pytest.approx(1.5)
    scaled = resolve_atom_radii(sources, 3, atom_radius=1.5, radius_mode="vdw", vdw_scale=0.5)
    assert scaled[0] == pytest.approx(0.5 * vdw_for_element("O"))
    custom = resolve_atom_radii(
        sources, 3, atom_radius=1.5, radius_mode="vdw", vdw_scale=1.0,
        point_radii=[None, 2.2, None],
    )
    assert custom[0] == pytest.approx(vdw_for_element("O"))
    assert custom[1] == pytest.approx(2.2)
    assert custom[2] == pytest.approx(1.5)


def test_asa_per_point_radii_use_expanded_spheres():
    mesh = Surface(
        [(0.0, 0.0, 0.0), (8.0, 0.0, 0.0)],
        atom_radius=1.5,
        probe_radius=1.4,
        point_radii=[1.0, 2.0],
        algorithm="ASA",
        quality=1,
        bypass_colormap=True,
    )
    r0 = 1.0 + 1.4
    r1 = 2.0 + 1.4
    dist0 = np.linalg.norm(mesh.vertices - np.array([0.0, 0.0, 0.0]), axis=1)
    dist1 = np.linalg.norm(mesh.vertices - np.array([8.0, 0.0, 0.0]), axis=1)
    assert np.any(np.abs(dist0 - r0) < 1e-3)
    assert np.any(np.abs(dist1 - r1) < 1e-3)
    assert np.all((np.abs(dist0 - r0) < 1e-3) | (np.abs(dist1 - r1) < 1e-3))


def test_vdw_mode_single_atom_uses_element_radius():
    src = AtomPoint("prot", 1, name="O", elem="O", last_xyz=(0.0, 0.0, 0.0))
    mesh = Surface(
        [src], atom_radius=1.5, probe_radius=1.4,
        radius_mode="vdw", vdw_scale=1.0,
        algorithm="ASA", quality=1, bypass_colormap=True,
    )
    radii = np.linalg.norm(mesh.vertices - np.zeros(3), axis=1)
    assert radii == pytest.approx(vdw_for_element("O") + 1.4, abs=1e-6)
