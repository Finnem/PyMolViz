"""Unit tests for mesh rebuild and Displayable ids."""

from __future__ import annotations

import numpy as np

import pytest

from pymolviz.Displayable import Displayable
from pymolviz.meshes.Sphere import Sphere
from pymolviz.meshes.Cylinder import Cylinder
from pymolviz.points import FixedPoint, AtomPoint


class _DummyDisplayable(Displayable):
    def _script_string(self):
        return ""


def test_displayable_id_stable_and_not_name():
    obj = _DummyDisplayable(name="my_name", obj_id="deadbeef")
    assert obj.id == "deadbeef"
    assert obj.name == "my_name"
    obj.name = "other_name"
    assert obj.id == "deadbeef"


def test_sphere_rebuild_changes_vertices():
    sphere = Sphere((0.0, 0.0, 0.0), 1.0, bypass_colormap=True)
    before = np.array(sphere.vertices, copy=True)
    sphere.position = FixedPoint((5.0, 0.0, 0.0))
    sphere.rebuild(None)
    after = np.array(sphere.vertices)
    assert not np.allclose(before, after)
    center = after.mean(axis=0)
    assert abs(center[0] - 5.0) < 0.5


def test_sphere_cgo_token_shape():
    sphere = Sphere((0.0, 0.0, 0.0), 1.0, bypass_colormap=True)
    tokens = sphere._create_CGO_list()
    kinds = [t for t in tokens if isinstance(t, str)]
    assert kinds[:2] == ["ENABLE", "LIGHTING"]
    assert "BEGIN" in kinds
    assert "TRIANGLES" in kinds
    assert "END" in kinds


def test_mesh_cgo_smooth_lighting_layout():
    sphere = Sphere(
        (0.0, 0.0, 0.0), 1.0, color=(0.2, 0.6, 0.9),
        bypass_colormap=True, frequency=2,
    )
    tokens = sphere._create_CGO_list()
    assert tokens[:4] == ["ENABLE", "LIGHTING", "BEGIN", "TRIANGLES"]
    assert tokens[4] == "COLOR"
    assert tokens.count("COLOR") == 1
    assert tokens.count("NORMAL") == tokens.count("VERTEX")
    assert tokens.count("VERTEX") == 3 * len(sphere.faces)
    for i, kind in enumerate(tokens):
        if kind == "VERTEX":
            assert tokens[i - 4] == "NORMAL"
    n0 = np.array([float(tokens[9]), float(tokens[10]), float(tokens[11])])
    n1 = np.array([float(tokens[17]), float(tokens[18]), float(tokens[19])])
    n2 = np.array([float(tokens[25]), float(tokens[26]), float(tokens[27])])
    assert float(np.dot(n0, n1)) < 0.999
    assert float(np.dot(n1, n2)) < 0.999


def test_matte_triangle_cgo_bakes_lambert_and_disables_lighting():
    sphere = Sphere(
        (0.0, 0.0, 0.0), 1.0, color=(1.0, 0.0, 0.0),
        bypass_colormap=True, frequency=2,
    )
    sphere.specular = False
    tokens = sphere._create_CGO_list()
    assert tokens[:4] == ["DISABLE", "LIGHTING", "BEGIN", "TRIANGLES"]
    assert tokens.count("COLOR") == tokens.count("VERTEX")
    assert tokens.count("NORMAL") == tokens.count("VERTEX")
    reds = []
    for i, kind in enumerate(tokens):
        if kind == "COLOR":
            reds.append(float(tokens[i + 1]))
    assert len(reds) > 1
    assert max(reds) - min(reds) > 0.05


def test_sphere_wireframe_cgo_uses_unique_cones():
    from pymolviz.meshes.Mesh import _unique_undirected_edges

    sphere = Sphere(
        (0.0, 0.0, 0.0), 1.0, color=(1.0, 0.0, 0.0),
        bypass_colormap=True, wireframe=True, frequency=2,
    )
    tokens = sphere._create_CGO_list()
    kinds = [t for t in tokens if isinstance(t, str)]
    assert "CONE" in kinds
    assert "TRIANGLES" not in kinds
    n_unique = len(_unique_undirected_edges(sphere.faces))
    assert kinds.count("CONE") == n_unique
    assert n_unique < 3 * len(sphere.faces)


def test_cylinder_construct_and_rebuild():
    cyl = Cylinder((0, 0, 0), (0, 0, 5), 0.2, bypass_colormap=True)
    assert len(cyl.vertices) > 0
    cyl.start = FixedPoint((1.0, 0.0, 0.0))
    cyl.end = FixedPoint((1.0, 0.0, 6.0))
    cyl.rebuild(None)
    zvals = cyl.vertices[:, 2]
    assert zvals.min() < 1.0
    assert zvals.max() > 5.0


def test_sphere_with_atom_point_last_xyz_bakes():
    src = AtomPoint("prot", 1, last_xyz=(2.0, 3.0, 4.0))
    sphere = Sphere(src, 0.5, bypass_colormap=True)
    sphere.rebuild(None)
    center = np.asarray(sphere.vertices).mean(axis=0)
    assert abs(center[0] - 2.0) < 0.2
    assert abs(center[1] - 3.0) < 0.2
    assert abs(center[2] - 4.0) < 0.2


def test_sphere_shift_vertices_keeps_mesh_and_patches_cgo():
    sphere = Sphere((0.0, 0.0, 0.0), 1.0, bypass_colormap=True)
    tokens = sphere._create_CGO_list()
    before = np.array(sphere.vertices, copy=True)
    vertex_idx = next(i for i, t in enumerate(tokens) if str(t) == "VERTEX")
    x0 = float(tokens[vertex_idx + 1])
    sphere.shift_vertices((5.0, 0.0, 0.0))
    after = np.array(sphere.vertices)
    assert np.allclose(after, before + np.array([5.0, 0.0, 0.0]))
    assert tokens is sphere._cached_cgo
    assert float(tokens[vertex_idx + 1]) == pytest.approx(x0 + 5.0)
    again = sphere._create_CGO_list()
    assert again is tokens


def test_sphere_shift_vertices_patches_resolved_float_cgo(monkeypatch):
    """replace_cgo loads _cached_resolved; real PyMOL opcodes are floats."""
    import pymol.cgo as cgo_mod

    monkeypatch.setattr(cgo_mod, "BEGIN", 2.0)
    monkeypatch.setattr(cgo_mod, "END", 3.0)
    monkeypatch.setattr(cgo_mod, "VERTEX", 4.0)
    monkeypatch.setattr(cgo_mod, "TRIANGLES", 4.0)
    monkeypatch.setattr(cgo_mod, "NORMAL", 5.0)
    monkeypatch.setattr(cgo_mod, "COLOR", 6.0)
    monkeypatch.setattr(cgo_mod, "SPHERE", 7.0)
    monkeypatch.setattr(cgo_mod, "CYLINDER", 9.0)
    monkeypatch.setattr(cgo_mod, "CONE", 27.0)
    monkeypatch.setattr(cgo_mod, "LINEWIDTH", 10.0)
    monkeypatch.setattr(cgo_mod, "ALPHA", 25.0)
    monkeypatch.setattr(cgo_mod, "POINTS", 0.0)
    monkeypatch.setattr(cgo_mod, "LINES", 1.0)

    from pymolviz.runtime.renderer import resolved_cgo_tokens

    sphere = Sphere((0.0, 0.0, 0.0), 1.0, bypass_colormap=True)
    resolved = resolved_cgo_tokens(sphere, None)
    begin_at = next(i for i, tok in enumerate(resolved) if tok == 2.0)
    assert resolved[begin_at + 1] == 4.0
    vertex_xs = []
    i = 0
    n = len(resolved)
    after_begin = False
    while i < n:
        tok = resolved[i]
        if after_begin:
            after_begin = False
            i += 1
            continue
        if tok == 2.0:
            after_begin = True
            i += 1
            continue
        if tok == 4.0 and i + 3 < n:
            vertex_xs.append(float(resolved[i + 1]))
            i += 4
            continue
        if tok == 6.0:
            i += 4
            continue
        if tok == 5.0:
            i += 4
            continue
        i += 1
    assert vertex_xs
    x0_mean = sum(vertex_xs) / len(vertex_xs)
    sphere.shift_vertices((5.0, 0.0, 0.0))
    assert sphere._cached_resolved is resolved
    vertex_xs_after = []
    i = 0
    after_begin = False
    while i < n:
        tok = resolved[i]
        if after_begin:
            after_begin = False
            i += 1
            continue
        if tok == 2.0:
            after_begin = True
            i += 1
            continue
        if tok == 4.0 and i + 3 < n:
            vertex_xs_after.append(float(resolved[i + 1]))
            i += 4
            continue
        if tok == 6.0:
            i += 4
            continue
        if tok == 5.0:
            i += 4
            continue
        i += 1
    x1_mean = sum(vertex_xs_after) / len(vertex_xs_after)
    assert x1_mean == pytest.approx(x0_mean + 5.0)


def test_clone_baked_shares_faces_and_isolates_vertices():
    sphere = Sphere((0.0, 0.0, 0.0), 1.0, bypass_colormap=True, frequency=4)
    tokens = sphere._create_CGO_list()
    cloned = sphere.clone_baked()
    assert cloned is not sphere
    assert cloned.id != sphere.id
    assert cloned.faces is sphere.faces
    assert cloned.vertices is not sphere.vertices
    assert np.allclose(cloned.vertices, sphere.vertices)
    assert cloned._cached_cgo is not tokens
    cloned.shift_vertices((3.0, 0.0, 0.0))
    assert np.allclose(sphere.vertices.mean(axis=0), [0.0, 0.0, 0.0], atol=0.5)
    assert cloned.vertices.mean(axis=0)[0] == pytest.approx(
        sphere.vertices.mean(axis=0)[0] + 3.0, abs=0.5,
    )


def _vertex_xs(tokens):
    from pymol import cgo

    xs = []
    i = 0
    n = len(tokens)
    after_begin = False
    while i < n:
        if after_begin:
            after_begin = False
            i += 1
            continue
        tok = tokens[i]
        if tok == cgo.BEGIN:
            after_begin = True
            i += 1
            continue
        if tok == cgo.VERTEX and i + 3 < n:
            xs.append(float(tokens[i + 1]))
            i += 4
            continue
        i += 1
    return xs


def test_collection_merged_tokens_track_child_shift():
    from pymolviz.meshes.CGOCollection import CGOCollection
    from pymolviz.runtime.renderer import resolved_cgo_tokens

    s1 = Sphere((0.0, 0.0, 0.0), 1.0, bypass_colormap=True, frequency=2)
    s2 = Sphere((8.0, 0.0, 0.0), 1.0, bypass_colormap=True, frequency=2)
    coll = CGOCollection([s1, s2])
    before = resolved_cgo_tokens(coll, None)
    xs0 = _vertex_xs(before)
    assert xs0
    cached = coll._cached_merged_resolved
    s1.shift_vertices((4.0, 0.0, 0.0))
    after = resolved_cgo_tokens(coll, None)
    assert after is not cached
    xs1 = _vertex_xs(after)
    assert sum(xs1) / len(xs1) == pytest.approx(sum(xs0) / len(xs0) + 2.0, abs=0.05)


def test_cylinder_and_cone_caps_add_disks():
    from pymolviz.util.cgo import mesh_cone_cgo, mesh_cylinder_cgo

    open_cyl = mesh_cylinder_cgo((0, 0, 0), (1, 0, 0), 0.1, (1, 0, 0), n_seg=6, caps=False)
    capped_cyl = mesh_cylinder_cgo((0, 0, 0), (1, 0, 0), 0.1, (1, 0, 0), n_seg=6, caps=True)
    assert capped_cyl.count("VERTEX") > open_cyl.count("VERTEX")
    open_cone = mesh_cone_cgo((0, 0, 0), (1, 0, 0), 0.2, (1, 0, 0), n_seg=6, cap_base=False)
    capped_cone = mesh_cone_cgo((0, 0, 0), (1, 0, 0), 0.2, (1, 0, 0), n_seg=6, cap_base=True)
    assert capped_cone.count("VERTEX") > open_cone.count("VERTEX")


def test_cone_slant_normals_and_axial_rings():
    from pymolviz.util.cgo import mesh_cone_cgo

    tokens = mesh_cone_cgo(
        (0, 0, 0), (2, 0, 0), 1.0, (1, 0, 0), n_seg=8, n_rings=1, cap_base=False,
    )
    normals = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "NORMAL":
            normals.append((float(tokens[i + 1]), float(tokens[i + 2]), float(tokens[i + 3])))
            i += 4
            continue
        i += 1
    assert normals
    # Radial (cylinder-style) normals would have nx ~= 0 on an x-aligned cone.
    assert max(n[0] for n in normals) > 0.2
    fan = mesh_cone_cgo(
        (0, 0, 0), (2, 0, 0), 1.0, (1, 0, 0), n_seg=8, n_rings=1, cap_base=False,
    )
    stacked = mesh_cone_cgo(
        (0, 0, 0), (2, 0, 0), 1.0, (1, 0, 0), n_seg=8, n_rings=3, cap_base=False,
    )
    assert stacked.count("VERTEX") > fan.count("VERTEX")


def test_dashed_arrow_keeps_head():
    from pymolviz.meshes.Arrows import build_styled_arrow_cgo
    from pymolviz.util.line_style import LineStyle

    color = (1.0, 0.0, 0.0)
    start, end = (0.0, 0.0, 0.0), (10.0, 0.0, 0.0)
    dashed_none = build_styled_arrow_cgo(
        start, end, color, 3, LineStyle(dash="Dashed", ends="None"),
    )
    dashed_arrow = build_styled_arrow_cgo(
        start, end, color, 3, LineStyle(dash="Dashed", ends="Arrow"),
    )
    assert dashed_none.count("BEGIN") >= 2
    assert dashed_arrow.count("VERTEX") > dashed_none.count("VERTEX")
    squeezed_arrow = build_styled_arrow_cgo(
        start, (2.0, 0.0, 0.0), color, 3,
        LineStyle(dash="Dashed", margin=8.0, ends="Arrow"),
        radius=0.045, head_length=0.36,
    )
    squeezed_none = build_styled_arrow_cgo(
        start, (2.0, 0.0, 0.0), color, 3,
        LineStyle(dash="Dashed", margin=8.0, ends="None"),
        radius=0.045, head_length=0.36,
    )
    assert squeezed_arrow
    assert squeezed_arrow.count("VERTEX") > squeezed_none.count("VERTEX")


def test_thin_shaft_wide_head_does_not_explode_cone():
    from pymolviz.meshes.Arrows import build_styled_arrow_cgo
    from pymolviz.util.line_style import LineStyle

    color = (1.0, 0.0, 0.0)
    start, end = (0.0, 0.0, 0.0), (10.0, 0.0, 0.0)
    style = LineStyle(ends="Arrow")
    normal = build_styled_arrow_cgo(
        start, end, color, 3, style, radius=0.045, head_radius=0.108,
    )
    skinny = build_styled_arrow_cgo(
        start, end, color, 3, style, radius=0.001, head_radius=0.108,
    )
    assert skinny.count("VERTEX") < normal.count("VERTEX") * 2


def _vertex_radial_max(tokens):
    verts = []
    i = 0
    n = len(tokens)
    while i < n:
        if tokens[i] == "VERTEX" and i + 3 < n:
            verts.append((float(tokens[i + 1]), float(tokens[i + 2]), float(tokens[i + 3])))
            i += 4
            continue
        i += 1
    assert verts
    return max((v[1] ** 2 + v[2] ** 2) ** 0.5 for v in verts)


def test_head_radius_follows_shaft_helper():
    from pymolviz.meshes.Arrows import default_head_radius, head_radius_follows_shaft

    assert default_head_radius(0.045) == pytest.approx(0.108)
    assert head_radius_follows_shaft(0.045, None)
    assert head_radius_follows_shaft(0.045, 0.108)
    assert not head_radius_follows_shaft(0.045, 0.2)


def test_styled_arrow_cone_scales_with_shaft_when_head_radius_unset():
    from pymolviz.meshes.Arrows import HEAD_WIDTH, build_styled_arrow_cgo
    from pymolviz.util.line_style import LineStyle

    color = (1.0, 0.0, 0.0)
    start, end = (0.0, 0.0, 0.0), (10.0, 0.0, 0.0)
    style = LineStyle(ends="Arrow")
    thin = build_styled_arrow_cgo(start, end, color, 3, style, radius=0.045)
    fat = build_styled_arrow_cgo(start, end, color, 3, style, radius=0.2)
    r_thin = _vertex_radial_max(thin)
    r_fat = _vertex_radial_max(fat)
    assert r_thin == pytest.approx(0.045 * HEAD_WIDTH, rel=0.15)
    assert r_fat == pytest.approx(0.2 * HEAD_WIDTH, rel=0.15)
    assert r_fat > r_thin * 2


def test_fixed_head_radius_masks_small_shaft_change():
    from pymolviz.meshes.Arrows import build_styled_arrow_cgo
    from pymolviz.util.line_style import LineStyle

    color = (1.0, 0.0, 0.0)
    start, end = (0.0, 0.0, 0.0), (10.0, 0.0, 0.0)
    style = LineStyle(ends="Arrow")
    thin = build_styled_arrow_cgo(
        start, end, color, 3, style, radius=0.045, head_radius=0.108,
    )
    slightly = build_styled_arrow_cgo(
        start, end, color, 3, style, radius=0.08, head_radius=0.108,
    )
    assert _vertex_radial_max(thin) == pytest.approx(0.108, rel=0.15)
    assert _vertex_radial_max(slightly) == pytest.approx(0.108, rel=0.15)


def test_quality0_arrow_linewidth_tracks_radius():
    from pymolviz.meshes.Arrows import build_styled_arrow_cgo
    from pymolviz.util.line_style import LineStyle

    color = (1.0, 0.0, 0.0)
    start, end = (0.0, 0.0, 0.0), (10.0, 0.0, 0.0)
    style = LineStyle(ends="Arrow")
    thin = build_styled_arrow_cgo(start, end, color, 0, style, radius=0.045)
    fat = build_styled_arrow_cgo(start, end, color, 0, style, radius=0.2)

    def linewidths(tokens):
        return [float(tokens[i + 1]) for i, tok in enumerate(tokens) if tok == "LINEWIDTH"]

    assert linewidths(fat)[0] > linewidths(thin)[0]


def test_retarget_arrow_width_rebuilds_cgo():
    from dataclasses import replace

    from pymolviz.points import FixedPoint
    from pymolviz.util.line_style import LineStyle
    from pymolviz.wizards.builders.pairs import VisualPair
    from pymolviz.wizards.builders.points import VisualPoint
    from pymolviz.wizards.builders.preview import (
        build_arrow_collection,
        retarget_arrow_collection,
    )

    def pt(xyz):
        return VisualPoint(
            "p", "manual", xyz[0], xyz[1], xyz[2], point_source=FixedPoint(xyz),
        )

    pair = VisualPair(pt((0.0, 0.0, 0.0)), pt((10.0, 0.0, 0.0)), width=0.045)
    coll = build_arrow_collection([pair], 3, LineStyle(), "a", head_radius=None)
    before = coll[0]._create_CGO_list()
    fat = replace(pair, width=0.2)
    assert retarget_arrow_collection(coll, [fat], head_radius=None) is True
    after = coll[0]._create_CGO_list()
    assert coll[0].pair_radii[0] == pytest.approx(0.2)
    assert len(coll[0].pair_radii) == 1
    assert coll[0].shaft_radius == pytest.approx(0.2)
    assert coll[0].head_radius is None
    assert _vertex_radial_max(after) > _vertex_radial_max(before) * 2


def test_independent_heads_cgo():
    from pymolviz.meshes.Arrows import build_styled_arrow_cgo
    from pymolviz.util.line_style import LineStyle

    color = (1.0, 0.0, 0.0)
    start, end = (0.0, 0.0, 0.0), (10.0, 0.0, 0.0)
    none = build_styled_arrow_cgo(start, end, color, 3, LineStyle(ends="None"))
    end_arrow = build_styled_arrow_cgo(start, end, color, 3, LineStyle(ends="Arrow"))
    both_arrow = build_styled_arrow_cgo(
        start, end, color, 3, LineStyle(start_head="Arrow", end_head="Arrow"),
    )
    start_circle = build_styled_arrow_cgo(
        start, end, color, 3, LineStyle(start_head="Circles", end_head="None"),
    )
    mixed = build_styled_arrow_cgo(
        start, end, color, 3, LineStyle(start_head="Circles", end_head="Arrow"),
    )
    assert end_arrow.count("VERTEX") > none.count("VERTEX")
    assert both_arrow.count("VERTEX") > end_arrow.count("VERTEX")
    assert start_circle.count("SPHERE") == 1
    assert none.count("SPHERE") == 0
    assert mixed.count("SPHERE") == 1
    assert mixed.count("VERTEX") > start_circle.count("VERTEX")


def test_surface_rebuild_and_cgo_tokens():
    from pymolviz.meshes.Surface import Surface

    mesh = Surface(
        [(0.0, 0.0, 0.0)], algorithm="GAUSS", quality=1, bypass_colormap=True,
    )
    tokens = mesh._create_CGO_list()
    kinds = [t for t in tokens if isinstance(t, str)]
    assert "BEGIN" in kinds
    assert "TRIANGLES" in kinds
    assert "END" in kinds
    before = np.array(mesh.vertices, copy=True)
    mesh.point_sources = [FixedPoint((3.0, 0.0, 0.0))]
    mesh.rebuild(None)
    assert np.allclose(mesh.vertices, before + np.array([3.0, 0.0, 0.0]))


def test_clip_gizmo_cgo_has_rectangle_and_eye():
    from pymolviz.meshes.ClipGizmo import ClipGizmo

    gizmo = ClipGizmo((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), scale=5.0, draft=True)
    tokens = gizmo._create_CGO_list()
    kinds = [t for t in tokens if isinstance(t, str)]
    assert kinds.count("BEGIN") >= 1
    assert kinds.count("TRIANGLES") >= 1
    assert kinds.count("CYLINDER") >= 4
    assert kinds.count("SPHERE") >= 3
    assert "ALPHA" in kinds
    assert kinds.count("VERTEX") == 12
    wide = ClipGizmo(
        (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
        points=[[12.0, 1.0, 0.0], [-12.0, -1.0, 0.0]],
        draft=True,
    )
    wide_tokens = wide._create_CGO_list()
    assert [t for t in wide_tokens if isinstance(t, str)].count("VERTEX") == 12


def _box_face_geom_normals(box):
    verts = np.asarray(box.vertices, dtype=float)
    faces = np.asarray(box.faces, dtype=int)
    out = []
    for i0, i1, i2 in faces:
        n = np.cross(verts[i1] - verts[i0], verts[i2] - verts[i0])
        ln = float(np.linalg.norm(n))
        out.append(n / ln if ln > 1e-12 else n)
    return np.asarray(out, dtype=float)


def test_centered_box_face_winding_is_outward():
    from pymolviz.meshes.CenteredBox import CenteredBox

    box = CenteredBox((0.0, 0.0, 0.0), (2.0, 4.0, 6.0), bypass_colormap=True)
    verts = np.asarray(box.vertices, dtype=float)
    for face, normal in zip(box.faces, _box_face_geom_normals(box)):
        centroid = verts[face].mean(axis=0) - np.array((0.0, 0.0, 0.0))
        assert float(np.dot(normal, centroid)) > 0.0
    flipped = CenteredBox((1.0, -2.0, 3.0), (-2.0, 4.0, -6.0), bypass_colormap=True)
    origin = np.array((1.0, -2.0, 3.0))
    verts = np.asarray(flipped.vertices, dtype=float)
    for face, normal in zip(flipped.faces, _box_face_geom_normals(flipped)):
        centroid = verts[face].mean(axis=0) - origin
        assert float(np.dot(normal, centroid)) > 0.0


def test_centered_box_cgo_uses_outward_face_normals():
    from pymolviz.meshes.CenteredBox import CenteredBox

    box = CenteredBox(
        (0.0, 0.0, 0.0), (2.0, 2.0, 2.0),
        color=(1.0, 0.0, 0.0), bypass_colormap=True,
    )
    tokens = box._create_CGO_list()
    assert tokens.count("NORMAL") == tokens.count("VERTEX") == 36
    i = 0
    while i < len(tokens):
        if tokens[i] != "NORMAL":
            i += 1
            continue
        n = np.array([float(tokens[i + 1]), float(tokens[i + 2]), float(tokens[i + 3])])
        assert tokens[i + 4] == "VERTEX"
        v = np.array([float(tokens[i + 5]), float(tokens[i + 6]), float(tokens[i + 7])])
        axis = int(np.argmax(np.abs(n)))
        assert abs(abs(n[axis]) - 1.0) < 1e-6
        assert abs(float(v[axis] - np.sign(n[axis]))) < 1e-6
        i += 8


def test_solid_box_cgo_normals_match_winding():
    from pymolviz.util.cgo import _BOX_FACES, _box_corners

    corners = np.asarray(_box_corners((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)), dtype=float)
    for face, expected in _BOX_FACES:
        v0, v1, v2 = corners[list(face)]
        geom = np.cross(v1 - v0, v2 - v0)
        geom = geom / float(np.linalg.norm(geom))
        assert float(np.dot(geom, expected)) > 0.99


