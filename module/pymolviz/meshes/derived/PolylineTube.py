from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ..Mesh import Mesh
from ..Points import pmv_default_color_counter, pmv_default_color_palette
from ...util.colors import get_distinct_colors
from ...util.math import get_perp


def quad_strip_to_triangles(n_u: int, n_v: int) -> np.ndarray:
    """Triangle indices for a grid with ``n_u + 1`` u-rings and ``n_v`` vertices per ring (v wraps)."""
    faces = []
    for i in range(n_u):
        for j in range(n_v):
            jn = (j + 1) % n_v
            a = i * n_v + j
            b = (i + 1) * n_v + j
            c = (i + 1) * n_v + jn
            d = i * n_v + jn
            faces.append([a, b, c])
            faces.append([a, c, d])
    return np.array(faces, dtype=int)


def resolve_mesh_color_kwargs(n_verts: int, color) -> Dict[str, Any]:
    """Build ``color`` / ``bypass_colormap`` kwargs for :class:`Mesh` construction."""
    global pmv_default_color_counter
    global pmv_default_color_palette

    if color is None:
        rgb = pmv_default_color_palette[pmv_default_color_counter]
        pmv_default_color_counter += 1
        if pmv_default_color_counter >= len(pmv_default_color_palette):
            pmv_default_color_palette = get_distinct_colors(pmv_default_color_counter * 2)
        vertex_rgb = np.broadcast_to(np.asarray(rgb, dtype=float).reshape(1, 3), (n_verts, 3)).copy()
        return {"color": vertex_rgb, "bypass_colormap": True}

    ca = np.asarray(color, dtype=float)
    if ca.size == 3 and ca.ndim == 1:
        vertex_rgb = np.broadcast_to(ca.reshape(1, 3), (n_verts, 3)).copy()
        return {"color": vertex_rgb, "bypass_colormap": True}

    return {"color": color, "bypass_colormap": False}


def _skew_symmetric(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float).reshape(3)
    return np.array(
        [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]],
        dtype=float,
    )


def _rotation_minimal_unit(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3×3 rotation with minimal angle mapping unit vector ``a`` onto ``b``."""
    a = np.asarray(a, dtype=float).reshape(3)
    b = np.asarray(b, dtype=float).reshape(3)
    axb = np.cross(a, b)
    s = np.linalg.norm(axb)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if s < 1e-12:
        if c > -1e-12:
            return np.eye(3)
        w = np.asarray(get_perp(a), dtype=float).reshape(3)
        w /= np.linalg.norm(w) + 1e-15
        K = _skew_symmetric(w)
        return np.eye(3) + 2.0 * (K @ K)
    wu = axb / s
    K = _skew_symmetric(wu)
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def polyline_tangents(path: np.ndarray) -> np.ndarray:
    """Unit tangents at each vertex: miter-style interior, segment ends at endpoints."""
    P = np.asarray(path, dtype=float).reshape(-1, 3)
    n = P.shape[0]
    if n < 2:
        raise ValueError("path must contain at least two distinct points.")
    T = np.zeros_like(P)
    T[0] = P[1] - P[0]
    T[-1] = P[-1] - P[-2]
    for i in range(1, n - 1):
        a = P[i] - P[i - 1]
        b = P[i + 1] - P[i]
        la = np.linalg.norm(a)
        lb = np.linalg.norm(b)
        if la < 1e-14:
            T[i] = b
        elif lb < 1e-14:
            T[i] = a
        else:
            T[i] = a / la + b / lb
    norms = np.linalg.norm(T, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return T / norms


def parallel_transport_normal_frames(tangents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reference normals ``n`` (perp to ``tangents``) by parallel transport; binormal ``b = t × n``."""
    T = np.asarray(tangents, dtype=float).reshape(-1, 3)
    n_pts = T.shape[0]
    n = np.zeros_like(T)
    n0 = np.asarray(get_perp(T[0]), dtype=float).reshape(3)
    n0 /= np.linalg.norm(n0) + 1e-15
    n[0] = n0
    for i in range(n_pts - 1):
        R = _rotation_minimal_unit(T[i], T[i + 1])
        ni = R @ n[i]
        ti = T[i + 1]
        ni = ni - float(np.dot(ni, ti)) * ti
        ln = np.linalg.norm(ni)
        if ln < 1e-12:
            ni = np.asarray(get_perp(ti), dtype=float).reshape(3)
            ln = np.linalg.norm(ni)
        n[i + 1] = ni / (ln + 1e-15)
    b = np.cross(T, n, axis=1)
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-15)
    return n, b


def path_from_line_vertices(lines: np.ndarray) -> np.ndarray:
    """Centerline polyline from segment pairs ``[[start, end], ...]`` (shape ``N×2×3`` or ``N×6``)."""
    L = np.asarray(lines, dtype=float).reshape(-1, 2, 3)
    if L.shape[0] < 1:
        raise ValueError("lines must contain at least one segment.")
    return np.vstack([L[0, 0], L[:, 1]])


def arc_centerline(
    center_position: np.ndarray,
    rotation_axis: np.ndarray,
    rotation_start: np.ndarray,
    angle: float,
    linewidth: float,
    n_samples: int,
) -> np.ndarray:
    """Sampled centerline on a circular arc (major radius includes ``linewidth`` offset)."""
    C = np.asarray(center_position, dtype=float).reshape(3)
    w = np.asarray(rotation_axis, dtype=float).reshape(3)
    w = w / (np.linalg.norm(w) + 1e-15)
    rs = np.asarray(rotation_start, dtype=float).reshape(3)
    rad0 = np.linalg.norm(rs)
    if rad0 < 1e-12:
        raise ValueError("outer_start must not coincide with center_position.")
    u0 = rs / rad0
    u0 = u0 - float(np.dot(u0, w)) * w
    u0 = u0 / (np.linalg.norm(u0) + 1e-15)
    v0 = np.cross(w, u0)
    R_major = rad0 * (1.0 + float(linewidth))
    u_arr = np.linspace(0.0, float(angle), max(2, int(n_samples)) + 1)
    return C + R_major * (np.cos(u_arr)[:, None] * u0 + np.sin(u_arr)[:, None] * v0)


def tube_radius_per_vertex(
    tube_radius: Union[float, np.ndarray],
    n_pts: int,
) -> np.ndarray:
    tr = np.asarray(tube_radius, dtype=float).reshape(-1)
    if tr.size == 1:
        return np.full(n_pts, float(tr[0]))
    if tr.size == n_pts:
        return tr
    raise ValueError("tube_radius must be a scalar or one value per path vertex.")


def cone_vertex_normals_from_winding(
    apex: np.ndarray,
    base_ring: np.ndarray,
) -> np.ndarray:
    """Per-vertex normals for cone side mesh (apex, then base ring), matching triangle winding."""
    apex = np.asarray(apex, dtype=float).reshape(3)
    br = np.asarray(base_ring, dtype=float).reshape(-1, 3)
    ns = br.shape[0]
    n_face = np.zeros((ns, 3))
    for j in range(ns):
        jn = (j + 1) % ns
        e1 = br[j] - apex
        e2 = br[jn] - apex
        c = np.cross(e1, e2)
        ln = np.linalg.norm(c)
        if ln > 1e-15:
            n_face[j] = c / ln
    apex_n = np.sum(n_face, axis=0)
    apex_n /= np.linalg.norm(apex_n) + 1e-15
    base_n = np.zeros((ns, 3))
    for j in range(ns):
        v = n_face[(j - 1) % ns] + n_face[j]
        base_n[j] = v / (np.linalg.norm(v) + 1e-15)
    return np.vstack([apex_n.reshape(1, 3), base_n])


def append_cone_arrow_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    base_center: np.ndarray,
    tangent: np.ndarray,
    plane_normal: np.ndarray,
    plane_binormal: np.ndarray,
    r_base: float,
    height: float,
    arrow_sides: Optional[int] = None,
    default_sides: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Append a meshed cone arrow; ``tangent`` points from base toward apex."""
    base_center = np.asarray(base_center, dtype=float).reshape(3)
    t_hat = np.asarray(tangent, dtype=float).reshape(3)
    t_hat /= np.linalg.norm(t_hat) + 1e-15
    e1 = np.asarray(plane_normal, dtype=float).reshape(3)
    e1 /= np.linalg.norm(e1) + 1e-15
    e2 = np.asarray(plane_binormal, dtype=float).reshape(3)
    e2 /= np.linalg.norm(e2) + 1e-15

    n_sides = int(arrow_sides) if arrow_sides is not None else int(default_sides)
    n_sides = max(6, n_sides)

    apex = base_center + float(height) * t_hat
    phi = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)
    base_ring = base_center + float(r_base) * (np.outer(np.cos(phi), e1) + np.outer(np.sin(phi), e2))

    n0 = vertices.shape[0]
    apex_i = n0
    base_start = n0 + 1
    cone_verts = np.vstack([vertices, apex.reshape(1, 3), base_ring])
    cone_normals_block = cone_vertex_normals_from_winding(apex, base_ring)
    new_normals = np.vstack([normals, cone_normals_block])

    cone_faces = []
    for j in range(n_sides):
        jn = (j + 1) % n_sides
        cone_faces.append([apex_i, base_start + j, base_start + jn])
    new_faces = np.vstack([faces, np.array(cone_faces, dtype=int)])
    return cone_verts, new_faces, new_normals


def tube_mesh_from_polyline(
    path: np.ndarray,
    tube_radius: Union[float, np.ndarray],
    tubular_resolution: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tube surface vertices, triangle faces, and vertex normals for a 3D polyline centerline."""
    P = np.asarray(path, dtype=float).reshape(-1, 3)
    n_pts = P.shape[0]
    if n_pts < 2:
        raise ValueError("path must have at least two points.")

    r = tube_radius_per_vertex(tube_radius, n_pts)
    n_v = max(6, int(tubular_resolution))
    n_u = n_pts - 1

    T = polyline_tangents(P)
    n_ref, b_ref = parallel_transport_normal_frames(T)

    phi = np.linspace(0.0, 2.0 * np.pi, n_v + 1, dtype=float)[:-1]
    cosv = np.cos(phi)[None, :, None]
    sinv = np.sin(phi)[None, :, None]

    pos = (
        P[:, None, :]
        + r[:, None, None] * (cosv * n_ref[:, None, :] + sinv * b_ref[:, None, :])
    )

    du = np.gradient(pos, axis=0, edge_order=1)
    dv = np.gradient(pos, axis=1, edge_order=1)
    nml = np.cross(du, dv, axis=-1)
    nml /= np.maximum(np.linalg.norm(nml, axis=-1, keepdims=True), 1e-15)

    verts = pos.reshape(-1, 3)
    normals = nml.reshape(-1, 3)
    faces = quad_strip_to_triangles(n_u, n_v)
    return verts, faces, normals


class PolylineTube(Mesh):
    """Triangle mesh of a tube swept along a 3D polyline (segmented, non-straight paths).

    ``path_vertices`` is an ``N×3`` array of centerline points (one per joint / sample).
    A reference frame at each point uses **parallel transport** of a normal perpendicular
    to the local tangent (miter-style tangents at interior joints).

    Build from segment pairs via :meth:`from_line_vertices`.
    """

    def __init__(
        self,
        path_vertices,
        tube_radius: Union[float, np.ndarray] = 0.05,
        color=None,
        name=None,
        state=1,
        transparency=0,
        colormap="RdYlBu_r",
        tubular_resolution=24,
        show_arrow=False,
        arrow_base_scale=1.2,
        arrow_height_scale=2.5,
        arrow_sides=None,
        *args,
        **kwargs,
    ) -> None:
        self.path_vertices = np.asarray(path_vertices, dtype=float).reshape(-1, 3)
        self.tube_radius = tube_radius
        self.tubular_resolution = int(tubular_resolution)
        self.show_arrow = bool(show_arrow)
        self.arrow_base_scale = float(arrow_base_scale)
        self.arrow_height_scale = float(arrow_height_scale)
        self.arrow_sides = arrow_sides

        verts, faces, normals = tube_mesh_from_polyline(
            self.path_vertices,
            self.tube_radius,
            self.tubular_resolution,
        )

        if self.show_arrow:
            r = tube_radius_per_vertex(self.tube_radius, self.path_vertices.shape[0])
            T = polyline_tangents(self.path_vertices)
            n_ref, b_ref = parallel_transport_normal_frames(T)
            verts, faces, normals = append_cone_arrow_mesh(
                verts,
                faces,
                normals,
                base_center=self.path_vertices[-1],
                tangent=T[-1],
                plane_normal=n_ref[-1],
                plane_binormal=b_ref[-1],
                r_base=self.arrow_base_scale * r[-1],
                height=self.arrow_height_scale * r[-1],
                arrow_sides=self.arrow_sides,
                default_sides=max(16, self.tubular_resolution),
            )

        color_kw = resolve_mesh_color_kwargs(verts.shape[0], color)
        super().__init__(
            verts,
            normals=normals,
            faces=faces,
            name=name,
            state=state,
            transparency=transparency,
            colormap=colormap,
            *args,
            **color_kw,
            **kwargs,
        )

    @classmethod
    def from_line_vertices(
        cls,
        lines,
        tube_radius: Union[float, np.ndarray] = 0.05,
        **kwargs,
    ) -> "PolylineTube":
        """Create a tube along ``[[start, end], ...]`` segments (``N×2×3`` or ``N×6``)."""
        path = path_from_line_vertices(lines)
        return cls(path, tube_radius=tube_radius, **kwargs)
