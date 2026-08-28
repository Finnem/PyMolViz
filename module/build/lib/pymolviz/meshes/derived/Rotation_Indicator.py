from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..Mesh import Mesh
from ..Points import pmv_default_color_counter, pmv_default_color_palette
from ...util.colors import get_distinct_colors


def _quad_strip_to_triangles(n_u: int, n_v: int) -> np.ndarray:
    """Triangle indices for a grid with n_u+1 u-rings and n_v vertices per ring (v wraps)."""
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


def _partial_torus_geometry(
    center_position: np.ndarray,
    rotation_axis: np.ndarray,
    rotation_start: np.ndarray,
    angle: float,
    linewidth: float,
    n_u: int,
    n_v: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tube centerline on a circle of radius R, tube radius r, swept from u=0 to u=angle."""
    C = np.asarray(center_position, dtype=float).reshape(3)
    w = np.asarray(rotation_axis, dtype=float).reshape(3)
    w = w / (np.linalg.norm(w) + 1e-15)
    rs = np.asarray(rotation_start, dtype=float).reshape(3)
    rad0 = np.linalg.norm(rs)
    if rad0 < 1e-12:
        raise ValueError("outer_start must not coincide with center_position.")
    u0 = rs / rad0
    u0 = u0 - float(np.dot(u0, w)) * w
    nu0 = np.linalg.norm(u0)
    if nu0 < 1e-12:
        raise ValueError("rotation_axis must not be parallel to (outer_start - center_position).")
    u0 = u0 / nu0
    v0 = np.cross(w, u0)

    R_major = rad0 * (1.0 + float(linewidth))
    r_minor = rad0 * float(linewidth)

    u_arr = np.linspace(0.0, float(angle), int(n_u) + 1)
    v_arr = np.linspace(0.0, 2.0 * np.pi, int(n_v) + 1)[:-1]

    cosu = np.cos(u_arr)[:, None]
    sinu = np.sin(u_arr)[:, None]
    cosv = np.cos(v_arr)[None, :]
    sinv = np.sin(v_arr)[None, :]

    n_tor = cosu * u0 + sinu * v0
    pos = (
        C
        + (R_major + r_minor * cosv)[..., None] * n_tor[:, None, :]
        + (r_minor * sinv)[..., None] * w
    )

    du = np.gradient(pos, axis=0, edge_order=1)
    dv = np.gradient(pos, axis=1, edge_order=1)
    nml = np.cross(du, dv, axis=-1)
    ln = np.linalg.norm(nml, axis=-1, keepdims=True)
    ln = np.maximum(ln, 1e-15)
    nml = nml / ln

    nv = pos.shape[1]
    pos_flat = pos.reshape(-1, 3)
    nml_flat = nml.reshape(-1, 3)
    faces = _quad_strip_to_triangles(int(n_u), nv)
    return pos_flat, faces, nml_flat


def _cone_vertex_normals_from_winding(
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


def _append_cone_arrow(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    center_position: np.ndarray,
    rotation_axis: np.ndarray,
    u0: np.ndarray,
    R_major: float,
    angle: float,
    r_minor: float,
    n_v: int,
    arrow_base_scale: float,
    arrow_height_scale: float,
    arrow_sides: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right circular cone along arc tangent; base radius = arrow_base_scale * r_minor."""
    C = np.asarray(center_position, dtype=float).reshape(3)
    w = np.asarray(rotation_axis, dtype=float).reshape(3)
    w = w / (np.linalg.norm(w) + 1e-15)
    v0 = np.cross(w, u0)
    ca, sa = np.cos(angle), np.sin(angle)
    n_end = ca * u0 + sa * v0
    L_end = C + R_major * n_end
    t = R_major * (-sa * u0 + ca * v0)
    t_hat = t / (np.linalg.norm(t) + 1e-15)

    e1 = n_end / (np.linalg.norm(n_end) + 1e-15)
    e2 = w

    r_base = float(arrow_base_scale) * r_minor
    h = float(arrow_height_scale) * r_minor
    n_sides = int(arrow_sides) if arrow_sides is not None else max(16, int(n_v))

    apex = L_end + h * t_hat
    phi = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)
    base_ring = L_end + r_base * (np.outer(np.cos(phi), e1) + np.outer(np.sin(phi), e2))

    n0 = vertices.shape[0]
    apex_i = n0
    base_start = n0 + 1
    cone_verts = np.vstack([vertices, apex.reshape(1, 3), base_ring])
    cone_normals_block = _cone_vertex_normals_from_winding(apex, base_ring)
    new_normals = np.vstack([normals, cone_normals_block])

    cone_faces = []
    for j in range(n_sides):
        jn = (j + 1) % n_sides
        cone_faces.append([apex_i, base_start + j, base_start + jn])
    new_faces = np.vstack([faces, np.array(cone_faces, dtype=int)])
    return cone_verts, new_faces, new_normals


class Rotation_Indicator(Mesh):
    """Rotation indicator as a partial torus (cut donut) built from triangle CGO.

    The surface is a swept tube around a circular arc, emitted as ``TRIANGLES`` so it
    ray-traces like ordinary mesh geometry (unlike chained ``CONE``/cylinder CGO).

    With ``show_arrow=True``, a separate **meshed right circular cone** is appended along
    the arc tangent: base lies in the torus end-plane (centered like the tube cut), base
    radius ``arrow_base_scale * r_minor`` (default ``1.2``), height ``arrow_height_scale * r_minor``.
    """

    def __init__(
        self,
        center_position,
        outer_start,
        rotation_axis,
        angle,
        color=None,
        name=None,
        state=1,
        transparency=0,
        colormap="RdYlBu_r",
        linewidth=0.05,
        resolution=64,
        tubular_resolution=24,
        show_arrow=True,
        arrow_base_scale=1.2,
        arrow_height_scale=2.5,
        arrow_sides=None,
        *args,
        **kwargs,
    ) -> None:
        global pmv_default_color_counter
        global pmv_default_color_palette

        self.center_position = np.asarray(center_position, dtype=float).reshape(3)
        self.outer_start = np.asarray(outer_start, dtype=float).reshape(3)
        self.rotation_axis = np.asarray(rotation_axis, dtype=float).reshape(3)
        self.angle = float(angle)
        self.linewidth = float(linewidth)
        self.arrow_base_scale = float(arrow_base_scale)
        self.arrow_height_scale = float(arrow_height_scale)
        self.arrow_sides = arrow_sides

        rotation_start = self.outer_start - self.center_position
        w = self.rotation_axis / (np.linalg.norm(self.rotation_axis) + 1e-15)
        rad0 = np.linalg.norm(rotation_start)
        if rad0 < 1e-12:
            raise ValueError("outer_start must not coincide with center_position.")
        u0 = rotation_start / rad0
        u0 = u0 - float(np.dot(u0, w)) * w
        u0 = u0 / (np.linalg.norm(u0) + 1e-15)
        R_major = rad0 * (1.0 + self.linewidth)
        r_minor = rad0 * self.linewidth

        n_u = max(2, int(resolution))
        n_v = max(6, int(tubular_resolution))

        verts, faces, normals = _partial_torus_geometry(
            self.center_position,
            w,
            rotation_start,
            self.angle,
            self.linewidth,
            n_u,
            n_v,
        )

        if show_arrow:
            verts, faces, normals = _append_cone_arrow(
                verts,
                faces,
                normals,
                self.center_position,
                w,
                u0,
                R_major,
                self.angle,
                r_minor,
                n_v,
                arrow_base_scale,
                arrow_height_scale,
                arrow_sides,
            )

        n_verts = verts.shape[0]

        if color is None:
            rgb = pmv_default_color_palette[pmv_default_color_counter]
            pmv_default_color_counter += 1
            if pmv_default_color_counter >= len(pmv_default_color_palette):
                pmv_default_color_palette = get_distinct_colors(pmv_default_color_counter * 2)
            vertex_rgb = np.broadcast_to(np.asarray(rgb, dtype=float).reshape(1, 3), (n_verts, 3)).copy()
            super().__init__(
                verts,
                color=vertex_rgb,
                normals=normals,
                faces=faces,
                name=name,
                state=state,
                transparency=transparency,
                colormap=colormap,
                bypass_colormap=True,
                *args,
                **kwargs,
            )
        else:
            ca = np.asarray(color, dtype=float)
            if ca.size == 3 and ca.ndim == 1:
                vertex_rgb = np.broadcast_to(ca.reshape(1, 3), (n_verts, 3)).copy()
                super().__init__(
                    verts,
                    color=vertex_rgb,
                    normals=normals,
                    faces=faces,
                    name=name,
                    state=state,
                    transparency=transparency,
                    colormap=colormap,
                    bypass_colormap=True,
                    *args,
                    **kwargs,
                )
            else:
                super().__init__(
                    verts,
                    color=color,
                    normals=normals,
                    faces=faces,
                    name=name,
                    state=state,
                    transparency=transparency,
                    colormap=colormap,
                    *args,
                    **kwargs,
                )
