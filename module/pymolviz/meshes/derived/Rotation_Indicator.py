from __future__ import annotations

from typing import Tuple

import numpy as np

from ..Mesh import Mesh
from .PolylineTube import (
    append_cone_arrow_mesh,
    quad_strip_to_triangles,
    resolve_mesh_color_kwargs,
)


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
    faces = quad_strip_to_triangles(int(n_u), nv)
    return pos_flat, faces, nml_flat


class Rotation_Indicator(Mesh):
    """Rotation indicator as a partial torus (cut donut) built from triangle CGO.

    The surface is a swept tube around a circular arc, emitted as ``TRIANGLES`` so it
    ray-traces like ordinary mesh geometry (unlike chained ``CONE``/cylinder CGO).

    With ``show_arrow=True``, a meshed cone is appended along the arc tangent at the end.
    For arbitrary segmented paths, use :class:`PolylineTube` instead.
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
        v0 = np.cross(w, u0)
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
            ca, sa = np.cos(self.angle), np.sin(self.angle)
            n_end = ca * u0 + sa * v0
            L_end = self.center_position + R_major * n_end
            t_hat = R_major * (-sa * u0 + ca * v0)
            t_hat /= np.linalg.norm(t_hat) + 1e-15
            verts, faces, normals = append_cone_arrow_mesh(
                verts,
                faces,
                normals,
                base_center=L_end,
                tangent=t_hat,
                plane_normal=n_end / (np.linalg.norm(n_end) + 1e-15),
                plane_binormal=w,
                r_base=self.arrow_base_scale * r_minor,
                height=self.arrow_height_scale * r_minor,
                arrow_sides=self.arrow_sides,
                default_sides=max(16, n_v),
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
