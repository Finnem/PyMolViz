"""Marching-cubes conversion of an IsoSurface Field Visual to an explicit Surface."""

from __future__ import annotations

import numpy as np


def isosurface_mesh_from_grid(grid, level=0.0):
    """Triangle mesh of ``grid`` at ``level``. Isotropic step is assumed.

    Returns ``(vertices, normals, faces)``; any may be empty.
    """
    from ..util.field_sample import grid_values_3d
    from ..util.marching_cubes import march_cubes_mesh
    from ..util.solvent_surface import _cubes_with_sign_change, _face_normals, _orient_faces_to_vertex_normals

    empty = (
        np.zeros((0, 3), dtype=float),
        np.zeros((0, 3), dtype=float),
        np.zeros((0, 3), dtype=int),
    )
    if grid is None:
        return empty
    values = grid_values_3d(grid) - float(level)
    origin = np.asarray(grid.origin, dtype=float).reshape(3)
    step = np.asarray(grid.step_sizes, dtype=float).reshape(3)
    h = float(np.mean(np.abs(step)))
    if h <= 1e-18:
        return empty
    cubes = _cubes_with_sign_change(values)
    if cubes.shape[0] == 0:
        return empty
    vertices, faces = march_cubes_mesh(origin, h, values, cubes)
    if faces.shape[0] == 0:
        return empty
    gx, gy, gz = np.gradient(values, h, h, h)
    grad = np.stack((gx, gy, gz), axis=-1)
    from ..util.solvent_surface import _trilinear_sample

    normals = _trilinear_sample(grad, origin, h, vertices)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    tiny = lengths[:, 0] < 1e-12
    normals = normals / np.maximum(lengths, 1e-12)
    if np.any(tiny):
        geom = _face_normals(vertices, faces)
        normals = np.array(normals, copy=True)
        normals[tiny] = geom[tiny]
    faces = _orient_faces_to_vertex_normals(vertices, faces, normals)
    return vertices, normals, faces


def convert_isosurface_to_surface(visual, name=None, color=None):
    """Build a frozen :class:`Surface` from a volumetric isosurface visual."""
    from ..meshes.Surface import Surface
    from .clip import crop_grid_to_aabb, normalize_clip_aabb
    from .field import ensure_brick
    from .isovalues import primary_isovalue

    grid = getattr(visual, "grid_data", None)
    if grid is None:
        field = None
        fid = getattr(visual, "geometry_field_id", None)
        if fid:
            try:
                from ..runtime.session import get as session_get

                field = session_get(str(fid))
            except Exception:
                field = None
        grid = ensure_brick(field) if field is not None else None
    aabb = normalize_clip_aabb(getattr(visual, "clip_aabb", None))
    if aabb is not None:
        grid = crop_grid_to_aabb(grid, aabb)
    level = primary_isovalue(
        getattr(visual, "isovalues", None),
        default_level=float(getattr(visual, "level", 0.0) or 0.0),
    )
    vertices, normals, faces = isosurface_mesh_from_grid(grid, level)
    field_id = str(getattr(visual, "geometry_field_id", "") or "")
    if not field_id:
        nested = getattr(visual, "grid_data", None)
        field_id = str(getattr(nested, "id", "") or "")
    label = name or ((getattr(visual, "_name", None) or getattr(visual, "name", None) or "iso") + "_surface")
    rgb = color
    if rgb is None:
        raw = getattr(visual, "color", None)
        if raw is not None and not hasattr(raw, "name"):
            try:
                rgb = (float(raw[0]), float(raw[1]), float(raw[2]))
            except (TypeError, IndexError, ValueError):
                rgb = (0.20, 0.60, 0.90)
        else:
            rgb = (0.20, 0.60, 0.90)
    color_fid = str(getattr(visual, "color_field_id", "") or "")
    created = {"field_id": field_id, "isovalue": float(level)}
    if color_fid:
        created["color_field_id"] = color_fid
    surface = Surface(
        [],
        color=rgb,
        name=label,
        bypass_colormap=True,
        created_from=created,
        source_vertices=vertices,
        source_normals=normals,
        source_faces=faces,
        clip_planes=getattr(visual, "clip_planes", None),
    )
    if color_fid:
        from ..util.field_sample import paint_mesh_by_field

        paint_mesh_by_field(
            surface,
            field_id=color_fid,
            colormap=getattr(visual, "field_colormap", None),
            refine=False,
            smooth=0.0,
        )
    return surface
