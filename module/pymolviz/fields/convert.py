"""Marching-cubes conversion of an IsoSurface Field Visual to an explicit Surface."""

from __future__ import annotations

import numpy as np

# Align with wireframe mesh budget in ``wireframe_quality.MAX_MESH_TRIANGLES``.
HEAVY_EXPLICIT_TRIANGLES = 80000
# Exact marching-cubes cube count is cheap only below this voxel count.
_EXACT_CUBE_VOXEL_LIMIT = 2_000_000


def _grid_and_level_for_convert(visual):
    """Resolve cropped grid and isolevel the same way as :func:`convert_isosurface_to_surface`."""
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
    if aabb is not None and grid is not None:
        grid = crop_grid_to_aabb(grid, aabb)
    level = primary_isovalue(
        getattr(visual, "isovalues", None),
        default_level=float(getattr(visual, "level", 0.0) or 0.0),
    )
    return grid, float(level)


def estimate_isosurface_convert_job(visual) -> dict:
    """Cheap estimate of explicit mesh size before marching cubes."""
    from ..util.field_sample import grid_values_3d
    from ..util.solvent_params import (
        HEAVY_SURFACE_SECONDS,
        HEAVY_SURFACE_VOXELS,
        _MC_SEC_PER_CUBE,
    )
    from ..util.solvent_surface import _cubes_with_sign_change

    job = {
        "algorithm": "EXPLICIT_ISO",
        "quality": 0,
        "n_atoms": 0,
        "voxels": 0,
        "cubes": 0,
        "triangles": 0,
        "seconds": 0.0,
        "heavy": False,
    }
    grid, level = _grid_and_level_for_convert(visual)
    if grid is None:
        return job
    values = grid_values_3d(grid) - float(level)
    n_vox = int(values.size)
    job["voxels"] = n_vox
    if n_vox <= _EXACT_CUBE_VOXEL_LIMIT:
        cubes = _cubes_with_sign_change(values)
        n_cubes = int(cubes.shape[0])
    else:
        n_cubes = int(min(float(n_vox), 0.08 * float(n_vox)))
    est_tri = int(n_cubes * 4)
    seconds = float(n_cubes) * _MC_SEC_PER_CUBE
    job["cubes"] = int(n_cubes)
    job["triangles"] = est_tri
    job["seconds"] = seconds
    job["heavy"] = bool(
        seconds >= HEAVY_SURFACE_SECONDS
        or n_vox >= HEAVY_SURFACE_VOXELS
        or est_tri >= HEAVY_EXPLICIT_TRIANGLES
    )
    return job


def explicit_convert_job_fingerprint(job) -> tuple:
    return (
        str(job.get("algorithm") or ""),
        int(job.get("voxels") or 0),
        int(job.get("cubes") or 0),
        int(job.get("triangles") or 0),
        round(float(job.get("seconds") or 0.0), 1),
    )


def format_explicit_convert_message(job) -> str:
    seconds = max(2, int(round(float(job.get("seconds") or 0.0))))
    voxels = int(job.get("voxels") or 0)
    triangles = int(job.get("triangles") or 0)
    parts = []
    if triangles:
        parts.append("about %s triangles" % format(triangles, ","))
    if voxels:
        parts.append("%s field voxels" % format(voxels, ","))
    detail = " (%s)" % ", ".join(parts) if parts else ""
    return (
        "Converting this implicit isosurface to an explicit mesh may take about %s seconds%s. "
        "PyMOL will not respond until it finishes. Large meshes are slow to edit and export. "
        "Convert anyway?"
        % (seconds, detail)
    )


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

    grid, level = _grid_and_level_for_convert(visual)
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
