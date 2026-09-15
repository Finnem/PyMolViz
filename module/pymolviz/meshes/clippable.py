"""Store unclipped source geometry and apply clip planes on triangle meshes."""

from __future__ import annotations

import numpy as np

from ..util.mesh_clip import clip_mesh_by_planes, normalize_clip_planes


def store_source_geometry(mesh, vertices, normals, faces) -> None:
    mesh._source_vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    mesh._source_normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    mesh._source_faces = np.asarray(faces, dtype=int).reshape(-1, 3)


def clipped_geometry(mesh):
    vertices = np.asarray(mesh._source_vertices, dtype=float).reshape(-1, 3)
    normals = np.asarray(mesh._source_normals, dtype=float).reshape(-1, 3)
    faces = np.asarray(mesh._source_faces, dtype=int).reshape(-1, 3)
    planes = getattr(mesh, "clip_planes", None) or ()
    if not planes:
        return (
            np.array(vertices, copy=True),
            np.array(normals, copy=True),
            np.array(faces, copy=True),
        )
    return clip_mesh_by_planes(vertices, faces, normals, planes)


def apply_source_clips(mesh) -> None:
    vertices, normals, faces = clipped_geometry(mesh)
    mesh.vertices = vertices
    mesh.normals = normals
    mesh.faces = faces
    mesh.invalidate_cgo_cache()


def set_clip_planes(mesh, planes) -> None:
    mesh.clip_planes = normalize_clip_planes(planes)
    apply_source_clips(mesh)


def shift_source_vertices(mesh, delta) -> bool:
    """Shift stored source vertices.

    Returns True when clip planes were reapplied, so the caller should not
    also shift already-clipped baked vertices.
    """
    src = getattr(mesh, "_source_vertices", None)
    if src is None:
        return False
    d = np.asarray(delta, dtype=float).reshape(3)
    mesh._source_vertices = np.asarray(src, dtype=float).reshape(-1, 3) + d
    if getattr(mesh, "clip_planes", None):
        apply_source_clips(mesh)
        return True
    return False


def clone_clip_state(src, cloned):
    for attr in ("_source_vertices", "_source_normals", "_source_faces"):
        val = getattr(src, attr, None)
        if val is not None:
            setattr(cloned, attr, np.array(val, copy=True))
    cloned.clip_planes = [dict(p) for p in (getattr(src, "clip_planes", None) or [])]
    cloned.enabled = bool(getattr(src, "enabled", True))
    return cloned
