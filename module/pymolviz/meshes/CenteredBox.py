from . import Mesh
import numpy as np

from ..points import as_point_source, resolve_xyz
from ..util.mesh_clip import normalize_clip_planes
from .clippable import (
    apply_source_clips,
    clipped_geometry,
    clone_clip_state,
    set_clip_planes as apply_clip_planes,
    shift_source_vertices,
    store_source_geometry,
)


# Outward winding, matching ``util.cgo._BOX_FACES``. The -X pair used to be
# (0, 3, 7) / (0, 7, 4), which crossed to +X (into the box).
_BOX_FACES = np.array([
    [0, 2, 1], [0, 3, 2],
    [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4],
    [2, 3, 7], [2, 7, 6],
    [0, 7, 3], [0, 4, 7],
    [1, 2, 6], [1, 6, 5],
], dtype=int)


def _build_box_mesh(center, extent):
    cx, cy, cz = center
    dx, dy, dz = (abs(float(extent[0])), abs(float(extent[1])), abs(float(extent[2])))
    vertices = np.array([
        [cx - dx / 2, cy - dy / 2, cz - dz / 2],
        [cx + dx / 2, cy - dy / 2, cz - dz / 2],
        [cx + dx / 2, cy + dy / 2, cz - dz / 2],
        [cx - dx / 2, cy + dy / 2, cz - dz / 2],
        [cx - dx / 2, cy - dy / 2, cz + dz / 2],
        [cx + dx / 2, cy - dy / 2, cz + dz / 2],
        [cx + dx / 2, cy + dy / 2, cz + dz / 2],
        [cx - dx / 2, cy + dy / 2, cz + dz / 2],
    ], dtype=float)
    faces = _BOX_FACES.copy()
    normals = np.zeros_like(vertices, dtype=float)
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        normal = np.cross(v1 - v0, v2 - v0)
        length = np.linalg.norm(normal)
        if length > 1e-12:
            normal /= length
        for idx in face:
            normals[idx] += normal
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths < 1e-12] = 1.0
    normals /= lengths
    return vertices, normals, faces


class CenteredBox(Mesh):
    def __init__(self, center, extent, color=None, wireframe=False, *args, **kwargs) -> None:
        clip_planes = kwargs.pop("clip_planes", None)
        enabled = kwargs.pop("enabled", True)
        self.center = as_point_source(center)
        self.extent = tuple(abs(float(v)) for v in extent)
        self.wireframe = bool(wireframe)
        self.flat_shading = True
        self.enabled = bool(enabled)
        center_xyz = resolve_xyz(self.center)
        vertices, normals, faces = _build_box_mesh(center_xyz, self.extent)
        store_source_geometry(self, vertices, normals, faces)
        self.clip_planes = normalize_clip_planes(clip_planes)
        vertices, normals, faces = clipped_geometry(self)
        super().__init__(vertices, color, normals, faces, *args, **kwargs)

    def set_clip_planes(self, planes) -> None:
        apply_clip_planes(self, planes)

    def shift_vertices(self, delta) -> None:
        if shift_source_vertices(self, delta):
            return
        super().shift_vertices(delta)

    def clone_baked(self):
        return clone_clip_state(self, super().clone_baked())

    def rebuild(self, context=None) -> None:
        center_xyz = resolve_xyz(self.center, context)
        vertices, normals, faces = _build_box_mesh(center_xyz, self.extent)
        store_source_geometry(self, vertices, normals, faces)
        apply_source_clips(self)
        from ..util.field_sample import paint_mesh_by_field
        paint_mesh_by_field(self)

    def from_corners(corner1, corner2, color="red", *args, **kwargs):
        cx1, cy1, cz1 = corner1
        cx2, cy2, cz2 = corner2
        dx, dy, dz = (cx2 - cx1, cy2 - cy1, cz2 - cz1)
        center = ((cx1 + cx2) / 2, (cy1 + cy2) / 2, (cz1 + cz2) / 2)
        return CenteredBox(center, (dx, dy, dz), color, *args, **kwargs)
