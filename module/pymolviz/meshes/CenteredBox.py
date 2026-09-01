from . import Mesh
import numpy as np

from ..points import as_point_source, resolve_xyz


def _build_box_mesh(center, extent):
    cx, cy, cz = center
    dx, dy, dz = extent
    vertices = np.array([
        [cx - dx / 2, cy - dy / 2, cz - dz / 2],
        [cx + dx / 2, cy - dy / 2, cz - dz / 2],
        [cx + dx / 2, cy + dy / 2, cz - dz / 2],
        [cx - dx / 2, cy + dy / 2, cz - dz / 2],
        [cx - dx / 2, cy - dy / 2, cz + dz / 2],
        [cx + dx / 2, cy - dy / 2, cz + dz / 2],
        [cx + dx / 2, cy + dy / 2, cz + dz / 2],
        [cx - dx / 2, cy + dy / 2, cz + dz / 2],
    ])
    faces = np.array([
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [0, 3, 7], [0, 7, 4],
        [1, 2, 6], [1, 6, 5],
    ])
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
        self.center = as_point_source(center)
        self.extent = tuple(float(v) for v in extent)
        self.wireframe = bool(wireframe)
        center_xyz = resolve_xyz(self.center)
        vertices, normals, faces = _build_box_mesh(center_xyz, self.extent)
        super().__init__(vertices, color, normals, faces, *args, **kwargs)

    def rebuild(self, context=None) -> None:
        center_xyz = resolve_xyz(self.center, context)
        vertices, normals, faces = _build_box_mesh(center_xyz, self.extent)
        self.vertices = vertices
        self.normals = normals
        self.faces = faces
        self.invalidate_cgo_cache()

    def from_corners(corner1, corner2, color="red", *args, **kwargs):
        cx1, cy1, cz1 = corner1
        cx2, cy2, cz2 = corner2
        dx, dy, dz = (cx2 - cx1, cy2 - cy1, cz2 - cz1)
        center = ((cx1 + cx2) / 2, (cy1 + cy2) / 2, (cz1 + cz2) / 2)
        return CenteredBox(center, (dx, dy, dz), color, *args, **kwargs)
