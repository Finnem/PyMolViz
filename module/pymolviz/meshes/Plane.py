from . import Mesh
import numpy as np
from ..points import as_point_source, resolve_xyz
from ..util.math import get_perp


class Plane(Mesh):
    def __init__(self, position, normal, scale=5, color=None, *args, **kwargs) -> None:
        self.position = as_point_source(position)
        self.normal = np.asarray(normal, dtype=float).reshape(3)
        self.normal = self.normal / np.linalg.norm(self.normal)
        self.scale = float(scale)
        pos = resolve_xyz(self.position)
        vertices, normals, faces = self._build_mesh(pos)
        super().__init__(vertices, color, normals, faces=faces, *args, **kwargs)

    def _build_mesh(self, position):
        v1 = get_perp(self.normal)
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(self.normal, v1)
        v2 = v2 / np.linalg.norm(v2)
        v1 *= self.scale
        v2 *= self.scale
        vertices = np.array([
            position,
            position + v1,
            position + v2,
            position + v1 + v2,
        ])
        vertices -= (v1 + v2) / 2
        faces = np.array([[0, 1, 2], [1, 3, 2]])
        normals = np.full(vertices.shape, self.normal)
        return vertices, normals, faces

    def rebuild(self, context=None) -> None:
        pos = resolve_xyz(self.position, context)
        vertices, normals, faces = self._build_mesh(pos)
        self.vertices = vertices
        self.normals = normals
        self.faces = faces
