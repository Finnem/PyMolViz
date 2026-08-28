from . import Mesh
import numpy as np
from scipy.spatial import ConvexHull as scpConvexHull

from ..points import as_point_source, point_sources_from_sequence, resolve_xyz


class ConvexHull(Mesh):
    def __init__(self, points, color=None, *args, **kwargs) -> None:
        self.point_sources = point_sources_from_sequence(points)
        xyz = np.array([resolve_xyz(p) for p in self.point_sources])
        vertices, normals, faces = self._build_mesh(xyz)
        super().__init__(vertices, color, normals, faces, *args, **kwargs)

    def _build_mesh(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        hull = scpConvexHull(points)
        vertices = points
        faces = hull.simplices
        normals = np.zeros_like(points, dtype=float)
        for simplex in hull.simplices:
            p0, p1, p2 = points[simplex]
            edge1 = p1 - p0
            edge2 = p2 - p0
            normal = np.cross(edge1, edge2)
            normal = normal / np.linalg.norm(normal)
            for vertex in simplex:
                normals[vertex] += normal
        normals = np.nan_to_num([tuple(n / np.linalg.norm(n)) for n in normals], nan=0.0)
        return vertices, normals, faces

    def rebuild(self, context=None) -> None:
        xyz = np.array([resolve_xyz(p, context) for p in self.point_sources])
        vertices, normals, faces = self._build_mesh(xyz)
        self.vertices = vertices
        self.normals = normals
        self.faces = faces
