from . import Mesh
import numpy as np
from scipy.spatial import ConvexHull as scpConvexHull
class ConvexHull(Mesh):
    def __init__(self, points, color = None, *args, **kwargs) -> None:
        """
        Create a convex hull mesh from a set of points.

        Parameters:
        points : array-like, list of points (shape Nx3)

        Returns:
        Mesh
        """
        points = np.array(points)
        hull = scpConvexHull(points)

        vertices = points
        faces = hull.simplices

        # Calculate normals
        normals = np.zeros_like(points, dtype = float)
        for simplex in hull.simplices:
            p0, p1, p2 = points[simplex]
            edge1 = p1 - p0
            edge2 = p2 - p0
            normal = np.cross(edge1, edge2)
            normal = normal / np.linalg.norm(normal)
            for vertex in simplex:
                normals[vertex] += normal

        # Normalize the vertex normals
        normals = np.nan_to_num([tuple(n / np.linalg.norm(n)) for n in normals], nan=0.0)

        super().__init__(vertices, color, normals, faces, *args, **kwargs)

