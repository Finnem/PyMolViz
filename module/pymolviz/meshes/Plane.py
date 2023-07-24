from . import Mesh
import numpy as np
from ..util.math import get_perp

class Plane(Mesh):
    def __init__(self, position, normal, scale = 5, color = "red", *args, **kwargs) -> None:
        """ Creates a plane mesh at the given position and with the given normal.
        
        Args:
            position (np.array): The position of the plane.
            normal (np.array): The normal of the plane.
            scale (float): The scale of the plane.
            color (str): Optional. Defaults to "red". The color of the plane.
            
            
            """


        self.position = position
        self.normal = normal / np.linalg.norm(normal)

        v1 = get_perp(self.normal); v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(self.normal, v1); v2 = v2 / np.linalg.norm(v2)
        v1 *= scale; v2 *= scale
        vertices = np.array([self.position, self.position + v1, self.position + v2, self.position + v1 + v2])
        vertices -= (v1 + v2)/2
        faces = np.array([[0, 1, 2], [1, 3, 2]])
        normals = np.full(vertices.shape, normal)
        

        super().__init__(vertices, color, normals, faces = faces, *args, **kwargs)
