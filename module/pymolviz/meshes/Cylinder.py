from . import Mesh
import numpy as np
from ..util.geometries import generate_cylinder
from scipy.spatial.transform import Rotation


class Cylinder(Mesh):
    def __init__(self, start, end, width, color = "white", curvature = 0, resolution = 20, **kwargs) -> None:
        """ Creates a cylinder mesh.
        
        Args:
            start (np.array): The start position of the cylinder.
            end (np.array): The end position of the cylinder.
            width (float): The width of the cylinder.
            color (str, optional): The color of the cylinder. Defaults to "white". Can also be a hex color code. Can also be two values, one for each end of the cylinder.
            curvature (float): The curvature of the cylinders ends. How much the end vertices pertrude from the end planes.
            """

        # create cylinder
        
        start = np.array(start)
        end = np.array(end)

        base_cylinder = generate_cylinder(np.linalg.norm(end - start), resolution, width, curvature=curvature)
        self.vertices = base_cylinder["vertices"]
        self.normals = base_cylinder["normals"]
        self.faces = base_cylinder["faces"]
        # rotate cylinder
        direction = end - start; direction = direction / np.linalg.norm(direction)
        if not (np.allclose(direction, np.array([0, 0, 1]))):
            if np.allclose(direction, np.array([0, 0, -1])):
                rotation = Rotation.from_rotvec(np.pi * np.array([0, 1, 0]))
            else:
                rot_axis = np.cross(np.array([0, 0, 1]), direction); rot_axis = rot_axis / np.linalg.norm(rot_axis)
                rot_angle = np.arccos(np.dot(direction, np.array([0, 0, 1])))
                rotation = Rotation.from_rotvec(rot_angle * rot_axis)

            self.vertices = rotation.apply(self.vertices)
            self.normals = rotation.apply(self.normals)

        # translate cylinder
        self.vertices = self.vertices + start

        if (type(color) is list and len(color) == 2) or (type(color) is np.ndarray and color.shape[0] == 2):
            color = [Mesh._convert_string_color(c) if type(c) is str else c for c in color]
            color = np.vstack([np.full((int(len(self.vertices) / 2), 3), color[0]),
                               np.full((int(len(self.vertices) / 2), 3), color[1])])


        super().__init__(self.vertices, color, self.normals, self.faces, **kwargs)
