from . import Mesh
import numpy as np
class Cylinder(Mesh):
    def __init__(self, start, end, radius, color = None, resolution = 20, *args, **kwargs) -> None:
        """
        Create a cylinder mesh given a start point, end point, and radius, including normals.

        Parameters:
        start : array-like, start point of the cylinder (x1, y1, z1)
        end   : array-like, end point of the cylinder (x2, y2, z2)
        radius      : float, radius of the cylinder
        color       : str, color of the cylinder
        resolution: int, number of segments around the cylinder's circumference

        Returns:
        Mesh
        """
        # Convert start and end points to numpy arrays
        start_point = np.array(start)
        end_point = np.array(end)

        # Compute the cylinder's axis
        axis = end_point - start_point
        axis_length = np.linalg.norm(axis)
        axis_direction = axis / axis_length

        # Create a vector that is not parallel to the axis
        if axis_direction[0] == 0 and axis_direction[1] == 0:
            not_parallel = np.array([1, 0, 0])
        else:
            not_parallel = np.array([0, 0, 1])

        # Compute two perpendicular vectors in the plane orthogonal to the axis
        orthogonal_vector1 = np.cross(axis_direction, not_parallel)
        orthogonal_vector1 /= np.linalg.norm(orthogonal_vector1)
        orthogonal_vector2 = np.cross(axis_direction, orthogonal_vector1)

        # Compute the circle points
        circle_points = []
        for i in range(resolution):
            angle = 2 * np.pi * i / resolution
            point = (radius * np.cos(angle) * orthogonal_vector1 +
                    radius * np.sin(angle) * orthogonal_vector2)
            circle_points.append(point)

        # Compute the vertices and normals
        vertices = []
        normals = []
        for point in circle_points:
            vertices.append(tuple(start_point + point))
            normals.append(tuple(point / radius))  # Normal is the normalized circle point vector
            vertices.append(tuple(end_point + point))
            normals.append(tuple(point / radius))  # Normal is the normalized circle point vector

        # Compute the faces
        faces = []
        for i in range(resolution):
            next_i = (i + 1) % resolution

            faces.append((2 * i, 2 * next_i, 2 * i + 1))
            faces.append((2 * i + 1, 2 * next_i, 2 * next_i + 1))

        # Add top and bottom faces if needed
        vertices.append(tuple(start_point))
        vertices.append(tuple(end_point))
        normals.append(tuple(-axis_direction))  # Normal for bottom cap center
        normals.append(tuple(axis_direction))   # Normal for top cap center
        center_start_index = len(vertices) - 2
        center_end_index = len(vertices) - 1

        for i in range(resolution):
            next_i = (i + 1) % resolution
            faces.append((2 * i, 2 * next_i, center_start_index))
            faces.append((2 * i + 1, center_end_index, 2 * next_i + 1))

        vertices = np.array(vertices)
        normals = np.array(normals)
        faces = np.array(faces)
        self.radius = radius
        super().__init__(vertices, color, normals, faces, *args, **kwargs)

