from . import Mesh
import numpy as np

from ..points import as_point_source, resolve_xyz


def _build_cylinder_mesh(start, end, radius, resolution):
    start_point = np.asarray(start, dtype=float).reshape(3)
    end_point = np.asarray(end, dtype=float).reshape(3)
    axis = end_point - start_point
    axis_length = np.linalg.norm(axis)
    if axis_length < 1e-12:
        axis_direction = np.array([0.0, 0.0, 1.0])
        axis_length = 1e-12
    else:
        axis_direction = axis / axis_length

    if axis_direction[0] == 0 and axis_direction[1] == 0:
        not_parallel = np.array([1, 0, 0])
    else:
        not_parallel = np.array([0, 0, 1])

    orthogonal_vector1 = np.cross(axis_direction, not_parallel)
    orthogonal_vector1 /= np.linalg.norm(orthogonal_vector1)
    orthogonal_vector2 = np.cross(axis_direction, orthogonal_vector1)

    circle_points = []
    for i in range(resolution):
        angle = 2 * np.pi * i / resolution
        point = (
            radius * np.cos(angle) * orthogonal_vector1
            + radius * np.sin(angle) * orthogonal_vector2
        )
        circle_points.append(point)

    vertices = []
    normals = []
    for point in circle_points:
        vertices.append(tuple(start_point + point))
        normals.append(tuple(point / radius))
        vertices.append(tuple(end_point + point))
        normals.append(tuple(point / radius))

    faces = []
    for i in range(resolution):
        next_i = (i + 1) % resolution
        faces.append((2 * i, 2 * next_i, 2 * i + 1))
        faces.append((2 * i + 1, 2 * next_i, 2 * next_i + 1))

    vertices.append(tuple(start_point))
    vertices.append(tuple(end_point))
    normals.append(tuple(-axis_direction))
    normals.append(tuple(axis_direction))
    center_start_index = len(vertices) - 2
    center_end_index = len(vertices) - 1

    for i in range(resolution):
        next_i = (i + 1) % resolution
        faces.append((2 * i, 2 * next_i, center_start_index))
        faces.append((2 * i + 1, center_end_index, 2 * next_i + 1))

    return np.array(vertices), np.array(normals), np.array(faces)


class Cylinder(Mesh):
    def __init__(self, start, end, radius, color=None, resolution=20, *args, **kwargs) -> None:
        self.start = as_point_source(start)
        self.end = as_point_source(end)
        self.geom_radius = float(radius)
        self.radius = self.geom_radius
        self.resolution = int(resolution)
        start_xyz = resolve_xyz(self.start)
        end_xyz = resolve_xyz(self.end)
        vertices, normals, faces = _build_cylinder_mesh(start_xyz, end_xyz, self.geom_radius, self.resolution)
        super().__init__(vertices, color, normals, faces, *args, **kwargs)
        self.radius = self.geom_radius

    def rebuild(self, context=None) -> None:
        start_xyz = resolve_xyz(self.start, context)
        end_xyz = resolve_xyz(self.end, context)
        vertices, normals, faces = _build_cylinder_mesh(
            start_xyz, end_xyz, self.geom_radius, self.resolution
        )
        self.vertices = vertices
        self.normals = normals
        self.faces = faces
        self.invalidate_cgo_cache()
