from . import Mesh
import numpy as np
from ..util.geometries import get_surface_from_points
class Sphere(Mesh):
    def __init__(self, radius, position, color = "white", resolution = 20) -> None:
        """ Creates a sphere mesh.

        Args:
            radius (float): The radius of the sphere.
            position (np.array): The position of the sphere.
        """
        u, v = np.mgrid[0:2*np.pi:resolution*1j, 0:np.pi:resolution*1j]
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.cos(v)
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        vertices = np.vstack((x, y, z)).T * radius
        normals = -vertices / np.linalg.norm(vertices, axis=1)[:, None]
        vertices = vertices + position
        faces = []
        for i in range(1, resolution):
            for j in range(1, resolution):
                faces.append([i*resolution+j-1, i*resolution+j, (i-1)*resolution+j-1])
                faces.append([i*resolution+j, (i-1)*resolution+j, (i-1)*resolution+j-1])
        faces = np.array(faces)
        self.position = position
        self.radius = radius
        super().__init__(vertices, color, normals, faces)


    def get_filtered_points(spheres, tolerance = 1e-4):
        from pymolviz import Points
        vertices = np.vstack([sphere.vertices for sphere in spheres])
        normals = np.vstack([sphere.normals for sphere in spheres])
        colors = np.vstack([sphere.color for sphere in spheres])
        positions = np.vstack([sphere.position for sphere in spheres])
        radii = np.array([sphere.radius for sphere in spheres]) - tolerance
        distances = np.linalg.norm(vertices[None, :, :] - positions[:, None, :], axis=-1)
        filter = np.all(distances.T >= radii, axis=1)
        vertices = vertices[filter]
        normals = normals[filter]
        colors = colors[filter]
        return Points(vertices, normals, colors)

    def merge(spheres, tolerance = 1e-4):
        """ Combines multiple spheres into one mesh.

        Args:
            spheres (list): A list of spheres to combine.

        Returns:
            Mesh: The combined mesh.
        """
        vertices = np.vstack([sphere.vertices for sphere in spheres])
        normals = np.vstack([sphere.normals for sphere in spheres])
        colors = np.vstack([sphere.color for sphere in spheres])
        positions = np.vstack([sphere.position for sphere in spheres])
        radii = np.array([sphere.radius for sphere in spheres]) - tolerance
        distances = np.linalg.norm(vertices[None, :, :] - positions[:, None, :], axis=-1)
        filter = np.all(distances.T >= radii, axis=1)
        vertices = vertices[filter]
        normals = normals[filter]
        colors = colors[filter]

        return get_surface_from_points(vertices, normals, colors)