from . import Mesh
import numpy as np
class Sphere(Mesh):
    def __init__(self, position, radius, color = "red", resolution = 20, *args, **kwargs) -> None:
        """ Creates a sphere mesh.

        Args:
            radius (float): The radius of the sphere.
            position (np.array): The position of the sphere.
            color (str): Optional. Defaults to "red". The color of the sphere.
            resolution (int): Optional. Defaults to 20. The resolution of the sphere.

        """
        u, v = np.mgrid[0:2*np.pi:resolution*1j, 0:np.pi:resolution*1j]
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.cos(v)
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        vertices = np.vstack((x, y, z)).T * radius
        normals = vertices / np.linalg.norm(vertices, axis=1)[:, None]
        vertices = vertices + position
        faces = []
        for i in range(1, resolution):
            for j in range(1, resolution):
                faces.append([i*resolution+j-1, (i-1)*resolution+j-1, i*resolution+j])
                faces.append([i*resolution+j, (i-1)*resolution+j-1, (i-1)*resolution+j])
        faces = np.array(faces)
        self.position = position
        self.radius = radius
        super().__init__(vertices, color, normals, faces, *args, **kwargs)

