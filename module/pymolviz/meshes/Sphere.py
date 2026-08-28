from . import Mesh
import numpy as np

from ..util.geometries import icosphere


def _frequency_from_resolution(resolution):
    try:
        steps = int(resolution)
    except (TypeError, ValueError):
        return 4
    if steps <= 6:
        return 2
    if steps <= 10:
        return 3
    if steps <= 14:
        return 4
    if steps <= 18:
        return 6
    return 8


class Sphere(Mesh):
    def __init__(
        self,
        position,
        radius,
        color=None,
        resolution=20,
        subdivisions=None,
        frequency=None,
        *args,
        **kwargs
    ) -> None:
        """Creates a geodesic sphere mesh (uniform triangles, no polar pinch).

        Args:
            radius (float): The radius of the sphere.
            position (np.array): The position of the sphere.
            color (str): Optional. Defaults to "red". The color of the sphere.
            resolution (int): Optional. Defaults to 20. Legacy UV-sphere step
                count; mapped to geodesic frequency when ``frequency`` /
                ``subdivisions`` are omitted.
            subdivisions (int): Optional. Power-of-two depth (frequency = 2**n).
            frequency (int): Optional. Geodesic frequency (2/3/4/6/8 →
                80/180/320/720/1280 triangles).
        """
        if frequency is None and subdivisions is not None:
            frequency = 2 ** max(0, int(subdivisions))
        if frequency is None:
            frequency = _frequency_from_resolution(resolution)
        unit_verts, faces, _edges = icosphere(frequency=frequency)
        vertices = unit_verts * float(radius) + np.asarray(position, dtype=float).reshape(1, 3)
        normals = unit_verts.copy()
        self.position = position
        self.radius = radius
        self.frequency = int(frequency)
        self.subdivisions = int(subdivisions) if subdivisions is not None else None
        super().__init__(vertices, color, normals, faces, *args, **kwargs)
