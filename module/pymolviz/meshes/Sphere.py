from . import Mesh
import numpy as np

from ..points import as_point_source, resolve_xyz
from ..util.geometries import icosphere

_UNIT_ICOSPHERE = {}


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


def _unit_icosphere(frequency):
    key = int(frequency)
    cached = _UNIT_ICOSPHERE.get(key)
    if cached is None:
        verts, faces, _edges = icosphere(frequency=key)
        cached = (np.asarray(verts, dtype=float), np.asarray(faces, dtype=int))
        _UNIT_ICOSPHERE[key] = cached
    return cached


def _build_sphere_mesh(position, radius, frequency):
    unit_verts, faces = _unit_icosphere(frequency)
    vertices = unit_verts * float(radius) + np.asarray(position, dtype=float).reshape(1, 3)
    normals = unit_verts.copy()
    return vertices, normals, faces.copy()


class Sphere(Mesh):
    def __init__(
        self,
        position,
        radius,
        color=None,
        resolution=20,
        subdivisions=None,
        frequency=None,
        wireframe=False,
        *args,
        **kwargs
    ) -> None:
        if frequency is None and subdivisions is not None:
            frequency = 2 ** max(0, int(subdivisions))
        if frequency is None:
            frequency = _frequency_from_resolution(resolution)
        self.position = as_point_source(position)
        self.geom_radius = float(radius)
        self.radius = self.geom_radius
        self.frequency = int(frequency)
        self.subdivisions = int(subdivisions) if subdivisions is not None else None
        self.wireframe = bool(wireframe)
        self.resolution = int(resolution)
        pos = resolve_xyz(self.position)
        vertices, normals, faces = _build_sphere_mesh(pos, self.geom_radius, self.frequency)
        super().__init__(vertices, color, normals, faces, *args, **kwargs)
        self.radius = self.geom_radius

    def rebuild(self, context=None) -> None:
        pos = resolve_xyz(self.position, context)
        vertices, normals, faces = _build_sphere_mesh(pos, self.geom_radius, self.frequency)
        self.vertices = vertices
        self.normals = normals
        self.faces = faces
        self.invalidate_cgo_cache()
