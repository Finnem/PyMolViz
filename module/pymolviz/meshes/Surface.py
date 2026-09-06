from __future__ import annotations

import numpy as np

from . import Mesh
from ..points import point_sources_from_sequence, resolve_xyz
from ..util.mesh_clip import clip_mesh_by_planes, normalize_clip_planes
from ..util.solvent_surface import (
    DEFAULT_ATOM_RADIUS,
    DEFAULT_PROBE_RADIUS,
    DEFAULT_QUALITY,
    DEFAULT_RADIUS_MODE,
    DEFAULT_VDW_SCALE,
    build_solvent_surface,
    normalize_algorithm,
    normalize_point_radii,
    normalize_radius_mode,
    resolve_atom_radii,
)


def _xyz_of_sources(sources, context=None):
    if not sources:
        return np.zeros((0, 3), dtype=float)
    return np.array([resolve_xyz(p, context) for p in sources], dtype=float)


class Surface(Mesh):
    """Solvent surface around point spheres: rolling-ball SAS or accessible ASA."""

    def __init__(
        self,
        points,
        atom_radius=DEFAULT_ATOM_RADIUS,
        probe_radius=DEFAULT_PROBE_RADIUS,
        algorithm="SAS",
        quality=DEFAULT_QUALITY,
        color=None,
        wireframe=False,
        radius_mode=DEFAULT_RADIUS_MODE,
        vdw_scale=DEFAULT_VDW_SCALE,
        point_radii=None,
        clip_planes=None,
        *args,
        **kwargs
    ) -> None:
        context = kwargs.pop("context", None)
        self.point_sources = point_sources_from_sequence(points)
        self.atom_radius = float(atom_radius)
        self.probe_radius = float(probe_radius)
        self.algorithm = normalize_algorithm(algorithm)
        self.quality = max(1, min(5, int(quality)))
        self.wireframe = bool(wireframe)
        self.radius_mode = normalize_radius_mode(radius_mode)
        self.vdw_scale = float(vdw_scale) if vdw_scale else DEFAULT_VDW_SCALE
        self.point_radii = normalize_point_radii(point_radii, len(self.point_sources))
        self.clip_planes = normalize_clip_planes(clip_planes)
        xyz = _xyz_of_sources(self.point_sources, context)
        src_v, src_n, src_f = build_solvent_surface(
            xyz, self._atom_radii(context), self.probe_radius, self.algorithm, self.quality,
        )
        self._source_vertices = np.asarray(src_v, dtype=float).reshape(-1, 3)
        self._source_normals = np.asarray(src_n, dtype=float).reshape(-1, 3)
        self._source_faces = np.asarray(src_f, dtype=int).reshape(-1, 3)
        vertices, normals, faces = self._clipped_geometry()
        super().__init__(vertices, color, normals, faces, *args, **kwargs)

    def _atom_radii(self, context=None):
        return resolve_atom_radii(
            self.point_sources,
            len(self.point_sources),
            self.atom_radius,
            self.radius_mode,
            self.vdw_scale,
            self.point_radii,
            context,
        )

    def _clipped_geometry(self):
        vertices = np.asarray(self._source_vertices, dtype=float).reshape(-1, 3)
        normals = np.asarray(self._source_normals, dtype=float).reshape(-1, 3)
        faces = np.asarray(self._source_faces, dtype=int).reshape(-1, 3)
        if not self.clip_planes:
            return (
                np.array(vertices, copy=True),
                np.array(normals, copy=True),
                np.array(faces, copy=True),
            )
        return clip_mesh_by_planes(vertices, faces, normals, self.clip_planes)

    def _apply_source_clips(self) -> None:
        vertices, normals, faces = self._clipped_geometry()
        self.vertices = vertices
        self.normals = normals
        self.faces = faces
        self.invalidate_cgo_cache()

    def set_clip_planes(self, planes) -> None:
        self.clip_planes = normalize_clip_planes(planes)
        self._apply_source_clips()

    def clone_baked(self):
        cloned = super().clone_baked()
        for attr in ("_source_vertices", "_source_normals", "_source_faces"):
            val = getattr(self, attr, None)
            if val is not None:
                setattr(cloned, attr, np.array(val, copy=True))
        cloned.clip_planes = [dict(p) for p in getattr(self, "clip_planes", [])]
        return cloned

    def rebuild(self, context=None) -> None:
        xyz = _xyz_of_sources(self.point_sources, context)
        self.point_radii = normalize_point_radii(self.point_radii, len(self.point_sources))
        src_v, src_n, src_f = build_solvent_surface(
            xyz, self._atom_radii(context), self.probe_radius, self.algorithm, self.quality,
        )
        self._source_vertices = np.asarray(src_v, dtype=float).reshape(-1, 3)
        self._source_normals = np.asarray(src_n, dtype=float).reshape(-1, 3)
        self._source_faces = np.asarray(src_f, dtype=int).reshape(-1, 3)
        self._apply_source_clips()
