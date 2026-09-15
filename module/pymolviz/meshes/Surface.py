from __future__ import annotations

import numpy as np

from . import Mesh
from ..points import point_sources_from_sequence, resolve_xyz
from ..util.mesh_clip import clip_mesh_by_planes, normalize_clip_planes
from ..util.solvent_surface import (
    DEFAULT_ALGORITHM,
    DEFAULT_ATOM_RADIUS,
    DEFAULT_PROBE_RADIUS,
    DEFAULT_QUALITY,
    DEFAULT_RADIUS_MODE,
    DEFAULT_VDW_SCALE,
    build_solvent_surface,
    normalize_algorithm,
    normalize_point_enabled,
    normalize_point_radii,
    normalize_radius_mode,
    resolve_atom_elements,
    resolve_atom_radii,
)
from .clippable import (
    apply_source_clips,
    clipped_geometry,
    clone_clip_state,
    set_clip_planes as apply_clip_planes,
    store_source_geometry,
)


def _xyz_of_sources(sources, context=None):
    if not sources:
        return np.zeros((0, 3), dtype=float)
    return np.array([resolve_xyz(p, context) for p in sources], dtype=float)


class Surface(Mesh):
    """Solvent surface around point spheres: Connolly SASA, marching-cubes SES, or PyMOL Gaussian."""

    def __init__(
        self,
        points,
        atom_radius=DEFAULT_ATOM_RADIUS,
        probe_radius=DEFAULT_PROBE_RADIUS,
        algorithm=DEFAULT_ALGORITHM,
        quality=DEFAULT_QUALITY,
        color=None,
        wireframe=False,
        radius_mode=DEFAULT_RADIUS_MODE,
        vdw_scale=DEFAULT_VDW_SCALE,
        point_radii=None,
        clip_planes=None,
        point_enabled=None,
        *args,
        **kwargs
    ) -> None:
        context = kwargs.pop("context", None)
        created_from = kwargs.pop("created_from", None)
        source_vertices = kwargs.pop("source_vertices", None)
        source_normals = kwargs.pop("source_normals", None)
        source_faces = kwargs.pop("source_faces", None)
        self.point_sources = point_sources_from_sequence(points)
        self.atom_radius = float(atom_radius)
        self.probe_radius = float(probe_radius)
        self.algorithm = normalize_algorithm(algorithm)
        self.quality = max(1, min(5, int(quality)))
        self.wireframe = bool(wireframe)
        self.radius_mode = normalize_radius_mode(radius_mode)
        self.vdw_scale = float(vdw_scale) if vdw_scale else DEFAULT_VDW_SCALE
        self.point_radii = normalize_point_radii(point_radii, len(self.point_sources))
        self.point_enabled = normalize_point_enabled(point_enabled, len(self.point_sources))
        self.clip_planes = normalize_clip_planes(clip_planes)
        self.created_from = dict(created_from) if created_from else None
        if source_vertices is not None:
            src_v = np.asarray(source_vertices, dtype=float).reshape(-1, 3)
            src_n = np.asarray(source_normals, dtype=float).reshape(-1, 3) if source_normals is not None else np.zeros_like(src_v)
            src_f = np.asarray(source_faces, dtype=int).reshape(-1, 3) if source_faces is not None else np.zeros((0, 3), dtype=int)
        else:
            src_v, src_n, src_f = self._build_source_geometry(context)
        store_source_geometry(self, src_v, src_n, src_f)
        vertices, normals, faces = clipped_geometry(self)
        super().__init__(vertices, color, normals, faces, *args, **kwargs)

    def _enabled_flags(self):
        n = len(self.point_sources)
        flags = self.point_enabled
        if flags is None:
            return [True] * n
        if len(flags) < n:
            return list(flags) + [True] * (n - len(flags))
        return list(flags[:n])

    def _active_sources_and_radii(self):
        flags = self._enabled_flags()
        sources = [src for src, on in zip(self.point_sources, flags) if on]
        radii = self.point_radii
        if radii is None:
            active_radii = None
        else:
            active_radii = [r for r, on in zip(radii, flags) if on]
        return sources, active_radii

    def _build_source_geometry(self, context=None):
        sources, radii = self._active_sources_and_radii()
        if not sources:
            empty = np.zeros((0, 3), dtype=float)
            return empty, empty.copy(), np.zeros((0, 3), dtype=int)
        xyz = _xyz_of_sources(sources, context)
        src_v, src_n, src_f = build_solvent_surface(
            xyz,
            self._atom_radii(context, sources, radii),
            self.probe_radius,
            self.algorithm,
            self.quality,
            elements=self._atom_elements(sources),
        )
        return src_v, src_n, src_f

    def _atom_radii(self, context=None, sources=None, radii=None):
        src = self.point_sources if sources is None else sources
        rad = self.point_radii if radii is None else radii
        return resolve_atom_radii(
            src,
            len(src),
            self.atom_radius,
            self.radius_mode,
            self.vdw_scale,
            rad,
            context,
        )

    def _atom_elements(self, sources=None):
        src = self.point_sources if sources is None else sources
        return resolve_atom_elements(src, len(src))

    def _clipped_geometry(self):
        return clipped_geometry(self)

    def _apply_source_clips(self) -> None:
        apply_source_clips(self)

    def _repaint_colors(self, context=None) -> None:
        if getattr(self, "field_id", None):
            from ..util.field_sample import paint_mesh_by_field
            paint_mesh_by_field(self)
        elif getattr(self, "point_colors", None):
            from ..util.field_sample import paint_surface_mesh_from_anchors
            paint_surface_mesh_from_anchors(self, context)

    def set_clip_planes(self, planes) -> None:
        apply_clip_planes(self, planes)
        # Clip rewrites vertices; per-vertex RGB must be remapped or the
        # preview falls back to a single COLOR (restored only on commit).
        self._repaint_colors()

    def clone_baked(self):
        return clone_clip_state(self, super().clone_baked())

    def rebuild(self, context=None) -> None:
        if getattr(self, "created_from", None) and getattr(self, "_source_vertices", None) is not None:
            apply_source_clips(self)
            self._repaint_colors(context)
            return
        self.point_radii = normalize_point_radii(self.point_radii, len(self.point_sources))
        self.point_enabled = normalize_point_enabled(self.point_enabled, len(self.point_sources))
        src_v, src_n, src_f = self._build_source_geometry(context)
        store_source_geometry(self, src_v, src_n, src_f)
        apply_source_clips(self)
        self._repaint_colors(context)
