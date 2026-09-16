"""Solvent surfaces around a set of spheres (facade)."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from .solvent_connolly import _sas_mesh
from .solvent_gauss import _gauss_mesh
from .solvent_mc import _mc_mesh
from .solvent_params import *  # noqa: F401,F403
from .solvent_connolly import *  # noqa: F401,F403
from .solvent_mesh import *  # noqa: F401,F403
from .solvent_mc import *  # noqa: F401,F403
from .solvent_gauss import *  # noqa: F401,F403

# ``import *`` omits leading-underscore names; tests and convert.py import these from the facade.
from .solvent_params import (
    _ISO_SEC_PER_VOXEL,
    _MC_SEC_PER_CUBE,
    _convex_cap_frequency,
    _phong_refine_params,
    _torus_n_phi_floor,
    _torus_n_theta,
)
from .solvent_mesh import (
    _cubes_with_sign_change,
    _drop_overcovered_edge_faces,
    _face_normals,
    _fill_boundary_holes,
    _orient_faces_to_vertex_normals,
    _split_t_junctions,
    _trilinear_sample,
)
from .solvent_connolly import _decimate_torus_succ, _reduced_surface


def build_solvent_surface(
    points: Sequence[Sequence[float]],
    atom_radius=DEFAULT_ATOM_RADIUS,
    probe_radius: float = DEFAULT_PROBE_RADIUS,
    algorithm: str = DEFAULT_ALGORITHM,
    quality: int = DEFAULT_QUALITY,
    elements=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(vertices, normals, faces)`` for the chosen algorithm."""
    empty = (
        np.zeros((0, 3), dtype=float),
        np.zeros((0, 3), dtype=float),
        np.zeros((0, 3), dtype=int),
    )
    centers = np.asarray(points, dtype=float)
    if centers.size == 0:
        return empty
    centers = np.reshape(centers, (-1, 3))
    radii = expanded_radii(atom_radius, centers.shape[0], probe_radius)
    kind = normalize_algorithm(algorithm)
    if kind == "MC":
        return _mc_mesh(centers, radii, quality, float(probe_radius))
    if kind == "GAUSS":
        return _gauss_mesh(centers, elements, quality)
    return _sas_mesh(centers, radii, quality, float(probe_radius))
