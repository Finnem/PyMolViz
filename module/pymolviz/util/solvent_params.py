"""Solvent surface parameters, radii, spacing, and job estimates."""

from __future__ import annotations

from typing import Sequence, Tuple

import math

import numpy as np
from scipy.spatial import cKDTree

from .gaussian_map import DEFAULT_GAUSSIAN_RESOLUTION

SURFACE_ALGORITHMS = ("SASA", "MC", "GAUSS")
SURFACE_ALGORITHM_LABELS = {
    "GAUSS": "Gaussian Spheres",
    "MC": "Marching Cubes",
    "SASA": "Solvent Accessible Surface",
}
SURFACE_ALGORITHM_UI_ORDER = ("GAUSS", "MC", "SASA")
_ALGORITHM_ALIASES = {
    "SAS": "SASA",
    "SASA": "SASA",
    "SOLVENT_ACCESSIBLE_SURFACE": "SASA",
    "CUBES": "MC",
    "EDT": "MC",
    "MARCHING_CUBES": "MC",
    "MARCHINGCUBES": "MC",
    "GAUSSIAN": "GAUSS",
    "GAUSSIAN_SPHERES": "GAUSS",
    "BLOB": "GAUSS",
    "MAP": "GAUSS",
}
RADIUS_MODES = ("uniform", "vdw")
DEFAULT_ATOM_RADIUS = 1.0
DEFAULT_PROBE_RADIUS = 1.4
DEFAULT_QUALITY = 3
DEFAULT_VDW_SCALE = 1.0
DEFAULT_RADIUS_MODE = "vdw"
DEFAULT_ALGORITHM = "GAUSS"
BONDI_VDW = {
    "H": 1.20, "HE": 1.40, "LI": 1.82, "BE": 1.53, "B": 1.92, "C": 1.70,
    "N": 1.55, "O": 1.52, "F": 1.47, "NE": 1.54, "NA": 2.27, "MG": 1.73,
    "AL": 1.84, "SI": 2.10, "P": 1.80, "S": 1.80, "CL": 1.75, "AR": 1.88,
    "K": 2.75, "CA": 2.31, "MN": 1.61, "FE": 1.84, "CO": 1.52, "NI": 1.63,
    "CU": 1.40, "ZN": 1.39, "SE": 1.90, "BR": 1.85, "KR": 2.02, "I": 1.98,
    "XE": 2.16, "MO": 1.90,
}
_TWO_LETTER_ELEM = frozenset(
    key for key in BONDI_VDW if len(key) == 2
)
MAX_SAS_CUBES = 320000
MAX_SAS_VOXELS = 9600000
SAS_SPACING = {1: 1.80, 2: 1.25, 3: 0.90, 4: 0.65, 5: 0.45}
ISO_SPACING = {1: 0.325, 2: 0.225, 3: 0.16, 4: 0.11, 5: 0.075}
GAUSS_SPACING = ISO_SPACING
MC_SPACING = ISO_SPACING
SAS_CAP_FREQUENCY = {1: 4, 2: 8, 3: 12, 4: 14, 5: 16}
SAS_N_PHI_FLOOR = {1: 4, 2: 6, 3: 8, 4: 16, 5: 32}

def _convex_cap_frequency(quality: int) -> int:
    """Geodesic frequency for VDW contact caps."""
    return int(SAS_CAP_FREQUENCY[_quality_level(quality)])
def _torus_n_theta(quality: int) -> int:
    """Contact-circle samples so the torus rim matches the geodesic cap."""
    return max(8, 4 * int(_convex_cap_frequency(quality)))
def _torus_n_phi_floor(quality: int) -> int:
    """Lower bound on probe-fillet rings. Quality 1 stays a handful of quads."""
    return int(SAS_N_PHI_FLOOR[_quality_level(quality)])
def _phong_refine_params(quality: int):
    """Phong splits that would undo a coarse torus are skipped at quality 1."""
    q = _quality_level(quality)
    if q <= 1:
        return 0.70, 1
    if q == 2:
        return 0.85, 2
    return 0.92, 3
def normalize_algorithm(name) -> str:
    text = str(name or DEFAULT_ALGORITHM).strip()
    for key, label in SURFACE_ALGORITHM_LABELS.items():
        if text.lower() == label.lower():
            return key
    text = text.upper().replace("-", "_").replace(" ", "_")
    text = _ALGORITHM_ALIASES.get(text, text)
    if text in SURFACE_ALGORITHMS:
        return text
    return DEFAULT_ALGORITHM
def surface_algorithm_label(algorithm: str) -> str:
    """User-facing combo label for a normalized algorithm id."""
    key = normalize_algorithm(algorithm)
    return SURFACE_ALGORITHM_LABELS[key]
def normalize_radius_mode(name) -> str:
    text = str(name or DEFAULT_RADIUS_MODE).strip().lower().replace("-", "_")
    if text in ("vdw", "vdw_scale", "scale_vdw"):
        return "vdw"
    return "uniform"
def element_from_atom_name(name, elem="") -> str:
    raw = str(elem or "").strip()
    if raw:
        letters = "".join(ch for ch in raw if ch.isalpha()).upper()
        if len(letters) >= 2 and letters[:2] in _TWO_LETTER_ELEM:
            return letters[:2]
        if letters:
            return letters[0]
    letters = "".join(ch for ch in str(name or "") if ch.isalpha()).upper()
    if letters:
        return letters[0]
    return "C"
def vdw_for_atom(name="", elem="", default=DEFAULT_ATOM_RADIUS) -> float:
    """Bondi radius from the ``elem`` field, else the first letter of the atom name.

    Two-letter symbols (``CA`` calcium, ``CL`` chlorine) are only taken from
    ``elem``. A protein atom named ``CA`` with an empty ``elem`` is carbon.
    """
    key = element_from_atom_name(name, elem)
    return float(BONDI_VDW.get(key, default))
def vdw_for_element(elem, default=DEFAULT_ATOM_RADIUS) -> float:
    return vdw_for_atom(elem=elem, default=default)
def lookup_source_vdw(source, context=None):
    """Return an atom van der Waals radius in Å, or None if the source is not an atom."""
    from ..points import AtomPoint

    if not isinstance(source, AtomPoint):
        return None
    if context is not None:
        found = source.lookup_vdw(context)
        if found is not None:
            return float(found)
    cached = getattr(source, "last_vdw", None)
    if cached is not None:
        return float(cached)
    return vdw_for_atom(getattr(source, "name", ""), getattr(source, "elem", ""))
def normalize_point_enabled(values, n: int):
    """Length-*n* list of bools; ``None`` when every point is enabled."""
    if n <= 0:
        return None
    if values is None:
        return None
    raw = list(values)
    out = []
    for i in range(int(n)):
        if i >= len(raw):
            out.append(True)
        else:
            out.append(bool(raw[i]))
    if all(out):
        return None
    return out
def normalize_point_radii(values, n: int):
    """Length-*n* list of optional floats; ``None`` if every entry is inherited."""
    if n <= 0:
        return None
    if values is None:
        return None
    out = []
    raw = list(values)
    for i in range(int(n)):
        if i >= len(raw) or raw[i] is None:
            out.append(None)
            continue
        try:
            out.append(float(raw[i]))
        except (TypeError, ValueError):
            out.append(None)
    if all(item is None for item in out):
        return None
    return out
def resolve_atom_radii(
    sources,
    n: int,
    atom_radius=DEFAULT_ATOM_RADIUS,
    radius_mode=DEFAULT_RADIUS_MODE,
    vdw_scale=DEFAULT_VDW_SCALE,
    point_radii=None,
    context=None,
) -> np.ndarray:
    """Per-point atom radii before adding the probe."""
    count = max(int(n), 0)
    out = np.full(count, float(atom_radius), dtype=float)
    if count == 0:
        return out
    if normalize_radius_mode(radius_mode) == "vdw" and sources:
        scale = float(vdw_scale) if vdw_scale else DEFAULT_VDW_SCALE
        for i, source in enumerate(sources):
            if i >= count:
                break
            vdw = lookup_source_vdw(source, context)
            if vdw is not None:
                out[i] = float(vdw) * scale
    custom = normalize_point_radii(point_radii, count)
    if custom:
        for i, value in enumerate(custom):
            if value is not None:
                out[i] = float(value)
    return out
def resolve_atom_elements(sources, n: int):
    """Per-point element symbols; carbon when the source has no identity."""
    count = max(int(n), 0)
    out = ["C"] * count
    if not sources:
        return out
    for i, source in enumerate(sources):
        if i >= count:
            break
        out[i] = element_from_atom_name(
            getattr(source, "name", ""),
            getattr(source, "elem", ""),
        )
    return out
def _quality_level(quality: int) -> int:
    return max(1, min(5, int(quality)))
def sas_spacing(quality: int) -> float:
    return float(SAS_SPACING[_quality_level(quality)])
def edt_spacing(quality: int, probe_radius: float) -> float:
    """Voxel size for the rolling-ball EDT.

    Independent of the SAS draft ladder. ``probe_radius`` is kept for callers.
    """
    _ = probe_radius
    return float(MC_SPACING[_quality_level(quality)])
def gauss_spacing(quality: int, resolution: float = DEFAULT_GAUSSIAN_RESOLUTION) -> float:
    """Voxel size for the PyMOL Gaussian map.

    Independent of the SAS draft ladder. ``resolution`` is kept for callers.
    """
    _ = resolution
    return float(GAUSS_SPACING[_quality_level(quality)])


# Seconds or voxel counts above these AABB estimates require confirmation.
HEAVY_SURFACE_SECONDS = 1.5
HEAVY_SURFACE_VOXELS = 4000000
# Cromer–Mann window splat, from ~800 carbons at 0.65 Å ≈ 0.13 s.
_PAINT_SEC_PER_WINDOW_VOXEL = 1.2e-7
_ISO_SEC_PER_VOXEL = 4.0e-8
_MC_SEC_PER_CUBE = 8.0e-7
_SASA_SEC_PER_FACE = 8.0e-6
_GAUSS_WINDOW_EXTENT = 2.8


def _bounded_brick_voxels(centers, spacing, pad, max_voxels=MAX_SAS_VOXELS, max_depth=8):
    """Voxel count after the same coarsen loop as ``_gauss_grid`` / ``_edt_ses_grid``."""
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    h = float(spacing)
    if h < 1e-8 or centers.shape[0] == 0:
        return 0, h
    span = np.max(centers, axis=0) - np.min(centers, axis=0) + 2.0 * float(pad)
    depth = 0
    while True:
        shape = np.floor(span / h).astype(np.int64) + 3
        shape = np.maximum(shape, 2)
        n_vox = int(shape[0] * shape[1] * shape[2])
        if n_vox <= int(max_voxels) or depth >= int(max_depth):
            return n_vox, h
        scale = (float(n_vox) / float(max_voxels)) ** (1.0 / 3.0)
        h = h * max(scale, 1.12)
        depth += 1
def estimate_surface_job(
    points,
    algorithm="GAUSS",
    quality=DEFAULT_QUALITY,
    atom_radius=DEFAULT_ATOM_RADIUS,
    probe_radius=DEFAULT_PROBE_RADIUS,
    elements=None,
) -> dict:
    """Cheap AABB estimate of work. Never builds a mesh."""
    centers = np.asarray(points, dtype=float).reshape(-1, 3)
    n = int(centers.shape[0])
    kind = normalize_algorithm(algorithm)
    q = _quality_level(quality)
    job = {
        "algorithm": kind,
        "quality": q,
        "n_atoms": n,
        "voxels": 0,
        "seconds": 0.0,
        "heavy": False,
    }
    if n == 0:
        return job
    if kind == "SASA":
        freq = float(_convex_cap_frequency(q))
        cap_faces = n * 20.0 * freq * freq
        contacts = min(n * 3.0, max(n - 1.0, 0.0))
        torus_faces = contacts * 2.0 * float(_torus_n_theta(q)) * float(_torus_n_phi_floor(q))
        seconds = (cap_faces + torus_faces) * _SASA_SEC_PER_FACE
    else:
        if kind == "MC":
            radii = expanded_radii(atom_radius, n, probe_radius)
            pad = float(np.max(radii)) + 2.0 * edt_spacing(q, probe_radius)
            spacing = edt_spacing(q, probe_radius)
            extent = pad
        else:
            spacing = gauss_spacing(q)
            pad = _GAUSS_WINDOW_EXTENT + 2.0 * spacing
            extent = _GAUSS_WINDOW_EXTENT
        n_vox, h = _bounded_brick_voxels(centers, spacing, pad)
        edge = max(1.0, 2.0 * extent / max(h, 1e-8) + 1.0)
        window = edge * edge * edge
        splat = float(n) * min(window, float(n_vox))
        cubes = min(float(MAX_SAS_CUBES), 0.05 * float(n_vox))
        seconds = (
            splat * _PAINT_SEC_PER_WINDOW_VOXEL
            + float(n_vox) * _ISO_SEC_PER_VOXEL
            + cubes * _MC_SEC_PER_CUBE
        )
        job["voxels"] = int(n_vox)
    job["seconds"] = float(seconds)
    job["heavy"] = bool(
        seconds >= HEAVY_SURFACE_SECONDS
        or int(job["voxels"]) >= HEAVY_SURFACE_VOXELS
    )
    return job
def surface_job_fingerprint(job) -> tuple:
    return (
        str(job.get("algorithm") or ""),
        int(job.get("quality") or 0),
        int(job.get("n_atoms") or 0),
        int(job.get("voxels") or 0),
        round(float(job.get("seconds") or 0.0), 1),
    )
def confirm_heavy_surface_job(job, previous_ok=None, previous_denied=None):
    """Return ``('allow'|'deny'|'ask', fingerprint)`` without showing UI."""
    fp = surface_job_fingerprint(job)
    if not job.get("heavy"):
        return "allow", fp
    if previous_ok == fp:
        return "allow", fp
    if previous_denied == fp:
        return "deny", fp
    return "ask", fp
def format_heavy_surface_message(job) -> str:
    label = SURFACE_ALGORITHM_LABELS.get(job.get("algorithm"), "surface")
    seconds = max(2, int(round(float(job.get("seconds") or 0.0))))
    n = int(job.get("n_atoms") or 0)
    q = int(job.get("quality") or 0)
    voxels = int(job.get("voxels") or 0)
    extra = " (%s voxels)" % format(voxels, ",") if voxels else ""
    return (
        "Building this %s mesh at quality %s for %s points may take about %s seconds%s. "
        "PyMOL will not respond until it finishes. Build it anyway?"
        % (label, q, n, seconds, extra)
    )
def expanded_radii(atom_radius, n: int, probe_radius: float) -> np.ndarray:
    radii = np.asarray(atom_radius, dtype=float).reshape(-1)
    if radii.size == 1:
        radii = np.repeat(radii, max(int(n), 1))
    elif radii.size != n:
        radii = np.resize(radii, n)
    return radii + float(probe_radius)
def signed_distance(xyz: np.ndarray, centers: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """min_i (|x - c_i| - R_i). Negative inside the union of spheres."""
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    radii = np.asarray(radii, dtype=float).reshape(-1)
    if xyz.shape[0] == 0 or centers.shape[0] == 0:
        return np.zeros((xyz.shape[0],), dtype=float)
    if centers.shape[0] == 1:
        return np.linalg.norm(xyz - centers[0], axis=1) - float(radii[0])
    if float(np.max(np.abs(radii - radii[0]))) < 1e-12:
        tree = cKDTree(centers)
        dist, idx = tree.query(xyz)
        return dist - radii[np.asarray(idx, dtype=int)]
    offset = xyz[:, None, :] - centers[None, :, :]
    return np.min(np.linalg.norm(offset, axis=2) - radii.reshape(1, -1), axis=1)
