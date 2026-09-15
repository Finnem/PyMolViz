"""Per-algorithm surface geometry parameter snapshots and visibility."""

from __future__ import annotations

from typing import Dict, Optional

from ...util.solvent_surface import (
    DEFAULT_ALGORITHM,
    DEFAULT_ATOM_RADIUS,
    DEFAULT_PROBE_RADIUS,
    DEFAULT_RADIUS_MODE,
    DEFAULT_VDW_SCALE,
    normalize_algorithm,
)

COLOR_MODE_UNIFORM = "uniform"
COLOR_MODE_PER_POINT = "per_point"
COLOR_MODE_FIELD = "field"

# Visible color-field controls per Appearance mode (hide irrelevant rows).
COLOR_MODE_CONTROL_VISIBILITY = {
    COLOR_MODE_UNIFORM: {"field": False, "colormap": False, "clim": False},
    COLOR_MODE_PER_POINT: {"field": False, "colormap": False, "clim": False},
    COLOR_MODE_FIELD: {"field": True, "colormap": True, "clim": True},
}

# Visible geometry controls per algorithm (hide irrelevant rows entirely).
ALGORITHM_CONTROL_VISIBILITY = {
    "GAUSS": {"atom_radius": True, "probe": False, "vdw": True},
    "MC": {"atom_radius": True, "probe": True, "vdw": True},
    "SASA": {"atom_radius": True, "probe": True, "vdw": True},
}


def algorithm_shows_probe(algorithm: str) -> bool:
    key = normalize_algorithm(algorithm)
    return bool(ALGORITHM_CONTROL_VISIBILITY.get(key, {}).get("probe", True))


def algorithm_shows_atom_radius(algorithm: str) -> bool:
    key = normalize_algorithm(algorithm)
    return bool(ALGORITHM_CONTROL_VISIBILITY.get(key, {}).get("atom_radius", True))


def algorithm_shows_vdw(algorithm: str) -> bool:
    key = normalize_algorithm(algorithm)
    return bool(ALGORITHM_CONTROL_VISIBILITY.get(key, {}).get("vdw", True))


def color_mode_shows(mode, control) -> bool:
    key = str(mode or COLOR_MODE_UNIFORM)
    return bool(COLOR_MODE_CONTROL_VISIBILITY.get(key, {}).get(control, False))


def default_param_snapshot() -> Dict[str, dict]:
    base = {
        "atom_radius": float(DEFAULT_ATOM_RADIUS),
        "probe": float(DEFAULT_PROBE_RADIUS),
        "radius_mode": str(DEFAULT_RADIUS_MODE),
        "vdw_scale": float(DEFAULT_VDW_SCALE),
    }
    return {algo: dict(base) for algo in ALGORITHM_CONTROL_VISIBILITY}


class SurfaceParamSnapshots:
    """Store per-algorithm values so switching GAUSS ↔ SAS does not wipe fields."""

    def __init__(self):
        self._by_algo = default_param_snapshot()
        self._algorithm = DEFAULT_ALGORITHM

    @property
    def algorithm(self) -> str:
        return self._algorithm

    def set_algorithm(self, algorithm: str) -> None:
        self._algorithm = normalize_algorithm(algorithm)

    def snapshot_from(
        self,
        algorithm: str,
        *,
        atom_radius: float,
        probe: float,
        radius_mode: str,
        vdw_scale: float,
    ) -> None:
        key = normalize_algorithm(algorithm)
        self._by_algo[key] = {
            "atom_radius": float(atom_radius),
            "probe": float(probe),
            "radius_mode": str(radius_mode),
            "vdw_scale": float(vdw_scale),
        }

    def restore(self, algorithm: Optional[str] = None) -> dict:
        key = normalize_algorithm(algorithm or self._algorithm)
        return dict(self._by_algo.get(key, default_param_snapshot()[key]))
