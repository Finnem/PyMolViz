"""From-selection Field algorithm labels and control visibility (no Qt)."""

from __future__ import annotations

from ...fields.identity import (
    GEN_DISTANCE,
    GEN_GAUSSIAN,
    GEN_NEAREST_PROP,
    GEN_SIGNED_VDW,
)
from ...fields.domain import (
    BOUNDS_AROUND_SELECTION,
    BOUNDS_CUSTOM_BOX,
    BOUNDS_MODES,
    BOUNDS_OBJECT,
)

FIELD_MODEL_ORDER = (
    GEN_GAUSSIAN,
    GEN_DISTANCE,
    GEN_SIGNED_VDW,
    GEN_NEAREST_PROP,
)

FIELD_MODEL_LABELS = {
    GEN_GAUSSIAN: "Gaussian atoms",
    GEN_DISTANCE: "Distance to closest atom",
    GEN_SIGNED_VDW: "Signed VDW distance",
    GEN_NEAREST_PROP: "Nearest-atom property",
}

FIELD_CONTROL_VISIBILITY = {
    GEN_GAUSSIAN: {
        "quality": True,
        "resolution": True,
        "property": False,
        "iso_value": True,
        "live_preview": True,
    },
    GEN_DISTANCE: {
        "quality": False,
        "resolution": False,
        "property": False,
        "iso_value": True,
        "live_preview": True,
    },
    GEN_SIGNED_VDW: {
        "quality": False,
        "resolution": False,
        "property": False,
        "iso_value": True,
        "live_preview": True,
    },
    GEN_NEAREST_PROP: {
        "quality": False,
        "resolution": False,
        "property": True,
        "iso_value": True,
        "live_preview": True,
    },
}

# Distance iso=0 never crosses zero (values are >= 0), so the default is a 1.5 Å shell.
DEFAULT_DISTANCE_ISOLEVEL = 1.5

NEAREST_PROPERTIES = (
    ("b_factor", "B-factor"),
    ("occupancy", "Occupancy"),
    ("elem", "Element"),
    ("chain", "Chain"),
    ("name", "Atom name"),
)

BOUNDS_MODE_LABELS = {
    BOUNDS_AROUND_SELECTION: "Around selection",
    BOUNDS_OBJECT: "Object extent",
    BOUNDS_CUSTOM_BOX: "Custom box",
}

CATEGORICAL_PROPERTIES = frozenset({"elem", "element", "chain", "name", "resn", "resi"})


def normalize_field_model(algorithm) -> str:
    text = str(algorithm or GEN_GAUSSIAN).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "gaussian": GEN_GAUSSIAN,
        "gauss": GEN_GAUSSIAN,
        "implicit": GEN_GAUSSIAN,
        "distance": GEN_DISTANCE,
        "signed_vdw": GEN_SIGNED_VDW,
        "vdw": GEN_SIGNED_VDW,
        "nearest": GEN_NEAREST_PROP,
        "property": GEN_NEAREST_PROP,
    }
    text = aliases.get(text, text)
    if text in FIELD_MODEL_LABELS:
        return text
    return GEN_GAUSSIAN


def field_model_shows(algorithm, control) -> bool:
    key = normalize_field_model(algorithm)
    return bool(FIELD_CONTROL_VISIBILITY.get(key, {}).get(control, False))


def default_preset_iso_level(algorithm, grid=None) -> float:
    """IsoSurface level when From Selection commits a preset visual.

    Gaussian matches PyMOL ``map_new gaussian`` (1.0). Distance and signed-VDW
    contour at 0 (atom centers / VDW surface). Nearest-atom property uses the
    grid mean when ``grid`` is given, otherwise 0.

    Live preview uses :func:`default_field_iso_level`, which raises Distance
    to 1.5 Å so the marching-cubes iso is not degenerate.
    """
    from ...util.gaussian_map import DEFAULT_GAUSSIAN_ISOLEVEL

    algo = normalize_field_model(algorithm)
    if algo == GEN_GAUSSIAN:
        return float(DEFAULT_GAUSSIAN_ISOLEVEL)
    if algo == GEN_NEAREST_PROP and grid is not None:
        values = getattr(grid, "values", None)
        if values is not None:
            import numpy as np

            arr = np.asarray(values, dtype=float).reshape(-1)
            if arr.size:
                return float(np.mean(arr))
    return 0.0


def iso_spin_range(algorithm):
    """``(min, max, step, decimals)`` for the From Selection iso knob."""
    algo = normalize_field_model(algorithm)
    if algo == GEN_GAUSSIAN:
        return (0.05, 8.00, 0.05, 2)
    if algo == GEN_DISTANCE:
        return (0.0, 50.0, 0.1, 2)
    if algo == GEN_SIGNED_VDW:
        return (-20.0, 20.0, 0.1, 2)
    return (-1.0e4, 1.0e4, 0.1, 2)


def property_is_categorical(key) -> bool:
    return str(key or "").strip().lower() in CATEGORICAL_PROPERTIES
