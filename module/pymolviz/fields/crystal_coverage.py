"""Place a map on the crystal lattice so a selection lies inside the brick.

Repeating physics is integer unit-cell translations (and space-group
operators when available), not a free Cartesian drag. If one translated
copy of the stored brick cannot cover the selection, neighboring cells are
tiled by wrapping sample coordinates through those operators.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .identity import GEN_IMPORTED, GEN_PYMOL_MAP

_DEG = np.pi / 180.0
_EPS = 1e-6
_MAX_VOXELS = 192 ** 3
_SHIFT_SEARCH = 2
# PyMOL reports this for objects with no CRYST1; it is not a real unit cell.
_DUMMY_CELL_LENGTH = 1.0


from .crystal_cell import CrystalError, map_world_aabb, point_coverage_status


def selection_points(cmd, selection=None) -> np.ndarray:
    if cmd is None:
        raise CrystalError("No PyMOL session to read a selection from.")
    candidates = []
    if selection:
        candidates.append(str(selection))
    candidates.extend(("sele", "(sele)"))
    try:
        names = [str(n) for n in cmd.get_names("objects", 1)]
    except Exception:
        names = []
    for name in names:
        try:
            typ = str(cmd.get_type(name) or "")
        except Exception:
            typ = ""
        if "molecule" in typ.lower():
            candidates.append(name)
    candidates.append("all")
    seen = set()
    for expr in candidates:
        if expr in seen:
            continue
        seen.add(expr)
        try:
            n = int(cmd.count_atoms(expr) or 0)
        except Exception:
            continue
        if n <= 0:
            continue
        try:
            coords = cmd.get_coords(expr)
        except Exception:
            coords = None
        pts = np.asarray(coords if coords is not None else [], dtype=float)
        if pts.size == 0:
            continue
        return pts.reshape(-1, 3)
    raise CrystalError("Select some atoms first (the named selection \"sele\").")


_SKIP_EXTEND_PREFIXES = ("_pmv_",)
_SKIP_EXTEND_NAMES = frozenset({"pmv_camera_center"})
_SKIP_EXTEND_TYPE_TOKENS = (
    "map", "volume", "mesh", "cgo", "ramp", "surface", "callback", "gadget", "group",
)
_COVERAGE_SORT = {"outside": 0, "partial": 1, "inside": 2, "empty": 3}


def _is_extend_skip_name(name: str) -> bool:
    text = str(name or "")
    if text in _SKIP_EXTEND_NAMES:
        return True
    return any(text.startswith(prefix) for prefix in _SKIP_EXTEND_PREFIXES)


def _object_is_coordinate_target(cmd, name: str) -> bool:
    try:
        typ = str(cmd.get_type(name) or "").lower()
    except Exception:
        typ = ""
    if any(token in typ for token in _SKIP_EXTEND_TYPE_TOKENS):
        return False
    return True


def _coords_for_name(cmd, name):
    try:
        n = int(cmd.count_atoms(name) or 0)
    except Exception:
        n = 0
    if n <= 0:
        return None
    try:
        coords = cmd.get_coords(name)
    except Exception:
        coords = None
    pts = np.asarray(coords if coords is not None else [], dtype=float)
    if pts.size == 0 or not np.all(np.isfinite(pts)):
        return None
    return pts.reshape(-1, 3)


def iter_extend_target_names(cmd) -> List[Tuple[str, str]]:
    """``(kind, name)`` for molecule objects and named selections with atoms."""
    found = []
    seen = set()
    try:
        objects = [str(n) for n in cmd.get_names("objects") or []]
    except Exception:
        objects = []
    for name in objects:
        if not name or name in seen or _is_extend_skip_name(name):
            continue
        if not _object_is_coordinate_target(cmd, name):
            continue
        seen.add(name)
        found.append(("object", name))
    try:
        selections = [str(n) for n in cmd.get_names("selections") or []]
    except Exception:
        selections = []
    for name in selections:
        if not name or name in seen or _is_extend_skip_name(name):
            continue
        seen.add(name)
        found.append(("selection", name))
    return found


def extend_target_rows(cmd, grid, skip_names=()) -> List[dict]:
    """Coverage of each object/selection against the current map brick."""
    skip = {str(n) for n in (skip_names or ()) if n}
    rows = []
    if cmd is None or grid is None:
        return rows
    for kind, name in iter_extend_target_names(cmd):
        if name in skip:
            continue
        pts = _coords_for_name(cmd, name)
        if pts is None:
            continue
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        rows.append({
            "kind": kind,
            "name": name,
            "status": point_coverage_status(grid, pts),
            "n_atoms": int(pts.shape[0]),
            "lo": lo.tolist(),
            "hi": hi.tolist(),
        })
    rows.sort(
        key=lambda row: (
            0 if row["kind"] == "object" else 1,
            _COVERAGE_SORT.get(row["status"], 9),
            str(row["name"]).lower(),
        )
    )
    return rows


def default_extend_target_name(rows) -> Optional[str]:
    rows = list(rows or [])
    if not rows:
        return None
    for row in rows:
        if row.get("name") in ("sele", "(sele)") and row.get("status") != "empty":
            return str(row["name"])
    for row in rows:
        if row.get("status") in ("outside", "partial"):
            return str(row["name"])
    return str(rows[0]["name"])

