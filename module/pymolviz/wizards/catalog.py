"""Classify session visuals for the wizard library tables (no Qt)."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from ..points import iter_point_sources

FIELD_TYPES = frozenset({
    "Volume",
    "IsoVolume",
    "IsoSurface",
    "IsoMesh",
    "GridData",
})

_TYPE_LABELS = {
    "CenteredBox": "Box",
    "CGOCollection": "Collection",
    "PolylineTube": "Tube",
    "Rotation_Indicator": "Rotation",
}

_EDITOR_TYPES = {
    "Sphere": "Sphere",
    "CenteredBox": "Box",
    "Arrows": "Arrows",
}


def is_field(obj) -> bool:
    return type(obj).__name__ in FIELD_TYPES


def display_name(obj) -> str:
    stored = getattr(obj, "_name", None)
    if stored:
        return str(stored)
    return str(getattr(obj, "name", "") or "")


def type_label(obj) -> str:
    children = mesh_children(obj)
    if type(obj).__name__ == "CGOCollection" and children:
        kinds = []
        seen = set()
        for child in children:
            label = _TYPE_LABELS.get(type(child).__name__, type(child).__name__)
            if label not in seen:
                seen.add(label)
                kinds.append(label)
        if len(kinds) == 1:
            return kinds[0]
        if kinds:
            return "Mixed"
    return _TYPE_LABELS.get(type(obj).__name__, type(obj).__name__)


def mesh_children(obj) -> list:
    if type(obj).__name__ == "CGOCollection":
        return list(obj)
    return [obj]


def editor_kind(obj) -> Optional[str]:
    """Builder page to open, or None if this visual cannot be edited yet."""
    kinds = {type(child).__name__ for child in mesh_children(obj)}
    if len(kinds) != 1:
        return None
    name = next(iter(kinds))
    return _EDITOR_TYPES.get(name)


def point_count(obj) -> int:
    children = mesh_children(obj)
    if not children:
        return 0
    kind = type(children[0]).__name__
    if kind == "Arrows":
        starts = getattr(children[0], "_start_sources", None) or getattr(children[0], "starts", None) or ()
        return len(list(starts))
    if kind in ("Sphere", "CenteredBox"):
        return len(children)
    sources = list(iter_point_sources(obj))
    if sources:
        return len(sources)
    vertices = getattr(obj, "vertices", None)
    if vertices is not None:
        try:
            return int(len(vertices))
        except TypeError:
            return 0
    return 1


def center_of_mass(obj) -> Optional[Tuple[float, float, float]]:
    points = _xyz_samples(obj)
    if not points:
        return None
    n = float(len(points))
    return (
        sum(p[0] for p in points) / n,
        sum(p[1] for p in points) / n,
        sum(p[2] for p in points) / n,
    )


def format_com(xyz: Optional[Sequence[float]]) -> str:
    if xyz is None or len(xyz) < 3:
        return ""
    return "%.2f, %.2f, %.2f" % (float(xyz[0]), float(xyz[1]), float(xyz[2]))


def object_row(obj) -> dict:
    return {
        "id": str(obj.id),
        "name": display_name(obj),
        "type": type_label(obj),
        "n_points": point_count(obj),
        "com": format_com(center_of_mass(obj)),
        "editor": editor_kind(obj),
    }


def object_rows(objects: Iterable) -> List[dict]:
    return [object_row(obj) for obj in objects if not is_field(obj)]


def _xyz_of_source(src) -> Optional[Tuple[float, float, float]]:
    last = getattr(src, "last_xyz", None)
    if last is not None:
        try:
            return (float(last[0]), float(last[1]), float(last[2]))
        except (TypeError, IndexError, ValueError):
            pass
    try:
        xyz = src.resolve(None)
        return (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    except Exception:
        return None


def _xyz_samples(obj) -> List[Tuple[float, float, float]]:
    samples = []
    for src in iter_point_sources(obj):
        xyz = _xyz_of_source(src)
        if xyz is not None:
            samples.append(xyz)
    if samples:
        return samples
    vertices = getattr(obj, "vertices", None)
    if vertices is None and type(obj).__name__ == "CGOCollection":
        for child in obj:
            verts = getattr(child, "vertices", None)
            if verts is None:
                continue
            try:
                for row in verts:
                    samples.append((float(row[0]), float(row[1]), float(row[2])))
            except (TypeError, IndexError, ValueError):
                continue
        return samples
    if vertices is None:
        return samples
    try:
        for row in vertices:
            samples.append((float(row[0]), float(row[1]), float(row[2])))
    except (TypeError, IndexError, ValueError):
        return samples
    return samples
