"""Who uses a Field: visuals (geometry/color) and meshes with field_id."""

from __future__ import annotations

from typing import Iterable, List, Optional


def field_identity_keys(obj) -> List[str]:
    """Ids and names used to match a field to its visuals."""
    from ..util.field_sample import PYMOL_MAP_ID_PREFIX, field_label, resolve_grid

    keys = []
    seen = set()

    def _add(value):
        text = str(value or "")
        if not text or text in seen:
            return
        seen.add(text)
        keys.append(text)

    _add(getattr(obj, "id", None))
    label = field_label(obj)
    _add(label)
    if label:
        _add(PYMOL_MAP_ID_PREFIX + label)
    generator = getattr(obj, "generator", None) or {}
    if str(generator.get("type") or "") == "pymol_map" and generator.get("map_name"):
        map_name = str(generator["map_name"])
        _add(map_name)
        _add(PYMOL_MAP_ID_PREFIX + map_name)
    grid = resolve_grid(obj)
    if grid is not None and grid is not obj:
        _add(getattr(grid, "id", None))
        nested = field_label(grid)
        _add(nested)
        if nested:
            _add(PYMOL_MAP_ID_PREFIX + nested)
    return keys


def _obj_id(obj) -> str:
    return str(getattr(obj, "id", "") or "")


def _display_name(obj) -> str:
    stored = getattr(obj, "_name", None)
    if stored:
        return str(stored)
    return str(getattr(obj, "name", "") or "")


def _mesh_children(obj) -> list:
    from ..meshes.CGOCollection import cgo_children

    return cgo_children(obj)


def referenced_field_ids(obj) -> List[str]:
    """Field ids this object depends on (geometry, color, or mesh sampling)."""
    ids = []
    seen = set()

    def _add(value):
        text = str(value or "")
        if not text or text in seen:
            return
        seen.add(text)
        ids.append(text)

    _add(getattr(obj, "geometry_field_id", None))
    _add(getattr(obj, "color_field_id", None))
    _add(getattr(obj, "field_id", None))
    grid = getattr(obj, "grid_data", None)
    if grid is not None:
        for key in field_identity_keys(grid):
            _add(key)
    for child in _mesh_children(obj):
        if child is obj:
            continue
        for fid in referenced_field_ids(child):
            _add(fid)
    return ids


def field_identity_set(field) -> set:
    keys = set()
    oid = _obj_id(field)
    if oid:
        keys.add(oid)
    grid = getattr(field, "grid_data", None)
    if grid is not None:
        gid = _obj_id(grid)
        if gid:
            keys.add(gid)
    gen = getattr(field, "generator", None) or {}
    if str(gen.get("type") or "") == "pymol_map":
        map_name = str(gen.get("map_name") or "")
        if map_name:
            keys.add(map_name)
            keys.add("pymol_map:" + map_name)
    return keys


def dependents_of_field(field, objects: Optional[Iterable] = None) -> List[object]:
    """Visuals and objects that reference ``field``. Does not include the Field itself."""
    if objects is None:
        try:
            from ..runtime.session import all_objects

            objects = all_objects()
        except Exception:
            objects = []
    keys = field_identity_set(field)
    field_id = _obj_id(field)
    out = []
    seen = set()
    for obj in objects:
        if obj is field:
            continue
        oid = _obj_id(obj)
        if field_id and oid == field_id:
            continue
        if type(obj).__name__ == "Field":
            continue
        refs = set(referenced_field_ids(obj))
        if not (refs & keys):
            continue
        marker = oid or id(obj)
        if marker in seen:
            continue
        seen.add(marker)
        out.append(obj)
    return out


def dependent_labels(dependents) -> List[str]:
    labels = []
    for obj in dependents or ():
        name = _display_name(obj) or type(obj).__name__
        labels.append("%s (%s)" % (name, type(obj).__name__))
    return labels
