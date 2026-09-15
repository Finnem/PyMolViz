"""Classify session visuals for the wizard library tables (no Qt)."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from ..points import iter_point_sources
from .widgets.theme import PRIMARY, TREE_LINE

FIELD_SOURCE_TYPES = frozenset({"GridData", "Field"})
FIELD_VISUAL_TYPES = frozenset({
    "Volume",
    "IsoVolume",
    "IsoSurface",
    "IsoMesh",
})
FIELD_TYPES = FIELD_SOURCE_TYPES | FIELD_VISUAL_TYPES

KIND_FIELD = "field"
KIND_VISUAL = "visual"
KIND_ADD_VISUAL = "add_visual"
ADD_VISUAL_ROW_PREFIX = "__pmv_add_visual__:"
NEST_INDENT_PX = 26
NEST_LINE_RGB = TREE_LINE
NEST_NONE = "none"
NEST_TEE = "tee"
NEST_ELL = "ell"

# Pastel pill fill + ink, sampled from the Fields catalog mock.
_KIND_BADGE_RGB = {
    "Distance": ((186, 232, 214), (22, 110, 78)),
    "Gaussian": ((198, 226, 248), (28, 90, 160)),
    "Map": ((186, 214, 246), (24, 82, 158)),
    "Grid": ((210, 222, 236), (70, 88, 108)),
    "Signed VDW": ((186, 232, 214), (22, 110, 78)),
    "Atom property": ((232, 220, 246), (88, 64, 140)),
    "Atom color": ((232, 220, 246), (88, 64, 140)),
    "Derived": ((226, 232, 238), (80, 90, 102)),
    "Gradient": ((210, 222, 236), (70, 88, 108)),
    "Field": ((210, 222, 236), (70, 88, 108)),
    "IsoSurface": ((252, 226, 196), (176, 88, 28)),
    "IsoMesh": ((252, 232, 176), (148, 108, 28)),
    "Volume": ((220, 210, 246), (88, 64, 148)),
    "IsoVolume": ((206, 228, 248), (40, 88, 150)),
    "Spheres": ((198, 226, 248), (28, 90, 160)),
    "Boxes": ((210, 222, 236), (70, 88, 108)),
    "Surface": ((252, 226, 196), (176, 88, 28)),
    "Arrows": ((186, 232, 214), (22, 110, 78)),
    "Collection": ((226, 232, 238), (80, 90, 102)),
    "Tube": ((206, 228, 248), (40, 88, 150)),
    "Rotation": ((232, 220, 246), (88, 64, 140)),
    "Mixed": ((226, 232, 238), (80, 90, 102)),
}
_KIND_BADGE_FALLBACK = ((226, 232, 238), (80, 90, 102))
_VISUAL_ICON_KIND = {
    "IsoSurface": "surface",
    "IsoMesh": "isomesh",
    "Volume": "volume",
    "IsoVolume": "volume",
    "Spheres": "sphere",
    "Boxes": "cube",
    "Surface": "surface",
    "Arrows": "arrow",
}
_VISUAL_ICON_RGB = {
    "IsoSurface": (248, 152, 16),
    "IsoMesh": (40, 168, 88),
    "Volume": PRIMARY,
    "IsoVolume": (56, 168, 196),
    "Spheres": PRIMARY,
    "Boxes": (70, 88, 108),
    "Surface": (248, 152, 16),
    "Arrows": (40, 168, 88),
}
_TYPE_CARD_ICON_RGB = {
    "sphere": PRIMARY,
    "cube": (70, 88, 108),
    "surface": (248, 152, 16),
    "arrow": (40, 168, 88),
    "volume": PRIMARY,
    "isomesh": (40, 168, 88),
}

_GENERATOR_TYPE_LABELS = {
    "imported": "Grid",
    "pymol_map": "Map",
    "gaussian_atoms": "Gaussian",
    "distance_to_atoms": "Distance",
    "signed_vdw_distance": "Signed VDW",
    "nearest_atom_property": "Atom property",
    "nearest_atom_color": "Atom color",
    "derived": "Derived",
    "gradient": "Gradient",
}

_TYPE_LABELS = {
    "Sphere": "Spheres",
    "CenteredBox": "Boxes",
    "Surface": "Surface",
    "CGOCollection": "Collection",
    "PolylineTube": "Tube",
    "Rotation_Indicator": "Rotation",
    "Field": "Field",
    "Lines": "Arrows",
}

_EDITOR_TYPES = {
    "Sphere": "Sphere",
    "CenteredBox": "Box",
    "Surface": "Surface",
    "Arrows": "Arrows",
}

_FIELD_EDITOR_TYPES = {
    "Volume": "Volume",
    "IsoVolume": "IsoVolume",
    "IsoSurface": "IsoSurface",
    "IsoMesh": "IsoMesh",
}


def is_field(obj) -> bool:
    return type(obj).__name__ in FIELD_TYPES


def is_field_source(obj) -> bool:
    return type(obj).__name__ in FIELD_SOURCE_TYPES


def is_field_visual(obj) -> bool:
    return type(obj).__name__ in FIELD_VISUAL_TYPES


def is_native_field(obj) -> bool:
    from ..util.field_sample import PYMOL_MAP_ID_PREFIX

    oid = str(getattr(obj, "id", "") or "")
    return oid.startswith(PYMOL_MAP_ID_PREFIX)


def add_visual_row_id(field_id) -> str:
    return ADD_VISUAL_ROW_PREFIX + str(field_id)


def parse_add_visual_row_id(marker):
    text = str(marker or "")
    if not text.startswith(ADD_VISUAL_ROW_PREFIX):
        return None
    return text[len(ADD_VISUAL_ROW_PREFIX):]


def row_indent_px(row) -> int:
    return NEST_INDENT_PX * int((row or {}).get("depth", 0) or 0)


def nest_connector(row) -> str:
    """Tree glyph for a library row: none on fields, tee/L on nested children."""
    if int((row or {}).get("depth", 0) or 0) <= 0:
        return NEST_NONE
    if (row or {}).get("last_child"):
        return NEST_ELL
    return NEST_TEE


def field_row_shows_edit(row) -> bool:
    """True when the library should show an Edit control for this row."""
    data = row or {}
    kind = data.get("kind")
    if kind in (KIND_VISUAL, KIND_FIELD):
        return bool(data.get("editor"))
    return False


def field_row_shows_symmetrize(row) -> bool:
    """True when a stored map can be lattice-wrapped onto the current selection."""
    data = row or {}
    return data.get("kind") == KIND_FIELD and bool(data.get("symmetrize"))


def kind_badge_rgb(type_label) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Pastel (fill, ink) for a Kind chip. Unknown labels use a slate fallback."""
    key = str(type_label or "").strip()
    return _KIND_BADGE_RGB.get(key, _KIND_BADGE_FALLBACK)


def visual_icon_kind(type_label) -> Optional[str]:
    """Glyph key for a nested visual row, or None when the row has no icon."""
    return _VISUAL_ICON_KIND.get(str(type_label or "").strip())


def visual_icon_rgb(type_label) -> Optional[Tuple[int, int, int]]:
    return _VISUAL_ICON_RGB.get(str(type_label or "").strip())


def type_card_icon_rgb(icon_key) -> Optional[Tuple[int, int, int]]:
    """Glyph color for a type-picker card (sphere, cube, surface, …)."""
    return _TYPE_CARD_ICON_RGB.get(str(icon_key or "").strip())


def field_library_detail_text(row) -> str:
    """Used-by (fields) or Geometry / Color (visuals) for the shared detail column."""
    data = row or {}
    kind = data.get("kind")
    if kind == KIND_FIELD:
        return str(data.get("used_by") or "")
    if kind != KIND_VISUAL:
        return ""
    geom = str(data.get("geometry_field") or "").strip()
    color = str(data.get("color_field") or "").strip()
    if geom and color and color != geom:
        return "Geometry: %s · Color: %s" % (geom, color)
    if geom:
        return "Geometry: %s" % geom
    if color:
        return "Color: %s" % color
    return ""


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
    if kind == "Surface":
        sources = getattr(children[0], "point_sources", None) or ()
        return len(list(sources))
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


def accent_rgb(obj) -> Optional[Tuple[float, float, float]]:
    """Representative RGB for a library swatch, or None if unknown."""
    children = mesh_children(obj)
    if not children:
        return None
    mesh = children[0]
    if getattr(mesh, "bypass_colormap", False):
        rgb = _first_rgb(getattr(mesh, "color", None))
        if rgb is not None:
            return rgb
    getter = getattr(mesh, "_cgo_vertex_rgb", None)
    if callable(getter):
        try:
            colors = getter()
            row = colors[0]
            return (float(row[0]), float(row[1]), float(row[2]))
        except Exception:
            pass
    return _first_rgb(getattr(mesh, "color", None))


def object_row(obj) -> dict:
    return {
        "id": str(obj.id),
        "name": display_name(obj),
        "type": type_label(obj),
        "n_points": point_count(obj),
        "com": format_com(center_of_mass(obj)),
        "editor": editor_kind(obj),
        "color": accent_rgb(obj),
    }


def object_rows(objects: Iterable) -> List[dict]:
    return [object_row(obj) for obj in objects if not is_field(obj)]


def field_type_label(obj) -> str:
    if is_native_field(obj):
        return "Map"
    generator = getattr(obj, "generator", None) or {}
    gen_type = str(generator.get("type") or "")
    if gen_type in _GENERATOR_TYPE_LABELS:
        return _GENERATOR_TYPE_LABELS[gen_type]
    if is_field_source(obj):
        return "Grid"
    return type(obj).__name__


def field_editor_kind(obj) -> Optional[str]:
    return _FIELD_EDITOR_TYPES.get(type(obj).__name__)


def field_source_editor_kind(obj) -> Optional[str]:
    """Builder for a reusable Field (From Selection), or None for imported maps."""
    if type(obj).__name__ != "Field":
        return None
    from ..fields.identity import (
        GEN_DISTANCE,
        GEN_GAUSSIAN,
        GEN_NEAREST_COLOR,
        GEN_NEAREST_PROP,
        GEN_SIGNED_VDW,
    )

    gen = str((getattr(obj, "generator", None) or {}).get("type") or "")
    if gen in (
        GEN_GAUSSIAN,
        GEN_DISTANCE,
        GEN_SIGNED_VDW,
        GEN_NEAREST_PROP,
        GEN_NEAREST_COLOR,
    ):
        return "FromSelection"
    return None


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


def visual_field_keys(obj) -> List[str]:
    keys = []
    seen = set()

    def _add(value):
        text = str(value or "")
        if not text or text in seen:
            return
        seen.add(text)
        keys.append(text)

    _add(getattr(obj, "geometry_field_id", None))
    _add(getattr(obj, "color_field_id", None))
    _add(getattr(obj, "field_id", None))
    grid = getattr(obj, "grid_data", None)
    if grid is not None:
        for key in field_identity_keys(grid):
            _add(key)
    return keys


def field_row(obj, used_by=None) -> dict:
    from ..fields.crystal import field_supports_symmetrize
    from ..util.field_sample import field_label

    return {
        "kind": KIND_FIELD,
        "id": str(getattr(obj, "id", "") or ""),
        "name": field_label(obj),
        "type": field_type_label(obj),
        "native": is_native_field(obj),
        "depth": 0,
        "editor": field_source_editor_kind(obj),
        "symmetrize": field_supports_symmetrize(obj),
        "last_child": False,
        "used_by": used_by or "",
        "geometry_field": "",
        "color_field": "",
    }


def field_visual_row(obj, field_id, last_child=False, geometry_field="", color_field="") -> dict:
    geom = geometry_field or str(getattr(obj, "geometry_field_id", "") or field_id or "")
    color = color_field or str(getattr(obj, "color_field_id", "") or "")
    return {
        "kind": KIND_VISUAL,
        "id": str(getattr(obj, "id", "") or ""),
        "field_id": str(field_id),
        "name": display_name(obj),
        "type": type(obj).__name__,
        "native": False,
        "depth": 1,
        "editor": field_editor_kind(obj),
        "last_child": bool(last_child),
        "used_by": "",
        "geometry_field": geom,
        "color_field": color,
    }


def add_visual_row(field_id) -> dict:
    fid = str(field_id)
    return {
        "kind": KIND_ADD_VISUAL,
        "id": add_visual_row_id(fid),
        "field_id": fid,
        "name": "+ Add Visual",
        "type": "",
        "native": False,
        "depth": 1,
        "editor": None,
        "last_child": True,
        "used_by": "",
        "geometry_field": "",
        "color_field": "",
    }


KIND_CMAP = "map"
KIND_CMAP_HEADER = "header"
KIND_CMAP_USER = "user"
CMAP_SECTION_SESSION = "In this session"
CMAP_SECTION_UNUSED = "Unused"


def colormap_names_on_object(obj) -> List[str]:
    """Named colormaps referenced by a session visual (not field sources)."""
    from ..util.colormap_spec import named_colormap_from_attrs

    names = []
    seen = set()

    def add(colormap, spec):
        text = named_colormap_from_attrs(colormap, spec)
        if not text or text in seen:
            return
        seen.add(text)
        names.append(text)

    targets = list(mesh_children(obj))
    if obj not in targets:
        targets.append(obj)
    for target in targets:
        add(getattr(target, "colormap", None), getattr(target, "colormap_spec", None))
        add(getattr(target, "field_colormap", None), getattr(target, "field_colormap_spec", None))
    return names


def session_colormap_users(objects=None) -> dict:
    """``{preset: [{"name", "type"}, ...]}`` for visuals in the current session."""
    if objects is None:
        from ..runtime.session import all_objects, is_ephemeral

        objects = [obj for obj in all_objects() if not is_ephemeral(obj)]
    users = {}
    for obj in objects:
        if is_field_source(obj):
            continue
        label = display_name(obj) or str(getattr(obj, "id", "") or "")
        kind = type_label(obj)
        for cmap in colormap_names_on_object(obj):
            bucket = users.setdefault(cmap, [])
            if any(item.get("name") == label for item in bucket):
                continue
            bucket.append({"name": label, "type": kind})
    return users


def colormap_catalog_rows(custom_rows, users_by_name=None, expanded=()) -> List[dict]:
    """Custom maps with session users first, then unused, plus optional child rows."""
    usage = users_by_name or {}
    open_names = {str(name) for name in (expanded or ())}
    used = []
    unused = []
    for row in custom_rows or ():
        name = str((row or {}).get("name") or "")
        if not name:
            continue
        users = list(usage.get(name) or [])
        entry = {
            "kind": KIND_CMAP,
            "name": name,
            "definition": (row or {}).get("definition") or {},
            "users": users,
            "used": bool(users),
            "custom": True,
        }
        (used if users else unused).append(entry)
    used.sort(key=lambda item: (-len(item["users"]), item["name"].lower()))
    unused.sort(key=lambda item: item["name"].lower())
    out = []
    if used:
        out.append({"kind": KIND_CMAP_HEADER, "name": CMAP_SECTION_SESSION})
        for entry in used:
            out.append(entry)
            if entry["name"] in open_names:
                for index, user in enumerate(entry["users"]):
                    out.append({
                        "kind": KIND_CMAP_USER,
                        "name": str(user.get("name") or ""),
                        "type": str(user.get("type") or ""),
                        "parent": entry["name"],
                        "last_child": index == len(entry["users"]) - 1,
                    })
    if unused:
        out.append({"kind": KIND_CMAP_HEADER, "name": CMAP_SECTION_UNUSED})
        out.extend(unused)
    return out


def _field_label_by_id(field_id, fields) -> str:
    from ..util.field_sample import field_label

    want = str(field_id or "")
    if not want:
        return ""
    for field in fields:
        if str(getattr(field, "id", "") or "") == want:
            return field_label(field)
    return want


def _used_by_text(field, children, objects) -> str:
    n_vis = len(children)
    keys = set(field_identity_keys(field))
    extra = 0
    for obj in objects:
        if is_field(obj) or is_field_visual(obj):
            continue
        hit = False
        for child in mesh_children(obj):
            fid = getattr(child, "field_id", None)
            if fid and str(fid) in keys:
                hit = True
                break
        if not hit:
            fid = getattr(obj, "field_id", None)
            if fid and str(fid) in keys:
                hit = True
        if hit:
            extra += 1
    parts = [
        "%d visual%s" % (n_vis, "" if n_vis == 1 else "s"),
    ]
    if extra:
        parts.append("%d object%s" % (extra, "" if extra == 1 else "s"))
    return ", ".join(parts)


def field_library_rows(objects: Optional[Iterable] = None, cmd=None) -> List[dict]:
    """Nested library rows: each field, its visuals, then an Add Visual control."""
    from ..util.field_sample import discover_fields, field_label

    if objects is None:
        try:
            from ..runtime.session import all_objects
            objects = all_objects()
        except Exception:
            objects = []
    objects = list(objects)
    visuals = [obj for obj in objects if is_field_visual(obj)]
    skip_names = {display_name(obj) for obj in visuals if display_name(obj)}

    fields = []
    seen_ids = set()
    for obj in discover_fields(objects, cmd=cmd):
        oid = str(getattr(obj, "id", "") or "")
        label = field_label(obj)
        if label in skip_names:
            continue
        if oid and oid in seen_ids:
            continue
        if oid:
            seen_ids.add(oid)
        fields.append(obj)

    assigned = set()
    groups = []
    for field in fields:
        keys = set(field_identity_keys(field))
        children = []
        for vis in visuals:
            vid = str(getattr(vis, "id", "") or "")
            if vid in assigned:
                continue
            if keys & set(visual_field_keys(vis)):
                children.append(vis)
                if vid:
                    assigned.add(vid)
        groups.append((field, children))

    orphans_by_grid = {}
    for vis in visuals:
        vid = str(getattr(vis, "id", "") or "")
        if vid in assigned:
            continue
        grid = getattr(vis, "grid_data", None)
        if grid is None:
            continue
        gid = str(getattr(grid, "id", "") or id(grid))
        orphans_by_grid.setdefault(gid, (grid, []))
        orphans_by_grid[gid][1].append(vis)
        if vid:
            assigned.add(vid)
    for grid, children in orphans_by_grid.values():
        groups.append((grid, children))

    rows = []
    for field, children in groups:
        fid = str(getattr(field, "id", "") or "")
        used = _used_by_text(field, children, objects)
        rows.append(field_row(field, used_by=used))
        for vis in children:
            geom_id = getattr(vis, "geometry_field_id", None) or fid
            color_id = getattr(vis, "color_field_id", None)
            rows.append(field_visual_row(
                vis,
                fid,
                last_child=False,
                geometry_field=_field_label_by_id(geom_id, fields),
                color_field=_field_label_by_id(color_id, fields) if color_id else "",
            ))
        rows.append(add_visual_row(fid))
    return rows


def field_library_field_count(rows: Sequence[dict]) -> int:
    return sum(1 for row in rows if row.get("kind") == KIND_FIELD)


def _first_rgb(value) -> Optional[Tuple[float, float, float]]:
    if value is None:
        return None
    try:
        if hasattr(value, "reshape"):
            flat = value.reshape(-1)
            if len(flat) >= 3:
                return (float(flat[0]), float(flat[1]), float(flat[2]))
        seq = list(value)
        if not seq:
            return None
        first = seq[0]
        if hasattr(first, "__len__") and not isinstance(first, (str, bytes)):
            return (float(first[0]), float(first[1]), float(first[2]))
        if len(seq) >= 3:
            return (float(seq[0]), float(seq[1]), float(seq[2]))
    except (TypeError, IndexError, ValueError):
        return None
    return None


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
