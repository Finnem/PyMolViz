"""Schema-1 persistence: plain dict/list/str/int/float/bool/None only."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Dict, Optional

import numpy as np

from .points import (
    AtomPoint,
    FixedPoint,
    PointSource,
    PseudoAtomPoint,
    as_point_source,
)

SCHEMA_VERSION = 1

_POINT_SOURCE_TYPES = {
    "FixedPoint": FixedPoint,
    "AtomPoint": AtomPoint,
    "PseudoAtomPoint": PseudoAtomPoint,
}

_DISPLAYABLE_TYPES: Optional[Dict[str, type]] = None


class SerializationError(ValueError):
    pass


def as_plain(value: Any) -> Any:
    """Convert numpy / tuples into JSON-safe Python types."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, np.ndarray):
        return as_plain(value.tolist())
    if isinstance(value, dict):
        return {str(k): as_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_plain(v) for v in value]
    raise SerializationError("Cannot persist %s (%r)" % (type(value).__name__, value))


def assert_plain(value: Any) -> None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise SerializationError("Dict keys must be str, got %r" % type(k))
            assert_plain(v)
        return
    if isinstance(value, list):
        for v in value:
            assert_plain(v)
        return
    raise SerializationError("Non-plain value %s: %r" % (type(value).__name__, value))


def persist_color(obj) -> Any:
    """RGB actually used (bypass_colormap path), never a matplotlib object."""
    if getattr(obj, "bypass_colormap", False):
        arr = np.asarray(obj.color, dtype=float)
    elif hasattr(obj, "colormap") and obj.colormap is not None:
        arr = np.asarray(obj.colormap.get_color(obj.color), dtype=float)
    else:
        arr = np.asarray(getattr(obj, "color", [1.0, 0.0, 0.0]), dtype=float)
    if arr.size == 0:
        return [1.0, 0.0, 0.0]
    if arr.ndim == 1:
        return [float(x) for x in arr.reshape(-1)[:3]]
    return [[float(x) for x in row[:3]] for row in arr]


def persist_transparency(obj) -> Any:
    t = getattr(obj, "transparency", 0)
    try:
        t[0]
        return [float(x) for x in np.asarray(t, dtype=float).reshape(-1)]
    except (TypeError, IndexError):
        return float(t) if t is not None else 0.0


def persist_vector(value) -> list:
    return [float(x) for x in np.asarray(value, dtype=float).reshape(-1)]


def persist_matrix(value) -> list:
    arr = np.asarray(value, dtype=float)
    return [[float(x) for x in row] for row in arr.reshape(-1, arr.shape[-1])]


def point_source_from_dict(data: dict) -> PointSource:
    if not isinstance(data, dict) or "type" not in data:
        return as_point_source(data)
    typ = data["type"]
    if typ == "FixedPoint":
        xyz = data.get("xyz") or (data.get("x"), data.get("y"), data.get("z"))
        return FixedPoint(xyz)
    if typ == "AtomPoint":
        return AtomPoint(
            data["object"],
            int(data["atom_id"]),
            chain=data.get("chain", ""),
            resi=data.get("resi", ""),
            name=data.get("name", ""),
            last_xyz=data.get("last_xyz"),
        )
    if typ == "PseudoAtomPoint":
        return PseudoAtomPoint(
            data["object"],
            int(data.get("atom_id", 0)),
            last_xyz=data.get("last_xyz"),
        )
    raise SerializationError("Unknown PointSource type %r" % typ)


def _base_fields(obj) -> dict:
    return {
        "type": type(obj).__name__,
        "id": str(obj.id),
        "name": obj._name,
        "state": int(getattr(obj, "state", 1) or 1),
        "transparency": persist_transparency(obj),
    }


def _common_mesh_fields(obj) -> dict:
    data = _base_fields(obj)
    data["color"] = persist_color(obj)
    return data


def _sources_to_dict(sources) -> list:
    return [s.to_dict() for s in sources]


def _sources_from_dict(items) -> list:
    return [point_source_from_dict(item) for item in items]


def _line_style_dict(obj) -> dict:
    style = getattr(obj, "line_style", None)
    if style is None:
        return {
            "dash": "Solid",
            "dash_scale": 1.0,
            "margin": 0.0,
            "ends": "Arrow",
        }
    if hasattr(style, "to_dict"):
        return style.to_dict()
    return {
        "dash": getattr(style, "dash", "Solid"),
        "dash_scale": float(getattr(style, "dash_scale", 1.0)),
        "margin": float(getattr(style, "margin", 0.0)),
        "ends": getattr(style, "ends", "Arrow"),
    }


def _dump_sphere(obj) -> dict:
    data = _common_mesh_fields(obj)
    data.update({
        "position": obj.position.to_dict(),
        "radius": float(obj.radius),
        "frequency": int(obj.frequency),
        "subdivisions": obj.subdivisions,
        "resolution": int(getattr(obj, "resolution", 20)),
        "wireframe": bool(getattr(obj, "wireframe", False)),
    })
    return data


def _load_sphere(cls, data: dict):
    return cls(
        point_source_from_dict(data["position"]),
        data["radius"],
        color=data.get("color"),
        frequency=data.get("frequency"),
        subdivisions=data.get("subdivisions"),
        resolution=data.get("resolution", 20),
        wireframe=data.get("wireframe", False),
        name=data.get("name"),
        obj_id=data.get("id"),
        state=data.get("state", 1),
        transparency=data.get("transparency", 0),
        bypass_colormap=True,
    )


def _dump_cylinder(obj) -> dict:
    data = _common_mesh_fields(obj)
    data.update({
        "start": obj.start.to_dict(),
        "end": obj.end.to_dict(),
        "radius": float(obj.radius),
        "resolution": int(obj.resolution),
    })
    return data


def _load_cylinder(cls, data: dict):
    return cls(
        point_source_from_dict(data["start"]),
        point_source_from_dict(data["end"]),
        data["radius"],
        color=data.get("color"),
        resolution=data.get("resolution", 20),
        name=data.get("name"),
        obj_id=data.get("id"),
        state=data.get("state", 1),
        transparency=data.get("transparency", 0),
        bypass_colormap=True,
    )


def _dump_box(obj) -> dict:
    data = _common_mesh_fields(obj)
    data.update({
        "center": obj.center.to_dict(),
        "extent": [float(v) for v in obj.extent],
        "wireframe": bool(getattr(obj, "wireframe", False)),
    })
    return data


def _load_box(cls, data: dict):
    return cls(
        point_source_from_dict(data["center"]),
        data["extent"],
        color=data.get("color"),
        wireframe=data.get("wireframe", False),
        name=data.get("name"),
        obj_id=data.get("id"),
        state=data.get("state", 1),
        transparency=data.get("transparency", 0),
        bypass_colormap=True,
    )


def _line_endpoints(obj):
    starts = getattr(obj, "_start_sources", None)
    ends = getattr(obj, "_end_sources", None)
    if starts and ends:
        return starts, ends
    return list(obj.starts), list(obj.ends)


def _dump_lines(obj) -> dict:
    starts, ends = _line_endpoints(obj)
    data = _common_mesh_fields(obj)
    data.update({
        "starts": _sources_to_dict(starts),
        "ends": _sources_to_dict(ends),
        "linewidth": float(obj.linewidth),
        "render_as": obj.render_as,
        "render_ends": bool(getattr(obj, "render_ends", False)),
    })
    return data


def _load_lines(cls, data: dict):
    return cls(
        starts=_sources_from_dict(data["starts"]),
        ends=_sources_from_dict(data["ends"]),
        color=data.get("color"),
        name=data.get("name"),
        obj_id=data.get("id"),
        state=data.get("state", 1),
        transparency=data.get("transparency", 0),
        linewidth=data.get("linewidth", 0.05),
        render_as=data.get("render_as", "cylinders"),
        render_ends=data.get("render_ends", False),
        bypass_colormap=True,
    )


def _dump_arrows(obj) -> dict:
    data = _dump_lines(obj)
    data["type"] = "Arrows"
    data.update({
        "head_length": float(getattr(obj, "head_length", 0.25)),
        "head_width": float(getattr(obj, "head_width", 1.618)),
        "quality": int(getattr(obj, "quality", 3)),
        "shaft_radius": float(getattr(obj, "shaft_radius", 0.045)),
        "use_styled_cgo": bool(getattr(obj, "use_styled_cgo", False)),
        "line_style": _line_style_dict(obj),
    })
    mask = getattr(obj, "arrow_mask", None)
    if mask is not None:
        data["arrow_mask"] = [bool(v) for v in np.asarray(mask).reshape(-1)]
    return data


def _load_arrows(cls, data: dict):
    from .util.line_style import LineStyle

    style = LineStyle.from_dict(data.get("line_style") or {})
    return cls(
        starts=_sources_from_dict(data["starts"]),
        ends=_sources_from_dict(data["ends"]),
        color=data.get("color"),
        name=data.get("name"),
        obj_id=data.get("id"),
        state=data.get("state", 1),
        transparency=data.get("transparency", 0),
        linewidth=data.get("linewidth", 0.05),
        head_length=data.get("head_length", 0.25),
        head_width=data.get("head_width", 1.618),
        render_as=data.get("render_as", "cylinders"),
        quality=data.get("quality", 3),
        line_style=style,
        shaft_radius=data.get("shaft_radius", 0.045),
        use_styled_cgo=data.get("use_styled_cgo", False),
        arrow_mask=data.get("arrow_mask"),
        bypass_colormap=True,
    )


def _dump_points(obj) -> dict:
    sources = getattr(obj, "vertex_sources", None)
    if not sources:
        sources = [as_point_source(v) for v in np.asarray(obj.vertices, dtype=float).reshape(-1, 3)]
    data = _common_mesh_fields(obj)
    data.update({
        "vertices": _sources_to_dict(sources),
        "render_as": getattr(obj, "render_as", "Spheres"),
        "radius": float(getattr(obj, "radius", 0.3)),
    })
    return data


def _load_points(cls, data: dict):
    return cls(
        None,
        color=data.get("color"),
        name=data.get("name"),
        obj_id=data.get("id"),
        state=data.get("state", 1),
        transparency=data.get("transparency", 0),
        render_as=data.get("render_as", "Spheres"),
        radius=data.get("radius", 0.3),
        vertex_sources=_sources_from_dict(data["vertices"]),
        bypass_colormap=True,
    )


def _dump_mesh(obj) -> dict:
    data = _dump_points(obj)
    data["type"] = type(obj).__name__
    if getattr(obj, "faces", None) is not None:
        data["faces"] = [[int(i) for i in face] for face in np.asarray(obj.faces)]
    if getattr(obj, "normals", None) is not None:
        data["normals"] = persist_matrix(obj.normals)
    return data


def _load_mesh(cls, data: dict):
    from .meshes.Mesh import Mesh

    verts = np.array([point_source_from_dict(v).resolve(None) for v in data["vertices"]])
    kwargs = {
        "color": data.get("color"),
        "name": data.get("name"),
        "obj_id": data.get("id"),
        "state": data.get("state", 1),
        "transparency": data.get("transparency", 0),
        "bypass_colormap": True,
    }
    if data.get("faces") is not None:
        kwargs["faces"] = data["faces"]
    if data.get("normals") is not None:
        kwargs["normals"] = data["normals"]
    if cls is Mesh or cls.__name__ == "Mesh":
        return Mesh(verts, **kwargs)
    return cls(verts, **kwargs)


def _dump_plane(obj) -> dict:
    data = _common_mesh_fields(obj)
    data.update({
        "position": obj.position.to_dict(),
        "normal": persist_vector(obj.normal),
        "scale": float(obj.scale),
    })
    return data


def _load_plane(cls, data: dict):
    return cls(
        point_source_from_dict(data["position"]),
        data["normal"],
        scale=data.get("scale", 5),
        color=data.get("color"),
        name=data.get("name"),
        obj_id=data.get("id"),
        state=data.get("state", 1),
        transparency=data.get("transparency", 0),
        bypass_colormap=True,
    )


def _dump_hull(obj) -> dict:
    data = _common_mesh_fields(obj)
    data["points"] = _sources_to_dict(obj.point_sources)
    return data


def _load_hull(cls, data: dict):
    return cls(
        _sources_from_dict(data["points"]),
        color=data.get("color"),
        name=data.get("name"),
        obj_id=data.get("id"),
        state=data.get("state", 1),
        transparency=data.get("transparency", 0),
        bypass_colormap=True,
    )


def _dump_tube(obj) -> dict:
    data = _common_mesh_fields(obj)
    radius = obj.tube_radius
    try:
        radius = [float(x) for x in np.asarray(radius, dtype=float).reshape(-1)]
        if len(radius) == 1:
            radius = radius[0]
    except Exception:
        radius = float(radius)
    data.update({
        "path": _sources_to_dict(obj.path_sources),
        "tube_radius": radius,
        "tubular_resolution": int(obj.tubular_resolution),
        "show_arrow": bool(obj.show_arrow),
        "arrow_base_scale": float(obj.arrow_base_scale),
        "arrow_height_scale": float(obj.arrow_height_scale),
        "arrow_sides": obj.arrow_sides if obj.arrow_sides is None else int(obj.arrow_sides),
    })
    return data


def _load_tube(cls, data: dict):
    return cls(
        _sources_from_dict(data["path"]),
        tube_radius=data.get("tube_radius", 0.05),
        color=data.get("color"),
        name=data.get("name"),
        obj_id=data.get("id"),
        state=data.get("state", 1),
        transparency=data.get("transparency", 0),
        tubular_resolution=data.get("tubular_resolution", 24),
        show_arrow=data.get("show_arrow", False),
        arrow_base_scale=data.get("arrow_base_scale", 1.2),
        arrow_height_scale=data.get("arrow_height_scale", 2.5),
        arrow_sides=data.get("arrow_sides"),
    )


def _dump_rotation(obj) -> dict:
    data = _common_mesh_fields(obj)
    data.update({
        "center_position": obj.center_position.to_dict(),
        "outer_start": obj.outer_start.to_dict(),
        "rotation_axis": persist_vector(obj.rotation_axis),
        "angle": float(obj.angle),
        "linewidth": float(obj.linewidth),
        "resolution": int(obj.resolution),
        "tubular_resolution": int(obj.tubular_resolution),
        "show_arrow": bool(obj.show_arrow),
        "arrow_base_scale": float(obj.arrow_base_scale),
        "arrow_height_scale": float(obj.arrow_height_scale),
        "arrow_sides": obj.arrow_sides if obj.arrow_sides is None else int(obj.arrow_sides),
    })
    return data


def _load_rotation(cls, data: dict):
    return cls(
        point_source_from_dict(data["center_position"]),
        point_source_from_dict(data["outer_start"]),
        data["rotation_axis"],
        data["angle"],
        color=data.get("color"),
        name=data.get("name"),
        obj_id=data.get("id"),
        state=data.get("state", 1),
        transparency=data.get("transparency", 0),
        linewidth=data.get("linewidth", 0.05),
        resolution=data.get("resolution", 64),
        tubular_resolution=data.get("tubular_resolution", 24),
        show_arrow=data.get("show_arrow", True),
        arrow_base_scale=data.get("arrow_base_scale", 1.2),
        arrow_height_scale=data.get("arrow_height_scale", 2.5),
        arrow_sides=data.get("arrow_sides"),
    )


def _dump_collection(obj) -> dict:
    data = _base_fields(obj)
    data["schema"] = SCHEMA_VERSION
    data["objects"] = [displayable_to_dict(child) for child in obj]
    return data


def _load_collection(cls, data: dict):
    children = [displayable_from_dict(item) for item in data.get("objects", [])]
    obj = cls(
        children,
        name=data.get("name"),
        state=data.get("state", 1),
        transparency=data.get("transparency", 0),
        obj_id=data.get("id"),
    )
    return obj


def _dump_volumetric(obj) -> dict:
    """Displayable.id only — volumetric objects are not live-followed."""
    return _base_fields(obj)


def _ensure_displayable_types() -> Dict[str, type]:
    global _DISPLAYABLE_TYPES
    if _DISPLAYABLE_TYPES is not None:
        return _DISPLAYABLE_TYPES
    from .meshes.Arrows import Arrows
    from .meshes.CenteredBox import CenteredBox
    from .meshes.CGOCollection import CGOCollection
    from .meshes.ConvexHull import ConvexHull
    from .meshes.Cylinder import Cylinder
    from .meshes.Lines import Lines
    from .meshes.Mesh import Mesh
    from .meshes.Plane import Plane
    from .meshes.Points import Points
    from .meshes.Sphere import Sphere
    from .meshes.derived.PolylineTube import PolylineTube
    from .meshes.derived.Rotation_Indicator import Rotation_Indicator
    from .volumetric.IsoMesh import IsoMesh
    from .volumetric.IsoSurface import IsoSurface
    from .volumetric.IsoVolume import IsoVolume
    from .volumetric.Volume import Volume

    _DISPLAYABLE_TYPES = {
        "Sphere": Sphere,
        "Cylinder": Cylinder,
        "CenteredBox": CenteredBox,
        "Lines": Lines,
        "Arrows": Arrows,
        "Points": Points,
        "Mesh": Mesh,
        "Plane": Plane,
        "ConvexHull": ConvexHull,
        "PolylineTube": PolylineTube,
        "Rotation_Indicator": Rotation_Indicator,
        "CGOCollection": CGOCollection,
        "IsoSurface": IsoSurface,
        "IsoMesh": IsoMesh,
        "IsoVolume": IsoVolume,
        "Volume": Volume,
    }
    return _DISPLAYABLE_TYPES


_DUMPERS: Dict[str, Callable] = {
    "Sphere": _dump_sphere,
    "Cylinder": _dump_cylinder,
    "CenteredBox": _dump_box,
    "Lines": _dump_lines,
    "Arrows": _dump_arrows,
    "Points": _dump_points,
    "Mesh": _dump_mesh,
    "Plane": _dump_plane,
    "ConvexHull": _dump_hull,
    "PolylineTube": _dump_tube,
    "Rotation_Indicator": _dump_rotation,
    "CGOCollection": _dump_collection,
    "IsoSurface": _dump_volumetric,
    "IsoMesh": _dump_volumetric,
    "IsoVolume": _dump_volumetric,
    "Volume": _dump_volumetric,
}

_LOADERS: Dict[str, Callable] = {
    "Sphere": _load_sphere,
    "Cylinder": _load_cylinder,
    "CenteredBox": _load_box,
    "Lines": _load_lines,
    "Arrows": _load_arrows,
    "Points": _load_points,
    "Mesh": _load_mesh,
    "Plane": _load_plane,
    "ConvexHull": _load_hull,
    "PolylineTube": _load_tube,
    "Rotation_Indicator": _load_rotation,
    "CGOCollection": _load_collection,
}


def displayable_to_dict(obj) -> dict:
    name = type(obj).__name__
    dumper = _DUMPERS.get(name)
    if dumper is None:
        if hasattr(obj, "vertices"):
            data = _dump_mesh(obj)
        else:
            data = _base_fields(obj)
    else:
        data = dumper(obj)
    data = as_plain(data)
    assert_plain(data)
    return data


def displayable_from_dict(data: dict):
    if not isinstance(data, dict) or "type" not in data:
        raise SerializationError("Displayable dict must include a type")
    typ = data["type"]
    registry = _ensure_displayable_types()
    cls = registry.get(typ)
    loader = _LOADERS.get(typ)
    if cls is None or loader is None:
        if typ in ("IsoSurface", "IsoMesh", "IsoVolume", "Volume"):
            raise SerializationError(
                "Volumetric type %r is not restored from session (id-only persistence)" % typ
            )
        raise SerializationError("Unknown Displayable type %r" % typ)
    return loader(cls, data)


def to_dict(obj) -> dict:
    if isinstance(obj, PointSource):
        data = as_plain(obj.to_dict())
        assert_plain(data)
        return data
    return displayable_to_dict(obj)


def from_dict(data: dict):
    if not isinstance(data, dict) or "type" not in data:
        raise SerializationError("from_dict expects a dict with a type key")
    if data["type"] in _POINT_SOURCE_TYPES:
        return point_source_from_dict(data)
    return displayable_from_dict(data)


def session_document(objects) -> dict:
    """Plain session blob for ``pymol.session.pymolviz``."""
    doc = {
        "schema": SCHEMA_VERSION,
        "objects": [displayable_to_dict(obj) for obj in objects],
    }
    assert_plain(doc)
    return doc


def session_from_document(data: dict) -> list:
    """Deserialize objects from a session document."""
    if not isinstance(data, dict):
        return []
    out = []
    for item in data.get("objects", []):
        if not isinstance(item, dict):
            continue
        try:
            out.append(displayable_from_dict(item))
        except Exception:
            continue
    return out


def style_hash(obj) -> str:
    data = to_dict(obj)

    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k != "last_xyz"}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value

    blob = json.dumps(strip(data), sort_keys=True, separators=(",", ":"))
    return hashlib.md5(blob.encode("utf-8")).hexdigest()
