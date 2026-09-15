"""Schema-1 persistence: plain dict/list/str/int/float/bool/None only."""

from __future__ import annotations

import contextlib
import contextvars
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
ARRAY_REF_KEY = "$npy"

_ARRAY_STORE: contextvars.ContextVar[Optional["ArrayStore"]] = contextvars.ContextVar(
    "pmv_array_store", default=None,
)

_POINT_SOURCE_TYPES = {
    "FixedPoint": FixedPoint,
    "AtomPoint": AtomPoint,
    "PseudoAtomPoint": PseudoAtomPoint,
}

_DISPLAYABLE_TYPES: Optional[Dict[str, type]] = None
_INFLIGHT: Dict[str, Any] = {}


class SerializationError(ValueError):
    pass


class ArrayStore:
    """Holds numpy arrays beside a JSON document (native ``.pmv`` packs)."""

    def __init__(self) -> None:
        self.arrays: Dict[str, np.ndarray] = {}

    def put(self, values) -> dict:
        arr = np.ascontiguousarray(values)
        key = "a%d" % len(self.arrays)
        self.arrays[key] = arr
        return {
            ARRAY_REF_KEY: key,
            "dtype": str(arr.dtype),
            "shape": [int(n) for n in arr.shape],
        }

    def get(self, key) -> np.ndarray:
        try:
            return self.arrays[str(key)]
        except KeyError:
            raise SerializationError("Missing native array %r" % key)

    def add(self, key, values) -> None:
        self.arrays[str(key)] = np.ascontiguousarray(values)


def persist_numeric_array(values, *, dtype=None) -> Any:
    """JSON list, or an array ref when a native ``ArrayStore`` is active."""
    arr = np.asarray(values) if dtype is None else np.asarray(values, dtype=dtype)
    store = _ARRAY_STORE.get()
    if store is not None:
        return store.put(arr)
    flat = np.asarray(arr, dtype=float).reshape(-1)
    return [float(v) for v in flat]


def resolve_numeric_array(value, *, dtype=float) -> np.ndarray:
    if isinstance(value, dict) and value.get(ARRAY_REF_KEY):
        store = _ARRAY_STORE.get()
        if store is None:
            raise SerializationError("Native array ref with no array store")
        return store.get(value[ARRAY_REF_KEY])
    return np.asarray(value, dtype=dtype)


@contextlib.contextmanager
def using_array_store(store: Optional["ArrayStore"]):
    """Bind ``store`` for the duration of a dump/load."""
    token = _ARRAY_STORE.set(store)
    try:
        yield store
    finally:
        _ARRAY_STORE.reset(token)


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
            elem=data.get("elem", ""),
            last_xyz=data.get("last_xyz"),
            last_vdw=data.get("last_vdw"),
        )
    if typ == "PseudoAtomPoint":
        return PseudoAtomPoint(
            data["object"],
            int(data.get("atom_id", 0)),
            last_xyz=data.get("last_xyz"),
        )
    raise SerializationError("Unknown PointSource type %r" % typ)


def _base_fields(obj) -> dict:
    data = {
        "type": type(obj).__name__,
        "id": str(obj.id),
        "name": obj._name,
        "state": int(getattr(obj, "state", 1) or 1),
        "transparency": persist_transparency(obj),
    }
    mode = getattr(obj, "preview_mode", None)
    if mode:
        from .wizards.builders.preview_mode import normalize_preview_mode

        data["preview_mode"] = normalize_preview_mode(mode)
    return data


def _restore_preview_mode(obj, data: dict) -> None:
    if obj is None or not isinstance(data, dict) or "preview_mode" not in data:
        return
    from .wizards.builders.preview_mode import stamp_preview_mode

    stamp_preview_mode(obj, data.get("preview_mode"))


def _common_mesh_fields(obj) -> dict:
    data = _base_fields(obj)
    data["color"] = persist_color(obj)
    field_id = getattr(obj, "field_id", None)
    if field_id:
        data["field_id"] = str(field_id)
        cmap = getattr(obj, "field_colormap", None)
        if cmap:
            data["field_colormap"] = str(cmap)
        spec = getattr(obj, "field_colormap_spec", None)
        if spec:
            data["field_colormap_spec"] = spec
        clims = getattr(obj, "field_clims", None)
        if clims is not None:
            from .util.field_sample import normalize_clims
            normalized = normalize_clims(clims)
            if normalized is not None:
                data["field_clims"] = normalized
        clim_mode = getattr(obj, "field_clim_mode", None)
        if clim_mode:
            data["field_clim_mode"] = str(clim_mode)
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
            "start_margin": 0.0,
            "end_margin": 0.0,
            "start_head": "None",
            "end_head": "Arrow",
            "ends": "Arrow",
        }
    if hasattr(style, "to_dict"):
        return style.to_dict()
    start_head = getattr(style, "start_head", None)
    end_head = getattr(style, "end_head", None)
    ends = getattr(style, "ends", "Arrow")
    if start_head is None and end_head is None:
        from .util.line_style import split_ends

        start_head, end_head = split_ends(ends)
    return {
        "dash": getattr(style, "dash", "Solid"),
        "dash_scale": float(getattr(style, "dash_scale", 1.0)),
        "margin": float(getattr(style, "margin", 0.0)),
        "start_margin": float(getattr(style, "start_margin", getattr(style, "margin", 0.0))),
        "end_margin": float(getattr(style, "end_margin", getattr(style, "margin", 0.0))),
        "start_head": start_head or "None",
        "end_head": end_head or "Arrow",
        "ends": ends,
    }


def _attach_clip_planes(obj, data: dict) -> None:
    from .util.mesh_clip import normalize_clip_planes
    planes = normalize_clip_planes(getattr(obj, "clip_planes", None))
    if planes:
        data["clip_planes"] = planes


def _dump_sphere(obj) -> dict:
    data = _common_mesh_fields(obj)
    data.update({
        "position": obj.position.to_dict(),
        "radius": float(obj.radius),
        "frequency": int(obj.frequency),
        "subdivisions": obj.subdivisions,
        "resolution": int(getattr(obj, "resolution", 20)),
        "wireframe": bool(getattr(obj, "wireframe", False)),
        "enabled": bool(getattr(obj, "enabled", True)),
    })
    from .util.mesh_clip import normalize_clip_planes
    planes = normalize_clip_planes(getattr(obj, "clip_planes", None))
    if planes:
        data["clip_planes"] = planes
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
        clip_planes=data.get("clip_planes"),
        enabled=data.get("enabled", True),
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
        "enabled": bool(getattr(obj, "enabled", True)),
    })
    from .util.mesh_clip import normalize_clip_planes
    planes = normalize_clip_planes(getattr(obj, "clip_planes", None))
    if planes:
        data["clip_planes"] = planes
    return data


def _load_box(cls, data: dict):
    return cls(
        point_source_from_dict(data["center"]),
        data["extent"],
        color=data.get("color"),
        wireframe=data.get("wireframe", False),
        clip_planes=data.get("clip_planes"),
        enabled=data.get("enabled", True),
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


def _load_lines(cls, data: dict):
    """Sessions that stored type ``Lines`` load as Arrows with line Options."""
    from .meshes.Arrows import Arrows
    from .util.line_style import LineStyle

    data = dict(data)
    if not data.get("line_style"):
        ends = "Circles" if data.get("render_ends") else "None"
        data["line_style"] = LineStyle(ends=ends).to_dict()
    if data.get("quality") is None:
        render_as = str(data.get("render_as", "cylinders") or "cylinders")
        data["quality"] = 0 if render_as in ("line", "lines") else 3
    if data.get("shaft_radius") is None:
        data["shaft_radius"] = data.get("linewidth", 0.05)
    data.setdefault("use_styled_cgo", True)
    return _load_arrows(Arrows, data)


def _dump_arrows(obj) -> dict:
    starts, ends = _line_endpoints(obj)
    data = _common_mesh_fields(obj)
    data["type"] = "Arrows"
    shaft = float(getattr(obj, "shaft_radius", obj.linewidth))
    quality = int(getattr(obj, "quality", 3))
    data.update({
        "starts": _sources_to_dict(starts),
        "ends": _sources_to_dict(ends),
        "shaft_radius": shaft,
        "quality": quality,
        "head_length": float(getattr(obj, "head_length", 0.25)),
        "head_width": float(getattr(obj, "head_width", 1.618)),
        "use_styled_cgo": bool(getattr(obj, "use_styled_cgo", True)),
        "line_style": _line_style_dict(obj),
        "linewidth": shaft,
        "render_as": "lines" if quality == 0 else "cylinders",
    })
    radii = getattr(obj, "pair_radii", None)
    if radii:
        data["pair_radii"] = [float(x) for x in radii]
    heads = getattr(obj, "pair_heads", None)
    if heads:
        data["pair_heads"] = [float(x) for x in heads]
    styles = getattr(obj, "pair_styles", None)
    if styles:
        data["pair_styles"] = [
            style.to_dict() if hasattr(style, "to_dict") else dict(style)
            for style in styles
        ]
    mask = getattr(obj, "arrow_mask", None)
    if mask is not None:
        data["arrow_mask"] = [bool(v) for v in np.asarray(mask).reshape(-1)]
    head_radius = getattr(obj, "head_radius", None)
    if head_radius is not None:
        data["head_radius"] = float(head_radius)
    _attach_clip_planes(obj, data)
    return data


def _load_arrows(cls, data: dict):
    from .util.line_style import LineStyle

    style = LineStyle.from_dict(data.get("line_style") or {})
    quality = data.get("quality")
    if quality is None:
        render_as = str(data.get("render_as", "cylinders") or "cylinders")
        quality = 0 if render_as in ("line", "lines") else 3
    shaft = data.get("shaft_radius", data.get("linewidth", 0.045))
    obj = cls(
        starts=_sources_from_dict(data["starts"]),
        ends=_sources_from_dict(data["ends"]),
        color=data.get("color"),
        name=data.get("name"),
        obj_id=data.get("id"),
        state=data.get("state", 1),
        transparency=data.get("transparency", 0),
        head_length=data.get("head_length", 0.25),
        head_width=data.get("head_width", 1.618),
        quality=int(quality),
        line_style=style,
        shaft_radius=shaft,
        use_styled_cgo=data.get("use_styled_cgo", True),
        arrow_mask=data.get("arrow_mask"),
        head_radius=data.get("head_radius"),
        clip_planes=data.get("clip_planes"),
        bypass_colormap=True,
    )
    if data.get("pair_radii"):
        obj.pair_radii = [float(x) for x in data["pair_radii"]]
    if data.get("pair_heads"):
        obj.pair_heads = [float(x) for x in data["pair_heads"]]
    raw_styles = data.get("pair_styles") or []
    if raw_styles:
        obj.pair_styles = [
            LineStyle.from_dict(item) if isinstance(item, dict) else item
            for item in raw_styles
        ]
    return obj


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


def _dump_surface(obj) -> dict:
    data = _common_mesh_fields(obj)
    n = len(getattr(obj, "point_sources", None) or ())
    from .util.solvent_surface import (
        DEFAULT_ALGORITHM,
        DEFAULT_ATOM_RADIUS,
        DEFAULT_RADIUS_MODE,
        DEFAULT_VDW_SCALE,
        normalize_point_radii,
        normalize_radius_mode,
    )
    data.update({
        "points": _sources_to_dict(obj.point_sources),
        "atom_radius": float(getattr(obj, "atom_radius", DEFAULT_ATOM_RADIUS)),
        "probe_radius": float(getattr(obj, "probe_radius", 1.4)),
        "algorithm": str(getattr(obj, "algorithm", DEFAULT_ALGORITHM)),
        "quality": int(getattr(obj, "quality", 3)),
        "wireframe": bool(getattr(obj, "wireframe", False)),
        "radius_mode": normalize_radius_mode(getattr(obj, "radius_mode", DEFAULT_RADIUS_MODE)),
        "vdw_scale": float(getattr(obj, "vdw_scale", DEFAULT_VDW_SCALE) or DEFAULT_VDW_SCALE),
        "point_radii": normalize_point_radii(getattr(obj, "point_radii", None), n),
    })
    from .util.solvent_surface import normalize_point_enabled
    enabled = normalize_point_enabled(getattr(obj, "point_enabled", None), n)
    if enabled is not None:
        data["point_enabled"] = enabled
    point_colors = getattr(obj, "point_colors", None)
    if point_colors and not getattr(obj, "field_id", None):
        data["point_colors"] = [
            [float(c) for c in row[:3]] for row in point_colors
        ]
    created_from = getattr(obj, "created_from", None)
    if created_from:
        data["created_from"] = dict(created_from)
        src_v = getattr(obj, "_source_vertices", None)
        src_n = getattr(obj, "_source_normals", None)
        src_f = getattr(obj, "_source_faces", None)
        if src_v is not None:
            data["source_vertices"] = persist_matrix(src_v)
        if src_n is not None:
            data["source_normals"] = persist_matrix(src_n)
        if src_f is not None:
            data["source_faces"] = [[int(i) for i in face] for face in np.asarray(src_f)]
    from .util.mesh_clip import normalize_clip_planes
    planes = normalize_clip_planes(getattr(obj, "clip_planes", None))
    if planes:
        data["clip_planes"] = planes
    return data


def _load_surface(cls, data: dict):
    from .util.solvent_surface import DEFAULT_ALGORITHM

    surface = cls(
        _sources_from_dict(data["points"]),
        atom_radius=data.get("atom_radius", 1.5),
        probe_radius=data.get("probe_radius", 1.4),
        algorithm=data.get("algorithm", DEFAULT_ALGORITHM),
        quality=data.get("quality", 3),
        wireframe=data.get("wireframe", False),
        radius_mode=data.get("radius_mode", "uniform"),
        vdw_scale=data.get("vdw_scale", 1.0),
        point_radii=data.get("point_radii"),
        point_enabled=data.get("point_enabled"),
        clip_planes=data.get("clip_planes"),
        color=data.get("color"),
        name=data.get("name"),
        obj_id=data.get("id"),
        state=data.get("state", 1),
        transparency=data.get("transparency", 0),
        bypass_colormap=True,
        created_from=data.get("created_from"),
        source_vertices=data.get("source_vertices"),
        source_normals=data.get("source_normals"),
        source_faces=data.get("source_faces"),
    )
    point_colors = data.get("point_colors")
    if point_colors and not data.get("field_id"):
        surface.point_colors = [
            (float(row[0]), float(row[1]), float(row[2])) for row in point_colors
        ]
        from .util.field_sample import paint_surface_mesh_from_anchors
        paint_surface_mesh_from_anchors(surface)
    return surface


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
    data["specular"] = bool(getattr(obj, "specular", True))
    return data


def _load_collection(cls, data: dict):
    children = [displayable_from_dict(item) for item in data.get("objects", [])]
    obj = cls(
        children,
        name=data.get("name"),
        state=data.get("state", 1),
        transparency=data.get("transparency", 0),
        obj_id=data.get("id"),
        specular=bool(data.get("specular", True)),
    )
    return obj


def _dump_field(obj) -> dict:
    from .fields.identity import (
        GEN_DISTANCE,
        GEN_GAUSSIAN,
        GEN_IMPORTED,
        GEN_NEAREST_COLOR,
        GEN_NEAREST_PROP,
        GEN_PYMOL_MAP,
        GEN_SIGNED_VDW,
        canonical_field_spec,
    )

    data = {
        "type": "Field",
        "id": str(obj.id),
        "name": obj._name,
        "kind": str(getattr(obj, "kind", "scalar") or "scalar"),
        "units": getattr(obj, "units", None),
        "generator": dict(getattr(obj, "generator", None) or {}),
        "domain": obj.domain.to_dict() if getattr(obj, "domain", None) is not None else {},
        "provenance": dict(getattr(obj, "provenance", None) or {}),
        "spec": canonical_field_spec(obj),
    }
    color_fid = getattr(obj, "default_color_field_id", None) or (
        (getattr(obj, "provenance", None) or {}).get("default_color_field_id")
    )
    if color_fid:
        data["default_color_field_id"] = str(color_fid)
    categories = getattr(obj, "categories", None)
    if categories:
        dumped = []
        for item in categories:
            if isinstance(item, (list, tuple)):
                dumped.append([float(v) for v in item])
            else:
                dumped.append(str(item))
        data["categories"] = dumped
    gen_type = str((getattr(obj, "generator", None) or {}).get("type") or "")
    grid = getattr(obj, "grid_data", None)
    store_brick = gen_type == GEN_IMPORTED or (
        grid is not None
        and gen_type not in (
            GEN_PYMOL_MAP,
            GEN_GAUSSIAN,
            GEN_DISTANCE,
            GEN_SIGNED_VDW,
            GEN_NEAREST_PROP,
            GEN_NEAREST_COLOR,
        )
    )
    if grid is not None and (store_brick or _ARRAY_STORE.get() is not None):
        data["brick"] = {
            "values": persist_numeric_array(grid.values),
            "step_sizes": persist_vector(grid.step_sizes),
            "step_counts": [int(v) for v in np.asarray(grid.step_counts).reshape(-1)],
            "origin": persist_vector(grid.origin),
            "name": getattr(grid, "_name", None) or getattr(grid, "name", None),
        }
    mode = getattr(obj, "preview_mode", None)
    if mode:
        from .wizards.builders.preview_mode import normalize_preview_mode

        data["preview_mode"] = normalize_preview_mode(mode)
    return data


def _load_field(cls, data: dict):
    from .fields.domain import Domain
    from .volumetric.GridData import GridData

    grid = None
    brick = data.get("brick")
    if brick:
        grid = GridData(
            resolve_numeric_array(brick["values"]),
            step_sizes=brick.get("step_sizes"),
            step_counts=brick.get("step_counts"),
            origin=brick.get("origin"),
            name=brick.get("name") or data.get("name"),
        )
        if brick.get("name"):
            grid._name = brick["name"]
    field = cls(
        name=data.get("name"),
        kind=data.get("kind"),
        units=data.get("units"),
        generator=data.get("generator"),
        domain=Domain.from_dict(data.get("domain")),
        provenance=data.get("provenance"),
        grid_data=grid,
        categories=data.get("categories"),
        obj_id=data.get("id"),
        default_color_field_id=data.get("default_color_field_id"),
    )
    if data.get("name"):
        field._name = data["name"]
    from .fields.field import remember_wrap

    remember_wrap(field)
    return field


def _colormap_name(obj) -> Optional[str]:
    cmap = getattr(obj, "colormap", None)
    if cmap is None:
        return None
    if isinstance(cmap, str):
        return cmap
    name = getattr(cmap, "name", None) or getattr(cmap, "_name", None)
    if name:
        return str(name)
    inner = getattr(cmap, "colormap", None)
    if isinstance(inner, str):
        return inner
    return None


def _dump_volumetric(obj) -> dict:
    data = _base_fields(obj)
    geom = getattr(obj, "geometry_field_id", None)
    if geom:
        data["geometry_field_id"] = str(geom)
    color_fid = getattr(obj, "color_field_id", None)
    if color_fid:
        data["color_field_id"] = str(color_fid)
    cmap = _colormap_name(obj)
    if cmap:
        data["colormap"] = cmap
    spec = getattr(obj, "colormap_spec", None)
    if spec:
        data["colormap_spec"] = spec
    isovalues = getattr(obj, "isovalues", None)
    if isovalues:
        from .fields.isovalues import normalize_isovalues
        data["isovalues"] = normalize_isovalues(isovalues)
    if getattr(obj, "level", None) is not None:
        data["level"] = float(obj.level)
    if getattr(obj, "side", None) is not None:
        data["side"] = int(obj.side)
    color = getattr(obj, "color", None)
    if color is not None and not hasattr(color, "name"):
        try:
            data["color"] = [float(color[0]), float(color[1]), float(color[2])]
        except (TypeError, IndexError, ValueError):
            pass
    aabb = getattr(obj, "clip_aabb", None)
    if aabb:
        data["clip_aabb"] = aabb
    sel = getattr(obj, "selection", None)
    if sel:
        data["selection"] = str(sel)
    carve = getattr(obj, "carve", None)
    if carve is not None:
        try:
            data["carve"] = float(carve)
        except (TypeError, ValueError):
            pass
    stops = getattr(obj, "transfer_stops", None)
    if stops:
        data["transfer_stops"] = list(stops)
    if getattr(obj, "clims", None) is not None:
        data["clims"] = [float(v) for v in np.asarray(obj.clims, dtype=float).reshape(-1)]
    if getattr(obj, "alphas", None) is not None:
        data["alphas"] = [float(v) for v in np.asarray(obj.alphas, dtype=float).reshape(-1)]
    return data


def _resolve_geometry_grid(data: dict):
    from .fields.field import ensure_brick
    from .runtime.session import get as session_get
    from .util.field_sample import resolve_grid_from_session

    fid = data.get("geometry_field_id") or data.get("field_id")
    if not fid:
        return None, None
    key = str(fid)
    field = session_get(key)
    if field is None:
        field = _INFLIGHT.get(key)
    if field is not None:
        return field, ensure_brick(field)
    grid = resolve_grid_from_session(fid)
    return field, grid


def _load_volumetric(cls, data: dict):
    field, grid = _resolve_geometry_grid(data)
    if grid is None:
        from .volumetric.GridData import GridData

        grid = GridData(
            np.zeros(8),
            step_sizes=(1.0, 1.0, 1.0),
            step_counts=(1, 1, 1),
            origin=(0.0, 0.0, 0.0),
            name=data.get("name") or "field",
        )
    geom_id = data.get("geometry_field_id")
    if not geom_id and field is not None:
        geom_id = field.id
    name = data.get("name")
    obj_id = data.get("id")
    kind = data.get("type")
    from .util.colormap_spec import volume_colormap_arg

    cmap = volume_colormap_arg(data.get("colormap", "RdYlBu_r"), data.get("colormap_spec"))
    kwargs = dict(
        geometry_field_id=geom_id,
        color_field_id=data.get("color_field_id"),
        clip_aabb=data.get("clip_aabb"),
        selection=data.get("selection"),
        carve=data.get("carve"),
    )
    if kind in ("Volume", "IsoVolume"):
        visual = cls(
            grid,
            name=name,
            colormap=cmap,
            alphas=data.get("alphas"),
            clims=data.get("clims"),
            transfer_stops=data.get("transfer_stops"),
            **kwargs
        )
    else:
        visual = cls(
            grid,
            float(data.get("level", 0.0) or 0.0),
            name=name,
            color=data.get("color"),
            transparency=data.get("transparency", 0),
            side=int(data.get("side", 1) or 1),
            isovalues=data.get("isovalues"),
            **kwargs
        )
    if obj_id:
        visual.id = str(obj_id)
    spec = data.get("colormap_spec")
    if spec:
        visual.colormap_spec = spec
    return visual


def _ensure_displayable_types() -> Dict[str, type]:
    global _DISPLAYABLE_TYPES
    if _DISPLAYABLE_TYPES is not None:
        return _DISPLAYABLE_TYPES
    from .fields.field import Field
    from .meshes.Arrows import Arrows
    from .meshes.CenteredBox import CenteredBox
    from .meshes.CGOCollection import CGOCollection
    from .meshes.ConvexHull import ConvexHull
    from .meshes.Cylinder import Cylinder
    from .meshes.Mesh import Mesh
    from .meshes.Plane import Plane
    from .meshes.Points import Points
    from .meshes.Sphere import Sphere
    from .meshes.Surface import Surface
    from .meshes.derived.PolylineTube import PolylineTube
    from .meshes.derived.Rotation_Indicator import Rotation_Indicator
    from .volumetric.IsoMesh import IsoMesh
    from .volumetric.IsoSurface import IsoSurface
    from .volumetric.IsoVolume import IsoVolume
    from .volumetric.Volume import Volume

    _DISPLAYABLE_TYPES = {
        "Sphere": Sphere,
        "Surface": Surface,
        "Cylinder": Cylinder,
        "CenteredBox": CenteredBox,
        "Lines": Arrows,
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
        "Field": Field,
    }
    return _DISPLAYABLE_TYPES


_DUMPERS: Dict[str, Callable] = {
    "Sphere": _dump_sphere,
    "Surface": _dump_surface,
    "Cylinder": _dump_cylinder,
    "CenteredBox": _dump_box,
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
    "Field": _dump_field,
}

_LOADERS: Dict[str, Callable] = {
    "Sphere": _load_sphere,
    "Surface": _load_surface,
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
    "IsoSurface": _load_volumetric,
    "IsoMesh": _load_volumetric,
    "IsoVolume": _load_volumetric,
    "Volume": _load_volumetric,
    "Field": _load_field,
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
        raise SerializationError("Unknown Displayable type %r" % typ)
    obj = loader(cls, data)
    _restore_preview_mode(obj, data)
    oid = getattr(obj, "id", None)
    if oid:
        _INFLIGHT[str(oid)] = obj
    if typ not in ("CGOCollection", "Field", "IsoSurface", "IsoMesh", "IsoVolume", "Volume"):
        from .util.field_sample import apply_stored_field_color
        apply_stored_field_color(obj, data)
    return obj


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
    objects = list(objects)
    fields = [obj for obj in objects if type(obj).__name__ == "Field"]
    rest = [obj for obj in objects if type(obj).__name__ != "Field"]
    doc = {
        "schema": SCHEMA_VERSION,
        "objects": [displayable_to_dict(obj) for obj in fields + rest],
    }
    assert_plain(doc)
    return doc


def session_from_document(data: dict, *, strict: bool = False) -> list:
    """Deserialize objects from a session document."""
    if not isinstance(data, dict):
        return []
    _INFLIGHT.clear()
    out = []
    for item in data.get("objects", []):
        if not isinstance(item, dict):
            continue
        try:
            out.append(displayable_from_dict(item))
        except Exception:
            if strict:
                raise
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
