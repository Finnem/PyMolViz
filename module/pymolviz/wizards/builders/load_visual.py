"""Rebuild wizard editor state from a persisted mesh collection."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ...points import AtomPoint, PointSource
from ...serialization import persist_color
from ...util.line_style import LineStyle, default_head_length
from ...util.solvent_surface import (
    DEFAULT_ALGORITHM,
    DEFAULT_ATOM_RADIUS,
    DEFAULT_PROBE_RADIUS,
    DEFAULT_QUALITY,
    DEFAULT_RADIUS_MODE,
    DEFAULT_VDW_SCALE,
)
from ..catalog import editor_kind, mesh_children
from .pairs import DEFAULT_ARROW_WIDTH, VisualPair
from .points import VisualPoint, atom_point_name, atom_ref_from_point_source, manual_fallback_name
from .wireframe_quality import DEFAULT_WIREFRAME_QUALITY, WIREFRAME_QUALITY_PRESETS

RGB = Tuple[float, float, float]


def visual_point_from_source(
    source: PointSource,
    color: Sequence[float],
    alpha: float,
    existing: Sequence[VisualPoint] = (),
    radius=None,
) -> VisualPoint:
    xyz = _xyz(source)
    rgb = (float(color[0]), float(color[1]), float(color[2]))
    if isinstance(source, AtomPoint):
        name = atom_point_name(
            source.object,
            elem=source.name or "",
            resi=source.resi,
            chain=source.chain,
            index=source.atom_id,
        )
        used = {pt.name for pt in existing}
        base = name
        n = 1
        while name in used:
            name = "%s_%d" % (base, n)
            n += 1
        return VisualPoint(
            name,
            "selection",
            xyz[0],
            xyz[1],
            xyz[2],
            color=rgb,
            alpha=float(alpha),
            point_source=source,
            atom_ref=atom_ref_from_point_source(source),
            radius=None if radius is None else float(radius),
        )
    name = manual_fallback_name("pt", existing)
    return VisualPoint(
        name,
        "manual",
        xyz[0],
        xyz[1],
        xyz[2],
        color=rgb,
        alpha=float(alpha),
        point_source=source,
        radius=None if radius is None else float(radius),
    )


def points_from_mesh(obj) -> List[VisualPoint]:
    points: List[VisualPoint] = []
    children = mesh_children(obj)
    if children and type(children[0]).__name__ == "Surface":
        surf = children[0]
        color = _rgb(surf)
        alpha = _alpha(surf)
        for i, src in enumerate(getattr(surf, "point_sources", None) or ()):
            custom = None
            radii = getattr(surf, "point_radii", None) or ()
            if i < len(radii) and radii[i] is not None:
                custom = float(radii[i])
            points.append(visual_point_from_source(src, color, alpha, points, radius=custom))
        return points
    for child in children:
        src = getattr(child, "position", None) or getattr(child, "center", None)
        if src is None:
            continue
        color = _rgb(child)
        alpha = _alpha(child)
        points.append(visual_point_from_source(src, color, alpha, points))
    return points


def pairs_from_mesh(obj) -> List[VisualPair]:
    pairs: List[VisualPair] = []
    existing: List[VisualPoint] = []
    for child in mesh_children(obj):
        starts = getattr(child, "_start_sources", None) or getattr(child, "starts", None) or ()
        ends = getattr(child, "_end_sources", None) or getattr(child, "ends", None) or ()
        colors = _rgbs(child, len(starts))
        alphas = _alphas(child, len(starts))
        for i, (start, end) in enumerate(zip(starts, ends)):
            start_pt = visual_point_from_source(start, colors[i], alphas[i], existing)
            existing.append(start_pt)
            end_pt = visual_point_from_source(end, colors[i], alphas[i], existing)
            existing.append(end_pt)
            width = float(getattr(child, "shaft_radius", DEFAULT_ARROW_WIDTH) or DEFAULT_ARROW_WIDTH)
            radii = getattr(child, "pair_radii", None) or ()
            heads = getattr(child, "pair_heads", None) or ()
            styles = getattr(child, "pair_styles", None) or ()
            if i < len(radii):
                width = float(radii[i])
            if i < len(heads):
                head = float(heads[i])
            else:
                head = default_head_length(width)
            if i < len(styles) and styles[i] is not None:
                style = styles[i].copy() if hasattr(styles[i], "copy") else styles[i]
            else:
                mesh_style = getattr(child, "line_style", None)
                style = mesh_style.copy() if mesh_style is not None and hasattr(mesh_style, "copy") else (mesh_style or LineStyle())
            pairs.append(VisualPair(start_pt, end_pt, width=width, head=head, style=style))
    return pairs


def sphere_options(obj) -> dict:
    child = _first_child(obj)
    frequency = int(getattr(child, "frequency", 4) or 4)
    return {
        "radius": float(getattr(child, "radius", 1.0) or 1.0),
        "wireframe": bool(getattr(child, "wireframe", False)),
        "quality": _quality_from_frequency(frequency),
    }


def box_options(obj) -> dict:
    child = _first_child(obj)
    extent = getattr(child, "extent", (1.0, 1.0, 1.0))
    return {
        "extent": (float(extent[0]), float(extent[1]), float(extent[2])),
        "wireframe": bool(getattr(child, "wireframe", False)),
    }


def arrow_options(obj) -> dict:
    child = _first_child(obj)
    style = getattr(child, "line_style", None)
    return {
        "quality": int(getattr(child, "quality", 3) or 3),
        "line_style": style,
    }


def surface_options(obj) -> dict:
    child = _first_child(obj)
    return {
        "radius": float(getattr(child, "atom_radius", DEFAULT_ATOM_RADIUS) or DEFAULT_ATOM_RADIUS),
        "probe_radius": float(getattr(child, "probe_radius", DEFAULT_PROBE_RADIUS) or DEFAULT_PROBE_RADIUS),
        "algorithm": str(getattr(child, "algorithm", DEFAULT_ALGORITHM) or DEFAULT_ALGORITHM),
        "quality": int(getattr(child, "quality", DEFAULT_QUALITY) or DEFAULT_QUALITY),
        "wireframe": bool(getattr(child, "wireframe", False)),
        "radius_mode": str(getattr(child, "radius_mode", DEFAULT_RADIUS_MODE) or DEFAULT_RADIUS_MODE),
        "vdw_scale": float(getattr(child, "vdw_scale", DEFAULT_VDW_SCALE) or DEFAULT_VDW_SCALE),
        "point_radii": getattr(child, "point_radii", None),
        "clip_planes": list(getattr(child, "clip_planes", None) or []),
        "color": _rgb(child),
        "alpha": _alpha(child),
    }


def loadable_kind(obj) -> Optional[str]:
    return editor_kind(obj)


def _first_child(obj):
    children = mesh_children(obj)
    return children[0] if children else obj


def _xyz(source: PointSource) -> Tuple[float, float, float]:
    last = getattr(source, "last_xyz", None)
    if last is not None:
        return (float(last[0]), float(last[1]), float(last[2]))
    xyz = source.resolve(None)
    return (float(xyz[0]), float(xyz[1]), float(xyz[2]))


def _rgb(obj) -> RGB:
    raw = persist_color(obj)
    if raw and isinstance(raw[0], (list, tuple)):
        row = raw[0]
        return (float(row[0]), float(row[1]), float(row[2]))
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def _rgbs(obj, n: int) -> List[RGB]:
    raw = persist_color(obj)
    if not raw:
        return [(1.0, 0.85, 0.15)] * max(n, 0)
    if isinstance(raw[0], (int, float)):
        rgb = (float(raw[0]), float(raw[1]), float(raw[2]))
        return [rgb] * max(n, 0)
    out = []
    for i in range(max(n, 0)):
        row = raw[i] if i < len(raw) else raw[-1]
        out.append((float(row[0]), float(row[1]), float(row[2])))
    return out


def _alpha(obj) -> float:
    t = getattr(obj, "transparency", 0) or 0
    try:
        t[0]
        t = float(t[0])
    except (TypeError, IndexError):
        t = float(t)
    return max(0.0, min(1.0, 1.0 - t))


def _alphas(obj, n: int) -> List[float]:
    t = getattr(obj, "transparency", 0) or 0
    try:
        values = [float(x) for x in t]
    except TypeError:
        values = [float(t)]
    if not values:
        values = [0.0]
    out = []
    for i in range(max(n, 0)):
        item = values[i] if i < len(values) else values[-1]
        out.append(max(0.0, min(1.0, 1.0 - float(item))))
    return out


def _quality_from_frequency(frequency: int) -> int:
    freq = int(frequency or 4)
    best = DEFAULT_WIREFRAME_QUALITY
    best_d = 10**9
    for level, preset in WIREFRAME_QUALITY_PRESETS.items():
        delta = abs(int(preset.frequency) - freq)
        if delta < best_d:
            best = level
            best_d = delta
    return best
