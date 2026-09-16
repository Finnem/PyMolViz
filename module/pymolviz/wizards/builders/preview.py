"""Live uncommitted CGO preview via PyMOLRuntime (not written to session)."""

from __future__ import annotations

import uuid
from typing import Optional, Sequence

import numpy as np

from ...meshes.Arrows import Arrows
from ...meshes.CenteredBox import CenteredBox
from ...meshes.CGOCollection import CGOCollection
from ...meshes.ClipGizmo import ClipGizmo
from ...meshes.Sphere import Sphere
from ...meshes.Surface import Surface
from ...util.clip_drag import (
    CLIP_DRAG_NAME,
    apply_axis_drag_matrix,
    apply_drag_matrix,
    matrix_changed,
    read_object_matrix,
    start_clip_drag,
    stop_clip_drag,
)
from ...util.clip_gizmo import fit_plane_rectangle
from ...util.mesh_clip import clip_planes_match, normalize_clip_planes
from ...util.pymol_helpers import purge_objects
from ...util.solvent_surface import DEFAULT_RADIUS_MODE, DEFAULT_VDW_SCALE, normalize_point_enabled
from .pairs import complete_pairs
from .points import VisualPoint, enabled_points
from .wireframe_quality import effective_wireframe_quality

PREVIEW_SPHERE_NAME = "_pmv_prev_spheres"
PREVIEW_BOX_NAME = "_pmv_prev_boxes"
PREVIEW_ARROW_NAME = "_pmv_prev_arrows"
PREVIEW_ARROW_PENDING = "_pmv_prev_arr_pend"
PREVIEW_SURFACE_NAME = "_pmv_prev_surface"
PREVIEW_SURFACE_CLIP_NAME = "_pmv_prev_surface_clip"
PREVIEW_FIELD_ISO_NAME = "_pmv_prev_field_iso"
PREVIEW_FIELD_VISUAL_NAME = "_pmv_prev_field_visual"
PREVIEW_FIELD_VOLUME_NAME = "_pmv_prev_field_volume"
PREVIEW_FIELD_CLIP_NAME = "_pmv_prev_field_clip"
PREVIEW_DOMAIN_NAME = "_pmv_prev_domain"
PREVIEW_CLIP_NAME = PREVIEW_SURFACE_CLIP_NAME

PREVIEW_SPHERE_PREFIX = "_pmv_sph_"
PREVIEW_BOX_PREFIX = "_pmv_box_"
PREVIEW_MARKER_PREFIX = "_pmv_pt_"
PREVIEW_BOX_MARKER_PREFIX = "_pmv_box_mk_"
PREVIEW_ARROW_PREFIX = "_pmv_arr_"
PREVIEW_ARROW_MARKER_PREFIX = "_pmv_arr_mk_"

_SHIFT_EPS2 = 1e-16


def _enabled_vertex_span(obj):
    if obj is None:
        return None
    children = _mesh_children(obj)
    if not children:
        children = [obj]
    chunks = []
    for child in children:
        if not getattr(child, "enabled", True):
            continue
        src = getattr(child, "_source_vertices", None)
        verts = src if src is not None else getattr(child, "vertices", None)
        if verts is None:
            continue
        arr = np.asarray(verts, dtype=float).reshape(-1, 3)
        if arr.size:
            chunks.append(arr)
    if not chunks:
        return None
    return np.vstack(chunks)


def _runtime(cmd_):
    from ...runtime.runtime import get_runtime
    return get_runtime(cmd_)


def _mesh_children(obj):
    from ...meshes.CGOCollection import cgo_children

    return cgo_children(obj)


def _clone_collection(source, name: str) -> CGOCollection:
    children = []
    for child in _mesh_children(source):
        if hasattr(child, "clone_baked"):
            children.append(child.clone_baked())
        else:
            children.append(child)
    collection = CGOCollection(children, name=name)
    collection.transparency = getattr(source, "transparency", 0)
    collection.specular = bool(getattr(source, "specular", True))
    collection.state = getattr(source, "state", 1)
    return collection


def _mesh_xyz(mesh) -> np.ndarray:
    verts = np.asarray(mesh.vertices, dtype=float).reshape(-1, 3)
    return verts.mean(axis=0)


def _set_anchor(mesh, source) -> None:
    if hasattr(mesh, "position"):
        mesh.position = source
    elif hasattr(mesh, "center"):
        mesh.center = source


def _retarget_point_mesh(mesh, pt: VisualPoint) -> None:
    """Reuse ``mesh``: retarget its PointSource and shift baked vertices."""
    _set_anchor(mesh, pt.point_source)
    delta = np.asarray(pt.xyz(), dtype=float) - _mesh_xyz(mesh)
    if float(np.dot(delta, delta)) > _SHIFT_EPS2:
        mesh.shift_vertices(delta)
    apply_point_color_to_mesh(mesh, pt)


def apply_point_color_to_mesh(mesh, pt: VisualPoint) -> None:
    """Apply the point's RGB (solid or already sampled from a field) onto a baked mesh."""
    from ...util.field_sample import clear_mesh_field, paint_mesh_by_field, sample_rgb_at

    mesh.transparency = 1.0 - float(pt.alpha)
    rgb = pt.color
    field_id = getattr(pt, "field_id", None)
    if field_id:
        painted = paint_mesh_by_field(
            mesh,
            field_id,
            colormap=getattr(pt, "field_colormap", None),
            clims=getattr(pt, "field_clims", None),
            colormap_spec=getattr(pt, "field_colormap_spec", None),
            clim_mode=getattr(pt, "field_clim_mode", None),
        )
        if painted:
            return
        sampled = sample_rgb_at(
            pt.xyz(), field_id,
            colormap=getattr(pt, "field_colormap_spec", None) or getattr(pt, "field_colormap", None),
            clims=getattr(pt, "field_clims", None),
        )
        if sampled is not None:
            rgb = sampled
        mesh.field_id = str(field_id)
        mesh.field_colormap = getattr(pt, "field_colormap", None)
        mesh.field_colormap_spec = getattr(pt, "field_colormap_spec", None)
        mesh.field_clims = getattr(pt, "field_clims", None)
        mesh.field_clim_mode = getattr(pt, "field_clim_mode", None)
        mesh.color = np.array(rgb, dtype=float)
        mesh.bypass_colormap = True
        if hasattr(mesh, "invalidate_cgo_cache"):
            mesh.invalidate_cgo_cache()
        return
    clear_mesh_field(mesh)
    mesh.color = np.array(rgb, dtype=float)
    if hasattr(mesh, "invalidate_cgo_cache"):
        mesh.invalidate_cgo_cache()


def apply_surface_color_to_mesh(mesh, points: Sequence[VisualPoint]) -> bool:
    """Blend each point's RGB onto the solvent surface (field RGB is already sampled).

    Color-only updates retarget this mesh; they must not rebuild the surface.
    Returns False when colors/centers/alpha are unchanged.
    """
    from ...points import resolve_xyz
    from ...util.field_sample import clear_mesh_field, paint_mesh_by_field, paint_mesh_by_point_colors
    from ...util.solvent_surface import normalize_point_enabled, normalize_point_radii, resolve_atom_radii
    from .points import enabled_points, infer_color_mode, _colors_equal
    from .surface_params import COLOR_MODE_FIELD

    active = enabled_points(points)
    if not active:
        return False
    alpha = 1.0 - min(float(pt.alpha) for pt in active)
    mesh.transparency = alpha
    if infer_color_mode(points) == COLOR_MODE_FIELD:
        pt0 = active[0]
        painted = paint_mesh_by_field(
            mesh,
            pt0.field_id,
            colormap=pt0.field_colormap,
            clims=pt0.field_clims,
            colormap_spec=getattr(pt0, "field_colormap_spec", None),
            clim_mode=getattr(pt0, "field_clim_mode", None),
        )
        mesh.point_colors = [tuple(float(c) for c in pt.color[:3]) for pt in points]
        return painted
    point_colors = [tuple(float(c) for c in pt.color[:3]) for pt in points]
    mesh.point_colors = point_colors
    flags = normalize_point_enabled([getattr(pt, "enabled", True) for pt in points], len(points))
    if flags is None:
        flags = [True] * len(points)
    radii_all = normalize_point_radii(getattr(mesh, "point_radii", None), len(points))
    centers = []
    colors = []
    active_sources = []
    active_radii = []
    for i, pt in enumerate(points):
        if not flags[i]:
            continue
        centers.append(tuple(float(v) for v in resolve_xyz(pt.point_source)))
        colors.append(pt.color[:3])
        active_sources.append(pt.point_source)
        active_radii.append(radii_all[i] if radii_all is not None else getattr(pt, "radius", None))
    verts = np.asarray(getattr(mesh, "vertices", []), dtype=float).reshape(-1, 3)
    vmean = tuple(np.round(verts.mean(axis=0), 5).tolist()) if verts.size else ()
    key = (
        int(verts.shape[0]),
        vmean,
        tuple(centers),
        tuple(tuple(float(c) for c in rgb[:3]) for rgb in colors),
        round(float(alpha), 5),
        tuple(bool(f) for f in flags),
    )
    if getattr(mesh, "_surface_color_key", None) == key:
        return False
    clear_mesh_field(mesh)
    if not centers:
        mesh._surface_color_key = key
        return False
    if len(colors) == 1 or all(_colors_equal(colors[0], rgb) for rgb in colors):
        mesh.color = np.array(colors[0], dtype=float)
        mesh._surface_color_key = key
        if hasattr(mesh, "invalidate_cgo_cache"):
            mesh.invalidate_cgo_cache()
        return True
    atom_radii = resolve_atom_radii(
        active_sources,
        len(active_sources),
        float(getattr(mesh, "atom_radius", 1.5)),
        getattr(mesh, "radius_mode", "uniform"),
        float(getattr(mesh, "vdw_scale", 1.0) or 1.0),
        active_radii,
        None,
    )
    probe = float(getattr(mesh, "probe_radius", 1.4))
    influence = [float(r) + probe for r in atom_radii]
    painted = paint_mesh_by_point_colors(mesh, centers, colors, influence)
    mesh._surface_color_key = key
    return painted


def _sphere_can_reuse(mesh, radius: float, wireframe: bool, frequency: int) -> bool:
    if type(mesh).__name__ != "Sphere":
        return False
    if bool(getattr(mesh, "wireframe", False)) != bool(wireframe):
        return False
    if int(getattr(mesh, "frequency", -1)) != int(frequency):
        return False
    current = float(getattr(mesh, "geom_radius", getattr(mesh, "radius", -1.0)))
    return abs(current - float(radius)) < 1e-7


def _box_can_reuse(mesh, extent: Sequence[float], wireframe: bool) -> bool:
    if type(mesh).__name__ != "CenteredBox":
        return False
    if bool(getattr(mesh, "wireframe", False)) != bool(wireframe):
        return False
    old = tuple(float(v) for v in getattr(mesh, "extent", ()))
    if len(old) != 3:
        return False
    return all(abs(a - float(b)) < 1e-7 for a, b in zip(old, extent))


def _style_close(a, b) -> bool:
    if a is b:
        return True
    if a is None or b is None:
        return False
    return (
        getattr(a, "dash", None) == getattr(b, "dash", None)
        and getattr(a, "start_head", None) == getattr(b, "start_head", None)
        and getattr(a, "end_head", None) == getattr(b, "end_head", None)
        and getattr(a, "ends", None) == getattr(b, "ends", None)
        and abs(float(getattr(a, "dash_scale", 0.0)) - float(getattr(b, "dash_scale", 0.0))) < 1e-7
        and abs(float(getattr(a, "start_margin", getattr(a, "margin", 0.0))) - float(getattr(b, "start_margin", getattr(b, "margin", 0.0)))) < 1e-7
        and abs(float(getattr(a, "end_margin", getattr(a, "margin", 0.0))) - float(getattr(b, "end_margin", getattr(b, "margin", 0.0)))) < 1e-7
    )


def _arrow_can_reuse(mesh, quality: int, n_pairs: int) -> bool:
    if type(mesh).__name__ != "Arrows":
        return False
    if int(getattr(mesh, "quality", -1)) != int(quality):
        return False
    if not bool(getattr(mesh, "use_styled_cgo", False)):
        return False
    verts = np.asarray(getattr(mesh, "vertices", []), dtype=float).reshape(-1, 3)
    if verts.shape[0] != n_pairs * 2:
        return False
    return True


def _invalidate_merged(collection) -> None:
    if collection is None:
        return
    if hasattr(collection, "invalidate_merged_cache"):
        collection.invalidate_merged_cache()
        return
    collection._cached_merged_resolved = None
    collection._child_spans = None
    collection._child_serials = None


def _mix_highlight(rgb, amount=0.4):
    return tuple(min(1.0, float(c) * (1.0 - amount) + amount) for c in rgb[:3])


def _preview_pairs(pairs, highlight_id=None):
    ready = complete_pairs(pairs)
    if not highlight_id:
        return ready
    out = []
    for pair in ready:
        if pair.pair_id != highlight_id:
            out.append(pair)
            continue
        brighter = pair.with_start(
            pair.start.with_color(_mix_highlight(pair.start.color)),
        ).with_end(
            pair.end.with_color(_mix_highlight(pair.end.color)),
        )
        out.append(brighter.with_width(float(pair.width) * 1.35))
    return out


def _style_copy(style):
    if style is None:
        from ...util.line_style import LineStyle
        return LineStyle()
    copy = getattr(style, "copy", None)
    if callable(copy):
        return copy()
    return style


def _style_pair_mesh(mesh, pairs) -> None:
    mesh.pair_radii = [float(pair.width) for pair in pairs]
    mesh.pair_heads = [float(pair.head) for pair in pairs]
    mesh.pair_styles = [_style_copy(getattr(pair, "style", None)) for pair in pairs]
    if pairs:
        width = float(pairs[0].width)
        mesh.shaft_radius = width
        mesh.linewidth = width
        mesh.line_style = _style_copy(getattr(pairs[0], "style", None))


def _arrow_endpoint_color(pt, fallback) -> tuple:
    if pt is None:
        return tuple(float(c) for c in fallback[:3])
    choice = pt.color_choice()
    rgb = tuple(float(c) for c in pt.color[:3])
    if not getattr(choice, "field_id", None):
        return rgb
    from ...util.field_sample import sample_rgb_at
    sampled = sample_rgb_at(
        pt.xyz(), choice.field_id, choice.colormap, choice.clims, smooth=0.0,
    )
    return sampled or rgb


def _arrow_endpoint_colors(pair) -> tuple:
    start = _arrow_endpoint_color(pair.start, pair.color)
    end = _arrow_endpoint_color(pair.end, start)
    return start, end


def _arrow_pair_color(pair) -> tuple:
    return _arrow_endpoint_colors(pair)[0]


def _arrow_colors_array(pairs):
    rows = []
    for pair in pairs:
        start, end = _arrow_endpoint_colors(pair)
        rows.append(start)
        rows.append(end)
    return np.array(rows, dtype=float)


def _apply_arrow_field_color(mesh, pairs) -> None:
    from ...util.field_sample import clear_mesh_field, paint_mesh_by_field

    field_ids = {getattr(pair.start, "field_id", None) for pair in pairs}
    if len(field_ids) == 1:
        field_id = next(iter(field_ids))
        if field_id:
            paint_mesh_by_field(
                mesh,
                field_id,
                colormap=pairs[0].start.field_colormap,
                clims=pairs[0].start.field_clims,
                refine=False,
                colormap_spec=getattr(pairs[0].start, "field_colormap_spec", None),
                clim_mode=getattr(pairs[0].start, "field_clim_mode", None),
            )
            return
    clear_mesh_field(mesh)
    mesh.color = _arrow_colors_array(pairs)
    if hasattr(mesh, "invalidate_cgo_cache"):
        mesh.invalidate_cgo_cache()


def _retarget_arrows(mesh, pairs) -> None:
    mesh._start_sources = [pair.start.point_source for pair in pairs]
    mesh._end_sources = [pair.end.point_source for pair in pairs]
    verts = []
    trans = []
    for pair in pairs:
        verts.append(pair.start.xyz())
        verts.append(pair.end.xyz())
        trans.append(1.0 - float(pair.alpha))
    mesh.vertices = np.array(verts, dtype=float)
    mesh.transparency = trans
    _style_pair_mesh(mesh, pairs)
    _apply_arrow_field_color(mesh, pairs)
    mesh._pair_spans = None


def _arrow_color_rows(mesh) -> np.ndarray:
    return np.asarray(mesh.color, dtype=float).reshape(-1, 3)


def _append_arrow_pair_arrays(mesh, pair) -> None:
    start = np.asarray(pair.start.xyz(), dtype=float)
    end = np.asarray(pair.end.xyz(), dtype=float)
    verts = np.asarray(mesh.vertices, dtype=float).reshape(-1, 3)
    n_pairs = verts.shape[0] // 2
    mesh.vertices = np.vstack([verts, start, end])
    mesh._start_sources = list(mesh._start_sources or []) + [pair.start.point_source]
    mesh._end_sources = list(mesh._end_sources or []) + [pair.end.point_source]
    colors = _arrow_color_rows(mesh)
    extra = np.asarray(_arrow_endpoint_colors(pair), dtype=float).reshape(2, 3)
    if colors.shape[0] == n_pairs:
        extra = extra[:1]
    elif colors.shape[0] != n_pairs * 2:
        extra = extra[:1]
    mesh.color = np.vstack([colors, extra])
    trans = [float(x) for x in np.atleast_1d(mesh.transparency)]
    trans.append(1.0 - float(pair.alpha))
    mesh.transparency = trans
    radii = list(getattr(mesh, "pair_radii", None) or [])
    heads = list(getattr(mesh, "pair_heads", None) or [])
    radii.append(float(getattr(pair, "width", getattr(mesh, "shaft_radius", 0.045))))
    heads.append(float(getattr(pair, "head", 0.36)))
    mesh.pair_radii = radii
    mesh.pair_heads = heads
    styles = list(getattr(mesh, "pair_styles", None) or [])
    styles.append(_style_copy(getattr(pair, "style", None)))
    mesh.pair_styles = styles


def _remove_arrow_pair_arrays(mesh, index: int) -> None:
    verts = np.asarray(mesh.vertices, dtype=float).reshape(-1, 2, 3)
    n_pairs = verts.shape[0]
    mesh.vertices = np.delete(verts, index, axis=0).reshape(-1, 3)
    starts = list(mesh._start_sources or [])
    ends = list(mesh._end_sources or [])
    if 0 <= index < len(starts):
        del starts[index]
    if 0 <= index < len(ends):
        del ends[index]
    mesh._start_sources = starts
    mesh._end_sources = ends
    colors = _arrow_color_rows(mesh)
    if colors.shape[0] == n_pairs * 2:
        mesh.color = np.delete(colors, [index * 2, index * 2 + 1], axis=0)
    elif colors.shape[0] > index:
        mesh.color = np.delete(colors, index, axis=0)
    trans = [float(x) for x in np.atleast_1d(mesh.transparency)]
    if 0 <= index < len(trans):
        del trans[index]
    mesh.transparency = trans
    radii = list(getattr(mesh, "pair_radii", None) or [])
    heads = list(getattr(mesh, "pair_heads", None) or [])
    if 0 <= index < len(radii):
        del radii[index]
    if 0 <= index < len(heads):
        del heads[index]
    mesh.pair_radii = radii
    mesh.pair_heads = heads
    styles = list(getattr(mesh, "pair_styles", None) or [])
    if 0 <= index < len(styles):
        del styles[index]
    mesh.pair_styles = styles


class RuntimeCollectionPreview:
    """One ephemeral CGOCollection synced through the runtime."""

    def __init__(self, cmd_, name: str):
        self.cmd = cmd_
        self.name = name
        self._obj = None
        self.specular = True

    def adopt(self, collection: CGOCollection):
        """Materialize already-baked meshes without calling ``rebuild``."""
        if collection is None or len(collection) == 0:
            self.cleanup()
            return
        collection.name = self.name
        self.specular = bool(getattr(collection, "specular", True))
        runtime = _runtime(self.cmd)
        if self._obj is not None:
            try:
                runtime.remove(self._obj)
            except Exception:
                pass
        collection.id = "preview_" + uuid.uuid4().hex
        self._obj = collection
        runtime.materialize(collection, rebuild=False)

    def update_collection(self, collection: Optional[CGOCollection]):
        if collection is None or len(collection) == 0:
            self.cleanup()
            return
        collection.name = self.name
        collection.specular = bool(self.specular)
        runtime = _runtime(self.cmd)
        if self._obj is None:
            collection.id = "preview_" + uuid.uuid4().hex
            self._obj = collection
            runtime.materialize(collection, rebuild=False)
            return
        collection.id = self._obj.id
        self._obj = collection
        runtime.replace_cgo(collection)

    def take(self):
        """Release the live collection without deleting the PyMOL object."""
        obj = self._obj
        self._obj = None
        return obj

    def set_specular(self, enabled: bool) -> None:
        self.specular = bool(enabled)
        if self._obj is None:
            return
        self._obj.specular = self.specular
        for child in self._obj:
            child.specular = self.specular
            if hasattr(child, "invalidate_cgo_cache"):
                child.invalidate_cgo_cache()
        if hasattr(self._obj, "invalidate_merged_cache"):
            self._obj.invalidate_merged_cache()
        self.push_tokens()

    def push_tokens(self):
        if self._obj is None:
            return
        self._obj.specular = bool(self.specular)
        _runtime(self.cmd).replace_cgo(self._obj)

    def set_children(self, children, transparency=0):
        if not children:
            self.cleanup()
            return
        if self._obj is None:
            collection = CGOCollection(children, name=self.name)
            collection.transparency = transparency
            collection.specular = bool(self.specular)
            self.update_collection(collection)
            return
        self._obj.clear()
        self._obj.extend(children)
        self._obj.transparency = transparency
        self._obj.specular = bool(self.specular)
        _invalidate_merged(self._obj)
        self.push_tokens()

    def append_child(self, mesh, transparency=None, push=True):
        if self._obj is None:
            collection = CGOCollection([mesh], name=self.name)
            if transparency is not None:
                collection.transparency = transparency
            collection.specular = bool(self.specular)
            self.update_collection(collection)
            return
        self._obj.append(mesh)
        _invalidate_merged(self._obj)
        if transparency is not None:
            self._obj.transparency = transparency
        self._obj.specular = bool(self.specular)
        if push:
            self.push_tokens()

    def remove_child_at(self, index: int, push=True):
        if self._obj is None:
            return
        if index < 0 or index >= len(self._obj):
            return
        if len(self._obj) == 1:
            self.cleanup()
            return
        del self._obj[index]
        _invalidate_merged(self._obj)
        if push:
            self.push_tokens()

    def cleanup(self):
        if self._obj is None:
            purge_objects(self.cmd, names=(self.name,), prefixes=(self.name,))
            return
        try:
            _runtime(self.cmd).remove(self._obj)
        except Exception:
            purge_objects(self.cmd, names=(self.name,))
        self._obj = None


def build_cgo_collection(
    points: Sequence[VisualPoint],
    radius: float,
    wireframe: bool,
    name: str,
    color=None,
    wireframe_quality: int = 3,
    clip_planes=None,
) -> CGOCollection:
    active = enabled_points(points)
    quality = effective_wireframe_quality(wireframe_quality, len(active), wireframe=wireframe)
    meshes = []
    clips = normalize_clip_planes(clip_planes)
    for pt in points:
        sphere = Sphere(
            pt.point_source,
            float(radius),
            color=pt.color,
            frequency=quality.frequency,
            wireframe=wireframe,
            clip_planes=clips,
            enabled=bool(getattr(pt, "enabled", True)),
            bypass_colormap=True,
            transparency=1.0 - float(pt.alpha),
        )
        apply_point_color_to_mesh(sphere, pt)
        meshes.append(sphere)
    collection = CGOCollection(meshes, name=name)
    if active:
        collection.transparency = 1.0 - min(float(pt.alpha) for pt in active)
    return collection


def build_box_cgo_collection(
    points: Sequence[VisualPoint],
    extent: Sequence[float],
    wireframe: bool,
    name: str,
    clip_planes=None,
) -> CGOCollection:
    active = enabled_points(points)
    meshes = []
    clips = normalize_clip_planes(clip_planes)
    for pt in points:
        box = CenteredBox(
            pt.point_source,
            extent,
            color=pt.color,
            wireframe=wireframe,
            clip_planes=clips,
            enabled=bool(getattr(pt, "enabled", True)),
            bypass_colormap=True,
            transparency=1.0 - float(pt.alpha),
        )
        apply_point_color_to_mesh(box, pt)
        meshes.append(box)
    collection = CGOCollection(meshes, name=name)
    if active:
        collection.transparency = 1.0 - min(float(pt.alpha) for pt in active)
    return collection


def build_arrow_collection(pairs, quality: int, style, name: str, clip_planes=None, head_radius=None) -> CGOCollection:
    ready = complete_pairs(pairs)
    if not ready:
        return CGOCollection([], name=name)
    arrows = Arrows(
        starts=[pair.start.point_source for pair in ready],
        ends=[pair.end.point_source for pair in ready],
        color=_arrow_colors_array(ready),
        transparency=[1.0 - float(pair.alpha) for pair in ready],
        quality=int(quality),
        line_style=style or getattr(ready[0], "style", None),
        shaft_radius=float(ready[0].width),
        use_styled_cgo=True,
        bypass_colormap=True,
        name=name,
        clip_planes=clip_planes,
        head_radius=head_radius,
    )
    _style_pair_mesh(arrows, ready)
    _apply_arrow_field_color(arrows, ready)
    collection = CGOCollection([arrows], name=name)
    collection.transparency = 1.0 - min(float(pair.alpha) for pair in ready)
    return collection


def build_surface_collection(
    points: Sequence[VisualPoint],
    atom_radius: float,
    probe_radius: float,
    algorithm: str,
    quality: int,
    wireframe: bool,
    name: str,
    radius_mode: str = DEFAULT_RADIUS_MODE,
    vdw_scale: float = DEFAULT_VDW_SCALE,
    clip_planes=None,
) -> CGOCollection:
    active = enabled_points(points)
    if not active:
        return CGOCollection([], name=name)
    alpha = 1.0 - min(float(pt.alpha) for pt in active)
    mesh = Surface(
        [pt.point_source for pt in points],
        atom_radius=float(atom_radius),
        probe_radius=float(probe_radius),
        algorithm=algorithm,
        quality=int(quality),
        color=active[0].color,
        wireframe=bool(wireframe),
        radius_mode=radius_mode,
        vdw_scale=float(vdw_scale),
        point_radii=[getattr(pt, "radius", None) for pt in points],
        point_enabled=[getattr(pt, "enabled", True) for pt in points],
        clip_planes=clip_planes,
        bypass_colormap=True,
        transparency=alpha,
        name=name,
    )
    apply_surface_color_to_mesh(mesh, points)
    collection = CGOCollection([mesh], name=name)
    collection.transparency = alpha
    return collection


def persist_collection(cmd_, collection: CGOCollection, obj_id=None):
    from ...runtime.persist import persist_collection as _persist_collection

    return _persist_collection(cmd_, collection, obj_id=obj_id)


def persist_promoted(cmd_, collection: CGOCollection, name: str, obj_id=None):
    """Move a live preview collection into the session and reload CGO tokens."""
    from ...runtime.bindings import PyMOLBinding
    from ...runtime.integration import install
    from ...runtime.runtime import get_runtime
    from ...runtime.session import add as session_add
    from ...runtime.session import get as session_get
    from ...serialization import style_hash
    from ...util.sanitize import sanitize_pymol_string

    try:
        install(cmd_)
    except Exception:
        pass

    runtime = get_runtime(cmd_)
    old_id = str(collection.id)
    old_binding = runtime.bindings.get(old_id)
    old_pymol_name = old_binding.pymol_name if old_binding is not None else None

    collection.name = name
    if obj_id:
        collection.id = str(obj_id)
    elif str(collection.id).startswith("preview_"):
        collection.id = uuid.uuid4().hex

    existing = session_get(collection.id)
    if existing is not None and existing is not collection:
        try:
            runtime.remove(existing)
        except Exception:
            pass

    new_pymol_name = sanitize_pymol_string(collection.name)
    promoted = False
    if old_binding is not None:
        runtime.bindings.pop(old_id)
        if old_pymol_name == new_pymol_name:
            promoted = True
        elif old_pymol_name:
            try:
                cmd_.set_name(old_pymol_name, new_pymol_name)
                promoted = True
            except Exception:
                promoted = False
        if promoted:
            runtime.bindings.put(
                PyMOLBinding(
                    collection.id, new_pymol_name, "cgo",
                    style_hash=style_hash(collection),
                )
            )
            # Reload tokens into the (possibly renamed) object. Width / head
            # radius live on the Python mesh; renaming the preview CGO is not
            # enough if those fields changed since the last load.
            if hasattr(collection, "invalidate_merged_cache"):
                collection.invalidate_merged_cache()
            for child in collection:
                if hasattr(child, "invalidate_cgo_cache"):
                    child.invalidate_cgo_cache()
            runtime.replace_cgo(collection)
            try:
                cmd_.enable(new_pymol_name)
            except Exception:
                pass
    if not promoted:
        runtime.materialize(collection, rebuild=False)
        if old_pymol_name and old_pymol_name != new_pymol_name:
            try:
                cmd_.delete(old_pymol_name)
            except Exception:
                pass
    session_add(collection)


def persist_live_preview(
    cmd_,
    preview,
    name: str,
    obj_id=None,
    *,
    retarget,
    fallback,
):
    """Commit preview meshes into the session; remesh only if they cannot be reused."""
    collection = preview.collection
    if collection is None or len(collection) == 0 or not retarget(collection):
        persist_collection(cmd_, fallback(), obj_id=obj_id)
        preview.cleanup()
        return
    try:
        _runtime(cmd_).replace_cgo(collection)
    except Exception:
        pass
    taken = preview.take()
    persist_promoted(cmd_, taken, name, obj_id)
    preview.cleanup()


def retarget_point_collection(
    collection: CGOCollection,
    points: Sequence[VisualPoint],
    clip_planes=None,
) -> bool:
    if len(collection) != len(points):
        return False
    incoming_clips = normalize_clip_planes(clip_planes)
    for mesh, pt in zip(collection, points):
        _retarget_point_mesh(mesh, pt)
        mesh.enabled = bool(getattr(pt, "enabled", True))
        if hasattr(mesh, "set_clip_planes"):
            if not clip_planes_match(getattr(mesh, "clip_planes", None), incoming_clips):
                mesh.set_clip_planes(incoming_clips)
        elif incoming_clips:
            mesh.clip_planes = incoming_clips
    active = enabled_points(points)
    if active:
        collection.transparency = 1.0 - min(float(pt.alpha) for pt in active)
    _invalidate_merged(collection)
    return True


def retarget_arrow_collection(collection: CGOCollection, pairs, clip_planes=None, head_radius=None) -> bool:
    if len(collection) != 1:
        return False
    mesh = collection[0]
    if type(mesh).__name__ != "Arrows":
        return False
    _retarget_arrows(mesh, pairs)
    incoming = normalize_clip_planes(clip_planes)
    mesh.clip_planes = incoming
    mesh.head_radius = None if head_radius is None else float(head_radius)
    if hasattr(mesh, "invalidate_cgo_cache"):
        mesh.invalidate_cgo_cache()
    if pairs:
        collection.transparency = 1.0 - min(float(pair.alpha) for pair in pairs)
    _invalidate_merged(collection)
    return True


def retarget_surface_collection(
    collection: CGOCollection,
    points: Sequence[VisualPoint],
    atom_radius: float,
    probe_radius: float,
    algorithm: str,
    quality: int,
    wireframe: bool,
    radius_mode: str = DEFAULT_RADIUS_MODE,
    vdw_scale: float = DEFAULT_VDW_SCALE,
    clip_planes=None,
) -> bool:
    if len(collection) != 1:
        return False
    mesh = collection[0]
    if type(mesh).__name__ != "Surface":
        return False
    if str(getattr(mesh, "algorithm", "")) != str(algorithm):
        return False
    if int(getattr(mesh, "quality", -1)) != int(quality):
        return False
    if abs(float(getattr(mesh, "atom_radius", -1.0)) - float(atom_radius)) > 1e-7:
        return False
    if abs(float(getattr(mesh, "probe_radius", -1.0)) - float(probe_radius)) > 1e-7:
        return False
    from ...util.solvent_surface import normalize_point_radii, normalize_radius_mode
    if normalize_radius_mode(getattr(mesh, "radius_mode", DEFAULT_RADIUS_MODE)) != normalize_radius_mode(radius_mode):
        return False
    if abs(float(getattr(mesh, "vdw_scale", 1.0) or 1.0) - float(vdw_scale or 1.0)) > 1e-7:
        return False
    incoming = normalize_point_radii(
        [getattr(pt, "radius", None) for pt in points], len(points),
    )
    stored = normalize_point_radii(getattr(mesh, "point_radii", None), len(points))
    if incoming != stored:
        return False
    incoming_en = normalize_point_enabled(
        [getattr(pt, "enabled", True) for pt in points], len(points),
    )
    stored_en = normalize_point_enabled(getattr(mesh, "point_enabled", None), len(points))
    if incoming_en != stored_en:
        return False
    sources = list(getattr(mesh, "point_sources", None) or [])
    if len(sources) != len(points):
        return False
    from ...points import resolve_xyz
    for src, pt in zip(sources, points):
        old = np.asarray(resolve_xyz(src, None), dtype=float).reshape(3)
        new = np.asarray(pt.resolve(None), dtype=float).reshape(3)
        if float(np.max(np.abs(old - new))) > 1e-5:
            return False
    mesh.point_sources = [pt.point_source for pt in points]
    mesh.point_enabled = incoming_en
    wireframe_changed = bool(getattr(mesh, "wireframe", False)) != bool(wireframe)
    mesh.wireframe = bool(wireframe)
    incoming_clips = normalize_clip_planes(clip_planes)
    clips_changed = not clip_planes_match(getattr(mesh, "clip_planes", None), incoming_clips)
    if clips_changed:
        if hasattr(mesh, "set_clip_planes"):
            mesh.set_clip_planes(incoming_clips)
        else:
            mesh.clip_planes = incoming_clips
    color_changed = False
    active = enabled_points(points)
    if active:
        collection.transparency = 1.0 - min(float(pt.alpha) for pt in active)
        color_changed = bool(apply_surface_color_to_mesh(mesh, points))
        mesh.transparency = collection.transparency
    dirty = color_changed or clips_changed or wireframe_changed
    if dirty:
        if hasattr(mesh, "invalidate_cgo_cache"):
            mesh.invalidate_cgo_cache()
        _invalidate_merged(collection)
    collection._preview_dirty = dirty
    return True


def _pymol_binding_name(cmd_, obj):
    from ...runtime.runtime import get_runtime

    binding = get_runtime(cmd_).bindings.get(getattr(obj, "id", None))
    if binding is None:
        return None
    return binding.pymol_name


def _pymol_object_name(cmd_, obj):
    name = _pymol_binding_name(cmd_, obj)
    if name:
        return name
    stored = getattr(obj, "_name", None)
    if stored:
        return str(stored)
    return None


def set_visual_enabled(cmd_, obj, enabled: bool) -> None:
    name = _pymol_object_name(cmd_, obj)
    if not name:
        return
    try:
        if enabled:
            cmd_.enable(name)
        else:
            cmd_.disable(name)
    except Exception:
        pass


def visual_is_enabled(cmd_, obj) -> bool:
    name = _pymol_object_name(cmd_, obj)
    if not name:
        return True
    try:
        names = cmd_.get_names("objects", enabled_only=1)
        return str(name) in names
    except TypeError:
        try:
            names = cmd_.get_names("objects", 1)
            return str(name) in names
        except Exception:
            return True
    except Exception:
        return True


def delete_visual(cmd_, obj) -> None:
    from ...runtime.runtime import get_runtime
    from ...runtime.session import remove as session_remove

    try:
        get_runtime(cmd_).remove(obj)
    except Exception:
        pass
    session_remove(obj)


from .clip_modifier import ClipGizmoPreview  # noqa: E402


class SpherePreview:
    def __init__(self, cmd_):
        self._preview = RuntimeCollectionPreview(cmd_, PREVIEW_SPHERE_NAME)
        self._gizmos = ClipGizmoPreview(cmd_)

    @property
    def collection(self):
        return self._preview._obj

    def take(self):
        return self._preview.take()

    def adopt(self, source):
        """Open an existing visual using its baked meshes (no icosphere rebuild)."""
        self._preview.adopt(_clone_collection(source, PREVIEW_SPHERE_NAME))

    def span_points(self):
        return _enabled_vertex_span(self._preview._obj)

    def set_gizmos(self, planes, selected_index=None, span_points=None, attach_drag=True):
        if span_points is None:
            span_points = self.span_points()
        self._gizmos.set_gizmos(planes, selected_index, span_points, attach_drag)

    def drag_is_live(self) -> bool:
        return self._gizmos.drag_is_live()

    def set_specular(self, enabled: bool) -> None:
        self._preview.set_specular(enabled)

    def poll_clip_drag(self):
        return self._gizmos.poll_clip_drag()

    def release_clip_drag(self):
        return self._gizmos.release_clip_drag()

    def clear_meshes(self):
        self._preview.cleanup()

    def update(self, points, radius, wireframe, wireframe_quality: int = 3, clip_planes=None):
        if not points:
            self.cleanup()
            return
        active = enabled_points(points)
        quality = effective_wireframe_quality(
            wireframe_quality, len(active) or len(points), wireframe=wireframe,
        )
        clips = normalize_clip_planes(clip_planes)
        existing = list(self._preview._obj) if self._preview._obj is not None else []
        children = []
        for i, pt in enumerate(points):
            if i < len(existing) and _sphere_can_reuse(
                existing[i], radius, wireframe, quality.frequency,
            ):
                _retarget_point_mesh(existing[i], pt)
                existing[i].enabled = bool(getattr(pt, "enabled", True))
                if hasattr(existing[i], "set_clip_planes"):
                    if not clip_planes_match(getattr(existing[i], "clip_planes", None), clips):
                        existing[i].set_clip_planes(clips)
                children.append(existing[i])
            else:
                sphere = Sphere(
                    pt.point_source,
                    float(radius),
                    color=pt.color,
                    frequency=quality.frequency,
                    wireframe=wireframe,
                    clip_planes=clips,
                    enabled=bool(getattr(pt, "enabled", True)),
                    bypass_colormap=True,
                    transparency=1.0 - float(pt.alpha),
                )
                apply_point_color_to_mesh(sphere, pt)
                children.append(sphere)
        alpha = 1.0 - min(float(pt.alpha) for pt in (active or points))
        self._preview.set_children(children, transparency=alpha)

    def add_points(self, points, radius, wireframe, wireframe_quality: int = 3):
        if not points:
            return True
        if self._preview._obj is None:
            self.update(points, radius, wireframe, wireframe_quality)
            return True
        sample = self._preview._obj[0]
        if getattr(sample, "clip_planes", None):
            return False
        n_total = len(self._preview._obj) + len(points)
        quality = effective_wireframe_quality(
            wireframe_quality, n_total, wireframe=wireframe,
        )
        if not _sphere_can_reuse(sample, radius, wireframe, quality.frequency):
            return False
        for pt in points:
            sphere = Sphere(
                pt.point_source,
                float(radius),
                color=pt.color,
                frequency=quality.frequency,
                wireframe=wireframe,
                enabled=bool(getattr(pt, "enabled", True)),
                bypass_colormap=True,
                transparency=1.0 - float(pt.alpha),
            )
            apply_point_color_to_mesh(sphere, pt)
            self._preview.append_child(sphere, transparency=1.0 - float(pt.alpha), push=False)
        self._preview.push_tokens()
        return True

    def remove_rows(self, rows: Sequence[int]):
        for index in sorted(set(int(i) for i in rows), reverse=True):
            self._preview.remove_child_at(index, push=False)
        if self._preview._obj is not None:
            self._preview.push_tokens()

    def cleanup(self):
        self._gizmos.cleanup()
        self._preview.cleanup()
        purge_objects(
            self._preview.cmd,
            prefixes=(PREVIEW_SPHERE_PREFIX, PREVIEW_MARKER_PREFIX, PREVIEW_SPHERE_NAME),
        )


class BoxPreview:
    def __init__(self, cmd_):
        self._preview = RuntimeCollectionPreview(cmd_, PREVIEW_BOX_NAME)
        self._gizmos = ClipGizmoPreview(cmd_)

    @property
    def collection(self):
        return self._preview._obj

    def take(self):
        return self._preview.take()

    def adopt(self, source):
        self._preview.adopt(_clone_collection(source, PREVIEW_BOX_NAME))

    def span_points(self):
        return _enabled_vertex_span(self._preview._obj)

    def set_gizmos(self, planes, selected_index=None, span_points=None, attach_drag=True):
        if span_points is None:
            span_points = self.span_points()
        self._gizmos.set_gizmos(planes, selected_index, span_points, attach_drag)

    def drag_is_live(self) -> bool:
        return self._gizmos.drag_is_live()

    def set_specular(self, enabled: bool) -> None:
        self._preview.set_specular(enabled)

    def poll_clip_drag(self):
        return self._gizmos.poll_clip_drag()

    def release_clip_drag(self):
        return self._gizmos.release_clip_drag()

    def clear_meshes(self):
        self._preview.cleanup()

    def update(self, points, extent, wireframe, clip_planes=None):
        if not points:
            self.cleanup()
            return
        active = enabled_points(points)
        clips = normalize_clip_planes(clip_planes)
        existing = list(self._preview._obj) if self._preview._obj is not None else []
        children = []
        for i, pt in enumerate(points):
            if i < len(existing) and _box_can_reuse(existing[i], extent, wireframe):
                _retarget_point_mesh(existing[i], pt)
                existing[i].enabled = bool(getattr(pt, "enabled", True))
                if hasattr(existing[i], "set_clip_planes"):
                    if not clip_planes_match(getattr(existing[i], "clip_planes", None), clips):
                        existing[i].set_clip_planes(clips)
                children.append(existing[i])
            else:
                box = CenteredBox(
                    pt.point_source,
                    extent,
                    color=pt.color,
                    wireframe=wireframe,
                    clip_planes=clips,
                    enabled=bool(getattr(pt, "enabled", True)),
                    bypass_colormap=True,
                    transparency=1.0 - float(pt.alpha),
                )
                apply_point_color_to_mesh(box, pt)
                children.append(box)
        alpha = 1.0 - min(float(pt.alpha) for pt in (active or points))
        self._preview.set_children(children, transparency=alpha)

    def add_points(self, points, extent, wireframe):
        if not points:
            return True
        if self._preview._obj is None:
            self.update(points, extent, wireframe)
            return True
        sample = self._preview._obj[0]
        if getattr(sample, "clip_planes", None):
            return False
        if not _box_can_reuse(sample, extent, wireframe):
            return False
        for pt in points:
            box = CenteredBox(
                pt.point_source,
                extent,
                color=pt.color,
                wireframe=wireframe,
                enabled=bool(getattr(pt, "enabled", True)),
                bypass_colormap=True,
                transparency=1.0 - float(pt.alpha),
            )
            apply_point_color_to_mesh(box, pt)
            self._preview.append_child(box, transparency=1.0 - float(pt.alpha), push=False)
        self._preview.push_tokens()
        return True

    def remove_rows(self, rows: Sequence[int]):
        for index in sorted(set(int(i) for i in rows), reverse=True):
            self._preview.remove_child_at(index, push=False)
        if self._preview._obj is not None:
            self._preview.push_tokens()

    def cleanup(self):
        self._gizmos.cleanup()
        self._preview.cleanup()
        purge_objects(
            self._preview.cmd,
            prefixes=(PREVIEW_BOX_PREFIX, PREVIEW_BOX_MARKER_PREFIX, PREVIEW_BOX_NAME),
        )


class ArrowPreview:
    def __init__(self, cmd_):
        self.cmd = cmd_
        self._preview = RuntimeCollectionPreview(cmd_, PREVIEW_ARROW_NAME)
        self._pending = RuntimeCollectionPreview(cmd_, PREVIEW_ARROW_PENDING)
        self._gizmos = ClipGizmoPreview(cmd_)

    @property
    def collection(self):
        return self._preview._obj

    def take(self):
        return self._preview.take()

    def adopt(self, source):
        self._preview.adopt(_clone_collection(source, PREVIEW_ARROW_NAME))

    def span_points(self):
        return _enabled_vertex_span(self._preview._obj)

    def set_gizmos(self, planes, selected_index=None, span_points=None, attach_drag=True):
        if span_points is None:
            span_points = self.span_points()
        self._gizmos.set_gizmos(planes, selected_index, span_points, attach_drag)

    def drag_is_live(self) -> bool:
        return self._gizmos.drag_is_live()

    def set_specular(self, enabled: bool) -> None:
        self._preview.set_specular(enabled)

    def poll_clip_drag(self):
        return self._gizmos.poll_clip_drag()

    def release_clip_drag(self):
        return self._gizmos.release_clip_drag()

    def clear_meshes(self):
        self._preview.cleanup()
        self._pending.cleanup()

    def update(self, pairs, quality: int, style, pending=None, highlight_id=None,
               clip_planes=None, head_radius=None):
        ready = _preview_pairs(pairs, highlight_id)
        clips = normalize_clip_planes(clip_planes)
        if ready:
            existing = list(self._preview._obj) if self._preview._obj is not None else []
            mesh = existing[0] if existing else None
            if mesh is not None and _arrow_can_reuse(mesh, quality, len(ready)):
                _retarget_arrows(mesh, ready)
                mesh.clip_planes = clips
                mesh.head_radius = None if head_radius is None else float(head_radius)
                if hasattr(mesh, "invalidate_cgo_cache"):
                    mesh.invalidate_cgo_cache()
                alpha = 1.0 - min(float(pair.alpha) for pair in ready)
                self._preview.set_children([mesh], transparency=alpha)
            else:
                self._preview.update_collection(
                    build_arrow_collection(
                        ready, quality, style, PREVIEW_ARROW_NAME,
                        clip_planes=clips, head_radius=head_radius,
                    )
                )
        else:
            self._preview.cleanup()
        self._update_pending(pending)

    def add_pairs(self, new_pairs, quality: int, style, current_pairs=()):
        if not new_pairs:
            return True
        if self._preview._obj is None:
            self.update(list(current_pairs) + list(new_pairs), quality, style)
            return True
        mesh = self._preview._obj[0]
        if type(mesh).__name__ != "Arrows":
            return False
        if int(getattr(mesh, "quality", -1)) != int(quality):
            return False
        if getattr(mesh, "clip_planes", None):
            return False
        for pair in new_pairs:
            _append_arrow_pair_arrays(mesh, pair)
        mesh.invalidate_cgo_cache()
        mesh._pair_spans = None
        self._sync_arrow_merged(mesh)
        return True

    def remove_rows(self, rows, quality: int, style, current_pairs):
        if self._preview._obj is None:
            return
        mesh = self._preview._obj[0]
        if type(mesh).__name__ != "Arrows":
            return
        n_pairs = np.asarray(mesh.vertices, dtype=float).reshape(-1, 3).shape[0] // 2
        for index in sorted(set(int(i) for i in rows), reverse=True):
            if index < 0 or index >= n_pairs:
                continue
            _remove_arrow_pair_arrays(mesh, index)
            n_pairs -= 1
        mesh.invalidate_cgo_cache()
        mesh._pair_spans = None
        if n_pairs <= 0:
            self._preview.cleanup()
            return
        self._sync_arrow_merged(mesh)

    def _sync_arrow_merged(self, mesh):
        coll = self._preview._obj
        if coll is None:
            return
        trans = getattr(mesh, "transparency", 0)
        try:
            alpha = min(float(x) for x in trans)
        except TypeError:
            alpha = float(trans or 0)
        coll.transparency = alpha
        _invalidate_merged(coll)
        self._preview.push_tokens()

    def _update_pending(self, pending):
        if pending is None:
            self._pending.cleanup()
            return
        existing = list(self._pending._obj) if self._pending._obj is not None else []
        if existing and type(existing[0]).__name__ == "Sphere":
            _retarget_point_mesh(existing[0], pending)
            self._pending.push_tokens()
            return
        marker = Sphere(
            pending.point_source,
            0.16,
            color=(0.2, 0.85, 1.0),
            frequency=2,
            bypass_colormap=True,
        )
        self._pending.update_collection(CGOCollection([marker], name=PREVIEW_ARROW_PENDING))

    def cleanup(self):
        self._gizmos.cleanup()
        self._preview.cleanup()
        self._pending.cleanup()
        purge_objects(
            self.cmd,
            prefixes=(PREVIEW_ARROW_PREFIX, PREVIEW_ARROW_MARKER_PREFIX, PREVIEW_ARROW_PENDING),
        )


class NativeFieldVisualPreview:
    """One ephemeral IsoSurface / IsoMesh / Volume loaded through ``visual.load``."""

    def __init__(self, cmd_, name: str):
        self.cmd = cmd_
        self.name = name
        self.visual = None
        self.key = None

    def update(self, visual=None, key=None):
        if visual is None:
            self.cleanup()
            return
        if key is not None and key == self.key and self.visual is not None:
            return
        self.cleanup()
        from .field_preview import load_preview_field_visual

        load_preview_field_visual(self.cmd, visual)
        self.visual = visual
        self.key = key

    def cleanup(self):
        visual = self.visual
        self.visual = None
        self.key = None
        names = []
        if visual is not None:
            names.append(str(visual.name))
            names.append("%s_volume_color_ramp" % visual.name)
        purge_objects(self.cmd, names=names, prefixes=(self.name,))


class FromSelectionFieldPreview:
    """Atom markers or native IsoSurface (optional Volume) for From Selection fields."""

    def __init__(self, cmd_):
        self.cmd = cmd_
        self._markers = SpherePreview(cmd_)
        self._native = NativeFieldVisualPreview(cmd_, PREVIEW_FIELD_ISO_NAME)
        self._volume = NativeFieldVisualPreview(cmd_, PREVIEW_FIELD_VOLUME_NAME)
        self._domain = RuntimeCollectionPreview(cmd_, PREVIEW_DOMAIN_NAME)
        self._iso_key = None
        self._domain_key = None

    @property
    def visual(self):
        return self._native.visual

    @property
    def collection(self):
        if self._native.visual is not None:
            return None
        return self._markers.collection

    def take(self):
        return self._markers.take()

    def adopt(self, source):
        self._markers.adopt(source)

    def span_points(self):
        return self._markers.span_points()

    def set_gizmos(self, planes, selected_index=None, span_points=None, attach_drag=True):
        self._markers.set_gizmos(planes, selected_index, span_points, attach_drag)

    def drag_is_live(self) -> bool:
        return self._markers.drag_is_live()

    def set_specular(self, enabled: bool) -> None:
        self._markers.set_specular(enabled)

    def poll_clip_drag(self):
        return self._markers.poll_clip_drag()

    def release_clip_drag(self):
        return self._markers.release_clip_drag()

    def update(
        self,
        points,
        marker_radius,
        live_iso=False,
        iso_visual=None,
        volume_visual=None,
        iso_key=None,
        wireframe_quality: int = 1,
        domain_aabb=None,
        show_domain=False,
    ):
        self._set_domain_box(domain_aabb, show_domain=show_domain)
        if not points:
            self._iso_key = None
            self._native.cleanup()
            self._volume.cleanup()
            self._markers.cleanup()
            return
        if live_iso and iso_visual is not None:
            self._markers._preview.cleanup()
            if iso_key is not None and iso_key == self._iso_key and self._native.visual is not None:
                return
            self._native.update(iso_visual, key=iso_key)
            self._volume.update(volume_visual, key=iso_key)
            self._iso_key = iso_key
            return
        self._native.cleanup()
        self._volume.cleanup()
        self._iso_key = None
        self._markers.update(points, marker_radius, False, wireframe_quality)

    def _set_domain_box(self, aabb, show_domain=False):
        apply_domain_box_preview(self, aabb, show_domain=show_domain)

    def cleanup(self):
        self._iso_key = None
        self._domain_key = None
        self._domain.cleanup()
        self._native.cleanup()
        self._volume.cleanup()
        self._markers.cleanup()
        from .field_preview import PREVIEW_NATIVE_PREFIXES

        purge_objects(self.cmd, prefixes=PREVIEW_NATIVE_PREFIXES)


def apply_domain_box_preview(holder, aabb, show_domain=False, grid=None):
    """Wire Domain AABB on ``holder._domain`` (RuntimeCollectionPreview)."""
    from .field_preview import build_domain_box_collection, domain_box_key

    key = domain_box_key(aabb, grid=grid) if show_domain else None
    if key is None:
        holder._domain_key = None
        holder._domain.cleanup()
        return
    if key == getattr(holder, "_domain_key", None) and holder._domain._obj is not None:
        return
    holder._domain.update_collection(build_domain_box_collection(aabb, PREVIEW_DOMAIN_NAME, grid=grid))
    holder._domain_key = key


class FieldVisualPreview:
    """Ephemeral native IsoSurface / IsoMesh / Volume (not interned)."""

    def __init__(self, cmd_):
        self.cmd = cmd_
        self._native = NativeFieldVisualPreview(cmd_, PREVIEW_FIELD_VISUAL_NAME)
        self._domain = RuntimeCollectionPreview(cmd_, PREVIEW_DOMAIN_NAME)
        self._gizmos = ClipGizmoPreview(cmd_, PREVIEW_FIELD_CLIP_NAME)
        self._iso_key = None
        self._domain_key = None

    @property
    def visual(self):
        return self._native.visual

    @property
    def collection(self):
        return None

    def set_gizmos(self, planes, selected_index=None, span_points=None, attach_drag=True, axis_lock=True):
        self._gizmos.set_gizmos(
            planes,
            selected_index=selected_index,
            span_points=span_points,
            attach_drag=attach_drag,
            axis_lock=axis_lock,
        )

    def drag_is_live(self) -> bool:
        return self._gizmos.drag_is_live()

    def poll_clip_drag(self):
        return self._gizmos.poll_clip_drag()

    def release_clip_drag(self):
        return self._gizmos.release_clip_drag()

    def update(self, visual=None, iso_key=None, domain_aabb=None, show_domain=False, grid=None):
        apply_domain_box_preview(self, domain_aabb, show_domain=show_domain, grid=grid)
        if visual is None:
            self._iso_key = None
            self._native.cleanup()
            if not show_domain:
                self._domain_key = None
                self._domain.cleanup()
            return
        if iso_key is not None and iso_key == self._iso_key and self._native.visual is not None:
            return
        self._native.update(visual, key=iso_key)
        self._iso_key = iso_key

    def cleanup(self):
        self._iso_key = None
        self._domain_key = None
        self._gizmos.cleanup()
        self._domain.cleanup()
        self._native.cleanup()
        from .field_preview import PREVIEW_NATIVE_PREFIXES

        purge_objects(self.cmd, prefixes=PREVIEW_NATIVE_PREFIXES)


class SurfacePreview:
    def __init__(self, cmd_):
        self.cmd = cmd_
        self._preview = RuntimeCollectionPreview(cmd_, PREVIEW_SURFACE_NAME)
        self._gizmos = ClipGizmoPreview(cmd_, PREVIEW_SURFACE_CLIP_NAME)

    @property
    def collection(self):
        return self._preview._obj

    def take(self):
        return self._preview.take()

    def adopt(self, source):
        self._preview.adopt(_clone_collection(source, PREVIEW_SURFACE_NAME))

    def span_points(self):
        return _enabled_vertex_span(self._preview._obj)

    def set_gizmos(self, planes, selected_index=None, span_points=None, attach_drag=True):
        if span_points is None:
            span_points = self.span_points()
        self._gizmos.set_gizmos(planes, selected_index, span_points, attach_drag)

    def drag_is_live(self) -> bool:
        return self._gizmos.drag_is_live()

    def set_specular(self, enabled: bool) -> None:
        self._preview.set_specular(enabled)

    def poll_clip_drag(self):
        return self._gizmos.poll_clip_drag()

    def release_clip_drag(self):
        return self._gizmos.release_clip_drag()

    def clear_meshes(self):
        self._preview.cleanup()

    def update(self, points, atom_radius, probe_radius, algorithm, quality, wireframe,
               radius_mode=DEFAULT_RADIUS_MODE, vdw_scale=DEFAULT_VDW_SCALE,
               clip_planes=None, gizmo_planes=None, gizmo_selected=None):
        active = enabled_points(points)
        if not active:
            self.cleanup()
            return
        existing = self._preview._obj
        if existing is not None and retarget_surface_collection(
            existing, points, atom_radius, probe_radius, algorithm, quality, wireframe,
            radius_mode=radius_mode, vdw_scale=vdw_scale, clip_planes=clip_planes,
        ):
            if getattr(existing, "_preview_dirty", True):
                self._preview.push_tokens()
        else:
            collection = build_surface_collection(
                points, atom_radius, probe_radius, algorithm, quality, wireframe, PREVIEW_SURFACE_NAME,
                radius_mode=radius_mode, vdw_scale=vdw_scale, clip_planes=clip_planes,
            )
            self._preview.update_collection(collection)
        self.set_gizmos(
            gizmo_planes,
            selected_index=gizmo_selected,
            attach_drag=not self.drag_is_live(),
        )

    def cleanup(self):
        self._gizmos.cleanup()
        self._preview.cleanup()
        purge_objects(
            self._preview.cmd,
            prefixes=(PREVIEW_SURFACE_NAME, PREVIEW_SURFACE_CLIP_NAME, CLIP_DRAG_NAME),
        )
