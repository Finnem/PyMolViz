"""Live uncommitted CGO preview via PyMOLRuntime (not written to session)."""

from __future__ import annotations

import uuid
from typing import Optional, Sequence

import numpy as np

from ...meshes.Arrows import Arrows
from ...meshes.CenteredBox import CenteredBox
from ...meshes.CGOCollection import CGOCollection
from ...meshes.Sphere import Sphere
from ...util.pymol_helpers import purge_objects
from .points import VisualPoint
from .wireframe_quality import effective_wireframe_quality

PREVIEW_SPHERE_NAME = "_pmv_prev_spheres"
PREVIEW_BOX_NAME = "_pmv_prev_boxes"
PREVIEW_ARROW_NAME = "_pmv_prev_arrows"
PREVIEW_ARROW_PENDING = "_pmv_prev_arr_pend"

PREVIEW_SPHERE_PREFIX = "_pmv_sph_"
PREVIEW_BOX_PREFIX = "_pmv_box_"
PREVIEW_MARKER_PREFIX = "_pmv_pt_"
PREVIEW_BOX_MARKER_PREFIX = "_pmv_box_mk_"
PREVIEW_ARROW_PREFIX = "_pmv_arr_"
PREVIEW_ARROW_MARKER_PREFIX = "_pmv_arr_mk_"

_SHIFT_EPS2 = 1e-16


def _runtime(cmd_):
    from ...runtime.runtime import get_runtime
    return get_runtime(cmd_)


def _mesh_children(obj):
    if type(obj).__name__ == "CGOCollection":
        return list(obj)
    return [obj]


def _clone_collection(source, name: str) -> CGOCollection:
    children = []
    for child in _mesh_children(source):
        if hasattr(child, "clone_baked"):
            children.append(child.clone_baked())
        else:
            children.append(child)
    collection = CGOCollection(children, name=name)
    collection.transparency = getattr(source, "transparency", 0)
    collection.state = getattr(source, "state", 1)
    return collection


def _rgb3(color) -> tuple:
    arr = np.asarray(color, dtype=float).reshape(-1)
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _mesh_xyz(mesh) -> np.ndarray:
    verts = np.asarray(mesh.vertices, dtype=float).reshape(-1, 3)
    return verts.mean(axis=0)


def _set_anchor(mesh, source) -> None:
    if hasattr(mesh, "position"):
        mesh.position = source
    elif hasattr(mesh, "center"):
        mesh.center = source


def _scalar_transparency(mesh) -> float:
    t = getattr(mesh, "transparency", 0)
    try:
        t[0]
        return float(min(t))
    except (TypeError, IndexError):
        try:
            return float(t)
        except (TypeError, ValueError):
            return 0.0


def _retarget_point_mesh(mesh, pt: VisualPoint) -> None:
    """Reuse ``mesh``: retarget its PointSource and shift baked vertices."""
    _set_anchor(mesh, pt.point_source)
    delta = np.asarray(pt.xyz(), dtype=float) - _mesh_xyz(mesh)
    if float(np.dot(delta, delta)) > _SHIFT_EPS2:
        mesh.shift_vertices(delta)
    recolor = False
    if not np.allclose(_rgb3(mesh.color), pt.color, atol=1e-5):
        mesh.color = np.array(pt.color, dtype=float)
        recolor = True
    new_t = 1.0 - float(pt.alpha)
    if abs(_scalar_transparency(mesh) - new_t) > 1e-5:
        mesh.transparency = new_t
        recolor = True
    if recolor:
        mesh.invalidate_cgo_cache()


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
        and getattr(a, "ends", None) == getattr(b, "ends", None)
        and abs(float(getattr(a, "dash_scale", 0.0)) - float(getattr(b, "dash_scale", 0.0))) < 1e-7
        and abs(float(getattr(a, "margin", 0.0)) - float(getattr(b, "margin", 0.0))) < 1e-7
    )


def _arrow_can_reuse(mesh, quality: int, style, n_pairs: int) -> bool:
    if type(mesh).__name__ != "Arrows":
        return False
    if int(getattr(mesh, "quality", -1)) != int(quality):
        return False
    if not bool(getattr(mesh, "use_styled_cgo", False)):
        return False
    verts = np.asarray(getattr(mesh, "vertices", []), dtype=float).reshape(-1, 3)
    if verts.shape[0] != n_pairs * 2:
        return False
    return _style_close(getattr(mesh, "line_style", None), style)


def _invalidate_merged(collection) -> None:
    if collection is None:
        return
    if hasattr(collection, "invalidate_merged_cache"):
        collection.invalidate_merged_cache()
        return
    collection._cached_merged_resolved = None
    collection._child_spans = None
    collection._child_serials = None


def _retarget_arrows(mesh, pairs) -> None:
    mesh._start_sources = [pair.start.point_source for pair in pairs]
    mesh._end_sources = [pair.end.point_source for pair in pairs]
    verts = []
    colors = []
    trans = []
    for pair in pairs:
        verts.append(pair.start.xyz())
        verts.append(pair.end.xyz())
        colors.append(pair.color)
        trans.append(1.0 - float(pair.alpha))
    mesh.vertices = np.array(verts, dtype=float)
    mesh.color = np.array(colors, dtype=float)
    mesh.transparency = trans
    mesh.invalidate_cgo_cache()
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
    extra = np.asarray(pair.color, dtype=float).reshape(1, 3)
    if colors.shape[0] == n_pairs * 2:
        extra = np.vstack([extra, extra])
    mesh.color = np.vstack([colors, extra])
    trans = [float(x) for x in np.atleast_1d(mesh.transparency)]
    trans.append(1.0 - float(pair.alpha))
    mesh.transparency = trans


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


class RuntimeCollectionPreview:
    """One ephemeral CGOCollection synced through the runtime."""

    def __init__(self, cmd_, name: str):
        self.cmd = cmd_
        self.name = name
        self._obj = None

    def adopt(self, collection: CGOCollection):
        """Materialize already-baked meshes without calling ``rebuild``."""
        if collection is None or len(collection) == 0:
            self.cleanup()
            return
        collection.name = self.name
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

    def push_tokens(self):
        if self._obj is None:
            return
        _runtime(self.cmd).replace_cgo(self._obj)

    def set_children(self, children, transparency=0):
        if not children:
            self.cleanup()
            return
        if self._obj is None:
            collection = CGOCollection(children, name=self.name)
            collection.transparency = transparency
            self.update_collection(collection)
            return
        self._obj.clear()
        self._obj.extend(children)
        self._obj.transparency = transparency
        _invalidate_merged(self._obj)
        self.push_tokens()

    def append_child(self, mesh, transparency=None, push=True):
        if self._obj is None:
            collection = CGOCollection([mesh], name=self.name)
            if transparency is not None:
                collection.transparency = transparency
            self.update_collection(collection)
            return
        self._obj.append(mesh)
        _invalidate_merged(self._obj)
        if transparency is not None:
            self._obj.transparency = transparency
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
) -> CGOCollection:
    quality = effective_wireframe_quality(wireframe_quality, len(points), wireframe=wireframe)
    meshes = []
    for pt in points:
        sphere = Sphere(
            pt.point_source,
            float(radius),
            color=pt.color,
            frequency=quality.frequency,
            wireframe=wireframe,
            bypass_colormap=True,
            transparency=1.0 - float(pt.alpha),
        )
        meshes.append(sphere)
    collection = CGOCollection(meshes, name=name)
    if points:
        collection.transparency = 1.0 - min(float(pt.alpha) for pt in points)
    return collection


def build_box_cgo_collection(
    points: Sequence[VisualPoint],
    extent: Sequence[float],
    wireframe: bool,
    name: str,
) -> CGOCollection:
    meshes = []
    for pt in points:
        box = CenteredBox(
            pt.point_source,
            extent,
            color=pt.color,
            wireframe=wireframe,
            bypass_colormap=True,
            transparency=1.0 - float(pt.alpha),
        )
        meshes.append(box)
    collection = CGOCollection(meshes, name=name)
    if points:
        collection.transparency = 1.0 - min(float(pt.alpha) for pt in points)
    return collection


def build_arrow_collection(pairs, quality: int, style, name: str) -> CGOCollection:
    if not pairs:
        return CGOCollection([], name=name)
    arrows = Arrows(
        starts=[pair.start.point_source for pair in pairs],
        ends=[pair.end.point_source for pair in pairs],
        color=[pair.color for pair in pairs],
        transparency=[1.0 - float(pair.alpha) for pair in pairs],
        quality=int(quality),
        line_style=style,
        use_styled_cgo=True,
        bypass_colormap=True,
        name=name,
    )
    collection = CGOCollection([arrows], name=name)
    collection.transparency = 1.0 - min(float(pair.alpha) for pair in pairs)
    return collection


def persist_collection(cmd_, collection: CGOCollection, obj_id=None):
    """Write to session and materialize. ``obj_id`` replaces an existing visual."""
    from ...runtime.integration import install
    from ...runtime.runtime import get_runtime
    from ...runtime.session import add as session_add
    from ...runtime.session import get as session_get

    try:
        install(cmd_)
    except Exception:
        pass
    if obj_id:
        collection.id = str(obj_id)
    runtime = get_runtime(cmd_)
    existing = session_get(collection.id)
    if existing is not None:
        try:
            runtime.remove(existing)
        except Exception:
            pass
    session_add(collection)
    runtime.materialize(collection, rebuild=False)


def persist_promoted(cmd_, collection: CGOCollection, name: str, obj_id=None):
    """Move a live preview collection into the session without remeshing."""
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
            runtime._apply_transparency(collection, new_pymol_name)
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


def retarget_point_collection(collection: CGOCollection, points: Sequence[VisualPoint]) -> bool:
    if len(collection) != len(points):
        return False
    for mesh, pt in zip(collection, points):
        _retarget_point_mesh(mesh, pt)
    if points:
        collection.transparency = 1.0 - min(float(pt.alpha) for pt in points)
    _invalidate_merged(collection)
    return True


def retarget_arrow_collection(collection: CGOCollection, pairs) -> bool:
    if len(collection) != 1:
        return False
    mesh = collection[0]
    if type(mesh).__name__ != "Arrows":
        return False
    _retarget_arrows(mesh, pairs)
    if pairs:
        collection.transparency = 1.0 - min(float(pair.alpha) for pair in pairs)
    _invalidate_merged(collection)
    return True


def set_visual_enabled(cmd_, obj, enabled: bool) -> None:
    from ...runtime.runtime import get_runtime

    binding = get_runtime(cmd_).bindings.get(getattr(obj, "id", None))
    if binding is None:
        return
    try:
        if enabled:
            cmd_.enable(binding.pymol_name)
        else:
            cmd_.disable(binding.pymol_name)
    except Exception:
        pass


class SpherePreview:
    def __init__(self, cmd_):
        self._preview = RuntimeCollectionPreview(cmd_, PREVIEW_SPHERE_NAME)

    @property
    def collection(self):
        return self._preview._obj

    def take(self):
        return self._preview.take()

    def adopt(self, source):
        """Open an existing visual using its baked meshes (no icosphere rebuild)."""
        self._preview.adopt(_clone_collection(source, PREVIEW_SPHERE_NAME))

    def update(self, points, radius, wireframe, wireframe_quality: int = 3):
        if not points:
            self.cleanup()
            return
        quality = effective_wireframe_quality(
            wireframe_quality, len(points), wireframe=wireframe,
        )
        existing = list(self._preview._obj) if self._preview._obj is not None else []
        children = []
        for i, pt in enumerate(points):
            if i < len(existing) and _sphere_can_reuse(
                existing[i], radius, wireframe, quality.frequency,
            ):
                _retarget_point_mesh(existing[i], pt)
                children.append(existing[i])
            else:
                children.append(
                    Sphere(
                        pt.point_source,
                        float(radius),
                        color=pt.color,
                        frequency=quality.frequency,
                        wireframe=wireframe,
                        bypass_colormap=True,
                        transparency=1.0 - float(pt.alpha),
                    )
                )
        alpha = 1.0 - min(float(pt.alpha) for pt in points)
        self._preview.set_children(children, transparency=alpha)

    def add_points(self, points, radius, wireframe, wireframe_quality: int = 3):
        if not points:
            return True
        if self._preview._obj is None:
            self.update(points, radius, wireframe, wireframe_quality)
            return True
        n_total = len(self._preview._obj) + len(points)
        quality = effective_wireframe_quality(
            wireframe_quality, n_total, wireframe=wireframe,
        )
        sample = self._preview._obj[0]
        if not _sphere_can_reuse(sample, radius, wireframe, quality.frequency):
            return False
        for pt in points:
            sphere = Sphere(
                pt.point_source,
                float(radius),
                color=pt.color,
                frequency=quality.frequency,
                wireframe=wireframe,
                bypass_colormap=True,
                transparency=1.0 - float(pt.alpha),
            )
            self._preview.append_child(sphere, transparency=1.0 - float(pt.alpha), push=False)
        self._preview.push_tokens()
        return True

    def remove_rows(self, rows: Sequence[int]):
        for index in sorted(set(int(i) for i in rows), reverse=True):
            self._preview.remove_child_at(index, push=False)
        if self._preview._obj is not None:
            self._preview.push_tokens()

    def cleanup(self):
        self._preview.cleanup()
        purge_objects(
            self._preview.cmd,
            prefixes=(PREVIEW_SPHERE_PREFIX, PREVIEW_MARKER_PREFIX, PREVIEW_SPHERE_NAME),
        )


class BoxPreview:
    def __init__(self, cmd_):
        self._preview = RuntimeCollectionPreview(cmd_, PREVIEW_BOX_NAME)

    @property
    def collection(self):
        return self._preview._obj

    def take(self):
        return self._preview.take()

    def adopt(self, source):
        self._preview.adopt(_clone_collection(source, PREVIEW_BOX_NAME))

    def update(self, points, extent, wireframe):
        if not points:
            self.cleanup()
            return
        existing = list(self._preview._obj) if self._preview._obj is not None else []
        children = []
        for i, pt in enumerate(points):
            if i < len(existing) and _box_can_reuse(existing[i], extent, wireframe):
                _retarget_point_mesh(existing[i], pt)
                children.append(existing[i])
            else:
                children.append(
                    CenteredBox(
                        pt.point_source,
                        extent,
                        color=pt.color,
                        wireframe=wireframe,
                        bypass_colormap=True,
                        transparency=1.0 - float(pt.alpha),
                    )
                )
        alpha = 1.0 - min(float(pt.alpha) for pt in points)
        self._preview.set_children(children, transparency=alpha)

    def add_points(self, points, extent, wireframe):
        if not points:
            return True
        if self._preview._obj is None:
            self.update(points, extent, wireframe)
            return True
        sample = self._preview._obj[0]
        if not _box_can_reuse(sample, extent, wireframe):
            return False
        for pt in points:
            box = CenteredBox(
                pt.point_source,
                extent,
                color=pt.color,
                wireframe=wireframe,
                bypass_colormap=True,
                transparency=1.0 - float(pt.alpha),
            )
            self._preview.append_child(box, transparency=1.0 - float(pt.alpha), push=False)
        self._preview.push_tokens()
        return True

    def remove_rows(self, rows: Sequence[int]):
        for index in sorted(set(int(i) for i in rows), reverse=True):
            self._preview.remove_child_at(index, push=False)
        if self._preview._obj is not None:
            self._preview.push_tokens()

    def cleanup(self):
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

    @property
    def collection(self):
        return self._preview._obj

    def take(self):
        return self._preview.take()

    def adopt(self, source):
        self._preview.adopt(_clone_collection(source, PREVIEW_ARROW_NAME))

    def update(self, pairs, quality: int, style, pending=None):
        if pairs:
            existing = list(self._preview._obj) if self._preview._obj is not None else []
            mesh = existing[0] if existing else None
            if mesh is not None and _arrow_can_reuse(mesh, quality, style, len(pairs)):
                _retarget_arrows(mesh, pairs)
                alpha = 1.0 - min(float(pair.alpha) for pair in pairs)
                self._preview.set_children([mesh], transparency=alpha)
            else:
                self._preview.update_collection(
                    build_arrow_collection(pairs, quality, style, PREVIEW_ARROW_NAME)
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
        if not _style_close(getattr(mesh, "line_style", None), style):
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
        self._preview.cleanup()
        self._pending.cleanup()
        purge_objects(
            self.cmd,
            prefixes=(PREVIEW_ARROW_PREFIX, PREVIEW_ARROW_MARKER_PREFIX, PREVIEW_ARROW_PENDING),
        )
