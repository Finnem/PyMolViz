"""Live uncommitted CGO preview via PyMOLRuntime (not written to session)."""

from __future__ import annotations

import uuid
from typing import Optional, Sequence

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


def _runtime(cmd_):
    from ...runtime.runtime import get_runtime
    return get_runtime(cmd_)


class RuntimeCollectionPreview:
    """One ephemeral CGOCollection synced through the runtime."""

    def __init__(self, cmd_, name: str):
        self.cmd = cmd_
        self.name = name
        self._obj = None

    def update_collection(self, collection: Optional[CGOCollection]):
        if collection is None or len(collection) == 0:
            self.cleanup()
            return
        collection.name = self.name
        runtime = _runtime(self.cmd)
        if self._obj is None:
            collection.id = "preview_" + uuid.uuid4().hex
            self._obj = collection
            runtime.materialize(collection)
            return
        collection.id = self._obj.id
        self._obj = collection
        runtime.sync(collection)

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


def persist_collection(cmd_, collection: CGOCollection):
    """Create CGO: write to session and materialize under the user-facing name."""
    from ...runtime.integration import install
    from ...runtime.runtime import get_runtime
    from ...runtime.session import add as session_add

    try:
        install(cmd_)
    except Exception:
        pass
    session_add(collection)
    get_runtime(cmd_).materialize(collection)


class SpherePreview:
    def __init__(self, cmd_):
        self._preview = RuntimeCollectionPreview(cmd_, PREVIEW_SPHERE_NAME)

    def update(self, points, radius, wireframe, wireframe_quality: int = 3):
        if not points:
            self.cleanup()
            return
        self._preview.update_collection(
            build_cgo_collection(
                points, radius, wireframe, PREVIEW_SPHERE_NAME,
                wireframe_quality=wireframe_quality,
            )
        )

    def cleanup(self):
        self._preview.cleanup()
        purge_objects(
            self._preview.cmd,
            prefixes=(PREVIEW_SPHERE_PREFIX, PREVIEW_MARKER_PREFIX, PREVIEW_SPHERE_NAME),
        )


class BoxPreview:
    def __init__(self, cmd_):
        self._preview = RuntimeCollectionPreview(cmd_, PREVIEW_BOX_NAME)

    def update(self, points, extent, wireframe):
        if not points:
            self.cleanup()
            return
        self._preview.update_collection(
            build_box_cgo_collection(points, extent, wireframe, PREVIEW_BOX_NAME)
        )

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

    def update(self, pairs, quality: int, style, pending=None):
        if pairs:
            self._preview.update_collection(
                build_arrow_collection(pairs, quality, style, PREVIEW_ARROW_NAME)
            )
        else:
            self._preview.cleanup()
        if pending is None:
            self._pending.cleanup()
        else:
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
