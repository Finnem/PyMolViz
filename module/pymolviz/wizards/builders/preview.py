"""Live uncommitted CGO preview for mesh builders."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ...meshes.CGOCollection import CGOCollection
from ...meshes.Sphere import Sphere
from ...util.cgo import (
    resolve_cgo_tokens,
    solid_box_cgo,
    solid_sphere_cgo,
    wireframe_box_cgo,
    wireframe_sphere_cgo,
    wireframe_sphere_mesh_cgo,
)
from ...util.pymol_helpers import (
    load_cgo_no_zoom,
    purge_objects,
    replace_cgo_no_zoom,
    restore_view,
    set_cgo_transparency,
)
from ...util.view import scale_translation_ttt
from .points import VisualPoint
from .wireframe_quality import WireframeQuality, effective_wireframe_quality

PREVIEW_SPHERE_PREFIX = "_pmv_sph_"
PREVIEW_BOX_PREFIX = "_pmv_box_"
PREVIEW_MARKER_PREFIX = "_pmv_pt_"
PREVIEW_BOX_MARKER_PREFIX = "_pmv_box_mk_"
PREVIEW_ARROW_PREFIX = "_pmv_arr_"
PREVIEW_ARROW_MARKER_PREFIX = "_pmv_arr_mk_"
PREVIEW_ARROW_PENDING = "_pmv_arr_pend"

SPHERE_COLOR = (1.0, 0.85, 0.15)
MARKER_COLOR = (0.2, 0.85, 1.0)
MARKER_RADIUS = 0.12


def _resolve_cgo_tokens(content: list) -> list:
    return resolve_cgo_tokens(content)


def _unit_sphere_cgo(
    wireframe: bool,
    color=SPHERE_COLOR,
    quality: WireframeQuality = None,
    alpha: float = 1.0,
) -> list:
    """Unit-radius geometry centered at the origin (scale via object TTT)."""
    q = quality or effective_wireframe_quality(3, 1, wireframe=wireframe)
    if wireframe:
        return wireframe_sphere_mesh_cgo(
            (0.0, 0.0, 0.0),
            1.0,
            color,
            frequency=q.frequency,
            alpha=alpha,
        )
    return solid_sphere_cgo(
        (0.0, 0.0, 0.0),
        1.0,
        color,
        frequency=q.frequency,
        alpha=alpha,
    )


def _unit_marker_cgo() -> list:
    return wireframe_sphere_cgo((0.0, 0.0, 0.0), 1.0, MARKER_COLOR, n_lon=4, n_lat=3)


def build_sphere_meshes(
    points: Sequence[VisualPoint],
    radius: float,
    color=SPHERE_COLOR,
    resolution: int = 14,
) -> list:
    """Triangle/cylinder wireframe meshes — wireframe commit/export only."""
    import numpy as np

    meshes = []
    for pt in points:
        sphere = Sphere(np.array(pt.xyz()), float(radius), color=color, resolution=resolution)
        meshes.append(sphere.to_wireframe())
    return meshes


def build_sphere_cgo_list(
    points: Sequence[VisualPoint],
    radius: float,
    wireframe: bool,
    color=SPHERE_COLOR,
    wireframe_quality: int = 3,
) -> list:
    quality = effective_wireframe_quality(wireframe_quality, len(points), wireframe=wireframe)
    merged = []
    for pt in points:
        if wireframe:
            merged.extend(
                wireframe_sphere_mesh_cgo(
                    pt.xyz(),
                    float(radius),
                    pt.color,
                    frequency=quality.frequency,
                    alpha=pt.alpha,
                )
            )
        else:
            merged.extend(
                solid_sphere_cgo(
                    pt.xyz(),
                    float(radius),
                    pt.color,
                    frequency=quality.frequency,
                    alpha=pt.alpha,
                )
            )
    return merged


def build_cgo_collection(
    points: Sequence[VisualPoint],
    radius: float,
    wireframe: bool,
    name: str,
    color=SPHERE_COLOR,
    wireframe_quality: int = 3,
) -> CGOCollection:
    quality = effective_wireframe_quality(wireframe_quality, len(points), wireframe=wireframe)
    import numpy as np

    meshes = []
    for pt in points:
        sphere = Sphere(
            np.array(pt.xyz()),
            float(radius),
            color=pt.color,
            frequency=quality.frequency,
            bypass_colormap=True,
        )
        sphere.transparency = 1.0 - float(pt.alpha)
        meshes.append(sphere.to_wireframe() if wireframe else sphere)
    return _collection_with_alpha(meshes, name, points)


def build_box_cgo_list(
    points: Sequence[VisualPoint],
    extent: Sequence[float],
    wireframe: bool,
) -> list:
    merged = []
    for pt in points:
        if wireframe:
            merged.extend(wireframe_box_cgo(pt.xyz(), extent, pt.color, alpha=pt.alpha))
        else:
            merged.extend(solid_box_cgo(pt.xyz(), extent, pt.color, alpha=pt.alpha))
    return merged


def build_box_cgo_collection(
    points: Sequence[VisualPoint],
    extent: Sequence[float],
    wireframe: bool,
    name: str,
) -> CGOCollection:
    from ...meshes.CenteredBox import CenteredBox

    meshes = []
    for pt in points:
        box = CenteredBox(
            pt.xyz(),
            extent,
            color=pt.color,
            bypass_colormap=True,
            transparency=1.0 - float(pt.alpha),
        )
        meshes.append(box.to_wireframe() if wireframe else box)
    return _collection_with_alpha(meshes, name, points)


def _collection_with_alpha(meshes, name: str, points: Sequence[VisualPoint]) -> CGOCollection:
    collection = CGOCollection(meshes, name=name)
    if points:
        collection.transparency = 1.0 - min(float(pt.alpha) for pt in points)
    return collection


def _shared_alpha(points: Sequence[VisualPoint]) -> Optional[float]:
    if not points:
        return 1.0
    alphas = [float(pt.alpha) for pt in points]
    if max(alphas) - min(alphas) < 1e-6:
        return alphas[0]
    return None


def _box_cgo_for_point(pt: VisualPoint, extent: Sequence[float], wireframe: bool) -> list:
    if wireframe:
        return wireframe_box_cgo(pt.xyz(), extent, pt.color, alpha=pt.alpha)
    return solid_box_cgo(pt.xyz(), extent, pt.color, alpha=pt.alpha)


class BoxPreview:
    """Incremental preview: one box CGO per point at full size."""

    def __init__(self, cmd_):
        self.cmd = cmd_
        self._wireframe = None
        self._extent = None
        self._positions: List[Tuple[float, float, float]] = []
        self._colors: List[Tuple[float, float, float]] = []
        self._alphas: List[float] = []
        self._box_names: List[str] = []
        self._marker_names: List[str] = []

    def update(
        self,
        points: Sequence[VisualPoint],
        extent: Sequence[float],
        wireframe: bool,
    ):
        if not points:
            self.cleanup()
            return

        extent = (float(extent[0]), float(extent[1]), float(extent[2]))
        positions = [pt.xyz() for pt in points]
        colors = [pt.color for pt in points]
        alphas = [float(pt.alpha) for pt in points]
        wireframe_changed = wireframe != self._wireframe
        count_changed = len(points) != len(self._box_names)
        extent_changed = self._extent != extent
        positions_changed = positions != self._positions
        colors_changed = colors != self._colors
        alphas_changed = alphas != self._alphas

        self._sync_object_count(len(points))

        geometry_changed = (
            wireframe_changed
            or count_changed
            or extent_changed
            or positions_changed
            or colors_changed
            or alphas_changed
        )
        if geometry_changed:
            for i, pt in enumerate(points):
                unit = _resolve_cgo_tokens(_box_cgo_for_point(pt, extent, wireframe))
                replace_cgo_no_zoom(self.cmd, unit, self._box_names[i])
                set_cgo_transparency(self.cmd, self._box_names[i], pt.alpha)
            marker_unit = _resolve_cgo_tokens(_unit_marker_cgo())
            for name in self._marker_names:
                replace_cgo_no_zoom(self.cmd, marker_unit, name)
            for i, pos in enumerate(positions):
                self.cmd.set_object_ttt(
                    self._marker_names[i],
                    scale_translation_ttt(pos, MARKER_RADIUS),
                )
            self._wireframe = wireframe
        elif alphas_changed:
            for i, pt in enumerate(points):
                set_cgo_transparency(self.cmd, self._box_names[i], pt.alpha)

        self._extent = extent
        self._positions = positions
        self._colors = list(colors)
        self._alphas = list(alphas)

    def _sync_object_count(self, count: int):
        while len(self._box_names) < count:
            idx = len(self._box_names)
            box_name = "%s%d" % (PREVIEW_BOX_PREFIX, idx)
            marker_name = "%s%d" % (PREVIEW_BOX_MARKER_PREFIX, idx)
            unit = _resolve_cgo_tokens(_box_cgo_for_point(
                VisualPoint("tmp", "tmp", 0.0, 0.0, 0.0),
                (2.0, 2.0, 2.0),
                False,
            ))
            marker_unit = _resolve_cgo_tokens(_unit_marker_cgo())
            load_cgo_no_zoom(self.cmd, unit, box_name)
            load_cgo_no_zoom(self.cmd, marker_unit, marker_name)
            self._box_names.append(box_name)
            self._marker_names.append(marker_name)

        while len(self._box_names) > count:
            name = self._box_names.pop()
            try:
                self.cmd.delete(name)
            except Exception:
                pass
        while len(self._marker_names) > count:
            name = self._marker_names.pop()
            try:
                self.cmd.delete(name)
            except Exception:
                pass

    def cleanup(self):
        purge_objects(
            self.cmd,
            names=tuple(self._box_names + self._marker_names),
            prefixes=(PREVIEW_BOX_PREFIX, PREVIEW_BOX_MARKER_PREFIX),
        )
        self._wireframe = None
        self._extent = None
        self._positions = []
        self._colors = []
        self._alphas = []
        self._box_names = []
        self._marker_names = []


def _load_merged_cgo(
    cmd_,
    name: str,
    token_list: list,
    state: int = 1,
    alpha: float = 1.0,
):
    """Commit path: load a merged CGO (creates/replaces one object)."""
    if not token_list:
        try:
            cmd_.delete(name)
        except Exception:
            pass
        return
    saved = cmd_.get_view()
    load_cgo_no_zoom(cmd_, _resolve_cgo_tokens(token_list), name, state)
    set_cgo_transparency(cmd_, name, alpha)
    restore_view(cmd_, saved)


def _commit_cgo(cmd_, name: str, points: Sequence[VisualPoint], build_all, build_one):
    """Load a committed CGO; group per-point objects when alphas differ."""
    shared = _shared_alpha(points)
    if shared is not None:
        _load_merged_cgo(cmd_, name, build_all(), alpha=shared)
        return
    members = []
    for i, pt in enumerate(points):
        member = "%s_%d" % (name, i + 1)
        members.append(member)
        _load_merged_cgo(cmd_, member, build_one(pt), alpha=pt.alpha)
    try:
        cmd_.group(name, " ".join(members))
    except Exception:
        pass


class SpherePreview:
    """Incremental preview: one unit CGO per point, moved/scaled with TTT."""

    def __init__(self, cmd_):
        self.cmd = cmd_
        self._wireframe = None
        self._wireframe_quality = None
        self._radius = None
        self._positions: List[Tuple[float, float, float]] = []
        self._colors: List[Tuple[float, float, float]] = []
        self._alphas: List[float] = []
        self._sphere_names: List[str] = []
        self._marker_names: List[str] = []

    def update(
        self,
        points: Sequence[VisualPoint],
        radius: float,
        wireframe: bool,
        wireframe_quality: int = 3,
    ):
        if not points:
            self.cleanup()
            return

        radius = float(radius)
        quality = effective_wireframe_quality(wireframe_quality, len(points), wireframe=wireframe)
        positions = [pt.xyz() for pt in points]
        colors = [pt.color for pt in points]
        alphas = [float(pt.alpha) for pt in points]
        wireframe_changed = wireframe != self._wireframe
        quality_changed = quality != self._wireframe_quality
        count_changed = len(points) != len(self._sphere_names)

        self._sync_object_count(len(points), wireframe, quality)

        colors_changed = colors != self._colors
        alphas_changed = alphas != self._alphas
        if wireframe_changed or quality_changed or count_changed or colors_changed or alphas_changed:
            for i, name in enumerate(self._sphere_names):
                unit = _resolve_cgo_tokens(
                    _unit_sphere_cgo(wireframe, colors[i], quality, alphas[i])
                )
                replace_cgo_no_zoom(self.cmd, unit, name)
                set_cgo_transparency(self.cmd, name, alphas[i])
            marker_unit = _resolve_cgo_tokens(_unit_marker_cgo())
            for name in self._marker_names:
                replace_cgo_no_zoom(self.cmd, marker_unit, name)
            self._wireframe = wireframe
            self._wireframe_quality = quality

        radius_changed = self._radius != radius
        positions_changed = positions != self._positions

        if radius_changed or positions_changed or wireframe_changed or quality_changed or count_changed:
            for i, pos in enumerate(positions):
                self.cmd.set_object_ttt(
                    self._sphere_names[i],
                    scale_translation_ttt(pos, radius),
                )
                self.cmd.set_object_ttt(
                    self._marker_names[i],
                    scale_translation_ttt(pos, MARKER_RADIUS),
                )

        self._radius = radius
        self._positions = positions
        self._colors = list(colors)
        self._alphas = list(alphas)

    def _sync_object_count(self, count: int, wireframe: bool, quality: WireframeQuality):
        while len(self._sphere_names) < count:
            idx = len(self._sphere_names)
            sphere_name = "%s%d" % (PREVIEW_SPHERE_PREFIX, idx)
            marker_name = "%s%d" % (PREVIEW_MARKER_PREFIX, idx)
            unit = _resolve_cgo_tokens(_unit_sphere_cgo(wireframe, SPHERE_COLOR, quality))
            marker_unit = _resolve_cgo_tokens(_unit_marker_cgo())
            load_cgo_no_zoom(self.cmd, unit, sphere_name)
            load_cgo_no_zoom(self.cmd, marker_unit, marker_name)
            self._sphere_names.append(sphere_name)
            self._marker_names.append(marker_name)

        while len(self._sphere_names) > count:
            name = self._sphere_names.pop()
            try:
                self.cmd.delete(name)
            except Exception:
                pass
        while len(self._marker_names) > count:
            name = self._marker_names.pop()
            try:
                self.cmd.delete(name)
            except Exception:
                pass

    def cleanup(self):
        purge_objects(
            self.cmd,
            names=tuple(self._sphere_names + self._marker_names),
            prefixes=(PREVIEW_SPHERE_PREFIX, PREVIEW_MARKER_PREFIX),
        )
        self._wireframe = None
        self._wireframe_quality = None
        self._radius = None
        self._positions = []
        self._colors = []
        self._alphas = []
        self._sphere_names = []
        self._marker_names = []


def _commit_pairs(cmd_, name: str, pairs, build_all, build_one):
    """Commit pair CGOs; group when alphas differ."""
    if not pairs:
        return
    alphas = [float(pair.alpha) for pair in pairs]
    if max(alphas) - min(alphas) < 1e-6:
        _load_merged_cgo(cmd_, name, build_all(), alpha=alphas[0])
        return
    members = []
    for i, pair in enumerate(pairs):
        member = "%s_%d" % (name, i + 1)
        members.append(member)
        _load_merged_cgo(cmd_, member, build_one(pair), alpha=pair.alpha)
    try:
        cmd_.group(name, " ".join(members))
    except Exception:
        pass


class ArrowPreview:
    """One arrow CGO per pair, plus a pending first-point marker."""

    def __init__(self, cmd_):
        self.cmd = cmd_
        self._signatures = []
        self._arrow_names: List[str] = []
        self._marker_names: List[str] = []
        self._pending_name = None

    def update(self, pairs, quality: int, style, pending=None):
        from .arrow_geom import arrow_cgo

        if not pairs and pending is None:
            self.cleanup()
            return

        signatures = [
            (pair.start.xyz(), pair.end.xyz(), pair.color, float(pair.alpha), int(quality), style)
            for pair in pairs
        ]
        count_changed = len(pairs) != len(self._arrow_names)
        self._sync_object_count(len(pairs))
        if signatures != self._signatures or count_changed:
            for i, pair in enumerate(pairs):
                tokens = _resolve_cgo_tokens(
                    arrow_cgo(pair.start.xyz(), pair.end.xyz(), pair.color, quality, style, pair.alpha)
                )
                replace_cgo_no_zoom(self.cmd, tokens, self._arrow_names[i])
                set_cgo_transparency(self.cmd, self._arrow_names[i], pair.alpha)
                marker = _resolve_cgo_tokens(_unit_marker_cgo())
                replace_cgo_no_zoom(self.cmd, marker, self._marker_names[i * 2])
                replace_cgo_no_zoom(self.cmd, marker, self._marker_names[i * 2 + 1])
                self.cmd.set_object_ttt(self._marker_names[i * 2], scale_translation_ttt(pair.start.xyz(), MARKER_RADIUS))
                self.cmd.set_object_ttt(self._marker_names[i * 2 + 1], scale_translation_ttt(pair.end.xyz(), MARKER_RADIUS))
            self._signatures = signatures

        if pending is None:
            if self._pending_name is not None:
                try:
                    self.cmd.delete(self._pending_name)
                except Exception:
                    pass
                self._pending_name = None
        else:
            if self._pending_name is None:
                self._pending_name = PREVIEW_ARROW_PENDING
                load_cgo_no_zoom(self.cmd, _resolve_cgo_tokens(_unit_marker_cgo()), self._pending_name)
            self.cmd.set_object_ttt(self._pending_name, scale_translation_ttt(pending.xyz(), MARKER_RADIUS * 1.4))

    def _sync_object_count(self, count: int):
        while len(self._arrow_names) < count:
            idx = len(self._arrow_names)
            arrow_name = "%s%d" % (PREVIEW_ARROW_PREFIX, idx)
            load_cgo_no_zoom(self.cmd, _resolve_cgo_tokens(_unit_marker_cgo()), arrow_name)
            self._arrow_names.append(arrow_name)
            for extra in (0, 1):
                marker_name = "%s%d" % (PREVIEW_ARROW_MARKER_PREFIX, idx * 2 + extra)
                load_cgo_no_zoom(self.cmd, _resolve_cgo_tokens(_unit_marker_cgo()), marker_name)
                self._marker_names.append(marker_name)
        while len(self._arrow_names) > count:
            name = self._arrow_names.pop()
            try:
                self.cmd.delete(name)
            except Exception:
                pass
        while len(self._marker_names) > count * 2:
            name = self._marker_names.pop()
            try:
                self.cmd.delete(name)
            except Exception:
                pass

    def cleanup(self):
        names = list(self._arrow_names + self._marker_names)
        if self._pending_name:
            names.append(self._pending_name)
        purge_objects(
            self.cmd,
            names=tuple(names),
            prefixes=(PREVIEW_ARROW_PREFIX, PREVIEW_ARROW_MARKER_PREFIX, PREVIEW_ARROW_PENDING),
        )
        self._signatures = []
        self._arrow_names = []
        self._marker_names = []
        self._pending_name = None
