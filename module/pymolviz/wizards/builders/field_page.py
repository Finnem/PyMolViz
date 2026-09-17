"""Builder page for volume / isosurface visuals bound to a field."""

from __future__ import annotations

from typing import Callable, Optional

from ...fields.clip import (
    aabb_corners,
    aabb_to_cardinal_planes,
    cardinal_planes_to_aabb,
    default_cardinal_plane,
    opposite_cardinal_plane,
    retarget_clip_aabb,
)
from ...fields.domain import field_display_aabb
from ...fields.isovalues import isovalues_for_side
from ...util.field_sample import field_label
from .appearance_section import AppearanceSection
from .base import BuilderPage
from ..pick import qt_modules, qt_widget_alive
from ..widgets.ascii_locale import apply_ascii_float_locale
from ..widgets.spin_step import STEP_TRANSPARENCY, apply_magnitude_step, apply_spin_step, bind_magnitude_step
from ..widgets.breadcrumb import (
    CRUMB_ISOMESH,
    CRUMB_ISOSURFACE,
    CRUMB_ISOVOLUME,
    CRUMB_VOLUME,
    create_field_visual_crumbs,
    edit_field_crumbs,
    set_breadcrumb,
)
from ..widgets.section import make_section
from ..widgets.theme import apply_secondary_button_style, swatch_button_css
from .field_picker import FieldPickerWidget
from .field_preview import (
    build_grid_preview_visual,
    estimate_grid_iso_job,
    field_visual_preview_key,
)
from .field_visual import (
    ISO_KINDS,
    VOLUME_KINDS,
    convert_isosurface_visual,
    default_iso_level,
    default_visual_name,
    field_visual_options,
    make_field_visual,
    persist_field_visual,
    resolve_field_grid,
)
from .object_names import unused_object_name
from .preview import FieldVisualPreview
from .preview_mode import (
    DEFAULT_PREVIEW_MODE,
    read_preview_mode,
    stamp_preview_mode,
    preview_grid,
    preview_iso_kind,
    preview_is_on,
    preview_is_simple,
)
from .surface_params import COLOR_MODE_FIELD, COLOR_MODE_UNIFORM
from .aabb_clip import CardinalClipGizmoController
from .carve_around import CarveAroundWidget
from .cardinal_clip_section import CLIP_TIP, CardinalClipSection
from .export import export_objects


def _field_with_preview_grid(field, grid):
    """Preview brick as a Field, keeping the interned geometry field id."""
    if field is None or grid is None:
        return field
    if grid is field:
        return field
    src_id = str(getattr(field, "id", "") or "")
    if type(grid).__name__ == "Field":
        if src_id and str(getattr(grid, "id", "") or "") != src_id:
            grid.id = src_id
        return grid
    return field


_KIND_CRUMBS = {
    "Volume": CRUMB_VOLUME,
    "IsoVolume": CRUMB_ISOVOLUME,
    "IsoSurface": CRUMB_ISOSURFACE,
    "IsoMesh": CRUMB_ISOMESH,
}
_LIVE_PREVIEW_TIP = (
    "No preview keeps the domain box only. Simple preview uses a coarser map "
    "and IsoMesh. Full preview is the native IsoSurface, IsoMesh, or Volume."
)
_LEVEL_TIP = (
    "Isovalue in the field's native units. Used by IsoSurface and IsoMesh. "
    "Volume and IsoVolume use the colormap transfer, not this level."
)
_CLIP_TIP = CLIP_TIP
_CLIP_CROP_TIP = CLIP_TIP


class FieldVisualBuilderPage(BuilderPage):
    """Name, Geometry, Appearance, Options, and Modifiers for a field visual."""

    DEFAULT_NAME = "pmv_volume"
    CONTEXT = "FieldVisualBuilderPage"

    def _init_editor(self):
        self._preview_section_widget = None
        self._appearance_section = None
        self._appearance = None
        self._iso_section = None
        self._clip_section = None
        self._level = None
        self._side = None
        self._transparency = None
        self._color = (1.0, 1.0, 1.0)
        self._color_btn = None
        self._iso_rest_widget = None
        self._geometry_picker = None
        self._cardinal_clip = None
        self._clip_gizmo = None
        self._carve = None
        self._convert_btn = None
        self._kind = "Volume"
        self._field = None
        self._level_geom_id = None
        self._preview = FieldVisualPreview(self.cmd)
        self._heavy_ok = None
        self._heavy_denied = None
        self._convert_heavy_ok = None
        self._convert_heavy_denied = None

    def cleanup_preview(self):
        if self._clip_gizmo is not None:
            self._clip_gizmo.clear()
        super().cleanup_preview()

    def _create_breadcrumb(self):
        return create_field_visual_crumbs(self._crumb_leaf())

    def _edit_breadcrumb(self):
        return create_field_visual_crumbs(self._crumb_leaf())

    def _crumb_leaf(self) -> str:
        return _KIND_CRUMBS.get(self._kind, self._kind)

    def reset_for_create(self, kind, field):
        self._suspend_preview = True
        try:
            self._reset_for_create_body(kind, field)
        finally:
            self._suspend_preview = False
        self._schedule_preview()

    def _reset_for_create_body(self, kind, field):
        self._kind = str(kind)
        self._field = field
        self._editing_id = None
        self._loaded_name = None
        self._color = (1.0, 1.0, 1.0)
        base = default_visual_name(self._kind, field)
        if self._object_name is not None:
            self._object_name.setText(unused_object_name(base, self.cmd))
        if self._action_bar is not None:
            self._action_bar.set_editing(False)
        set_breadcrumb(self._title, create_field_visual_crumbs(self._crumb_leaf()))
        if self._geometry_picker is not None:
            self._geometry_picker.refresh(getattr(field, "id", None))
        color_id = None
        try:
            from .load_field import default_color_colormap, default_color_field_id

            color_id = default_color_field_id(field)
            cmap = default_color_colormap(field)
            if cmap and self._appearance is not None:
                self._appearance.set_colormap(str(cmap))
        except Exception:
            color_id = getattr(field, "default_color_field_id", None) if field is not None else None
        if self._appearance is not None:
            if self._kind in VOLUME_KINDS:
                self._appearance.set_color_mode(COLOR_MODE_UNIFORM)
            else:
                self._appearance.refresh_color_field(color_id)
                if color_id:
                    self._appearance.set_color_mode(COLOR_MODE_FIELD)
                else:
                    self._appearance.set_color_mode(COLOR_MODE_UNIFORM)
        self._reset_clip_axes()
        if self._carve is not None:
            self._carve.reset()
        if self._appearance is not None:
            self._appearance.set_preview_mode(DEFAULT_PREVIEW_MODE)
        self._heavy_ok = None
        self._heavy_denied = None
        self._level_geom_id = None
        self._sync_form()
        self._sync_commit_enabled()

    def load_object(self, obj):
        self._suspend_preview = True
        try:
            self._load_object_body(obj)
        finally:
            self._suspend_preview = False
        self._schedule_preview()

    def _load_object_body(self, obj):
        self._kind = type(obj).__name__
        geom_id = getattr(obj, "geometry_field_id", None)
        self._field = getattr(obj, "grid_data", None)
        if geom_id:
            try:
                from ...runtime.session import get as session_get
                from ...util.field_sample import resolve_grid_from_session

                found = session_get(str(geom_id))
                if found is not None:
                    self._field = found
                else:
                    self._field = resolve_grid_from_session(geom_id) or self._field
            except Exception:
                pass
        self._editing_id = str(getattr(obj, "id", "") or "") or None
        name = getattr(obj, "_name", None) or getattr(obj, "name", "")
        self._loaded_name = name
        if self._object_name is not None:
            self._object_name.setText(str(name or ""))
        if self._action_bar is not None:
            self._action_bar.set_editing(True)
        opts = field_visual_options(obj)
        cmap = opts.get("colormap")
        spec = opts.get("colormap_spec")
        if self._appearance is not None and (isinstance(cmap, str) or spec):
            name = cmap if isinstance(cmap, str) else ((spec or {}).get("preset") or "RdYlBu_r")
            self._appearance.set_colormap(
                name,
                range_mode=opts.get("range_mode"),
                clims=opts.get("clims"),
                spec=spec,
            )
        if self._level is not None:
            self._level.setValue(float(opts.get("level") or 0.0))
        if self._side is not None:
            self._side.setCurrentIndex(int(opts.get("side_index") or 0))
        if self._transparency is not None:
            self._transparency.setValue(float(opts.get("transparency") or 0.0))
        if opts.get("color") is not None:
            self._color = opts["color"]
        if self._geometry_picker is not None:
            self._geometry_picker.refresh(opts.get("geometry_field_id") or getattr(self._field, "id", None))
        color_id = opts.get("color_field_id")
        if self._appearance is not None:
            if self._kind in VOLUME_KINDS:
                self._appearance.set_color_mode(COLOR_MODE_UNIFORM)
            else:
                self._appearance.refresh_color_field(color_id)
                if color_id:
                    self._appearance.set_color_mode(COLOR_MODE_FIELD)
                else:
                    self._appearance.set_color_mode(COLOR_MODE_UNIFORM)
        aabb = opts.get("clip_aabb")
        domain = self._field_domain_aabb(self._field, resolve_field_grid(self._field))
        self._set_cardinal_planes(aabb_to_cardinal_planes(aabb, domain))
        if self._cardinal_clip is not None:
            from ...util.clip_gizmo import read_clip_gizmo_state

            self._cardinal_clip.set_gizmo_state(opts.get("clip_gizmos") or read_clip_gizmo_state(obj))
        if self._carve is not None:
            self._carve.set_carve(opts.get("selection"), opts.get("carve"))
            self._carve.refresh(opts.get("selection"))
        field_name = field_label(self._field) if self._field is not None else ""
        crumbs = (field_name, self._crumb_leaf()) if field_name else (self._crumb_leaf(),)
        set_breadcrumb(self._title, edit_field_crumbs(*crumbs))
        if self._appearance is not None:
            self._appearance.set_preview_mode(read_preview_mode(obj))
        self._heavy_ok = None
        self._heavy_denied = None
        self._sync_form()
        self._sync_commit_enabled()

    def _typed_name(self) -> str:
        if self._object_name is None:
            return default_visual_name(self._kind, self._field)
        return self._object_name.text().strip() or default_visual_name(self._kind, self._field)

    def _on_geometry_changed(self):
        self._sync_form()
        self._sync_commit_enabled()
        self._schedule_preview()

    def _selected_field(self):
        fid = None
        if self._geometry_picker is not None:
            fid = self._geometry_picker.field_id()
        if fid:
            try:
                from ...runtime.session import get as session_get
                from ...util.field_sample import resolve_grid_from_session

                found = session_get(str(fid))
                if found is not None:
                    return found
                return resolve_grid_from_session(fid) or self._field
            except Exception:
                return self._field
        return self._field

    def _can_commit(self) -> bool:
        return resolve_field_grid(self._selected_field()) is not None

    def _color_from_field(self) -> bool:
        if self._kind in VOLUME_KINDS:
            return False
        if self._appearance is None:
            return False
        return self._appearance.color_mode() == COLOR_MODE_FIELD

    def _histogram_field_id(self):
        if self._color_from_field() and self._appearance is not None:
            fid = self._appearance.color_field_id()
            if fid:
                return fid
        field = self._selected_field()
        return getattr(field, "id", None) if field is not None else None

    def _sync_form(self):
        volume = self._kind in VOLUME_KINDS
        iso = self._kind in ISO_KINDS
        if self._preview_section_widget is not None:
            self._preview_section_widget.setVisible(True)
        if self._appearance_section is not None:
            self._appearance_section.widget.setVisible(True)
        if self._iso_section is not None:
            self._iso_section.widget.setVisible(iso)
        show_cmap = volume or self._color_from_field()
        if self._appearance is not None:
            self._appearance.set_colormap_visible(show_cmap)
            self._appearance.set_field_picker_visible(not volume)
        if self._iso_rest_widget is not None:
            self._iso_rest_widget.setVisible(iso)
        if self._color_btn is not None:
            self._color_btn.setVisible(iso and not self._color_from_field())
        if self._transparency is not None:
            self._transparency.setVisible(iso)
        if self._convert_btn is not None:
            self._convert_btn.setVisible(iso and self._editing_id is not None)
        grid = resolve_field_grid(self._selected_field())
        field = self._selected_field()
        geom_id = getattr(field, "id", None) if field is not None else None
        if geom_id != getattr(self, "_level_geom_id", None):
            if self._level is not None and grid is not None and self._editing_id is None:
                self._level.setValue(default_iso_level(grid))
            self._level_field_span = self._grid_value_span(grid)
            if self._level is not None:
                apply_magnitude_step(
                    self._level, min_decimals=4, extras=self._level_step_extras,
                )
        self._level_geom_id = geom_id
        self._paint_color_button()
        if self._cardinal_clip is not None:
            self._cardinal_clip.sync_widgets()

    def _grid_value_span(self, grid):
        values = getattr(grid, "values", None) if grid is not None else None
        if values is None:
            return ()
        import numpy as np

        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0:
            return ()
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return ()
        return (float(np.min(finite)), float(np.max(finite)))

    def _level_step_extras(self):
        return getattr(self, "_level_field_span", ()) or ()

    def _seed_clip_plane(self, axis: int):
        field = self._selected_field()
        grid = resolve_field_grid(field)
        return default_cardinal_plane(int(axis), self._field_domain_aabb(field, grid))

    def _seed_opposite_clip_plane(self, axis: int, existing: dict):
        field = self._selected_field()
        grid = resolve_field_grid(field)
        return opposite_cardinal_plane(existing, self._field_domain_aabb(field, grid))

    def _any_clip_enabled(self) -> bool:
        if self._cardinal_clip is None:
            return False
        return self._cardinal_clip.any_enabled()

    def _reset_clip_axes(self):
        if self._cardinal_clip is not None:
            self._cardinal_clip.reset()

    def _cardinal_planes(self):
        if self._cardinal_clip is None:
            return []
        return self._cardinal_clip.planes()

    def _set_cardinal_planes(self, planes):
        if self._cardinal_clip is not None:
            self._cardinal_clip.set_planes(planes)

    def _carve_args(self):
        if self._carve is None:
            return None, None
        return self._carve.carve_args()

    def _clip_aabb(self):
        planes = self._cardinal_planes()
        if not planes:
            return None
        field = self._selected_field()
        grid = resolve_field_grid(field)
        return cardinal_planes_to_aabb(planes, self._field_domain_aabb(field, grid))

    def _set_clip_aabb(self, aabb):
        field = self._selected_field()
        grid = resolve_field_grid(field)
        self._set_cardinal_planes(
            aabb_to_cardinal_planes(aabb, self._field_domain_aabb(field, grid))
        )

    def _clip_span_points(self):
        field = self._selected_field()
        grid = resolve_field_grid(field)
        box = self._field_domain_aabb(field, grid)
        return aabb_corners(box)

    def _sync_clip_gizmos(self):
        if self._clip_gizmo is None:
            return
        if not self._any_clip_enabled():
            self._clip_gizmo.clear()
            return
        self._clip_gizmo.refresh_gizmos()

    def _clip_axis_gizmo_visible(self, axis: int) -> bool:
        clip = getattr(self, "_cardinal_clip", None)
        if clip is None:
            return True
        return clip.axis_gizmo_visible(int(axis))

    def _paint_color_button(self):
        btn = self._color_btn
        if btn is None:
            return
        r, g, b = [max(0, min(255, int(round(c * 255.0)))) for c in self._color[:3]]
        btn.setStyleSheet(
            swatch_button_css("rgb(%d, %d, %d)" % (r, g, b), extra=" min-width: 72px;")
        )

    def _side_code(self) -> str:
        if self._side is None:
            return "positive"
        data = self._side.currentData()
        return str(data or "positive")

    def _live_preview_enabled(self) -> bool:
        if self._any_clip_enabled():
            return True
        if self._appearance is None:
            return False
        return preview_is_on(self._appearance.preview_mode())

    def _on_preview_setting(self):
        self._sync_form()
        self._schedule_preview()

    def _on_clip_axis_enabled(self, axis: int) -> None:
        if self._clip_gizmo is not None:
            self._clip_gizmo.select_axis(int(axis))
        self._sync_clip_gizmos()

    def _on_clip_axis_disabled(self, axis: int) -> None:
        if self._clip_gizmo is None:
            return
        axis = int(axis)
        if self._clip_gizmo.selected_axis != axis:
            return
        remaining = [
            i for i in (self._cardinal_clip.enabled_axes() if self._cardinal_clip else ())
            if i != axis
        ]
        if remaining:
            self._clip_gizmo.select_axis(remaining[0])
        else:
            self._clip_gizmo.clear()
        self._sync_clip_gizmos()

    def _on_clip_axis_focus(self, axis: int, hi=None) -> None:
        if self._clip_gizmo is not None and self._any_clip_enabled():
            self._clip_gizmo.select_axis(int(axis), hi=hi)

    def _on_clip_axis_flipped(self, axis: int) -> None:
        if self._clip_gizmo is not None:
            self._clip_gizmo.select_axis(int(axis))
            self._clip_gizmo.relatch()
        if self._preview is not None:
            self._preview._iso_key = None
        self._sync_clip_gizmos()

    def notify_field_brick_changed(self, field, info=None):
        """Move domain cube / crop gizmos after a lattice wrap of ``field``."""
        selected = self._selected_field()
        fid = str(getattr(field, "id", "") or "")
        selected_id = str(getattr(selected, "id", "") or "") if selected is not None else ""
        if fid and selected_id and fid != selected_id:
            return
        info = dict(info or {})
        grid = resolve_field_grid(self._selected_field() if selected is not None else field)
        new_aabb = field_display_aabb(field if field is not None else selected, grid)
        if self._any_clip_enabled():
            retargeted = retarget_clip_aabb(
                self._clip_aabb(),
                info.get("old_aabb"),
                new_aabb,
                origin_delta=info.get("origin_delta"),
                copied=bool(info.get("copied")),
            )
            if retargeted is not None:
                self._set_clip_aabb(retargeted)
            elif new_aabb is not None:
                self._set_clip_aabb(new_aabb)
        if self._preview is not None:
            self._preview._iso_key = None
        self._sync_clip_gizmos()
        self._schedule_preview()

    def _confirm_heavy_iso(self, grid) -> bool:
        job = estimate_grid_iso_job(grid)
        seconds = max(2, int(round(float(job.get("seconds") or 0.0))))
        voxels = int(job.get("voxels") or 0)
        extra = " (%s voxels)" % format(voxels, ",") if voxels else ""
        message = (
            "Building this field preview%s may take about %s seconds. "
            "PyMOL will not respond until it finishes. Build it anyway?"
            % (extra, seconds)
        )
        from .heavy_job import ask_heavy_job

        allowed, ok_fp, denied_fp = ask_heavy_job(
            self._page,
            job,
            title="Heavy field",
            message=message,
            previous_ok=self._heavy_ok,
            previous_denied=self._heavy_denied,
        )
        if ok_fp is not None:
            self._heavy_ok = ok_fp
        if denied_fp is not None:
            self._heavy_denied = denied_fp
        return allowed

    def _refresh_preview(self):
        if not qt_widget_alive(self._page):
            return
        if self._preview is None:
            return
        if self._carve is not None:
            self._carve.refresh()
        try:
            if not self._can_commit():
                if self._preview is not None:
                    self._preview.update(None)
                self._sync_clip_gizmos()
                return
            if not self._live_preview_enabled():
                if self._preview is not None:
                    field = self._selected_field()
                    grid = resolve_field_grid(field)
                    self._preview.update(
                        None,
                        domain_aabb=self._field_domain_aabb(field, grid),
                        show_domain=True,
                        grid=grid,
                    )
                self._sync_clip_gizmos()
                return
            field = self._selected_field()
            grid = resolve_field_grid(field)
            if grid is None:
                if self._preview is not None:
                    self._preview.update(None)
                self._sync_clip_gizmos()
                return
            mode = (
                self._appearance.preview_mode()
                if self._appearance is not None
                else DEFAULT_PREVIEW_MODE
            )
            display_grid = preview_grid(grid, mode)
            preview_kind = preview_iso_kind(self._kind, mode)
            if not preview_is_simple(mode) and not self._confirm_heavy_iso(display_grid):
                return
            colormap = self._colormap_name()
            colormap_spec = self._colormap_spec()
            level = float(self._level.value()) if self._level is not None else default_iso_level(grid)
            transparency = float(self._transparency.value()) if self._transparency is not None else 0.0
            color_field_id = None
            color_src = None
            if self._color_from_field() and self._appearance is not None:
                color_field_id = self._appearance.color_field_id()
                geom_id = str(getattr(field, "id", "") or "")
                if color_field_id and str(color_field_id) == geom_id:
                    color_src = display_grid
            carve_sel, carve_radius = self._carve_args()
            iso_key = field_visual_preview_key(
                kind=preview_kind,
                field_id=getattr(field, "id", None),
                iso_level=level,
                side=self._side_code(),
                clip_aabb=self._clip_aabb(),
                selection=carve_sel,
                carve=carve_radius,
                color_field_id=color_field_id,
                colormap=colormap,
                colormap_spec=colormap_spec,
                clims=self._editor_clims(grid),
                color=self._color,
                transparency=transparency,
                origin=getattr(display_grid, "origin", None),
            )
            if iso_key is not None:
                iso_key = iso_key + (mode,)
            visual = build_grid_preview_visual(
                _field_with_preview_grid(field, display_grid),
                kind=preview_kind,
                iso_level=level,
                side=self._side_code(),
                clip_aabb=self._clip_aabb(),
                selection=carve_sel,
                carve=carve_radius,
                color=self._color,
                transparency=transparency,
                color_field_id=color_field_id,
                color_src=color_src,
                colormap=colormap,
                colormap_spec=colormap_spec,
                cmd=self.cmd,
                clims=self._editor_clims(grid),
            )
            self._preview.update(
                visual,
                iso_key=iso_key,
                domain_aabb=self._field_domain_aabb(field, grid),
                show_domain=True,
                grid=grid,
            )
            self._sync_clip_gizmos()
        except RuntimeError:
            pass

    def _field_domain_aabb(self, field, grid):
        return field_display_aabb(field, grid)

    def _colormap_name(self) -> str:
        if self._appearance is None:
            return "RdYlBu_r"
        return self._appearance.colormap_name()

    def _colormap_arg(self):
        if self._appearance is None:
            return "RdYlBu_r"
        from ...util.colormap_spec import volume_colormap_arg

        return volume_colormap_arg(
            self._appearance.colormap_name(), self._appearance.colormap_spec(),
        )

    def _colormap_spec(self):
        if self._appearance is None:
            return None
        return self._appearance.colormap_spec()

    def _editor_clims(self, grid=None):
        if self._appearance is None:
            return None
        values = None
        if self._color_from_field():
            fid = self._appearance.color_field_id()
            if fid:
                try:
                    from ...util.colormap_spec import field_values_for_stats

                    values = field_values_for_stats(fid)
                except Exception:
                    values = None
        if values is None and grid is not None:
            values = getattr(grid, "values", None)
        return self._appearance.resolved_clims(values)

    def _commit_kwargs(self):
        field = self._selected_field()
        grid = resolve_field_grid(field)
        name = unused_object_name(self._typed_name(), self.cmd, keep=self._loaded_name)
        colormap = self._colormap_arg()
        level = float(self._level.value()) if self._level is not None else None
        transparency = float(self._transparency.value()) if self._transparency is not None else 0.0
        color_field_id = None
        if self._color_from_field() and self._appearance is not None:
            color_field_id = self._appearance.color_field_id()
        isovalues = None
        if self._kind in ISO_KINDS and level is not None:
            isovalues = isovalues_for_side(level, self._side_code())
        carve_sel, carve_radius = self._carve_args()
        return dict(
            kind=self._kind,
            grid=field if field is not None else grid,
            name=name,
            colormap=colormap,
            level=level,
            color=self._color,
            transparency=transparency,
            obj_id=self._editing_id,
            geometry_field_id=getattr(field, "id", None),
            color_field_id=color_field_id,
            isovalues=isovalues,
            clip_aabb=self._clip_aabb(),
            selection=carve_sel,
            carve=carve_radius,
            clims=self._editor_clims(grid),
            colormap_spec=self._colormap_spec(),
        )

    def _create_cgo(self):
        if not self._can_commit():
            return
        self.cleanup_preview()
        visual = make_field_visual(**self._commit_kwargs())
        mode = (
            self._appearance.preview_mode()
            if self._appearance is not None
            else DEFAULT_PREVIEW_MODE
        )
        stamp_preview_mode(visual, mode)
        from ...util.clip_gizmo import stamp_clip_gizmo_state

        if self._cardinal_clip is not None:
            stamp_clip_gizmo_state(visual, self._cardinal_clip.gizmo_state())
        persist_field_visual(self.cmd, visual)
        if self._on_create is not None:
            self._on_create()

    def _export_cgo(self):
        if not self._can_commit():
            return
        kwargs = self._commit_kwargs()
        name = kwargs["name"]
        kwargs["obj_id"] = None
        visual = make_field_visual(**kwargs)
        export_objects(self._page, visual, name, title="Export field visual")

    def _confirm_heavy_convert(self, visual) -> bool:
        from ...fields.convert import (
            estimate_isosurface_convert_job,
            explicit_convert_job_fingerprint,
            format_explicit_convert_message,
        )
        from .heavy_job import ask_heavy_job

        job = estimate_isosurface_convert_job(visual)
        allowed, ok_fp, denied_fp = ask_heavy_job(
            self._page,
            job,
            title="Large explicit surface",
            message=format_explicit_convert_message(job),
            previous_ok=self._convert_heavy_ok,
            previous_denied=self._convert_heavy_denied,
            fingerprint=explicit_convert_job_fingerprint,
        )
        if ok_fp is not None:
            self._convert_heavy_ok = ok_fp
        if denied_fp is not None:
            self._convert_heavy_denied = denied_fp
        return allowed

    def _convert(self):
        if self._editing_id is None:
            return
        try:
            from ...runtime.session import get as session_get

            obj = session_get(self._editing_id)
        except Exception:
            obj = None
        if obj is None:
            return
        if not self._confirm_heavy_convert(obj):
            return
        convert_isosurface_visual(self.cmd, obj, name=self._typed_name() + "_surface")
        if self._on_create is not None:
            self._on_create()

    def _pick_color(self):
        from .colors import pick_rgb

        def _done(choice):
            if choice is None:
                return
            rgba = getattr(choice, "rgba", None) or choice
            self._color = (float(rgba[0]), float(rgba[1]), float(rgba[2]))
            self._paint_color_button()
            self._schedule_preview()

        pick_rgb(self._page, initial=self._color, on_done=_done, cmd=self.cmd)

    def _build(self, parent):
        QtCore, _, QtWidgets = self._require_qt()
        page, body, back = self._mount_shell(parent, QtWidgets)
        self._clip_gizmo = CardinalClipGizmoController(
            page=page,
            preview=self._preview,
            get_planes=self._cardinal_planes,
            set_planes=self._set_cardinal_planes,
            span_points=self._clip_span_points,
            domain_aabb=lambda: self._field_domain_aabb(
                self._selected_field(), resolve_field_grid(self._selected_field()),
            ),
            on_changed=self._schedule_preview,
            gizmos_visible=self._clip_axis_gizmo_visible,
        )

        bind = make_section("Geometry", form=True)
        self._geometry_picker = FieldPickerWidget(
            page, self.cmd, self.CONTEXT, on_changed=self._on_geometry_changed,
        )
        bind.layout.addRow("Geometry field", self._geometry_picker.widget)
        body.addWidget(bind.widget)

        self._appearance = AppearanceSection(
            page,
            self.cmd,
            self.CONTEXT,
            show_wireframe=False,
            show_quality=False,
            show_per_point=False,
            show_live_preview=True,
            live_preview_tooltip=_LIVE_PREVIEW_TIP,
            show_color_actions=False,
            colormap_in_uniform=True,
            section_title=None,
            preview_as_sibling=True,
            field_picker_label="Color field",
            field_empty_label="Optional color field",
            field_id_provider=self._histogram_field_id,
            uniform_tooltip="Solid color or volume colormap, not sampled from another field.",
            field_tooltip="Color this visual by sampling another field.",
            on_changed=self._on_preview_setting,
            on_preview=self._schedule_preview,
        )
        preview_widget = self._appearance.preview_section_widget
        if preview_widget is not None:
            body.addWidget(preview_widget)
            self._preview_section_widget = preview_widget

        self._appearance_section = make_section("Appearance")
        self._appearance_section.layout.addWidget(self._appearance.widget)
        self._iso_rest_widget = QtWidgets.QWidget()
        rest_form = QtWidgets.QFormLayout(self._iso_rest_widget)
        rest_form.setContentsMargins(0, 0, 0, 0)
        self._transparency = QtWidgets.QDoubleSpinBox()
        self._transparency.setRange(0.0, 1.0)
        apply_spin_step(self._transparency, STEP_TRANSPARENCY, decimals=2)
        apply_ascii_float_locale(self._transparency, QtCore)
        self._transparency.valueChanged.connect(lambda *_: self._schedule_preview())
        self._color_btn = QtWidgets.QPushButton("Color")
        self._color_btn.setAutoDefault(False)
        self._color_btn.setDefault(False)
        self._color_btn.clicked.connect(self._pick_color)
        rest_form.addRow("Transparency", self._transparency)
        rest_form.addRow("Color", self._color_btn)
        self._appearance_section.layout.addWidget(self._iso_rest_widget)
        body.addWidget(self._appearance_section.widget)

        iso = make_section("Options")
        iso_form = QtWidgets.QFormLayout()
        iso_form.setContentsMargins(0, 0, 0, 0)
        self._level = QtWidgets.QDoubleSpinBox()
        self._level.setRange(-1e6, 1e6)
        apply_ascii_float_locale(self._level, QtCore)
        bind_magnitude_step(self._level, min_decimals=4, extras=self._level_step_extras)
        self._level.valueChanged.connect(lambda *_: self._schedule_preview())
        self._side = QtWidgets.QComboBox()
        self._side.addItem("Positive", "positive")
        self._side.addItem("Negative", "negative")
        self._side.addItem("Both", "both")
        self._side.currentIndexChanged.connect(lambda *_: self._schedule_preview())
        iso_form.addRow("Level", self._level)
        iso_form.addRow("Side", self._side)
        self._convert_btn = QtWidgets.QPushButton("Convert to explicit surface")
        self._convert_btn.setAutoDefault(False)
        self._convert_btn.setDefault(False)
        apply_secondary_button_style(self._convert_btn)
        self._convert_btn.clicked.connect(self._convert)
        iso.layout.addLayout(iso_form)
        iso.layout.addWidget(self._convert_btn)
        self._iso_section = iso
        body.addWidget(iso.widget)

        clip = make_section("Modifiers", form=True)
        self._carve = CarveAroundWidget(
            page, self.cmd, self.CONTEXT, on_changed=self._schedule_preview,
        )
        clip.layout.addRow("Carve around", self._carve.widget)
        self._cardinal_clip = CardinalClipSection(
            page,
            on_changed=self._schedule_preview,
            seed_plane=self._seed_clip_plane,
            seed_opposite=self._seed_opposite_clip_plane,
            on_axis_enabled=self._on_clip_axis_enabled,
            on_axis_disabled=self._on_clip_axis_disabled,
            on_axis_focus=self._on_clip_axis_focus,
            on_axis_flip=self._on_clip_axis_flipped,
            on_gizmo_changed=self._sync_clip_gizmos,
        )
        clip.layout.addRow("", self._cardinal_clip.widget)
        if clip.header is not None:
            clip.header.setToolTip(_CLIP_TIP)
        clip_tips = list(self._cardinal_clip.tooltips())
        self._clip_section = clip
        body.addWidget(clip.widget)
        body.addStretch(1)

        self._mount_action_bar(page, body)
        tips = [
            (self._geometry_picker.widget, "Field that defines the volume geometry.", "Geometry field"),
            *self._appearance.tooltips(),
            (self._level, _LEVEL_TIP, "Level"),
            (self._side, "Which side of the isosurface to keep.", "Side"),
            (self._transparency, "0 is opaque, 1 is invisible.", "Transparency"),
            (self._color_btn, "Solid color for this isosurface.", "Color"),
            (self._convert_btn, "Bake this isosurface into an editable Surface object.", "Convert"),
            *self._carve.tooltips(),
            *clip_tips,
        ]
        self._finish_build(page, back, tips)
        self._sync_form()
