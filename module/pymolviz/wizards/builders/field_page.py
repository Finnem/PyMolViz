"""Builder page for volume / isosurface visuals bound to a field."""

from __future__ import annotations

from typing import Callable, Optional

from ...fields.clip import aabb_corners, normalize_clip_aabb, retarget_clip_aabb
from ...fields.domain import aabb_has_extent, field_display_aabb
from ...fields.isovalues import isovalues_for_side, primary_isovalue, primary_side
from ...util.field_sample import field_label
from .colormap_editor import ColormapEditor
from ..pick import (
    DeferredCallback,
    overlay_get_save_file_name,
    overlay_question,
    qt_modules,
    qt_widget_alive,
)
from ..tooltips import apply_required_tooltips, warn_missing_setting_tooltips
from ..widgets.action_bar import BuilderActionBar
from ..widgets.ascii_locale import apply_ascii_float_locale
from ..widgets.breadcrumb import (
    BACK_TIP,
    CRUMB_ISOMESH,
    CRUMB_ISOSURFACE,
    CRUMB_ISOVOLUME,
    CRUMB_VOLUME,
    create_field_visual_crumbs,
    edit_field_crumbs,
    make_page_header,
    set_breadcrumb,
)
from ..widgets.name_section import BuilderNameSection
from ..widgets.scrolling import bind_width_to_scroll_viewport, make_scrolling_body
from ..widgets.section import make_section
from ..widgets.theme import (
    apply_page_layout,
    apply_secondary_button_style,
    apply_wizard_page_style,
    swatch_button_css,
)
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
    make_field_visual,
    persist_field_visual,
    resolve_field_grid,
)
from .object_names import unused_object_name
from .preview import FieldVisualPreview
from .aabb_clip import AabbClipGizmoController

_KIND_CRUMBS = {
    "Volume": CRUMB_VOLUME,
    "IsoVolume": CRUMB_ISOVOLUME,
    "IsoSurface": CRUMB_ISOSURFACE,
    "IsoMesh": CRUMB_ISOMESH,
}
_LIVE_PREVIEW_TIP = (
    "When on, show the same native PyMOL IsoSurface, IsoMesh, or Volume that "
    "Done creates. When off, hide the preview."
)
_LEVEL_TIP = (
    "Isovalue in the field's native units. Used by IsoSurface and IsoMesh. "
    "Volume and IsoVolume use the colormap transfer, not this level."
)
_CLIP_CROP_TIP = (
    "Crop the field to an axis-aligned box. Six preview planes sit on the "
    "faces; drag a plane along its axis (X, Y, or Z only). Edit a min/max "
    "value to select that face."
)


class FieldVisualBuilderPage:
    """Name + type-specific options for a field visual, with optional live native preview."""

    CONTEXT = "FieldVisualBuilderPage"

    def __init__(
        self,
        cmd_,
        on_back: Callable[[], None],
        on_create: Optional[Callable[[], None]] = None,
        parent=None,
    ):
        self.cmd = cmd_
        self._on_back = on_back
        self._on_create = on_create
        self._page = None
        self._title = None
        self._object_name = None
        self._action_bar = None
        self._name_section = None
        self._outer_layout = None
        self._volume_section = None
        self._appearance_section = None
        self._iso_section = None
        self._clip_section = None
        self._cmap_editor = None
        self._level = None
        self._side = None
        self._transparency = None
        self._color = (1.0, 1.0, 1.0)
        self._color_btn = None
        self._color_mode_uniform = None
        self._color_mode_field = None
        self._geometry_picker = None
        self._color_picker = None
        self._clip_enabled = None
        self._clip_lo = None
        self._clip_hi = None
        self._clip_gizmo = None
        self._clip_spin_suspend = False
        self._convert_btn = None
        self._kind = "Volume"
        self._field = None
        self._editing_id = None
        self._loaded_name = None
        self._level_geom_id = None
        self._deferred = DeferredCallback()
        self._preview = None
        self._live_preview = None
        self._suspend_preview = False
        self._heavy_ok = None
        self._heavy_denied = None
        self._build(parent)

    @property
    def widget(self):
        return self._page

    def cleanup_preview(self):
        if self._deferred is not None:
            self._deferred.cancel()
        if self._clip_gizmo is not None:
            self._clip_gizmo.clear()
        if self._preview is not None:
            self._preview.cleanup()

    def _go_back(self):
        self.cleanup_preview()
        self._on_back()

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
            if cmap and self._cmap_editor is not None:
                self._cmap_editor.set_colormap(str(cmap))
        except Exception:
            color_id = getattr(field, "default_color_field_id", None) if field is not None else None
        if self._color_picker is not None:
            self._color_picker.refresh(color_id)
        if color_id and self._color_mode_field is not None:
            self._color_mode_field.setChecked(True)
        elif self._color_mode_uniform is not None:
            self._color_mode_uniform.setChecked(True)
        if self._clip_enabled is not None:
            self._clip_enabled.setChecked(False)
        if self._live_preview is not None:
            self._live_preview.setChecked(False)
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
        cmap = getattr(getattr(obj, "colormap", None), "preset", None)
        if not cmap:
            cmap = getattr(obj, "colormap", None)
        spec = getattr(obj, "colormap_spec", None)
        if self._cmap_editor is not None and (isinstance(cmap, str) or spec):
            range_mode = getattr(getattr(obj, "colormap", None), "range_mode", None)
            clims = getattr(obj, "clims", None)
            if clims is not None and len(clims) >= 2:
                pair = (float(clims[0]), float(clims[-1]))
            else:
                pair = None
            name = cmap if isinstance(cmap, str) else ((spec or {}).get("preset") or "RdYlBu_r")
            self._cmap_editor.set_colormap(name, range_mode=range_mode, clims=pair, spec=spec)
        if self._level is not None:
            level = primary_isovalue(getattr(obj, "isovalues", None), default_level=float(getattr(obj, "level", 0) or 0))
            self._level.setValue(float(level))
        if self._side is not None:
            side = primary_side(getattr(obj, "isovalues", None), default_side=int(getattr(obj, "side", 1) or 1))
            idx = 0 if side >= 0 else 1
            entries = getattr(obj, "isovalues", None) or []
            if len(entries) >= 2:
                idx = 2
            self._side.setCurrentIndex(idx)
        if self._transparency is not None:
            self._transparency.setValue(float(getattr(obj, "transparency", 0) or 0))
        color = getattr(obj, "color", None)
        if color is not None and not hasattr(color, "name"):
            try:
                self._color = (float(color[0]), float(color[1]), float(color[2]))
            except (TypeError, IndexError, ValueError):
                pass
        if self._geometry_picker is not None:
            self._geometry_picker.refresh(geom_id or getattr(self._field, "id", None))
        color_id = getattr(obj, "color_field_id", None)
        if self._color_picker is not None:
            self._color_picker.refresh(color_id)
        if color_id and self._color_mode_field is not None:
            self._color_mode_field.setChecked(True)
        elif self._color_mode_uniform is not None:
            self._color_mode_uniform.setChecked(True)
        aabb = getattr(obj, "clip_aabb", None)
        if self._clip_enabled is not None:
            self._clip_enabled.setChecked(bool(aabb))
        if aabb and self._clip_lo and self._clip_hi:
            for i in range(3):
                self._clip_lo[i].setValue(float(aabb[0][i]))
                self._clip_hi[i].setValue(float(aabb[1][i]))
        field_name = field_label(self._field) if self._field is not None else ""
        crumbs = (field_name, self._crumb_leaf()) if field_name else (self._crumb_leaf(),)
        set_breadcrumb(self._title, edit_field_crumbs(*crumbs))
        if self._live_preview is not None:
            self._live_preview.setChecked(False)
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

    def _sync_commit_enabled(self):
        if self._action_bar is not None:
            self._action_bar.set_commit_enabled(self._can_commit())

    def _color_from_field(self) -> bool:
        if self._color_mode_field is None:
            return False
        return bool(self._color_mode_field.isChecked())

    def _histogram_field_id(self):
        if self._color_from_field() and self._color_picker is not None:
            fid = self._color_picker.field_id()
            if fid:
                return fid
        field = self._selected_field()
        return getattr(field, "id", None) if field is not None else None

    def _sync_form(self):
        volume = self._kind in VOLUME_KINDS
        iso = self._kind in ISO_KINDS
        if self._appearance_section is not None:
            self._appearance_section.widget.setVisible(True)
        if self._volume_section is not None:
            self._volume_section.widget.setVisible(False)
        if self._iso_section is not None:
            self._iso_section.widget.setVisible(iso)
        show_cmap = volume or self._color_from_field()
        if self._cmap_editor is not None:
            self._cmap_editor.widget.setVisible(show_cmap)
        if self._color_btn is not None:
            self._color_btn.setVisible(iso and not self._color_from_field())
        if self._transparency is not None:
            self._transparency.setVisible(iso)
        if self._color_picker is not None:
            self._color_picker.widget.setEnabled(self._color_from_field())
        if self._convert_btn is not None:
            self._convert_btn.setVisible(iso and self._editing_id is not None)
        grid = resolve_field_grid(self._selected_field())
        field = self._selected_field()
        geom_id = getattr(field, "id", None) if field is not None else None
        if self._level is not None and grid is not None and self._editing_id is None:
            if geom_id != self._level_geom_id:
                self._level.setValue(default_iso_level(grid))
        self._level_geom_id = geom_id
        self._paint_color_button()
        self._sync_clip_enabled()

    def _sync_clip_enabled(self):
        on = bool(self._clip_enabled.isChecked()) if self._clip_enabled is not None else False
        for spin in list(self._clip_lo or ()) + list(self._clip_hi or ()):
            spin.setEnabled(on)

    def _clip_aabb(self):
        if self._clip_enabled is None or not self._clip_enabled.isChecked():
            return None
        if not self._clip_lo or not self._clip_hi:
            return None
        return normalize_clip_aabb([
            [float(spin.value()) for spin in self._clip_lo],
            [float(spin.value()) for spin in self._clip_hi],
        ])

    def _set_clip_aabb(self, aabb):
        box = normalize_clip_aabb(aabb)
        if box is None or not self._clip_lo or not self._clip_hi:
            return
        self._clip_spin_suspend = True
        try:
            for i in range(3):
                self._clip_lo[i].setValue(float(box[0][i]))
                self._clip_hi[i].setValue(float(box[1][i]))
        finally:
            self._clip_spin_suspend = False

    def _clip_span_points(self):
        field = self._selected_field()
        grid = resolve_field_grid(field)
        box = self._field_domain_aabb(field, grid)
        return aabb_corners(box)

    def _ensure_clip_aabb(self):
        box = self._clip_aabb()
        if box is not None and aabb_has_extent(box):
            return
        field = self._selected_field()
        grid = resolve_field_grid(field)
        seeded = self._field_domain_aabb(field, grid)
        if seeded is not None and aabb_has_extent(seeded):
            self._set_clip_aabb(seeded)

    def _sync_clip_gizmos(self):
        if self._clip_gizmo is None:
            return
        if self._clip_enabled is None or not self._clip_enabled.isChecked():
            self._clip_gizmo.clear()
            return
        self._ensure_clip_aabb()
        self._clip_gizmo.refresh_gizmos()

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
        if self._clip_enabled is not None and self._clip_enabled.isChecked():
            return True
        if self._live_preview is None:
            return False
        return bool(self._live_preview.isChecked())

    def _schedule_preview(self):
        if self._suspend_preview:
            return
        if self._deferred is None:
            return
        self._deferred.schedule(self._refresh_preview, page=self._page)

    def _on_preview_setting(self):
        self._sync_form()
        self._schedule_preview()

    def _on_clip_toggled(self):
        self._sync_clip_enabled()
        if self._clip_enabled is not None and self._clip_enabled.isChecked():
            self._ensure_clip_aabb()
        self._sync_clip_gizmos()
        self._schedule_preview()

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
        if self._clip_enabled is not None and self._clip_enabled.isChecked():
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

    def _on_clip_spin(self, axis, is_hi):
        if self._clip_spin_suspend:
            return
        if self._clip_gizmo is not None and self._clip_enabled is not None and self._clip_enabled.isChecked():
            self._clip_gizmo.select_face(axis, is_hi)
        self._schedule_preview()

    def _confirm_heavy_iso(self, grid) -> bool:
        job = estimate_grid_iso_job(grid)
        from ...util.solvent_surface import confirm_heavy_surface_job

        decision, fingerprint = confirm_heavy_surface_job(
            job, previous_ok=self._heavy_ok, previous_denied=self._heavy_denied,
        )
        if decision == "allow":
            self._heavy_ok = fingerprint
            return True
        if decision == "deny":
            return False
        _, _, QtWidgets = qt_modules()
        if QtWidgets is None:
            self._heavy_ok = fingerprint
            return True
        seconds = max(2, int(round(float(job.get("seconds") or 0.0))))
        voxels = int(job.get("voxels") or 0)
        extra = " (%s voxels)" % format(voxels, ",") if voxels else ""
        message = (
            "Building this field preview%s may take about %s seconds. "
            "PyMOL will not respond until it finishes. Build it anyway?"
            % (extra, seconds)
        )
        result = overlay_question(
            self._page,
            "Heavy field",
            message,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if result == QtWidgets.QMessageBox.Yes:
            self._heavy_ok = fingerprint
            return True
        self._heavy_denied = fingerprint
        return False

    def _refresh_preview(self):
        if not qt_widget_alive(self._page):
            return
        if self._preview is None:
            return
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
            if not self._confirm_heavy_iso(grid):
                return
            colormap = self._colormap_name()
            colormap_spec = self._colormap_spec()
            level = float(self._level.value()) if self._level is not None else default_iso_level(grid)
            transparency = float(self._transparency.value()) if self._transparency is not None else 0.0
            color_field_id = None
            if self._color_from_field() and self._color_picker is not None:
                color_field_id = self._color_picker.field_id()
            iso_key = field_visual_preview_key(
                kind=self._kind,
                field_id=getattr(field, "id", None),
                iso_level=level,
                side=self._side_code(),
                clip_aabb=self._clip_aabb(),
                color_field_id=color_field_id,
                colormap=colormap,
                colormap_spec=colormap_spec,
                clims=self._editor_clims(grid),
                color=self._color,
                transparency=transparency,
                origin=getattr(grid, "origin", None),
            )
            visual = build_grid_preview_visual(
                field,
                kind=self._kind,
                iso_level=level,
                side=self._side_code(),
                clip_aabb=self._clip_aabb(),
                color=self._color,
                transparency=transparency,
                color_field_id=color_field_id,
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
        if self._cmap_editor is None:
            return "RdYlBu_r"
        return self._cmap_editor.colormap_name()

    def _colormap_arg(self):
        if self._cmap_editor is None:
            return "RdYlBu_r"
        from ...util.colormap_spec import volume_colormap_arg

        return volume_colormap_arg(self._cmap_editor.colormap_name(), self._cmap_editor.colormap_spec())

    def _colormap_spec(self):
        if self._cmap_editor is None:
            return None
        return self._cmap_editor.colormap_spec()

    def _editor_clims(self, grid=None):
        if self._cmap_editor is None:
            return None
        values = getattr(grid, "values", None) if grid is not None else None
        return self._cmap_editor.resolved_clims(values)

    def _commit_kwargs(self):
        field = self._selected_field()
        grid = resolve_field_grid(field)
        name = unused_object_name(self._typed_name(), self.cmd, keep=self._loaded_name)
        colormap = self._colormap_arg()
        level = float(self._level.value()) if self._level is not None else None
        transparency = float(self._transparency.value()) if self._transparency is not None else 0.0
        color_field_id = None
        if self._color_from_field() and self._color_picker is not None:
            color_field_id = self._color_picker.field_id()
        isovalues = None
        if self._kind in ISO_KINDS and level is not None:
            isovalues = isovalues_for_side(level, self._side_code())
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
            clims=self._editor_clims(grid),
            colormap_spec=self._colormap_spec(),
        )

    def _commit(self):
        if not self._can_commit():
            return
        self.cleanup_preview()
        visual = make_field_visual(**self._commit_kwargs())
        persist_field_visual(self.cmd, visual)
        if self._on_create is not None:
            self._on_create()

    def _export(self):
        _, _, QtWidgets = qt_modules()
        if QtWidgets is None or not self._can_commit():
            return
        kwargs = self._commit_kwargs()
        name = kwargs["name"]
        path, _ = overlay_get_save_file_name(
            self._page,
            "Export field visual script",
            "%s.py" % name,
            "Python (*.py)",
        )
        if not path:
            return
        kwargs["obj_id"] = None
        visual = make_field_visual(**kwargs)
        visual.write(path)

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
        QtCore, _, QtWidgets = qt_modules()
        if QtWidgets is None:
            raise RuntimeError("PyMOL Qt UI required")
        page = QtWidgets.QWidget(parent)
        self._page = page
        apply_wizard_page_style(page)
        outer = QtWidgets.QVBoxLayout(page)
        apply_page_layout(outer)
        header, back, self._title = make_page_header(
            QtWidgets,
            self._go_back,
            create_field_visual_crumbs(CRUMB_VOLUME),
            BACK_TIP,
        )
        outer.addLayout(header)
        self._name_section = BuilderNameSection(
            page,
            name=unused_object_name("pmv_volume", self.cmd),
            context=self.CONTEXT,
        )
        self._object_name = self._name_section.name_edit
        outer.addWidget(self._name_section.widget)
        scroll, body = make_scrolling_body(page)
        bind_width_to_scroll_viewport(self._name_section.widget, scroll)
        outer.addWidget(scroll, stretch=1)
        self._outer_layout = outer
        self._preview = FieldVisualPreview(self.cmd)
        self._clip_gizmo = AabbClipGizmoController(
            page=page,
            preview=self._preview,
            get_aabb=self._clip_aabb,
            set_aabb=self._set_clip_aabb,
            span_points=self._clip_span_points,
            on_changed=self._schedule_preview,
        )

        bind = make_section("Fields", form=True)
        self._geometry_picker = FieldPickerWidget(
            page, self.cmd, self.CONTEXT, on_changed=self._on_geometry_changed,
        )
        bind.layout.addRow("Geometry field", self._geometry_picker.widget)
        mode_row = QtWidgets.QWidget()
        mode_layout = QtWidgets.QHBoxLayout(mode_row)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        self._color_mode_uniform = QtWidgets.QRadioButton("Uniform")
        self._color_mode_field = QtWidgets.QRadioButton("From field")
        self._color_mode_uniform.setChecked(True)
        self._color_mode_uniform.toggled.connect(lambda *_: self._on_preview_setting())
        self._color_mode_field.toggled.connect(lambda *_: self._on_preview_setting())
        mode_layout.addWidget(self._color_mode_uniform)
        mode_layout.addWidget(self._color_mode_field)
        mode_layout.addStretch(1)
        bind.layout.addRow("Color mode", mode_row)
        self._color_picker = FieldPickerWidget(
            page, self.cmd, self.CONTEXT, empty_label="Optional color field",
            on_changed=self._schedule_preview,
        )
        bind.layout.addRow("Color field", self._color_picker.widget)
        body.addWidget(bind.widget)

        volume = make_section("Appearance")
        appear_form = QtWidgets.QFormLayout()
        appear_form.setContentsMargins(0, 0, 0, 0)
        self._live_preview = QtWidgets.QCheckBox("Live preview")
        self._live_preview.setObjectName("pmvLivePreview")
        self._live_preview.toggled.connect(lambda *_: self._schedule_preview())
        appear_form.addRow("", self._live_preview)
        volume.layout.addLayout(appear_form)
        self._cmap_editor = ColormapEditor(
            volume.widget,
            on_changed=self._schedule_preview,
            cmd=self.cmd,
            field_id_provider=self._histogram_field_id,
        )
        volume.layout.addWidget(self._cmap_editor.widget)
        rest_form = QtWidgets.QFormLayout()
        rest_form.setContentsMargins(0, 0, 0, 0)
        self._transparency = QtWidgets.QDoubleSpinBox()
        self._transparency.setDecimals(2)
        self._transparency.setRange(0.0, 1.0)
        self._transparency.setSingleStep(0.05)
        apply_ascii_float_locale(self._transparency, QtCore)
        self._transparency.valueChanged.connect(lambda *_: self._schedule_preview())
        self._color_btn = QtWidgets.QPushButton("Color")
        self._color_btn.setAutoDefault(False)
        self._color_btn.setDefault(False)
        self._color_btn.clicked.connect(self._pick_color)
        rest_form.addRow("Transparency", self._transparency)
        rest_form.addRow("Color", self._color_btn)
        volume.layout.addLayout(rest_form)
        self._appearance_section = volume
        self._volume_section = None
        body.addWidget(volume.widget)

        iso = make_section("IsoSurface")
        iso_form = QtWidgets.QFormLayout()
        iso_form.setContentsMargins(0, 0, 0, 0)
        self._level = QtWidgets.QDoubleSpinBox()
        self._level.setDecimals(4)
        self._level.setRange(-1e6, 1e6)
        self._level.setSingleStep(0.1)
        apply_ascii_float_locale(self._level, QtCore)
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

        clip = make_section("Clip / Crop", form=True)
        self._clip_enabled = QtWidgets.QCheckBox("Axis-aligned crop")
        self._clip_enabled.toggled.connect(lambda *_: self._on_clip_toggled())
        clip.layout.addRow("", self._clip_enabled)
        box = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(box)
        grid.setContentsMargins(0, 0, 0, 0)
        self._clip_lo = []
        self._clip_hi = []
        for i, axis in enumerate("XYZ"):
            lo = QtWidgets.QDoubleSpinBox()
            hi = QtWidgets.QDoubleSpinBox()
            for spin in (lo, hi):
                spin.setDecimals(2)
                spin.setRange(-1e4, 1e4)
                spin.setSingleStep(0.5)
                apply_ascii_float_locale(spin, QtCore)
            lo.valueChanged.connect(lambda *_a, ax=i: self._on_clip_spin(ax, False))
            hi.valueChanged.connect(lambda *_a, ax=i: self._on_clip_spin(ax, True))
            grid.addWidget(QtWidgets.QLabel(axis), i, 0)
            grid.addWidget(lo, i, 1)
            grid.addWidget(hi, i, 2)
            self._clip_lo.append(lo)
            self._clip_hi.append(hi)
        clip.layout.addRow("Min / max", box)
        self._clip_section = clip
        body.addWidget(clip.widget)
        body.addStretch(1)

        self._action_bar = BuilderActionBar(
            page,
            on_commit=self._commit,
            on_export=self._export,
            context=self.CONTEXT,
        )
        outer.addWidget(self._action_bar.widget)
        apply_required_tooltips(
            [
                (back, BACK_TIP, "Back"),
                (self._geometry_picker.widget, "Field that defines the volume geometry.", "Geometry field"),
                (self._color_mode_uniform, "Solid color or volume colormap, not sampled from another field.", "Uniform"),
                (self._color_mode_field, "Color this visual by sampling another field.", "From field"),
                (self._color_picker.widget, "Optional second field used as a color ramp.", "Color field"),
                *self._cmap_editor.tooltips(),
                (self._live_preview, _LIVE_PREVIEW_TIP, "Live preview"),
                (self._level, _LEVEL_TIP, "Level"),
                (self._side, "Which side of the isosurface to keep.", "Side"),
                (self._transparency, "0 is opaque, 1 is invisible.", "Transparency"),
                (self._color_btn, "Solid color for this isosurface.", "Color"),
                (self._convert_btn, "Bake this isosurface into an editable Surface object.", "Convert"),
                (self._clip_enabled, _CLIP_CROP_TIP, "Clip / Crop"),
            ],
            context=self.CONTEXT,
        )
        self._sync_form()
        self._sync_commit_enabled()
        warn_missing_setting_tooltips(page, context=self.CONTEXT)
