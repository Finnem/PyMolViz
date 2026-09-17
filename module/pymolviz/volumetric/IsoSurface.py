import numpy as np
import logging
from ..fields.field import Field
from .ColorRamp import ColorRamp
from ..Displayable import Displayable
from ..util.colors import _convert_string_color


class IsoSurface(Displayable):
    renders_cgo = False
    _native_iso_cmd = "isosurface"
    _native_iso_passes_side = True

    def __init__(self, grid_data : Field, level: float, name = None, color = None, transparency = 0, selection = '', carve = None, side = 1, geometry_field_id=None, color_field_id=None, isovalues=None, clip_aabb=None):
        """ 
        Computes and collects pymol commands to load in regular data and display an iso mesh at the given level.
        Note that, since this is based on volumetric data it is different from the pmv.Mesh class.

        Args:
            grid_data (pymolviz.GridData): The data to use for the iso surface.
            level (float): The level at which to display the iso surface.
            name (str): Optional. Defaults to None. The name of the object.
            color (str or list of float): Optional. Defaults to None. The color of the object.
            transparency (float): Optional. Defaults to 0. The transparency value of the object.
            selection (str): Optional. Defaults to None. The selection to use.
            carve (float): Optional. Defaults to None. The carve value to use.
            side (int): Optional. Defaults to 1. The side to use.
            
        """
        
        self.side = side
        self.transparency = transparency
        self.grid_data = grid_data

        self.level = level
        from ..fields.isovalues import normalize_isovalues, primary_isovalue, primary_side
        self.isovalues = normalize_isovalues(
            isovalues, default_level=level, default_color=color, default_side=side,
        )
        self.level = primary_isovalue(self.isovalues, default_level=level)
        self.side = primary_side(self.isovalues, default_side=side)
        if geometry_field_id:
            self.geometry_field_id = str(geometry_field_id)
        elif grid_data is not None and type(grid_data).__name__ == "Field":
            self.geometry_field_id = str(grid_data.id)
        else:
            self.geometry_field_id = None
        self.color_field_id = str(color_field_id) if color_field_id else None
        from ..fields.clip import normalize_clip_aabb
        self.clip_aabb = normalize_clip_aabb(clip_aabb)
        color = [1, 1, 1] if color is None else color
        if isinstance(color, str):
            self.color = _convert_string_color(color)
        else:
            self.color = color

        self.selection = selection
        self.carve = carve

        if issubclass(type(self.color), ColorRamp):
            dependencies = [self.grid_data, self.color]
        else:   
            dependencies = [self.grid_data]
        super().__init__(name = name, dependencies = dependencies)

    def _iso_optional_script_args(self):
        optional_arguments = []
        if not (self.selection is None):
            optional_arguments.append('selection = "%s"' % self.selection)
        if self.carve is not None:
            optional_arguments.append("carve = %s" % self.carve)
        return optional_arguments

    def _iso_color_script(self):
        if issubclass(type(self.color), ColorRamp):
            return 'cmd.color("%s", "%s")' % (self.color.name, self.name)
        return '''cmd.set_color("%s_color", %s)
cmd.color("%s_color", "%s")
''' % (self.name, self.color, self.name, self.name)

    def _script_string(self):
        optional_arguments = self._iso_optional_script_args()
        color_string = self._iso_color_script()
        cmd_name = getattr(self, "_native_iso_cmd", "isosurface")
        opt_join = (" , ".join(optional_arguments) + ", ") if optional_arguments else ""
        if getattr(self, "_native_iso_passes_side", True):
            opt_join += "side = %s" % self.side
        elif optional_arguments:
            opt_join = " , ".join(optional_arguments)
        else:
            opt_join = ""
        result = """
cmd.%s("%s", "%s", %s, %s)
%s
cmd.set("transparency", %s, "%s")
        """ % (
            cmd_name,
            self.name,
            self.grid_data.name,
            self.level,
            opt_join,
            color_string,
            self.transparency,
            self.name,
        )
        return result

    def load(self, cmd=None):
        if cmd is None:
            from pymol import cmd
        from ..Displayable import call_load
        from .map_load import bind_iso_color_ramp, load_geometry_map, sync_visual_grid_from_field

        bind_iso_color_ramp(self)
        sync_visual_grid_from_field(self, cmd)
        map_name, _rebuilt = load_geometry_map(cmd, self.grid_data, getattr(self, "clip_aabb", None))
        if not map_name:
            return
        iso_kwargs = {"level": self.level}
        if getattr(self, "_native_iso_passes_side", True):
            iso_kwargs["side"] = self.side
        sel = str(self.selection or "").strip()
        if sel and not (sel.startswith("(") and sel.endswith(")")):
            iso_kwargs["selection"] = sel
            if self.carve is not None:
                iso_kwargs["carve"] = self.carve
        cmd_name = getattr(self, "_native_iso_cmd", "isosurface")
        getattr(cmd, cmd_name)(self.name, map_name, **iso_kwargs)
        if issubclass(type(self.color), ColorRamp):
            call_load(self.color, cmd)
            cmd.color(self.color.name, self.name)
        else:
            cmd.set_color(self.name + "_color", self.color)
            cmd.color(self.name + "_color", self.name)
        cmd.set("transparency", self.transparency, self.name)
