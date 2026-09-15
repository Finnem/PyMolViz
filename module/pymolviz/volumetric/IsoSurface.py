import numpy as np
import logging
from .GridData import GridData
from .ColorRamp import ColorRamp
from ..Displayable import Displayable
from ..util.colors import _convert_string_color


class IsoSurface(Displayable):
    renders_cgo = False

    def __init__(self, grid_data : GridData, level: float, name = None, color = None, transparency = 0, selection = '', carve = None, side = 1, geometry_field_id=None, color_field_id=None, isovalues=None, clip_aabb=None):
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
        self.geometry_field_id = str(geometry_field_id) if geometry_field_id else None
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

    def _script_string(self):
        """ Creates a pymol script to create an isomesh representation of the given regular data.
        
        Returns:
            str: The script.
        """

        optional_arguments = []
        if not(self.selection is None):
            optional_arguments.append(f"selection = \"{self.selection}\"")
        if self.carve is not None:
            optional_arguments.append(f"carve = {self.carve}")

        if issubclass(type(self.color), ColorRamp):
            color_string = f'cmd.color("{self.color.name}", "{self.name}")'
        else:
            color_string = f'''cmd.set_color("{self.name}_color", {self.color})
cmd.color("{self.name}_color", "{self.name}")
'''

        
        result = f"""
cmd.isosurface("{self.name}", "{self.grid_data.name}", {self.level}, {" , ".join(optional_arguments)}{"," if len(optional_arguments) > 0 else ""} side = {self.side})
{color_string}
cmd.set("transparency", {self.transparency}, "{self.name}")
        """
        
        return result
    
    def load(self, cmd=None):
        if cmd is None:
            from pymol import cmd
        from ..Displayable import call_load
        from ..wizards.builders.field_visual import bind_iso_color_ramp

        bind_iso_color_ramp(self)
        from ..wizards.builders.field_visual import load_geometry_map, sync_visual_grid_from_field

        sync_visual_grid_from_field(self, cmd)
        map_name, _rebuilt = load_geometry_map(cmd, self.grid_data, getattr(self, "clip_aabb", None))
        if not map_name:
            return
        cmd.isosurface(self.name, map_name, level=self.level, side=self.side, selection=self.selection, carve=self.carve)
        if issubclass(type(self.color), ColorRamp):
            call_load(self.color, cmd)
            cmd.color(self.color.name, self.name)
        else:
            cmd.set_color(self.name + "_color", self.color)
            cmd.color(self.name + "_color", self.name)
        cmd.set("transparency", self.transparency, self.name)
