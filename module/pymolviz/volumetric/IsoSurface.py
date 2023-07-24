import numpy as np
import logging
from .GridData import GridData
from .ColorRamp import ColorRamp
from ..Displayable import Displayable
from ..util.colors import _convert_string_color


class IsoSurface(Displayable):
    def __init__(self, grid_data : GridData, level: float, name = None, color = None, transparency = 0, selection = None, carve = None, side = 1):
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
