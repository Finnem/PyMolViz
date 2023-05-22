import numpy as np
import logging
from .RegularData import RegularData
from ..ColorRamp import ColorRamp
from ..util.colors import _convert_string_color

_pmv_isosurface_counter= 0

class IsoSurface():
    def __init__(self, regular_data : RegularData, level: float, name = None, value_label = None, color = None, transparancy = 0, selection = None, carve = None, side = 1, in_sigma = False):
        """ 
        Computes and collects pymol commands to load in regular data and display an iso mesh at the given level.
        Note that, since this is based on volumetric data it is different from the pmv.Mesh class.

        Args:
            regular_data (pymolviz.RegularData): Regular data for which to show the isomesh.
            level (float): The level at which to display the isomesh.
            name (str, optional): The name of the mesh as displayed in PyMOL. Defaults to {regular_data.name}_{value_label}_IsoMesh_{i}.
            value_label (str, optional): The name of the value to use from the regular data. Defaults to None. Must be passed if regular_data has multiple values.
            color (str or rgb or pymolviz.ColorRamp, optional): The name of the color to use or rgb values or a pymolviz ColorRamp which will be used to color based on position. Defaults to white. 
            transparancy (float): Transparancy of the surface, defaults to 1.
            selection (str, optional): The selection to use. Defaults to None.
            carve (float, optional): The carve to use. Defaults to None.
            side (int, optional): The side of the isosurface to show. Defaults to 1 (outside/gradient facing).
            in_sigma (bool, optional): Whether the level is in sigma. Defaults to False.
        """
        global _pmv_isosurface_counter
        self.value_name = ("_" + value_label) if value_label else ""
        if name is None:
            name = "{}{}_IsoSurface_{}".format(regular_data.name, self.value_name, _pmv_isosurface_counter)
            logging.warning("No name provided for IsoSurface. Using default name: {}. It is highly recommended to provide meaningful names.".format(name))
            _pmv_isosurface_counter += 1
        else:
            self.name = name
        self.regular_data = regular_data
        self.value_label = value_label
        self.side = side
        self.transparancy = transparancy
        if in_sigma:
            self.level = level
        else:
            if value_label is None:
                values = self.regular_data._values[self.regular_data._values.__iter__().__next__()]
            else:
                values = self.regular_data.values[value_label]
            std = np.std(values)
            self.level = level / std
        if isinstance(color, str):
            self.color = _convert_string_color(color) if color else (1, 1, 1)
        else:
            self.color = color
        self.selection = selection
        self.carve = carve

    def _create_script(self, state = 0):
        """ Creates a pymol script to create an isomesh representation of the given regular data.
        
        Returns:
            str: The script.
        """

        optional_arguments = []
        if not(self.selection is None):
            optional_arguments.append(f"selection = \"{self.selection}\"")
        if self.carve is not None:
            optional_arguments.append(f"carve = {self.carve}")

        if isinstance(self.color, ColorRamp):
            color_string = self.color._create_script()
        else:
            color_string = f"""cmd.set_color("{self.name}_color", {self.color})"""
        result = f"""
cmd.isosurface("{self.name}", "{self.regular_data.name}{self.value_name}", {self.level}, {" , ".join(optional_arguments)}{"," if len(optional_arguments) > 0 else ""} state = {state}, side = {self.side})
{color_string}
cmd.color("{self.color.name}", "{self.name}")
cmd.set("transparency", {self.transparancy}, "{self.name}")
        """
        
        return result

    def to_script(self, state = 0):
        """ Creates a pymolviz script to create a volume representation of the given regular data.
        
        Returns:
            pymolviz.Script: The script.
        """
        from ..Script import Script
        return Script([self])
