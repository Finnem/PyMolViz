import numpy as np
import logging
from .RegularData import RegularData
from ..Displayable import Displayable
from ..ColorMap import ColorMap
from ..util.colors import _convert_string_color

class Volume(Displayable):
    def __init__(self, regular_data : RegularData, name = None, colormap = "RdYlBu_r", alphas = None, clims = None, selection = None, carve = None, state = 1):
        """ 
        Computes and collects pymol commands to load in regular data and display it volumetrically.

        Args:
            regular_data (pymolviz.RegularData): Regular data for which to show the volume.
            name (str, optional): The name of the volume as displayed in PyMOL. Defaults to {regular_data.name}_{value_label}_Volume_{i}.
            colormap (str, optional): The name of the colormap to use. Defaults to coolwarm.
            alphas (np.array, optional): The alphas to use. Defaults to [0.03, 0.005, 0.1].
            clims (np.array, optional): The clims to use. Defaults to [mean - 2 stddev, mean, mean + 2 stddev].
            selection (str, optional): The selection to use. Defaults to None.
            carve (float, optional): The carve to use. Defaults to None.
        """

        self.regular_data = regular_data
        self.selection = selection
        self.carve = carve
        self.state = state

        super().__init__(name = name)
        
        if clims is None:
            values = self.regular_data.values
            std = np.std(values)
            mean = np.mean(values)
            self.clims = [mean - 2 * std, mean, mean + 2 * std]
        else:
            self.clims = clims

        if not issubclass(type(colormap), ColorMap):
            colormap = ColorMap(self.regular_data.values, colormap)
        self.colormap = colormap

        if alphas is None:
            self.alphas = [0.03, 0.005, 0.1]
        else:
            self.alphas = alphas
        if len(self.alphas) != len(self.clims):
            raise Exception("Alphas and clims must have the same length.")
        
        self.dependencies.extend([self.regular_data])

        

    def _script_string(self):
        """ Creates a pymol script to create a volume representation of the given regular data.
        
        Returns:
            str: The script.
        """

        optional_arguments = []
        if not(self.selection is None):
            optional_arguments.append(f"selection = \"{self.selection}\"")
        if self.carve is not None:
            optional_arguments.append(f"carve = {self.carve}")

        string_list = []


        if len(self.alphas) != len(self.clims):
                raise ValueError("The number of volume alphas must be equal to the number of clims.")
        string_list = [f"""cmd.volume_ramp_new("{self.name}_volume_color_ramp", [\\"""]
        for i, c in enumerate(self.clims):
            string_list.append(f"""    {self.clims[i]}, {",".join([str(v) for v in self.colormap.get_color(c)[:3]])}, {self.alphas[i]},\\""")
        string_list.append("])")
        string_list.append(f"""
cmd.volume("{self.name}", "{self.regular_data.name}", "{self.name}_volume_color_ramp", {" , ".join(optional_arguments)}{"," if len(optional_arguments) > 0 else ""})
        """)

        result = "\n".join(string_list)
        return result

    def to_script(self, state = 0):
        """ Creates a pymolviz script to create a volume representation of the given regular data.
        
        Returns:
            pymolviz.Script: The script.
        """
        from ..Script import Script
        return Script([self])
