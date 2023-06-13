import numpy as np
import logging
from .RegularData import RegularData
from ..ColorRamp import ColorRamp
from ..util.colors import _convert_string_color

_pmv_volume_counter = 0

class Volume():
    def __init__(self, regular_data : RegularData, name = None, value_label = None, colormap = None, alphas = None, clims = None, selection = None, carve = None, state = 1):
        """ 
        Computes and collects pymol commands to load in regular data and display it volumetrically.

        Args:
            regular_data (pymolviz.RegularData): Regular data for which to show the volume.
            name (str, optional): The name of the volume as displayed in PyMOL. Defaults to {regular_data.name}_{value_label}_Volume_{i}.
            value_label (str, optional): The name of the values to use from the regular data. Defaults to None. Must be passed if regular_data has multiple values.
            colormap (str, optional): The name of the colormap to use. Defaults to coolwarm.
            alphas (np.array, optional): The alphas to use. Defaults to [0.03, 0.005, 0.1].
            clims (np.array, optional): The clims to use. Defaults to [mean - 2 stddev, mean, mean + 2 stddev].
            selection (str, optional): The selection to use. Defaults to None.
            carve (float, optional): The carve to use. Defaults to None.
        """

        global _pmv_volume_counter

        self.regular_data = regular_data
        self.value_label = value_label
        self.selection = selection
        self.carve = carve
        self.state = state
        self.value_name = ("_" + value_label) if value_label else ""
        if name is None:
            self.name = "{}{}_Volume_{}".format(regular_data.name, self.value_name, _pmv_volume_counter)
            logging.warning("No name provided for Volume. Using default name: {}. It is highly recommended to provide meaningful names.".format(self.name))
            _pmv_volume_counter += 1
        else:
            self.name = name

        
        if clims is None:
            if value_label is None:
                values = self.regular_data._values[self.regular_data._values.__iter__().__next__()]
            else:
                values = self.regular_data.values[value_label]

            std = np.std(values)
            mean = np.mean(values)
            self.clims = [mean - 2 * std, mean, mean + 2 * std]
        else:
            self.clims = clims

        self.color = ColorRamp(self.regular_data, name = self.name + "_ColorRamp", value_label = self.value_label, colormap=colormap, clims = self.clims)

        if alphas is None:
            self.alphas = [0.03, 0.005, 0.1]
        else:
            self.alphas = alphas
        if len(self.alphas) != len(self.clims):
            raise Exception("Alphas and clims must have the same length.")

        

    def _create_script(self):
        """ Creates a pymol script to create a volume representation of the given regular data.
        
        Returns:
            str: The script.
        """

        optional_arguments = []
        if not(self.selection is None):
            optional_arguments.append(f"selection = \"{self.selection}\"")
        if self.carve is not None:
            optional_arguments.append(f"carve = {self.carve}")
        result = f"""
{self.color._create_script(self.state, self.alphas)}
cmd.volume("{self.name}", "{self.regular_data.name}{self.value_name}", "{self.color.name}", {" , ".join(optional_arguments)}{"," if len(optional_arguments) > 0 else ""} state = {self.state})
        """

        return result

    def to_script(self, state = 0):
        """ Creates a pymolviz script to create a volume representation of the given regular data.
        
        Returns:
            pymolviz.Script: The script.
        """
        from ..Script import Script
        return Script([self])
