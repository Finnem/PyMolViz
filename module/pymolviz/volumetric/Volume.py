import numpy as np
import logging
from .GridData import GridData
from ..Displayable import Displayable
from ..ColorMap import ColorMap
from ..util.colors import _convert_string_color

class Volume(Displayable):
    def __init__(self, grid_data : GridData, name = None, colormap = "RdYlBu_r", alphas = None, clims = None, selection = None, carve = None, state = 1):
        """ 
        Computes and collects pymol commands to load in regular data and display it volumetrically.

        Args:
            grid_data (pymolviz.RegularData): Regular data for which to show the volume.
            name (str, optional): The name of the volume as displayed in PyMOL. Defaults to {grid_data.name}_{value_label}_Volume_{i}.
            colormap (str, optional): The name of the colormap to use. Defaults to coolwarm.
            alphas (np.array, optional): The alphas to use. Defaults to [0.03, 0.005, 0.1].
            clims (np.array, optional): The clims to use. Defaults to [mean - 2 stddev, mean, mean + 2 stddev].
            selection (str, optional): The selection to use. Defaults to None.
            carve (float, optional): The carve to use. Defaults to None.
            state (int, optional): The state to use. Defaults to 1.
        """

        self.grid_data = grid_data
        self.selection = selection
        self.carve = carve
        self.state = state

        super().__init__(name = name)
        
        if clims is None:
            min_val = max([np.min(grid_data.values), -np.std(grid_data.values) * 5 + np.mean(grid_data.values)])
            max_val = min([np.max(grid_data.values), np.std(grid_data.values) * 5 + np.mean(grid_data.values)])
            self.clims = np.linspace(min_val, max_val, 33)
            # getting number of values within each bin
            self.clims = np.vstack([self.clims[:-1], self.clims[1:]]).T.flatten()
            self.clims = np.hstack([self.clims, self.clims[-1]])
        else:
            self.clims = clims

        if not issubclass(type(colormap), ColorMap):
            colormap = ColorMap(self.clims, colormap, state = state, name = f"{self.name}_colormap")
        self.colormap = colormap

        if alphas is None:
            used_length = len(self.clims)-(len(self.clims) % 2) # if length is uneven, we forgo the last value for binning
            bins = np.reshape(self.clims[:used_length], (-1, 2))
            densities = np.sum((bins[:,0] < self.grid_data.values[:, None]) & (bins[:,1] >= self.grid_data.values[:, None]), axis = 0)
            if np.sum(densities) == 0: densities = np.ones(len(densities))
            densities = densities / np.sum(densities)
            densities = np.clip(densities, np.min(densities), 0.9)
            self.alphas = np.vstack([(1 - densities[:used_length]) * 0.03, np.full(len(densities[:used_length]), 0.005)]).T.flatten()
            if len(self.clims) % 2 == 1:
                self.alphas = np.hstack([self.alphas, (1 - densities[-1]) * 0.03])
        else:
            self.alphas = np.array(alphas)
        if len(self.alphas) != len(self.clims):
            raise Exception("Alphas and clims must have the same length.")
        
        self.dependencies.extend([self.grid_data])

        

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
cmd.volume("{self.name}", "{self.grid_data.name}", "{self.name}_volume_color_ramp", {" , ".join(optional_arguments)}{"," if len(optional_arguments) > 0 else ""} state={self.state})
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
