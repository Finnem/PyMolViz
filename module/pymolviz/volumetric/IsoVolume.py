import numpy as np
import logging
from .RegularData import RegularData
from .Volume import Volume
from ..util.colors import _convert_string_color
import pandas as pd

_pmv_isovolume_counter= 0

class IsoVolume(Volume):
    def __init__(self, regular_data : RegularData, name = None, value_label = None, colormap = None, alphas = None, clims = None, selection = None, carve = None, margin = 0.05, state = 0):
        """ 
        Computes and collects pymol commands to load in regular data and display it as multiple, transparent, same colored iso-surfaces using PyMOLs volume command.

        Args:
            regular_data (pymolviz.RegularData): Regular data for which to show the volume.
            name (str, optional): The name of the volume as displayed in PyMOL. Defaults to {regular_data.name}_{value_label}_IsoVolume_{i}.
            value_label (str, optional): The name of the value to use from the regular data. Defaults to None. Must be passed if regular_data has multiple values.
            colormap (str, optional): The name of the colormap to use. Defaults to coolwarm.
            alphas (np.array, optional): The alphas to use. Defaults to [0.1, 0.5, 0.8].
            clims (np.array, optional): The clims to use. Defaults to [mean - 2 stddev, mean, mean + 2 stddev].
            selection (str, optional): The selection to use. Defaults to None.
            carve (float, optional): The carve to use. Defaults to None.
        """
        global _pmv_isovolume_counter
        if clims is None:
            clims = list(pd.qcut(regular_data.values[value_label], 7, retbins=True)[1])
        else:
            self.clims = clims

        if alphas is None:
            alphas = np.linspace(0.1, 0.8, len(clims))
        self.value_name = ("_" + value_label) if value_label else ""
        if name is None:
            name = "{}{}_IsoVolume_{}".format(regular_data.name, self.value_name, _pmv_isovolume_counter)
            logging.warning("No name provided for IsoVolume. Using default name: {}. It is highly recommended to provide meaningful names.".format(name))
            _pmv_isovolume_counter += 1
        else:
            self.name = name

        self.regular_data = regular_data

        new_clims = []
        new_alphas = []
        for i in range(len(self.clims) - 1):
            difference = (self.clims[i + 1] - self.clims[i]) * margin
            new_alphas.append(alphas[i])
            new_clims.append(self.clims[i])
            new_alphas.append(0)
            new_clims.append(self.clims[i] + difference)
            new_alphas.append(0)
            new_clims.append(self.clims[i + 1] - difference)
        new_alphas.append(alphas[-1])
        new_clims.append(self.clims[-1])
        alphas = new_alphas
        clims = new_clims

        super().__init__(regular_data, name, value_label, colormap, alphas, clims, selection, carve, state)
        
       

