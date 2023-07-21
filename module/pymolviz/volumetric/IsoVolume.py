import numpy as np
import logging
from .GridData import GridData
from .Volume import Volume
import pandas as pd


class IsoVolume(Volume):
    def __init__(self, grid_data : GridData, name = None, colormap = "RdYlBu_r", alphas = None, clims = None, selection = None, carve = None, margin = 0.05, state = 1):
        """ 
        Computes and collects pymol commands to load in regular data and display it as multiple, transparent, same colored iso-surfaces using PyMOLs volume command.

        Args:
            grid_data (pymolviz.RegularData): Regular data for which to show the volume.
            name (str, optional): The name of the volume as displayed in PyMOL. Defaults to {grid_data.name}_{value_label}_IsoVolume_{i}.
            value_label (str, optional): The name of the value to use from the regular data. Defaults to None. Must be passed if grid_data has multiple values.
            colormap (str, optional): The name of the colormap to use. Defaults to coolwarm.
            alphas (np.array, optional): The alphas to use. Defaults to [0.1, 0.5, 0.8].
            clims (np.array, optional): The clims to use. Defaults to [mean - 2 stddev, mean, mean + 2 stddev].
            selection (str, optional): The selection to use. Defaults to None.
            carve (float, optional): The carve to use. Defaults to None.
        """
        if clims is None:
            # for some reason it seems that isolines are only shown on 16th of 5*std + mean
            self.clims = np.linspace(-np.std(grid_data.values) * 5 + np.mean(grid_data.values), np.std(grid_data.values) * 5 + np.mean(grid_data.values), 33)
        else:
            self.clims = clims
        
        self.grid_data = grid_data

        if alphas is None:
            self.alphas = np.full(len(self.clims), 0.3)
        else:
            self.alphas = alphas

        new_clims = []
        new_alphas = []
        for i in range(len(self.clims) - 1):
            difference = (self.clims[i + 1] - self.clims[i]) * margin
            new_alphas.append(self.alphas[i])
            new_clims.append(self.clims[i])
            new_alphas.append(0)
            new_clims.append(self.clims[i] + difference)
            new_alphas.append(0)
            new_clims.append(self.clims[i + 1] - difference)

        new_alphas.append(self.alphas[-1])
        new_clims.append(self.clims[-1])
        alphas = new_alphas
        clims = new_clims

        super().__init__(grid_data, name, colormap, alphas, clims, selection, carve, state)
        
       

