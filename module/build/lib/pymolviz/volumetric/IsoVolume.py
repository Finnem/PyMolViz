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
            grid_data (pymolviz.GridData): The data to use for the iso surface.
            name (str): Optional. Defaults to None. The name of the object.
            colormap (str or list of float): Optional. Defaults to "RdYlBu_r". The colormap of the object.
            alphas (list of float): Optional. Defaults to None. The alpha values to use.
            clims (list of float): Optional. Defaults to None. The color limits to use.
            selection (str): Optional. Defaults to None. The selection to use.
            carve (float): Optional. Defaults to None. The carve value to use.
            margin (float): Optional. Defaults to 0.05. The margin to use.
            state (int): Optional. Defaults to 1. The state to use.
            
           
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
        
       

