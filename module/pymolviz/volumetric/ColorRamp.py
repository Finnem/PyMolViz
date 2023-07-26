import numpy as np
from ..Displayable import Displayable
from ..ColorMap import ColorMap

class ColorRamp(Displayable):
    def __init__(self, data = None, name = None, colormap = "RdYlBu_r",  clims = None, state = 1):
        """ 
        Computes a color ramp which may be used for different colorings.

        Args:
            data (pymolviz.GridData): The data to use for the color ramp. Defaults to None.
            name (str): Optional. Defaults to None. The name of the object.
            colormap (str): Optional. Defaults to "RdYlBu_r". The colormap to use.
            clims (list of float): Optional. Defaults to None. The color limits to use.            
            state (int): Optional. Defaults to 1. The state to use.
        """

        self.data = data
        if not issubclass(type(colormap), ColorMap):
            colormap = ColorMap(self.data.values, colormap)
        else:
            if clims is None:
                clims = colormap.clims

        if clims is None:
            min_val = max([np.min(self.data.values), -np.std(self.data.values) * 5 + np.mean(self.data.values)])
            max_val = min([np.max(self.data.values), np.std(self.data.values) * 5 + np.mean(self.data.values)])
            self.clims = [min_val, max_val]
        else:
            self.clims = clims
        self.colormap = colormap
        self.state = state

        super().__init__(name = name, dependencies = [self.data])

    def _script_string(self):
        """ Creates a pymol script to create a volume representation of the given regular data.
        
        Args:
            state (int, optional): The state to use. Defaults to 0.
            volume_alphas (list of float, optional): When passed a volume color ramp is created and these values are used as alpha values.
                   
        Returns:
            str: The script.
        """
        sample_points = np.linspace(self.clims[0], self.clims[-1], 100)
        colors = self.colormap.get_color(sample_points)[:,:3]
        result = f"""cmd.ramp_new("{self.name}", "{self.data.name}", range = [{",".join([str(c) for c in sample_points])}], color = [{", ".join(["[" + ", ".join([str(c) for c in color]) + "]" for color in colors])}], state = {self.state})"""
        
        return result
