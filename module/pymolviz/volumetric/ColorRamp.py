import numpy as np
from ..Displayable import Displayable
from ..ColorMap import ColorMap

class ColorRamp(Displayable):
    def __init__(self, data = None, name = None, colormap = "RdYlBu_r",  clims = None):
        """ 
        Computes a color ramp which may be used for different colorings.

        Args:
            data (numpy.array or pymolviz.RegularData): Data from which to compute the color ramp. If none is given, the graphical generation functions still work based on clims.
            name (str, optional): The name of the colorramp when displayed in PyMOL. Defaults to ColorRamp_{i}.
            value_label (str, optional): The name of the values to use, if pymolviz.RegularData was passed as data. Defaults to None.
            colormap (str or matplotlib.colormap or list of colors, optional): The name of the colormap to use. Defaults to coolwarm.
            clims (np.array, optional): The clims to use. Defaults to [min(data), max(data)].
        """

        self.data = data
        if not issubclass(type(colormap), ColorMap):
            colormap = ColorMap(self.data.values, colormap)
        else:
            if clims is None:
                clims = colormap.clims

        if clims is None:
            self.clims = [np.min(self.data.values), np.max(self.data.values)]
        else:
            self.clims = clims
        self.colormap = colormap

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
        result = f"""cmd.ramp_new("{self.name}", "{self.data.name}", range = [{",".join([str(c) for c in sample_points])}], color = [{", ".join(["[" + ", ".join([str(c) for c in color]) + "]" for color in colors])}])"""
        
        return result
