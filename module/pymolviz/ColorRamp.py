import numpy as np
import logging
from .volumetric.RegularData import RegularData
from .util.colors import _convert_string_color

_pmv_colorramp_counter = 0

class ColorRamp():
    def __init__(self, data = None, name = None, value_label = None, colormap = None,  clims = None):
        """ 
        Computes a color ramp which may be used for different colorings.

        Args:
            data (numpy.array or pymolviz.RegularData): Data from which to compute the color ramp. If none is given, the graphical generation functions still work based on clims.
            name (str, optional): The name of the colorramp when displayed in PyMOL. Defaults to ColorRamp_{i}.
            value_label (str, optional): The name of the values to use, if pymolviz.RegularData was passed as data. Defaults to None.
            colormap (str or matplotlib.colormap or list of colors, optional): The name of the colormap to use. Defaults to coolwarm.
            clims (np.array, optional): The clims to use. Defaults to [min(data), max(data)].
        """

        global _pmv_colorramp_counter

        self.data = data
        self.value_label = value_label
        self.value_name = ("_" + value_label) if value_label else ""
        if name is None:
            self.name = "ColorRamp_{}".format(_pmv_colorramp_counter)
            logging.warning("No name provided for ColorRamp. Using default name: {}. It is highly recommended to provide meaningful names.".format(self.name))
            _pmv_colorramp_counter += 1
        else:
            self.name = name
        if clims is None:
            if isinstance(data, RegularData):
                if value_label is None:
                    self.values = self.data._values[self.data._values.__iter__().__next__()]
                else:
                    self.values = self.data.values[value_label]
            else:
                self.values = data
            minimum = np.min(self.values)
            maximum = np.max(self.values)
            self.clims = [minimum, maximum]
        else:
            self.clims = clims

        self.colormap = self._infer_colormap(colormap)

    def get_value_colors(self):
        """ Returns the colors for the values in the color ramp.
        
        Returns:
            list of (float, float, float): The colors.
        """
        return self._apply_cmap(self.values)

    def _infer_colormap(self, colormap = None):
        """ Infers a colormap from the given data.
        
        Returns:
            matplotlib.colormap: The colormap.
        """
        from matplotlib.colors import Colormap

        if colormap is None:
            colormap = "coolwarm"
        if type(colormap) is str:
            from matplotlib import cm
            colormap = cm.get_cmap(colormap)
        elif type(colormap) is list:
            if len(colormap) == len(self.clims):
                from matplotlib.colors import LinearSegmentedColormap
                colors = []
                for i, c in enumerate(colormap):
                    if type(c) is str:
                        c = _convert_string_color(c)
                    colors.append((self.clims[i], c))

                colormap = LinearSegmentedColormap.from_list(self.name, colors)
            else:
                raise ValueError("For a colormap that is a list of colors, it must have the same length as the clims.")
        elif isinstance(colormap, Colormap):
            self.colormap = colormap
        else:
            raise ValueError("The colormap must be a string, a matplotlib colormap or a list of colors.")
        return colormap 

    def _apply_cmap(self, value):
        """ Applies the colormap to the given value.
        
        Args:
            value (float): The value to apply the colormap to.
        
        Returns:
            (float, float, float): The color.
        """
        value = np.copy(value)
        value -= self.clims[0]
        value /= self.clims[-1] - self.clims[0]
        colors = np.array(self.colormap(value, alpha=None))
        if colors.ndim == 1:
            return colors[:3]
        else:
            return colors[:, :3]

    def _create_script(self, state = 0, volume_alphas = None):
        """ Creates a pymol script to create a volume representation of the given regular data.
        
        Args:
            state (int, optional): The state to use. Defaults to 0.
            volume_alphas (list of float, optional): When passed a volume color ramp is created and these values are used as alpha values.
                   
        Returns:
            str: The script.
        """
        if volume_alphas is None:
            sample_points = np.linspace(self.clims[0], self.clims[-1], 100)
            colors = self._apply_cmap(sample_points)
            result = f"""cmd.ramp_new("{self.name}", "{self.data.name}{self.value_name}", range = [{",".join([str(c) for c in sample_points])}], color = [{", ".join(["[" + ", ".join([str(c) for c in color]) + "]" for color in colors])}])"""
        else:
            if len(volume_alphas) != len(self.clims):
                raise ValueError("The number of volume alphas must be equal to the number of clims.")
            color_ramp_string_list = [f"""cmd.volume_ramp_new("{self.name}", [\\"""]
            for i, c in enumerate(self.clims):
                color_ramp_string_list.append(f"""    {self.clims[i]}, {",".join([str(v) for v in self._apply_cmap(c)[:3]])}, {volume_alphas[i]},\\""")
            color_ramp_string_list.append("])")
            result = "\n".join(color_ramp_string_list)
        return result

    def to_script(self, state = 0):
        """ Creates a pymolviz script to create a volume representation of the given regular data.
        
        Returns:
            pymolviz.Script: The script.
        """
        from .Script import Script
        return Script([self])
