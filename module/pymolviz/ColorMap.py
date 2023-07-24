import numpy as np
import logging
from .util.colors import get_colormap, _convert_string_color
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors 
import matplotlib.cm 
from matplotlib.colors import LinearSegmentedColormap
from .Displayable import Displayable

class ColorMap(Displayable):
    """
    Creates a colormap from a list of values.
    
    Args:
        values (list of float or list of (float, float) or list of (float, str) or list of (float, list of float)): The values to use for the colormap. If values_are_single_color is True, this is interpreted as a list of colors. If values_are_single_color is False, this is interpreted as a list of values to be mapped to a colormap. If values_are_single_color is None, this is interpreted as a list of values to be mapped to a colormap, unless it is a list of colors (strings or 3 or 4 long array-like), in which case it is interpreted as a list of colors.
        colormap (str or matplotlib colormap or pymolviz.ColorMap): Optional. Defaults to "RdYlBu_r". The colormap to use.
        values_are_single_color (bool): Optional. Defaults to None. If True, values is interpreted as a list of colors. If False, values is interpreted as a list of values to be mapped to a colormap. If None, values is interpreted as a list of values to be mapped to a colormap, unless it is a list of colors (strings or 3 or 4 long array-like), in which case it is interpreted as a list of colors.
        name (str): Optional. Defaults to None. The name of the object.
        state (int): Optional. Defaults to 1. The state of the object.
    """

    def __init__(self, values, colormap = "RdYlBu_r", values_are_single_color = None, name = None, state = 1):
        

        # get colormap
        self.colormap = None
        self._color_type = None
        # differentiating between values
        self.color = None
        self.state = state
        if (values_are_single_color is True) or (values_are_single_color is None):

            # if values is a single rgb(a) color
            try:
                if np.issubdtype(type(values), np.str_):
                    color, alpha = self._value2color(values)
                    self.clims = [0, 1]
                    self._norm = matplotlib.colors.Normalize(vmin = self.clims[0], vmax = self.clims[1])
                    self.colormap = LinearSegmentedColormap.from_list("custom", list(zip([0, 1], [color, color])))
                    self._color_type = "single_str"
                elif len(values) in [3, 4]:
                    color, alpha = self._value2color(values)
                    self.clims = [0, 1]
                    self._norm = matplotlib.colors.Normalize(vmin = self.clims[0], vmax = self.clims[1])
                    self.colormap = LinearSegmentedColormap.from_list("custom", list(zip([0, 1], [color, color])))
                    self._color_type = "single_rgb"
                    if (self.colormap is not None) and (values_are_single_color is None):
                        logging.warning(f"Interpreted {len(values)} values as a single color. If you intended them to be values to extrapolate from, set values_are_single_color to False. If you intended them to be a color, set values_are_single_color to True to suppress this warning.")
            except TypeError:
                pass

        # if values is a list of colors
        if ((values_are_single_color is False) or (values_are_single_color is None)) and (self.colormap is None):
            colors = []
            try:
                if (np.issubdtype(type(values[0]), np.str_)) or (len(values[0]) in [3, 4]):
                    for i, color in enumerate(values):
                        # lists of colors
                            color, alpha = self._value2color(color)
                            if color is None:
                                logging.error(f"Could not interpret {color} at index {i} as a color, but first element of values was a color (string or 3 or 4 long array-like).")
                                break
                            colors.append(color)
                    self.clims = [0, len(colors) - 1]
                    self._norm = matplotlib.colors.Normalize(vmin = self.clims[0], vmax = self.clims[1])
                    self.colormap = LinearSegmentedColormap.from_list("custom", list(zip(np.linspace(0, 1, len(colors)), colors)))
                    self._color_type = "multi_single"
            except TypeError:
                pass
            
            if self.colormap is None: # if any of the values are not colors
                try:
                    # list of values
                    values = [float(v) for v in values]
                    self.clims = [np.min(values), np.max(values)]
                    self._norm = matplotlib.colors.Normalize(vmin = self.clims[0], vmax = self.clims[1])
                    self.colormap = get_colormap(colormap)
                    self._color_type = "multi_inferred"
                except TypeError:
                    try:
                        # list of (value, color) pairs
                        colors = []
                        mapped_values = []
                        for i, (value, color) in enumerate(values):
                            color, alpha = self._value2color(color)
                            colors.append(color)
                            mapped_values.append(value)
                        self.clims = [np.min(mapped_values), np.max(mapped_values)]
                        self._norm = matplotlib.colors.Normalize(vmin = self.clims[0], vmax = self.clims[1])
                        self.colormap = LinearSegmentedColormap.from_list("custom", list(zip(self._norm(mapped_values), colors)))
                        self._color_type = "segmented_single"
                    except TypeError:
                        # list of (value, value) pairs
                        colors = []
                        mapped_values = []
                        _colormap = get_colormap(colormap)
                        for i, (value, color_value) in enumerate(values):
                            color_value = float(color_value)
                            if (color_value < 0) or (color_value > 1):
                                logging.error(f"Could not interpret {color_value} at index {i} as a color value, but first element of values was a list of (value, value) pairs.")
                                raise TypeError
                            color = _colormap(color_value)
                            colors.append(color)
                            mapped_values.append(value)
                        self.clims = [np.min(mapped_values), np.max(mapped_values)]
                        self._norm = matplotlib.colors.Normalize(vmin = self.clims[0], vmax = self.clims[1])
                        self.colormap = LinearSegmentedColormap.from_list("custom", list(zip(self._norm(mapped_values), colors)))
                        self._color_type = "segmented_inferred"
        if self.colormap is None:
            raise ValueError("Could not infer a colormap from the given values.")

        super().__init__(name = name)

    def get_color(self, values):
        return self.colormap(self._norm(values))
    
    def get_figure(self, figsize : tuple = (1,6), ax = None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize = figsize)
        else:
            fig = ax.figure

        fig.colorbar(matplotlib.cm.ScalarMappable(norm = self._norm, cmap = self.colormap), cax = ax, **kwargs)

    def _script_string(self):
        """ Creates a pymol script to create this colorbar as a pymol displayable.
        
        Returns:
            str: The script.
        """
        from .volumetric.GridData import GridData
        dummy_data = GridData(np.zeros(8), name="cbar_dummy", step_sizes=(1e-8,1e-8,1e-8), step_counts=(1,1,1)) 
        
        sample_points = np.linspace(self.clims[0], self.clims[-1], 100)
        colors = self.get_color(sample_points)[:,:3]
        result = []
        result.append(dummy_data._script_string())
        result.append(f"""cmd.ramp_new("{self.name}", "{dummy_data.name}", range = [{",".join([str(c) for c in sample_points])}], color = [{", ".join(["[" + ", ".join([str(c) for c in color]) + "]" for color in colors])}], state = {self.state})""")
        result.append(f"""cmd.delete("{dummy_data.name}")""")

        result = "\n".join(result)
        return result

    def _value2color(self, value):
        color = None
        alpha = None
        if np.issubdtype(type(value), np.str_):
            return _convert_string_color(value), 1
        try:
            if len(value) in [3, 4]:
                if all([np.issubdtype(type(v), np.integer) for v in value]):
                    color = [float(v) / 255 for v in value]
                if all([v <= 1 for v in value]):
                    color = np.array([float(v) for v in value])
                if len(value) != 4:
                    alpha = 1
                else:
                    alpha = color[3]
                    color = color[:3]
        except TypeError:
            pass
        if color is None:
            raise TypeError
        return color, alpha
