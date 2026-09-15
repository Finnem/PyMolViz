import numpy as np
from ..Displayable import Displayable
from ..ColorMap import ColorMap

class ColorRamp(Displayable):
    def __init__(self, data = None, name = None, colormap = "RdYlBu_r",  clims = None, state = 1, interpolate=None):
        """ 
        Computes a color ramp which may be used for different colorings.

        Args:
            data (pymolviz.GridData): The data to use for the color ramp. Defaults to None.
            name (str): Optional. Defaults to None. The name of the object.
            colormap (str): Optional. Defaults to "RdYlBu_r". The colormap to use.
            clims (list of float): Optional. Defaults to None. The color limits to use.            
            state (int): Optional. Defaults to 1. The state to use.
            interpolate (bool): Optional. Defaults to None. False holds each
                integer category color across ``[i-0.5, i+0.5]`` (nearest bin).
                None infers stepped slots for discrete RGB palettes.
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
        if interpolate is None:
            interpolate = getattr(colormap, "_color_type", None) != "multi_single"
        self.interpolate = bool(interpolate)

        super().__init__(name = name, dependencies = [self.data])

    def _ramp_table(self):
        """``(range_values, rgb_rows)`` for ``cmd.ramp_new``."""
        if self.interpolate:
            sample_points = np.linspace(self.clims[0], self.clims[-1], 100)
            colors = self.colormap.get_color(sample_points)[:, :3]
            return [float(c) for c in sample_points], [[float(c) for c in row] for row in colors]
        lo = float(self.clims[0])
        hi = float(self.clims[-1])
        i0 = int(np.floor(lo))
        i1 = int(np.ceil(hi))
        if i1 < i0:
            i1 = i0
        values = np.arange(i0, i1 + 1, dtype=float)
        colors = self.colormap.get_color(values)[:, :3]
        ranges = []
        rows = []
        half = 0.5
        for i, rgb in enumerate(colors):
            center = float(values[i])
            ranges.extend([center - half, center + half])
            color = [float(c) for c in rgb]
            rows.extend([color, color])
        return ranges, rows

    def _script_string(self):
        """ Creates a pymol script to create an isosurface color ramp.
        
        Returns:
            str: The script.
        """
        sample_points, colors = self._ramp_table()
        result = f"""cmd.ramp_new("{self.name}", "{self.data.name}", range = [{",".join([str(c) for c in sample_points])}], color = [{", ".join(["[" + ", ".join([str(c) for c in color]) + "]" for color in colors])}], state = {self.state})"""
        
        return result
    
    def load(self, cmd=None):
        if cmd is None:
            from pymol import cmd
        sample_points, colors = self._ramp_table()
        from ..Displayable import call_load
        call_load(self.data, cmd)
        cmd.ramp_new(self.name, self.data.name, range = sample_points, color = colors, state = self.state)
