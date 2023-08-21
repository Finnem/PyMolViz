import numpy as np
from ..Displayable import Displayable
from ..ColorMap import ColorMap

class Expressions(Displayable):
    def __init__(self, expressions, color = "red", colormap ="RdYlBu_r", *args, **kwargs):
        """ Represents a set of PyMOL expressions, which will then be colored according to colors. 
        Can be used to easily color objects accessed by expressions according to a color map.

        Args:

            expressions (array-like): List of expressions strings will be assigned a color using the 'color' command.
            color (array-like): Optional. Defaults to red. An array like of colors. Can be a single color, a list of colors, or a list of values to be mapped to a colormap.
            colormap: Optional. Defaults to "RdYlBu_r". Name of a colormap or a matplotlib colormap or a pymolviz.ColorMap object. Used to map values to colors.
        """
        super().__init__("dummy")
        if type(colormap) != ColorMap:
            self.colormap = ColorMap(color, colormap, name=f"{self.name}_colormap", *args, **kwargs)
        else:
            self.colormap = colormap
        if "single" in self.colormap._color_type: # colors were not inferred
            self.color = np.arange(len(self.expressions)) # color is just the index
        else:
            self.color = np.array(color).flatten()
        self.expressions = expressions


    def _script_string(self):
        result = []
        for i, (expression, color) in enumerate(zip(self.expressions, self.color)):
            result.append(f"""
cmd.set_color("{self.name}_{i}", {self.colormap.get_color(color)[:3]})
cmd.color("{self.name}_{i}", "{expression}", 1)""")
        return "\n".join(result)

