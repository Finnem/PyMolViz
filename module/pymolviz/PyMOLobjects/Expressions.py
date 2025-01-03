import numpy as np
from ..Displayable import Displayable
from ..ColorMap import ColorMap
from ..util.sanitize import sanitize_pymol_string

class Expressions(Displayable):
    def __init__(self, expressions, color = None, colormap ="RdYlBu_r", transparencies = None, *args, **kwargs):
        """ Represents a set of PyMOL expressions, which will then be colored according to colors. 
        Can be used to easily color objects accessed by expressions according to a color map.

        Args:

            expressions (array-like): List of expressions strings will be assigned a color using the 'color' command.
            color (array-like): Optional. Defaults to red. An array like of colors. Can be a single color, a list of colors, or a list of values to be mapped to a colormap.
            colormap: Optional. Defaults to "RdYlBu_r". Name of a colormap or a matplotlib colormap or a pymolviz.ColorMap object. Used to map values to colors.
        """
        super().__init__("dummy")
        self.expressions = expressions
        if not color is None:
            if type(colormap) != ColorMap:
                self.colormap = ColorMap(color, colormap, name=f"{self.name}_colormap", *args, **kwargs)
            else:
                self.colormap = colormap
            if "single" in self.colormap._color_type: # colors were not inferred
                self.color = np.arange(len(self.expressions)) # color is just the index
            else:
                self.color = np.array(color).flatten()
        else:
            self.color = None
        if transparencies is None:
            self.transparencies = [0] * len(self.expressions)
        else:
            self.transparencies = np.array(transparencies).flatten()


    def _script_string(self):
        result = []
        try:
            self.transparencies[0]
        except TypeError:
            self.transparencies = [self.transparencies] * len(self.expressions)
        try:
            self.color[0]
        except TypeError:
            self.color = [self.color] * len(self.expressions)
        for i, (expression, color, transparency) in enumerate(zip(self.expressions, self.color, self.transparencies)):
            if not color is None:
                result.append(f"""
cmd.set_color("{self.name}_{i}", {self.colormap.get_color(color)[:3]})
cmd.color("{self.name}_{i}", "{expression}", 1)""")
            if not transparency is None:
                result.append(f"""
cmd.set("stick_transparency", {transparency}, f"{expression}")""")
        return "\n".join(result)
    
    def load(self):
        from pymol import cmd
        for i, (expression, color, transparency) in enumerate(zip(self.expressions, self.color, self.transparencies)):
            expression_name = self.name + "_" + str(i)
            if not color is None:
                cmd.set_color(expression_name, self.colormap.get_color(color)[:3])
                cmd.color(expression_name, expression, 1)
            if not transparency is None:
                cmd.set("stick_transparency", transparency, expression)
