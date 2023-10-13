import numpy as np
from ..Displayable import Displayable
from ..ColorMap import ColorMap

class Labels(Displayable):
    def __init__(self, positions, labels, color = None, name = None, state = 1, size = 24, colormap = "RdYlBu_r", *args, **kwargs):
        """ Represents a set of labels.

        Args:
            positions (np.array): The positions of the labels.
            labels (list): The labels.
            color (array-like): Color for the labels. Can follow color scheme of points. If None PyMOLs default system for choosing color is used.
            name (str): Optional. The name of the data. Defaults to None.
            state (int): Optional. The state of the data. Defaults to 1.
            size (array-like): Optional. Single value or array like, indicating the labels size.
            colormap: Optional. Defaults to "RdYlBu_r". Name of a colormap or a matplotlib colormap or a pymolviz.ColorMap object. Used to map values to colors.

        """
        positions = np.array(positions)
        if len(positions.shape) == 1:
            positions = np.array([positions])
            labels = [labels]
        self.size = np.array(size)
        if len(self.size.shape) == 0:
            self.size = np.full(len(labels), size)
        self.positions = positions
        self.labels = labels
        self.state = state

        super().__init__(name = name, **kwargs)
        if not color is None:
            if type(colormap) != ColorMap:
                self.colormap = ColorMap(color, colormap, state = state, name=f"{self.name}_colormap", *args, **kwargs)
            else:
                self.colormap = colormap
            if "single" in self.colormap._color_type: # colors were not inferred
                self.color = np.arange(len(self.labels)) # color is just the index
            else:
                self.color = np.array(color).flatten()
        else:
            self.color = None
            self.colormap = colormap

    def load(self):
        from pymol import cmd
        colors = np.full(len(self.labels), None) if self.color is None else self.color
        for i, (label, position, color, size) in enumerate(zip(self.labels, self.positions, colors, self.size)):
            used_position = position.tolist()
            cmd.pseudoatom(self.name, label=label, pos = used_position, state = self.state)
            if not (color is None):
                cmd.set_color("{self.name}_{i}", self.colormap.get_color(color)[:3])
                cmd.set("label_color", "{self.name}_{i}", self.name)
            cmd.set("label_size", size, self.name)
        
    def _script_string(self):
        result = []
        colors = np.full(len(self.labels), None) if self.color is None else self.color
        for i, (label, position, color, size) in enumerate(zip(self.labels, self.positions, colors, self.size)):
            used_position = position.tolist()
            result.append(f"""cmd.pseudoatom("{self.name}", label="{label}", pos = {used_position}, state = {self.state})""")
            if not (color is None):
                result.append(f"""
cmd.set_color("{self.name}_{i}", {self.colormap.get_color(color)[:3]})
cmd.set("label_color", "{self.name}_{i}", "{self.name}")
""")
            result.append(f"""
cmd.set("label_size", {size}, "{self.name}")
""")
        return "\n".join(result)
