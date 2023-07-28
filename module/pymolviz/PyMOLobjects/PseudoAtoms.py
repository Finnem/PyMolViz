import numpy as np
from ..Displayable import Displayable
from ..ColorMap import ColorMap


class PseudoAtoms(Displayable):
    def __init__(self, positions, color = "red", name = None, state = 1, colormap = "RdYlBu_r", *args, **kwargs):
        """ Represents a set of labels.

        Args:
            positions (np.array): The positions of the labels.
            color (list): Color of the pseudoatoms.
            name (str): Optional. The name of the data. Defaults to None.
            state (int): Optional. The state of the data. Defaults to 1.

        """
        super().__init__(name = name, **kwargs)

        positions = np.array(positions)
        if len(positions.shape) == 1:
            positions = np.array([positions])
            color = [color]
        self.positions = positions
        if type(colormap) != ColorMap:
            self.colormap = ColorMap(color, colormap, state = state, name=f"{self.name}_colormap", *args, **kwargs)
        else:
            self.colormap = colormap
        if "single" in self.colormap._color_type: # colors were not inferred
            self.color = np.arange(self.positions.shape[0]) # color is just the index
        else:
            self.color = np.array(color).flatten()
        self.state = state


    def _script_string(self):
        result = []
        colors = [list(c[:3]) for c in self.colormap.get_color(self.color)]
        for color, position in zip(colors, self.positions):
            result.append(f"""cmd.set_color("{self.name}_{color}", {color})
cmd.pseudoatom("{self.name}", pos = {position.tolist()}, color = "{self.name}_{color}", state = {self.state})""")
        return "\n".join(result)
