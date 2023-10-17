import numpy as np
from ..Displayable import Displayable
from ..ColorMap import ColorMap
from ..util import ViewportCallback as vc


class Label2D(Displayable):
    def __init__(self, position, label, color = None, name = None, state = 1, size = 24, colormap = "RdYlBu_r", *args, **kwargs):
        """ Represents a label at 2D positions.

        Args:
            position (np.array): The position of the label.
            label (str): The label.
            color (array-like): Color for the label. If None PyMOLs default system for choosing color is used.
            name (str): Optional. The name of the data. Defaults to None.
            state (int): Optional. The state of the data. Defaults to 1.
            size (int): Optional. Single value indicating the labels size.
            colormap: Optional. Defaults to "RdYlBu_r". Name of a colormap or a matplotlib colormap or a pymolviz.ColorMap object. Used to map values to colors.

        """
        
        position = np.array(position)
        self.size = int(size)
        self.position = position
        self.label = str(label)
        self.state = int(state)
        
        super().__init__(name = name, **kwargs)
        
        
        existing_viewport = vc.get_viewport_callback(self.position)
        if existing_viewport != None:
            self.viewport = existing_viewport
            self.dependencies.extend([existing_viewport])
            if not self.name in self.viewport.names:
                self.viewport.add_object(self.name)
        else:
            self.viewport = vc.ViewportCallback([self.name], *self.position)
            self.dependencies.extend([self.viewport])
        
        if not color is None:
            if type(colormap) != ColorMap:
                self.colormap = ColorMap(color, colormap, state = state, name=f"{self.name}_colormap", *args, **kwargs)
            else:
                self.colormap = colormap
            if "single" in self.colormap._color_type: # colors were not inferred
                self.color = np.arange(1) # color is just the index
            else:
                self.color = np.array(color).flatten()
        else:
            self.color = None
            self.colormap = colormap
    
    def _script_string(self):
        result = []
        used_position = [0,0,0]
        result.append(f"""cmd.pseudoatom("{self.name}", label="{self.label}", pos = {used_position}, state = {self.state})""")
        if not (self.color is None):
            result.append(f"""
cmd.set_color("{self.name}_color", {list(self.colormap.get_color(self.color)[0][:3])})
cmd.set("label_color", "{self.name}_color", "{self.name}")
""")
        result.append(f"""
cmd.set("label_size", {self.size}, "{self.name}")
""")
        return "\n".join(result)
    
    def load(self):
        from pymol import cmd
        used_position = [0,0,0]
        cmd.pseudoatom(self.name, label=self.label, pos = used_position, state = self.state)
        if not (self.color is None):
            print(self.colormap.get_color(self.color)[0][:3])
            cmd.set_color("%s_color"%self.name, list(self.colormap.get_color(self.color)[0][:3]))
            cmd.set("label_color", "%s_color"%self.name, self.name)
        cmd.set("label_size", self.size, self.name)
        self.viewport.load()

        
            
        
        