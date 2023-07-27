from __future__ import annotations

import logging
import numpy as np

from ..ColorMap import ColorMap
from ..Displayable import Displayable
from ..PyMOLobjects.PseudoAtoms import PseudoAtoms

class Points(Displayable):
    """ Class to store points and associated colors which can be displayed as a point cloud.
    
    
    Attributes:
        vertices (array-like): An array-like of vertex positions.
        color (array-like): Optional. Defaults to red. An array like of colors. Can be a single color, a list of colors, or a list of values to be mapped to a colormap.
        name (str): Optional. Defaults to None. The name of the object.
        state (int): Optional. Defaults to 1. The state of the object.
        transparency (float): Optional. Defaults to 0. The transparency value of the object.
        colormap: Optional. Defaults to "RdYlBu_r". Name of a colormap or a matplotlib colormap or a pymolviz.ColorMap object. Used to map values to colors.
        render_as (str): Optional. Defaults to "Spheres". How to display the points. Can be "Spheres" or "Dots" or None. If None, the points are not displayed.
        radius (float): Optional. Defaults to .3. Only relevant if render_as is "Spheres". The radius of the spheres.
    """

    def __init__(self, vertices, color = "red", name = None, state = 1, transparency = 0, colormap = "RdYlBu_r", render_as = "Spheres", radius = .3, *args, **kwargs) -> None:
        super().__init__(name)
        self.vertices = np.array(vertices, dtype=float).reshape(-1, 3)
        if type(colormap) != ColorMap:
            self.colormap = ColorMap(color, colormap, state = state, name=f"{self.name}_colormap", *args, **kwargs)
        else:
            self.colormap = colormap
        if "single" in self.colormap._color_type: # colors were not inferred
            self.color = np.arange(self.vertices.shape[0]) # color is just the index
        else:
            self.color = np.array(color).flatten()

        self.render_as = render_as
        self.radius = radius
        self.state = state
        self.transparency = transparency


    def as_pseudoatoms(self) -> PseudoAtoms:
        return PseudoAtoms(self.vertices, self.color, name = self.name, state = self.state, colormap = self.colormap)


    def _create_CGO_list(self) -> list:
        """ Creates a CGO list from the mesh information. Points can be displayed as spheres or as points.

        Returns:
            List of str: The CGO list.
        """

        cgo_points = self.vertices
        cgo_colors = self.colormap.get_color(self.color)

        cgo_list = []
        
        
        if self.render_as == "Spheres":
            #vertices
            point_meshes = np.hstack([
                np.full(cgo_points.shape[0], "COLOR")[:,None], cgo_colors, \
                np.full(cgo_points.shape[0], "SPHERE")[:,None], cgo_points, \
                np.full(cgo_points.shape[0], self.radius)[:,None], \
                ]).flatten()
            cgo_list.extend(point_meshes)

        elif self.render_as == "Dots":
            cgo_list.extend(["BEGIN", "POINTS"])
            #vertices
            points = np.hstack([
                np.full(cgo_points.shape[0], "COLOR")[:,None], cgo_colors, \
                np.full(cgo_points.shape[0], "VERTEX")[:,None], cgo_points, \
                ]).flatten()
            cgo_list.extend(points)
            # ending
            cgo_list.append("END")

        return cgo_list

    def _script_string(self):
        cgo_string_builder = []
        state = "" if self.state is None else f", state={self.state}"
        cgo_name = self.name.replace(" ", "_")
        cgo_string_builder.append(f"""
{cgo_name} = [
        """)
        content = ",".join([str(e) for e in self._create_CGO_list()])
        cgo_string_builder.append(content)

        # ending
        cgo_string_builder.append(f"""
            ]
cmd.load_cgo({cgo_name}, "{cgo_name}"{state})
cmd.set("cgo_transparency", {self.transparency}, "{cgo_name}")
        """)
        return "\n".join(cgo_string_builder)
