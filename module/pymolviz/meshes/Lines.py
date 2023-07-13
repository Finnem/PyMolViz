from __future__ import annotations

import logging
import numpy as np
from .Points import Points

class Lines(Points):
    """ Class to store all relevant information required to create a CGO Line object.

    Attributes:
        lines (array-like): An array-like of vertex positions.
        color (array-like): Optional. Defaults to red. An array like of colors. Can be a single color, a list of colors, or a list of values to be mapped to a colormap. Lists of colors may either be for each vertex or for each line.
        name (str): Optional. Defaults to None. The name of the object.
        state (int): Optional. Defaults to 1. The state of the object.
        transparency (float): Optional. Defaults to 0. The transparency value of the object.
        colormap: Optional. Defaults to "RdYlBu_r". Name of a colormap or a matplotlib colormap or a pymolviz.ColorMap object. Used to map values to colors.
        linewidth (float): Optional. Defaults to 1. The width of the lines. Seems to be currently not supported by PyMol.

    """

    def __init__(self, lines : np.array, color = "red", name = None, state = 1, transparency = 0, colormap = "RdYlBu_r", linewidth = 1, **kwargs) -> None:
        try:
            if (not np.issubdtype(type(color), np.str_)):
                if (len(color) == (len(lines.reshape(-1, 3)) / 2)):
                    color = np.repeat(color, 2, axis = 0)
        except TypeError:
            pass
        super().__init__(lines.reshape(-1, 3), color, name, state, transparency, colormap, **kwargs)
        self.linewidth = linewidth


    def _create_CGO_list(self) -> str:
        """ Creates a CGO list from the mesh information. A line mesh will be created.
            CGO constants are kept as strings to avoid importing the pymol module.
        
        Returns:
            None
        """

        cgo_list = []
        
        cgo_list.extend(["LINEWIDTH", self.linewidth])

        cgo_list.extend(["BEGIN", "LINES"])


        cgo_vertices = self.vertices
        cgo_colors = self.colormap.get_color(self.color)

        #vertices
        triangles = np.hstack([
            np.full(cgo_colors.shape[0], "COLOR")[:,None], cgo_colors, \
            np.full(cgo_vertices.shape[0], "VERTEX")[:,None], cgo_vertices, \
            ]).flatten()
        cgo_list.extend(triangles)

        # ending
        cgo_list.append("END")

        return cgo_list

    def combine(lines):
        """ Combines multiple lines into one.
        
        Args:
            lines (list): A list of lines.
        
        Returns:
            Lines: A Lines object.
        """
        vertices = np.vstack([line.vertices for line in lines])
        colors = np.vstack([line.color for line in lines])
        return Lines(vertices, colors)

