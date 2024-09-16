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
        transparency (array-like): Optional. Defaults to 0. The transparency value of the object. If render_as = "cylinders", each cylinder can get its own transparency.
        colormap: Optional. Defaults to "RdYlBu_r". Name of a colormap or a matplotlib colormap or a pymolviz.ColorMap object. Used to map values to colors.
        linewidth (float): Optional. Defaults to 1. The width of the lines. Seems to be currently not supported by PyMol.
        render_as (str): Optional. Defaults to "lines". If "cylinders", lines will be drawn as 3D objects using cylinders.

    """

    def __init__(self, lines : np.array, color = "red", name = None, state = 1, transparency = 0, colormap = "RdYlBu_r", linewidth = 1, render_as = "cylinders", *args, **kwargs) -> None:
        try:
            if (not np.issubdtype(type(color), np.str_)):
                if (len(color) == (len(lines.reshape(-1, 3)) / 2)):
                    color = np.repeat(color, 2, axis = 0)
        except TypeError:
            pass
        super().__init__(lines.reshape(-1, 3), color, name, state, transparency, colormap, *args, **kwargs)
        self.linewidth = linewidth
        self.render_as = render_as
        self.transparency = transparency
        try:
            self.transparency[0]
        except TypeError:
            self.transparency = np.full(int(lines.reshape(-1,6).shape[0]), self.transparency)
        #catching common errors:
        if self.render_as == "line":
            self.render_as = "lines"
        if self.render_as == "cylinder":
            self.render_as = "cylinders"


    def _create_CGO_list(self) -> str:
        """ Creates a CGO list from the mesh information. A line mesh will be created.
            CGO constants are kept as strings to avoid importing the pymol module.
        
        Returns:
            None
        """

        cgo_list = []
        if self.render_as == "lines":
            
            cgo_list.extend(["LINEWIDTH", self.linewidth])

            cgo_list.extend(["BEGIN", "LINES"])

            cgo_vertices = self.vertices
            cgo_colors = self.colormap.get_color(self.color)[:,:3]

            #vertices
            triangles = np.hstack([
                np.full(cgo_colors.shape[0], "COLOR")[:,None], cgo_colors, \
                np.full(cgo_vertices.shape[0], "VERTEX")[:,None], cgo_vertices, \
                ]).flatten()
            cgo_list.extend(triangles)

            # ending
            cgo_list.append("END")

        elif self.render_as == "cylinders":
            cgo_vertices = self.vertices.reshape(-1, 6)
            cgo_colors = self.colormap.get_color(self.color)[:,:3].reshape(-1, 6)
            transparency = 1 - self.transparency
            

            triangles = np.hstack([
                np.full(cgo_vertices.shape[0], "ALPHA")[:,None], transparency[:,None], np.full(cgo_vertices.shape[0], "CYLINDER")[:,None], cgo_vertices, np.full(cgo_vertices.shape[0], self.linewidth)[:,None], cgo_colors \
            ]).flatten()
            cgo_list.extend(triangles)
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


    def as_dotted(self, max_length = 0.1):
        """
        Converts the lines to dotted lines.

        Args:
            fraction (float): The fraction of the line that should be visible.

        Returns:
            Lines: A Lines object.
        """
        vertices = self.vertices
        vertices = vertices.reshape(-1, 2, 3)
        # split up each pair of vertices into multiple pairs of vertices
        lines = vertices[:, 1] - vertices[:, 0]
        lengths = np.linalg.norm(lines, axis = 1)
        directions = lines / lengths[:, None]
        segments = np.ceil(lengths / max_length).astype(int)
        new_vertices = []
        new_colors = []
        for i, (segment, direction) in enumerate(zip(segments, directions)):
            cur_max_length = np.min([max_length, lengths[i]])
            for j in range(segment):
                if j % 2 == 1:
                    continue
                new_vertices.append(vertices[i, 0] + direction * j * max_length)
                new_vertices.append(vertices[i, 0] + direction * (j + 1) * max_length)
                new_colors.append(self.color[i])
                new_colors.append(self.color[i])
        vertices = np.array(new_vertices).reshape(-1, 3)
    
        return Lines(vertices, self.color, self.name, self.state, self.transparency, self.colormap, self.linewidth, self.render_as)