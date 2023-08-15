from __future__ import annotations

import numpy as np
from .Lines import Lines
from ..util.math import get_perp

class Arrows(Lines):
    """ Class to store all relevant information required to create a CGO Line object.

    Attributes:
        lines (np.array): A 3xN array of vertices. Each line is defined by two vertices. Each beginning and end of a line should directly follow each other: [start1, end1, start2, end2, start3, end3] 
        color (np.array): A 3xN array of colors. May have a color for each line or each vertex.
        name (str): Optional. Defaults to None. The name of the object.
        state (int): Optional. Defaults to 1. The state of the object.
        transparency (float): Optional. Defaults to 0. The transparency value of the object.
        colormap: Optional. Defaults to "RdYlBu_r". Name of a colormap or a matplotlib colormap or a pymolviz.ColorMap object. Used to map values to colors.
        linewidth (float): Optional. Defaults to 1. The width of the lines. Seems to be currently not supported by PyMol.
        head_length (float): The relative length of the arrow head. Does not influence the length of the arrow itself.
        head_width (float): The relative width of the arrow head.
        render_as (str): Optional. Defaults to "lines". If "cylinders", arrows will be drawn as 3D objects using cylinders and cones.
    """

    def __init__(self, lines, color = "red", name = None, state = 1, transparency = 0, colormap = "RdYlBu_r", linewidth = 1, head_length = .25, head_width = 1.618, render_as="lines", *args, **kwargs) -> None:
        self.original_color = color
        self.head_length = head_length
        self.head_width = head_width
        try:
            if (not np.issubdtype(type(color), np.str_)):
                if (len(color) == (len(lines.reshape(-1, 3)) / 2)):
                    self.original_color = np.repeat(color, 2, axis = 0)
                    color = np.repeat(color, 10, axis = 0)
                elif (len(color) == len(lines.reshape(-1, 3))):
                    color = np.hstack([color[::2, None], np.repeat(color[1::2], 9, axis = 0).reshape(-1, 9)]).flatten()
        except TypeError:
            pass
        original_lines = lines.reshape(-1, 6)
        new_lines = np.zeros((original_lines.shape[0] * 4, 6))
        for i, line in enumerate(original_lines):
            start = line[:3]
            end = line[3:]
            vector = end - start
            head_start = (vector) * (1 - head_length) + start
            perp = get_perp(vector)
            x1 = head_start + perp * head_width
            x2 = head_start - perp * head_width
            ortho = np.cross(vector, perp); ortho /= np.linalg.norm(ortho)
            y1 = head_start + ortho * head_width
            y2 = head_start - ortho * head_width
            new_lines[i * 4] = np.hstack([end, x1])
            new_lines[i * 4 + 1] = np.hstack([end, x2])
            new_lines[i * 4 + 2] = np.hstack([end, y1])
            new_lines[i * 4 + 3] = np.hstack([end, y2])
        lines = np.hstack([original_lines, new_lines.reshape(-1, 24)])
                

        super().__init__(lines.reshape(-1, 3), color, name, state, transparency, colormap, linewidth, render_as, *args, **kwargs)
        self.linewidth = linewidth

    def _create_CGO_list(self) -> str:
        """ Creates a CGO list from the mesh information. A line mesh will be created.
            CGO constants are kept as strings to avoid importing the pymol module.
        
        Returns:
            None
        """
        if self.render_as == "lines":
            return super()._create_CGO_list()
           
        elif self.render_as == "cylinders":
            cgo_list = []
            starts = self.vertices[::10]
            ends = self.vertices[1::10]
            cylinder_ends = starts + ((ends - starts) * (1-self.head_length))
            cgo_colors = self.colormap.get_color(self.color)[:,:3].reshape(-1, 3)
            start_colors = cgo_colors[::10]
            end_colors = cgo_colors[1::10]
            print(starts.shape, cylinder_ends.shape, start_colors.shape)
            cylinders = np.hstack([
                np.full(starts.shape[0], "CYLINDER")[:,None], starts, cylinder_ends, np.full(starts.shape[0], self.linewidth)[:,None], start_colors, end_colors \
            ]).flatten()
            cgo_list.extend(cylinders)
            cones = np.hstack([
                    np.full(starts.shape[0], "CONE")[:,None], cylinder_ends, ends, np.full(starts.shape[0], self.linewidth * self.head_width)[:,None], np.full(starts.shape[0], 0.0)[:,None], end_colors, end_colors, np.full((starts.shape[0],2), 1.0)
            ]).flatten()
            cgo_list.extend(cones)
        return cgo_list