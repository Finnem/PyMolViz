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
        transparency (np.array): Optional. Defaults to 0. The transparency value of the object.
        colormap: Optional. Defaults to "RdYlBu_r". Name of a colormap or a matplotlib colormap or a pymolviz.ColorMap object. Used to map values to colors.
        linewidth (float): Optional. Defaults to 1. The width of the lines. Seems to be currently not supported by PyMol.
        head_length (float): The relative length of the arrow head. Does not influence the length of the arrow itself.
        head_width (float): The relative width of the arrow head.
        render_as (str): Optional. Defaults to "lines". If "cylinders", arrows will be drawn as 3D objects using cylinders and cones.
        starts (np.array): Optional. Defaults to None. The start of the lines.
        ends (np.array): Optional. Defaults to None. The end of the lines.
    """

    def __init__(self, lines = None, color = None, name = None, state = 1, transparency = 0, colormap = "RdYlBu_r", linewidth = 0.05, head_length = .25, head_width = 1.618, render_as="cylinders", starts = None, ends = None, arrow_mask = None, *args, **kwargs) -> None:
        self.original_color = color
        self.head_length = head_length
        self.head_width = head_width

        # make sure that either lines or start and end are given
        if lines is None:
            if starts is None or ends is None:
                raise ValueError("Either lines or start and end must be given.")
            lines = np.hstack([starts, ends])

        lines = np.array(lines)
        self.arrow_mask = arrow_mask if arrow_mask is not None else np.ones(int(len(lines)), dtype=bool)
        
        try:
            if (not np.issubdtype(type(color), np.str_)) and (not (color is None)):
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
        self.transparency = transparency
        try:
            self.transparency[0]
        except TypeError:
            self.transparency = np.full(int(original_lines.shape[0]), self.transparency)
        super().__init__(lines.reshape(-1, 3), color, name, state, self.transparency, colormap, linewidth, render_as, *args, **kwargs)
        self.linewidth = linewidth

    def from_start_end(starts, ends, color = None, name = None, state = 1, transparency = 0, colormap = "RdYlBu_r", linewidth = 0.05, head_length = .25, head_width = 1.618, render_as="cylinders", *args, **kwargs):
        """ Creates an Arrows object from start and end points.
        
        Args:
            starts (array-like): An array-like of start points.
            ends (array-like): An array-like of end points.
            color (array-like): Optional. Defaults to red. An array like of colors. Can be a single color, a list of colors, or a list of values to be mapped to a colormap. Lists of colors may either be for each vertex or for each line.
            name (str): Optional. Defaults to None. The name of the object.
            state (int): Optional. Defaults to 1. The state of the object.
            transparency (array-like): Optional. Defaults to 0. The transparency value of the object. If render_as = "cylinders", each cylinder can get its own transparency.
            colormap: Optional. Defaults to "RdYlBu_r". Name of a colormap or a matplotlib colormap or a pymolviz.ColorMap object. Used to map values to colors.
            linewidth (float): Optional. Defaults to 1. The width of the lines. Seems to be currently not supported by PyMol.
            head_length (float): Optional. Defaults to .25. The relative length of the arrow head. Does not influence the length of the arrow itself.
            head_width (float): Optional. Defaults to 1.618. The relative width of the arrow head.
            render_as (str): Optional. Defaults to "cylinders". If "cylinders", arrows will be drawn as 3D objects using cylinders and cones.
            *args: Additional arguments for the Lines class.
            **kwargs: Additional keyword arguments for the Lines class.
        
        Returns:
            Arrows: An Arrows object.
        """
        #ensure that start and end dims are correct
        starts = np.array(starts).reshape(-1, 3)
        ends = np.array(ends).reshape(-1, 3)
        return Arrows(np.hstack([starts, ends]), color, name, state, transparency, colormap, linewidth, head_length, head_width, render_as, *args, **kwargs)

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
            cylinder_ends[~self.arrow_mask] = ends[~self.arrow_mask]
            cgo_colors = self.colormap.get_color(self.color)[:,:3].reshape(-1, 3)
            start_colors = cgo_colors[::10]
            end_colors = cgo_colors[1::10]
            transparency = 1 - self.transparency

            cylinders = np.hstack([
                np.full(starts.shape[0], "ALPHA")[:,None], transparency[:,None], np.full(starts.shape[0], "CONE")[:,None], starts, cylinder_ends, np.full(starts.shape[0], self.linewidth)[:,None], np.full(starts.shape[0], self.linewidth)[:,None], start_colors, end_colors, np.full((starts.shape[0],2), (1.0, 0.0)) \
            ]).flatten()
            cgo_list.extend(cylinders)
            cones = np.hstack([
                    np.full(starts.shape[0], "ALPHA")[:,None], transparency[:,None], np.full(starts.shape[0], "CONE")[:,None], cylinder_ends, ends, np.full(starts.shape[0], self.linewidth * self.head_width)[:,None], np.full(starts.shape[0], 0.0)[:,None], end_colors, end_colors, np.full((starts.shape[0],2), 0.0)
            ])
            cones = cones[self.arrow_mask]
            cgo_list.extend(cones.flatten())
        return cgo_list

    