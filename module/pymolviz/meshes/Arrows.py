from __future__ import annotations

import numpy as np
from .Lines import Lines
from ..util.geometries import get_perp

class Arrows(Lines):
    """ Class to store all relevant information required to create a CGO Line object.
    
    
    Attributes:
        vertices (np.array): A 3xN array of vertices. Each line is defined by two vertices. Each beginning and end of a line should directly follow each other: [start1, end1, start2, end2, start3, end3] 
        color (np.array): A 3xN array of colors. May have a color for each line or each vertex.
        head_length (float): The relative length of the arrow head. Does not influence the length of the arrow itself.
        head_width (float): The relative width of the arrow head.
        transformation (np.array): A 4x4 transformation matrix.
    """

    def __init__(self, lines : np.array, color : np.array = None, head_length = .2, head_width = .2, transformation : np.array = None, linewidth = 1, **kwargs) -> None:

        if type(lines) == list:
            lines = np.array(lines)
        if (not color is None) and (not type(color) is str):
            if len(color) == (len(lines.reshape(-1, 3)) / 2):
                color = np.repeat(color, 10, axis = 0)
            elif len(color) == (len(lines.reshape(-1, 3))):
                color = np.repeat(color, 5, axis = 0)
            else:
                color = color

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
                

        super().__init__(lines.reshape(-1, 3), color, transformation, linewidth, **kwargs)
        self.linewidth = linewidth

    def _create_CGO(self) -> str:
        """ Creates a CGO list from the mesh information. A line mesh will be created.
            CGO constants are kept as strings to avoid importing the pymol module.
        
        Returns:
            None
        """

        vertices = self.vertices @ self.transformation[:3,:3].T + self.transformation[:3,3]

        cgo_list = []
        
        cgo_list.extend(["LINEWIDTH", self.linewidth])

        cgo_list.extend(["BEGIN", "LINES"])


        cgo_vertices = vertices
        cgo_colors = self.color

        #vertices
        triangles = np.hstack([
            np.full(cgo_colors.shape[0], "COLOR")[:,None], cgo_colors, \
            np.full(cgo_vertices.shape[0], "VERTEX")[:,None], cgo_vertices, \
            ]).flatten()
        cgo_list.extend(triangles)

        # ending
        cgo_list.append("END")

        return cgo_list

    def from_o3d_lineset_mesh(o3d_mesh) -> Lines:
        """ Creates a Mesh object from an Open3D triangle mesh.
        
        Args:
            o3d_mesh (o3d.geometry.LineSet): An Open3D lineset.
        
        Returns:
            Mesh: A Mesh object.
        """
        vertices = np.asarray(o3d_mesh.vertices)
        color = np.asarray(o3d_mesh.vertex_colors)
        lines = np.asarray(o3d_mesh.lines_normals)
        vertices = vertices[lines].reshape(-1, 3)

        return Lines(vertices, color)

