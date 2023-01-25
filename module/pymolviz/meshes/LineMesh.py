from __future__ import annotations

import numpy as np
from .Mesh import Mesh

class LineMesh(Mesh):
    """ Class to store all relevant information required to create a CGO Line object.
    
    
    Attributes:
        vertices (np.array): A 3xN array of vertices.
        color (np.array): A 3xN array of colors. May have a color for each line or each vertex.
        normals (np.array): A 3xN array of normals.
        faces (np.array): A 3xN array of faces.
        transformation (np.array): A 4x4 transformation matrix.
    """

    def __init__(self, lines : np.array, color : np.array = None, transformation : np.array = None, linewidth = 1, **kwargs) -> None:

        if not color is None:
            if len(color) == (len(lines.reshape(-1, 3)) / 2):
                color = np.repeat(color, 2, axis = 0)
            else:
                color = color

        super().__init__(lines.reshape(-1, 3), color, None, None, transformation, **kwargs)
        self.linewidth = linewidth

    def create_CGO(self) -> str:
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

    def from_o3d_lineset_mesh(o3d_mesh) -> LineMesh:
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

        return LineMesh(vertices, color)

    
