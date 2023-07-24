from __future__ import annotations
from .Points import Points

import logging
import numpy as np


class Mesh(Points):
    """ Class to store all relevant information required to create a CGO object.
    
    
    Attributes:
        vertices (array-like): An array-like of vertex positions.
        color (array-like): Optional. Defaults to red. An array like of colors. Can be a single color, a list of colors, or a list of values to be mapped to a colormap.
        normals (np.array): A Nx3 array of normals.
        faces (np.array): A Nx3 array of faces.
        name (str): Optional. Defaults to None. The name of the object.
        state (int): Optional. Defaults to 1. The state of the object.
        transparency (float): Optional. Defaults to 0. The transparency value of the object.
        colormap: Optional. Defaults to "RdYlBu_r". Name of a colormap or a matplotlib colormap or a pymolviz.ColorMap object. Used to map values to colors.
    """

    def __init__(self, vertices, color = "red", normals : np.array = None, faces : np.array = None, name = None, state = 1, transparency = 0, colormap = "RdYlBu_r", *args, **kwargs) -> None:
        self.normals = np.array(normals, dtype=float).reshape(-1, 3) if normals is not None else np.zeros_like(vertices)
        self.faces = np.array(faces, dtype=int).reshape(-1, 3) if faces is not None else np.arange(vertices.shape[0]).reshape(-1, 3)
        super().__init__(vertices.reshape(-1, 3), color, name, state, transparency, colormap, *args, **kwargs)
        
    def to_wireframe(self, *args, **kwargs):
        """ Converts the mesh to a wireframe.
        
        Returns:
            Mesh: A wireframe mesh.
        """
        from .Lines import Lines
        vertex_indices = []
        if not "state" in kwargs:
            kwargs["state"] = self.state
        if not "colormap" in kwargs:
            kwargs["colormap"] = self.colormap

        for face in self.faces:
            vertex_indices.extend([face[0], face[1], face[1], face[2], face[2], face[0]])
        
        return Lines(self.vertices[vertex_indices], self.color[vertex_indices], *args, **kwargs)


    def _create_CGO_list(self) -> str:
        """ Creates a CGO list from the mesh information. The base class assumes a triangle mesh.
            CGO constants are kept as strings to avoid importing the pymol module.
        
        Returns:
            None
        """


        cgo_colors = self.colormap.get_color(self.color)[:,:3]
        cgo_triangles = self.vertices[self.faces].reshape(-1, 3)
        cgo_colors = cgo_colors[self.faces].reshape(-1, 3)
        cgo_normals = self.normals[self.faces].reshape(-1, 3)

        cgo_list = []
        
        cgo_list.extend(["BEGIN", "TRIANGLES"]),

        #vertices
        triangles = np.hstack([
            np.full(cgo_triangles.shape[0], "COLOR")[:,None], cgo_colors, \
            np.full(cgo_triangles.shape[0], "VERTEX")[:,None], cgo_triangles, \
            np.full(cgo_triangles.shape[0], "NORMAL")[:,None], cgo_normals, \
            ]).flatten()
        cgo_list.extend(triangles)

        # ending
        cgo_list.append("END")

        return cgo_list

         