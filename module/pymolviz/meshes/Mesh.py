from __future__ import annotations
from .Points import Points

import logging
import numpy as np


def _unique_undirected_edges(faces) -> np.ndarray:
    """Sorted unique ``(i, j)`` edges from a triangle mesh, ``i < j``."""
    faces = np.asarray(faces, dtype=int).reshape(-1, 3)
    if faces.size == 0:
        return np.zeros((0, 2), dtype=int)
    edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]),
        axis=0,
    )
    edges.sort(axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if edges.size == 0:
        return np.zeros((0, 2), dtype=int)
    return np.unique(edges, axis=0)


def _colors_are_uniform(colors, atol=1e-5) -> bool:
    colors = np.asarray(colors, dtype=float).reshape(-1, 3)
    if colors.shape[0] <= 1:
        return True
    return bool(np.allclose(colors, colors[0], atol=atol, rtol=0.0))


def _lit_triangle_cgo(vertices, colors, normals):
    """Triangle soup with lighting state and a NORMAL on every corner.

    PyMOL's default shader interpolates vertex normals in the fragment stage.
    If any triangle vertex is missing a NORMAL, ObjectCGO runs
    ``CGOGenerateNormalsForTriangles``, which *discards* supplied normals and
    writes one face normal per triangle — faceted speculars. COLOR once when
    the mesh is uniform, then NORMAL immediately before VERTEX (same order as
    ``pymol.cgo.torus``).
    """
    cgo_list = ["ENABLE", "LIGHTING", "BEGIN", "TRIANGLES"]
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    n = int(vertices.shape[0])
    if n == 0:
        cgo_list.append("END")
        return cgo_list
    colors = np.asarray(colors, dtype=float).reshape(-1, 3)
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    ln = np.linalg.norm(normals, axis=1, keepdims=True)
    ln = np.maximum(ln, 1e-18)
    normals = normals / ln
    if _colors_are_uniform(colors):
        r, g, b = (float(colors[0, 0]), float(colors[0, 1]), float(colors[0, 2]))
        cgo_list.extend(["COLOR", r, g, b])
        block = np.empty((n, 8), dtype=object)
        block[:, 0] = "NORMAL"
        block[:, 1:4] = normals
        block[:, 4] = "VERTEX"
        block[:, 5:8] = vertices
    else:
        block = np.empty((n, 12), dtype=object)
        block[:, 0] = "COLOR"
        block[:, 1:4] = colors
        block[:, 4] = "NORMAL"
        block[:, 5:8] = normals
        block[:, 8] = "VERTEX"
        block[:, 9:12] = vertices
    cgo_list.extend(block.ravel().tolist())
    cgo_list.append("END")
    return cgo_list


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

    def __init__(self, vertices, color = None, normals : np.array = None, faces : np.array = None, name = None, state = 1, transparency = 0, colormap = "RdYlBu_r", *args, **kwargs) -> None:
        self.normals = np.array(normals, dtype=float).reshape(-1, 3) if normals is not None else np.zeros_like(vertices)
        self.faces = np.array(faces, dtype=int).reshape(-1, 3) if faces is not None else np.arange(vertices.shape[0]).reshape(-1, 3)
        super().__init__(vertices.reshape(-1, 3), color, name, state, transparency, colormap, *args, **kwargs)
        
    def _colors_for_indices(self, indices):
        if getattr(self, "bypass_colormap", False):
            colors = np.asarray(self.color, dtype=float).reshape(-1, 3)
            if colors.shape[0] == 1:
                return np.repeat(colors, len(indices), axis=0)
            if colors.shape[0] == self.vertices.shape[0]:
                return colors[indices]
            return np.repeat(colors[:1], len(indices), axis=0)
        color = np.asarray(self.color)
        if color.ndim == 0 or color.shape[0] != self.vertices.shape[0]:
            return self.color
        return self.color[indices]

    def to_wireframe(self, *args, **kwargs):
        """ Converts the mesh to a wireframe.
        
        Returns:
            Mesh: A wireframe mesh.
        """
        from .Lines import Lines
        if not "state" in kwargs:
            kwargs["state"] = self.state
        if not "colormap" in kwargs and not getattr(self, "bypass_colormap", False):
            kwargs["colormap"] = self.colormap
        kwargs.setdefault("bypass_colormap", getattr(self, "bypass_colormap", False))
        kwargs.setdefault("transparency", getattr(self, "transparency", 0))
        if not "render_as" in kwargs:
            kwargs["render_as"] = "lines"

        edges = _unique_undirected_edges(self.faces)
        if edges.shape[0] == 0:
            empty = np.zeros((0, 3), dtype=float)
            return Lines(empty, empty, *args, **kwargs)
        vertex_indices = edges.reshape(-1)
        return Lines(
            self.vertices[vertex_indices],
            self._colors_for_indices(vertex_indices),
            *args,
            **kwargs,
        )


    def _create_CGO_list(self) -> str:
        """ Creates a CGO list from the mesh information. The base class assumes a triangle mesh.
            CGO constants are kept as strings to avoid importing the pymol module.
        
        Returns:
            None
        """


        cached = getattr(self, "_cached_cgo", None)
        if cached is not None:
            return cached

        if getattr(self, "wireframe", False):
            wire = self.to_wireframe(render_as="cylinders", linewidth=0.012)
            cgo_list = list(wire._create_CGO_list())
            self._cached_cgo = cgo_list
            return cgo_list

        if getattr(self, "bypass_colormap", False):
            cgo_colors = np.asarray(self.color, dtype=float).reshape(-1, 3)
            if cgo_colors.shape[0] != self.vertices.shape[0]:
                cgo_colors = np.repeat(cgo_colors[:1], self.vertices.shape[0], axis=0)
        else:
            cgo_colors = self.colormap.get_color(self.color)[:, :3]
        cgo_triangles = self.vertices[self.faces].reshape(-1, 3)
        cgo_colors = cgo_colors[self.faces].reshape(-1, 3)
        if getattr(self, "flat_shading", False) and self.faces.shape[0] > 0:
            p0 = self.vertices[self.faces[:, 0]]
            p1 = self.vertices[self.faces[:, 1]]
            p2 = self.vertices[self.faces[:, 2]]
            fn = np.cross(p1 - p0, p2 - p0)
            ln = np.linalg.norm(fn, axis=1, keepdims=True)
            ln = np.maximum(ln, 1e-18)
            cgo_normals = np.repeat(fn / ln, 3, axis=0)
        else:
            cgo_normals = self.normals[self.faces].reshape(-1, 3)

        cgo_list = _lit_triangle_cgo(cgo_triangles, cgo_colors, cgo_normals)
        self._cached_cgo = cgo_list
        return cgo_list

         