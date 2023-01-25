from __future__ import annotations

from ..util.element_colors import get_AA_color, get_element_color
import logging
import numpy as np

class Mesh():
    """ Class to store all relevant information required to create a CGO object.
    
    
    Attributes:
        vertices (np.array): A 3xN array of vertices.
        color (np.array): A 3xN array of colors.
        normals (np.array): A 3xN array of normals.
        faces (np.array): A 3xN array of faces.
        transformation (np.array): A 4x4 transformation matrix.
    """

    def __init__(self, vertices : np.array, color : np.array = None, normals : np.array = None, faces : np.array = None, transformation : np.array = None, colormap = None, clims = None) -> None:
        self.vertices = vertices
        self.color = self._convert_color(color, colormap, clims)
        self.normals = normals
        self.faces = faces
        self.transformation = transformation or np.eye(4)
        
    def _create_CGO(self) -> str:
        """ Creates a CGO list from the mesh information. The base class assumes a triangle mesh.
            CGO constants are kept as strings to avoid importing the pymol module.
        
        Returns:
            None
        """

        vertices = self.vertices @ self.transformation[:3,:3].T + self.transformation[:3,3]

        cgo_triangles = vertices[self.faces].reshape(-1, 3)
        cgo_colors = self.color[self.faces].reshape(-1, 3)
        cgo_normals = self.color[self.faces].reshape(-1, 3)

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

    def from_o3d_triangle_mesh(o3d_mesh) -> Mesh:
        """ Creates a Mesh object from an Open3D triangle mesh.
        
        Args:
            o3d_mesh (o3d.geometry.TriangleMesh): An Open3D triangle mesh.
        
        Returns:
            Mesh: A Mesh object.
        """
        vertices = np.asarray(o3d_mesh.vertices)
        color = np.asarray(o3d_mesh.vertex_colors)
        normals = np.asarray(o3d_mesh.vertex_normals)
        faces = np.asarray(o3d_mesh.triangles)

        return Mesh(vertices, color, normals, faces)


    def get_color_map(self, figsize : tuple = (1, 6), **kwargs):
        """ Creates a color map and saves it to a file.

        Args:
            out (str): Path to the output file.
            figsize (tuple, optional): Figure size. Defaults to (1, 6).
            dpi (int, optional): DPI of the figure. Defaults to 300.
            **kwargs: Additional arguments passed to fig.colorbar (see https://matplotlib.org/stable/tutorials/colors/colorbar_only.html).
        
        Returns:
            None
        """
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        fig, ax = plt.subplots(figsize = figsize)
        if self._norm is None:
            logging.warn("Could not find self._norm. Seems that no color was set.")
        fig.colorbar(mpl.cm.ScalarMappable(norm = self._norm, cmap = self._colormap), cax = ax, **kwargs)
        return fig

    def save_color_map(self, out : str, figsize : tuple (1,6), dpi : int = 300, **kwargs):
        fig = self.get_color_map(figsize, **kwargs)
        fig.savefig(out, dpi = kwargs.get("dpi", dpi), bbox_inches = "tight")

    
    def _convert_color(self, color, colormap, clims):
        """ Converts single color as string or numpy array to a 3xN array of colors.
            MUST BE CALLED AFTER VERTICES ARE SET.
        
        Returns:
            None
        """

        target_length = len(self.vertices)
        color_array = np.ones((target_length, 3))

        if (color is None) or (len(color) == 0):
            return color_array

        if isinstance(color, str):
            from matplotlib import colors
            if not (c := get_AA_color(color)) is None:
                color = c
            elif not (c := get_element_color(color)) is None:
                color = c
            else:
                color = colors.to_rgb(color)
            color_array = np.full((target_length, 3), color)
        elif isinstance(color, np.ndarray):
            if color.shape == (4,):
                color_array = np.full((target_length, 3), color[:3])
                logging.warning("Color was passed with an alpha value. Alpha can only be set explicitly and only for a whole CGO and is ignored here.")
            elif color.shape == (3,):
                color_array = np.full((target_length, 3), color)
            elif color.shape == (target_length, 4):
                color_array = color[:,:3]
                logging.warning("Color was passed with an alpha value. Alpha can only be set explicitly and only for a whole CGO and is ignored here.")
            elif color.shape == (target_length,):
                from matplotlib import cm, colors
                factor = 1.0
                if (colormap is None):
                    colormap = "coolwarm"
                if isinstance(colormap, str):
                    try:
                        colormap = cm.get_cmap(colormap)
                    except ValueError:
                        seperated_colormap = colormap.split("_")
                        factor = float(seperated_colormap[-1])
                        if factor > 1:
                            factor = 1.0
                            logging.warn("Shrinkage factor for colormap is larger than 1.0. Setting factor to 1.0.")
                        colormap = cm.get_cmap("_".join(seperated_colormap[:-1]))
                        factor_offset = (1 - factor) / 2
                        color_segments = colormap(np.linspace(factor_offset, 1 - factor_offset, 256))
                        colormap = colors.LinearSegmentedColormap.from_list(colormap.name + "_shrunk", color_segments)
                self._colormap = colormap
                if clims is not None:
                    self._norm = colors.Normalize(vmin = clims[0], vmax = clims[1]) 
                else:
                    self._norm = colors.Normalize(vmin=color.min(), vmax=color.max())

                color_array = self._colormap(self._norm(color))

            elif color.shape == (target_length, 3):
                pass
            else:
                raise ValueError(f"Color array has shape {color.shape} but should be (3,), ({target_length}) or ({target_length}, 3)")
        else:
            raise ValueError(f"Color has type {type(color)} but should be str or np.array")
        return color_array