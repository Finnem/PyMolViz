from __future__ import annotations
from collections import defaultdict

import logging
import numpy as np

_pmv_mesh_counter = 0

class Mesh():
    """ Class to store all relevant information required to create a CGO object.
    
    
    Attributes:
        vertices (np.array): A Nx3 array of vertices.
        color (np.array): A Nx3 array of colors.
        normals (np.array): A Nx3 array of normals.
        faces (np.array): A Nx3 array of faces.
        transformation (np.array): A 4x4 transformation matrix.
    """

    def __init__(self, vertices : np.array, color : np.array = None, normals : np.array = None, faces : np.array = None, transformation : np.array = None, name = None, colormap = None, clims = None, **kwargs) -> None:
        self.vertices = vertices
        self.color = self._convert_color(color, colormap, clims, **kwargs)
        self.normals = normals
        self.faces = faces
        self.name = name
        self.transformation = np.eye(4) if transformation is None else transformation
        

    def to_wireframe(self):
        """ Converts the mesh to a wireframe.
        
        Returns:
            Mesh: A wireframe mesh.
        """
        from .Lines import Lines
        vertex_indices = []
        for face in self.faces:
            vertex_indices.extend([face[0], face[1], face[1], face[2], face[2], face[0]])
        
        return Lines(self.vertices[vertex_indices], self.color[vertex_indices], transformation = self.transformation, name = self.name)

    def combine(meshes, **kwargs):
        """ Combines multiple meshes into one.
        
        Args:
            meshes (list): A list of meshes.
        
        Returns:
            Mesh: A combined mesh.
        """
        if type(meshes) == Mesh:
            return meshes
        else:
            from .Lines import Lines
            if all([isinstance(mesh, Lines) for mesh in meshes]):
                result = Lines.combine(meshes)
                norm_meshes = [mesh for mesh in meshes if hasattr(mesh, "_norm")]
                if len(norm_meshes) > 0:
                    result._norm = norm_meshes[0]._norm
                colormap_meshes = [mesh for mesh in meshes if hasattr(mesh, "_colormap")]
                if len(colormap_meshes) > 0:
                    result._colormap = norm_meshes[0]._colormap
                return result
            elif any([isinstance(mesh, Lines) for mesh in meshes]):
                raise ValueError("Cannot combine Lines with other mesh types.")
            else:
                vertices = np.vstack([mesh.vertices for mesh in meshes])
                face_offsets = np.cumsum([0] + [len(mesh.vertices) for mesh in meshes])
                faces = np.vstack([mesh.faces + face_offset for mesh, face_offset in zip(meshes, face_offsets)])
                colors = np.vstack([mesh.color for mesh in meshes])
                normals = np.vstack([mesh.normals for mesh in meshes])
                result = Mesh(vertices, colors, normals, faces, **kwargs)
                norm_meshes = [mesh for mesh in meshes if hasattr(mesh, "_norm")]
                if len(norm_meshes) > 0:
                    result._norm = norm_meshes[0]._norm
                colormap_meshes = [mesh for mesh in meshes if hasattr(mesh, "_colormap")]
                if len(colormap_meshes) > 0:
                    result._colormap = norm_meshes[0]._colormap
                return result
    
    def union(self, mesh):
        from ..util.bsp import BSP_Node
        my_node = BSP_Node.from_mesh(self)
        other_node = BSP_Node.from_mesh(mesh)
        new_node = my_node.union(other_node)
        return new_node.to_mesh()

    def difference(self, mesh):
        from ..util.bsp import BSP_Node
        my_node = BSP_Node.from_mesh(self)
        other_node = BSP_Node.from_mesh(mesh)
        new_node = my_node.subtract(other_node)
        return new_node.to_mesh()

    def intersect(self, mesh):
        from ..util.bsp import BSP_Node
        my_node = BSP_Node.from_mesh(self)
        other_node = BSP_Node.from_mesh(mesh)
        new_node = my_node.intersect(other_node)
        return new_node.to_mesh()



    def load(self, name = None):
        """ Loads the mesh into PyMOL. """
        from pymol import cmd
        if name: self.name = name
        if not self.name:
            global _pmv_mesh_counter
            logging.warning("No name provided for Mesh. Using default name: Mesh_{}. It is highly recommended to provide meaningful names.".format(_pmv_mesh_counter))
            self.name = "Mesh_{}".format(_pmv_mesh_counter)
            _pmv_mesh_counter += 1
        cmd.load_cgo(self._create_CGO(), self.name)

    def to_script(self, name = None):
        """ Creates a script from the mesh.
        
        Returns:
            Script: A script object.
        """
        if name: self.name = name
        from ..Script import Script
        return Script([self], name = self.name)

    def _create_CGO(self) -> str:
        """ Creates a CGO list from the mesh information. The base class assumes a triangle mesh.
            CGO constants are kept as strings to avoid importing the pymol module.
        
        Returns:
            None
        """

        vertices = self.vertices @ self.transformation[:3,:3].T + self.transformation[:3,3]
        if self.color.shape[1] == 4:
            cgo_colors = self.color[:,:3]
        else:
            cgo_colors = self.color
        cgo_triangles = vertices[self.faces].reshape(-1, 3)
        cgo_colors = cgo_colors[self.faces].reshape(-1, 3)
        cgo_normals = self.normals[self.faces].reshape(-1, 3)

        print(cgo_colors.shape, cgo_triangles.shape, cgo_normals.shape)
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
        from matplotlib import cm, colors

        target_length = len(self.vertices)
        color_array = np.ones((target_length, 3))

        if (color is None):
            return color_array

        if isinstance(color, (list, tuple)):
            if len(color) == 0:
                return color_array
            color = np.array(color)

        if type(color) is np.ndarray:
            color = color.squeeze()

        if isinstance(color, (str, np.str_)):
            from ..util.colors import _convert_string_color
            color = _convert_string_color(color)
            color_array = np.full((target_length, 3), color[:3])
        elif np.isscalar(color) and (clims is not None):
            self._norm = colors.Normalize(vmin = clims[0], vmax = clims[1])
            if (colormap is None):
                    colormap = "coolwarm"
            self._colormap = cm.get_cmap(colormap)
            color = self._colormap(self._norm(color))
            color_array = np.full((target_length, 3), color[:3])
            return color_array
        elif isinstance(color, np.ndarray):
            if len(color) == 0:
                return color_array
            elif color.shape == (3,) and not (color.dtype.kind in ["U", "S"]):
                print(color.dtype)
                color_array = np.full((target_length, 3), color)
            elif color.shape == (target_length, 4):
                color_array = color[:,:3]
                logging.warning("Color was passed with an alpha value. Alpha can only be set explicitly and only for a whole CGO and is ignored here.")
            elif color.shape == (target_length,):
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
                color_numbers = np.array([c for c in color if not c.dtype.kind in ["S", "U"]])
                from ..util.colors import _convert_string_color
                if len(color_numbers) > 0:
                    if clims is not None:
                        self._norm = colors.Normalize(vmin = clims[0], vmax = clims[1]) 
                    else:
                        self._norm = colors.Normalize(vmin=color_numbers.min(), vmax=color_numbers.max())
                    color_array = np.array([_convert_string_color(c) if c.dtype.kind in ["S", "U"] else self._colormap(self._norm(c)) for c in color])
                else:
                    color_array = np.array([_convert_string_color(c) for c in color])

            elif color.shape == (target_length, 3) and not color.dtype.kind in ["S", "U"]:
                color_array = color
            elif color.shape == (4,) and not color.dtype.kind in ["S", "U"]:
                color_array = np.full((target_length, 3), color[:3])
                logging.warning("Color was passed with an alpha value. Alpha can only be set explicitly and only for a whole CGO and is ignored here.")
            
            else:
                raise ValueError(f"Color array has shape {color.shape} but should be (3,), ({target_length}) or ({target_length}, 3)")
        else:
            raise ValueError(f"Color has type {type(color)} but should be str or np.array")
        return color_array

        

         