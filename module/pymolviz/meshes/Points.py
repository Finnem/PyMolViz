from __future__ import annotations

import logging
import numpy as np
import seaborn as sns

from ..ColorMap import ColorMap
from ..Displayable import Displayable
from ..PyMOLobjects.PseudoAtoms import PseudoAtoms
from ..util.colors import get_distinct_colors
from ..util.sanitize import sanitize_pymol_string

pmv_default_color_palette = get_distinct_colors(20)
pmv_default_color_counter = 0

class Points(Displayable):
    """ Class to store points and associated colors which can be displayed as a point cloud.
    
    
    Attributes:
        vertices (array-like): An array-like of vertex positions.
        color (array-like): Optional. Defaults to red. An array like of colors. Can be a single color, a list of colors, or a list of values to be mapped to a colormap.
        name (str): Optional. Defaults to None. The name of the object.
        state (int): Optional. Defaults to 1. The state of the object.
        transparency (float): Optional. Defaults to 0. The transparency value of the object.
        colormap: Optional. Defaults to "RdYlBu_r". Name of a colormap or a matplotlib colormap or a pymolviz.ColorMap object. Used to map values to colors.
        render_as (str): Optional. Defaults to "Spheres". How to display the points. Can be "Spheres" or "Dots" or None. If None, the points are not displayed.
        radius (float): Optional. Defaults to .3. Only relevant if render_as is "Spheres". The radius of the spheres.
    """

    def __init__(self, vertices, color = None, name = None, state = 1, transparency = 0, colormap = "RdYlBu_r", render_as = "Spheres", radius = .3, *args, **kwargs) -> None:
        global pmv_default_color_counter
        global pmv_default_color_palette
        super().__init__(name)
        if color is None:
            color = pmv_default_color_palette[pmv_default_color_counter]
            kwargs["values_are_single_color"] = True
            pmv_default_color_counter += 1
            if pmv_default_color_counter >= len(pmv_default_color_palette):
                pmv_default_color_palette = get_distinct_colors(pmv_default_color_counter * 2)

        self.vertices = np.array(vertices, dtype=float).reshape(-1, 3)
        if type(colormap) != ColorMap:
            self.colormap = ColorMap(color, colormap, state = state, name=f"{self.name}_colormap", *args, **kwargs)
        else:
            self.colormap = colormap
        if "single" in self.colormap._color_type: # colors were not inferred
            self.color = np.arange(self.vertices.shape[0]) # color is just the index
        else:
            self.color = np.array(color).flatten()

        self.render_as = render_as
        self.radius = radius
        self.state = state
        self.transparency = transparency


    def as_pseudoatoms(self) -> PseudoAtoms:
        return PseudoAtoms(self.vertices, self.color, name = self.name, state = self.state, colormap = self.colormap)


    def interpolate_to_grid_data(self, grid_spacing = None, method = "linear", margin = .1, *args, **kwargs):
        """ Converts the points to GridData.
        
        Parameters:
        grid_spacing (array-like or float): Spacing between grid points in the 3 directions. Defaults to 1 in all directions.
        method (str): Interpolation method, one of 'linear' or 'nearest'. Defaults to 'linear'.

        Returns:
            GridData: A GridData object.
        """
        
        # Define the grid range based on the given bounding box or the given points
        margin = 1 + margin
        x_min, x_max = self.vertices[:, 0].min() * margin, self.vertices[:, 0].max() * margin
        y_min, y_max = self.vertices[:, 1].min() * margin, self.vertices[:, 1].max() * margin
        z_min, z_max = self.vertices[:, 2].min() * margin, self.vertices[:, 2].max() * margin
        
        if grid_spacing is None:
            x_spacing = y_spacing = z_spacing = 1
        elif type(grid_spacing) == float:
            x_spacing = y_spacing = z_spacing = grid_spacing
        else:
            x_spacing, y_spacing, z_spacing = grid_spacing
        # Create grid
        xi = np.arange(x_min, x_max + x_spacing, x_spacing)
        yi = np.arange(y_min, y_max + y_spacing, y_spacing)
        zi = np.arange(z_min, z_max + z_spacing, z_spacing)
        X, Y, Z = np.meshgrid(xi, yi, zi)
        
        if "single" in self.colormap._color_type: # colors were not inferred
            values = np.ones(self.vertices.shape[0]) # values are just 1
        else:
            values = self.color
        # Interpolate values on the grid
        grid_values = griddata(self.vertices, values, (X, Y, Z), method=method)
        
        # Flatten the grid points and values
        grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        grid_values = grid_values.ravel()
        

        return GridData(grid_values, grid_points, *args, **kwargs)

    def to_surface(self, distance = 1, grid_spacing = 1, name = None, *args, **kwargs):
        """ Converts the points to a surface.
        
        Returns:
            IsoSurface: An IsoSurface object.
        """
        from ..volumetric.IsoSurface import IsoSurface
        from ..volumetric.GridData import GridData

        from scipy.spatial import KDTree
        tree = KDTree(self.vertices)
        # we determine grid points from the bounding box of the points

        min_grid, max_grid = self.vertices.min(axis=0) - 2*distance, self.vertices.max(axis=0) + 2*distance
        grid = np.mgrid[min_grid[0]:max_grid[0]:grid_spacing, min_grid[1]:max_grid[1]:grid_spacing, min_grid[2]:max_grid[2]:grid_spacing]
        grid_data = np.vstack([grid[0].ravel(), grid[1].ravel(), grid[2].ravel()]).T
        # we query the tree for the nearest point
        distances, indices = tree.query(grid_data)
        # we set the value of the grid point to the distance to the nearest point
        values = distances * grid_spacing

        if name is None:
            gdata = GridData(values, grid_data)
            return IsoSurface(gdata, distance, *args, **kwargs)
        else:
            gdata = GridData(values, grid_data, name=f"{name}_grid")
            return IsoSurface(gdata, distance, name = f"{name}_surface", *args, **kwargs)


    def _create_CGO_list(self) -> list:
        """ Creates a CGO list from the mesh information. Points can be displayed as spheres or as points.

        Returns:
            List of str: The CGO list.
        """

        cgo_points = self.vertices
        cgo_colors = self.colormap.get_color(self.color)

        cgo_list = []
        
        
        if self.render_as.lower() == "spheres":
            #vertices
            point_meshes = np.hstack([
                np.full(cgo_points.shape[0], "COLOR")[:,None], cgo_colors, \
                np.full(cgo_points.shape[0], "SPHERE")[:,None], cgo_points, \
                np.full(cgo_points.shape[0], self.radius)[:,None], \
                ]).flatten()
            cgo_list.extend(point_meshes)

        elif self.render_as.lower() == "dots":
            cgo_list.extend(["BEGIN", "POINTS"])
            #vertices
            points = np.hstack([
                np.full(cgo_points.shape[0], "COLOR")[:,None], cgo_colors, \
                np.full(cgo_points.shape[0], "VERTEX")[:,None], cgo_points, \
                ]).flatten()
            cgo_list.extend(points)
            # ending
            cgo_list.append("END")

        return cgo_list

    def _script_string(self):
        cgo_string_builder = []
        state = "" if self.state is None else f", state={self.state}"
        cgo_name = sanitize_pymol_string(self.name)
        cgo_string_builder.append(f"""
{cgo_name} = [
        """)
        content = ",".join([str(e) for e in self._create_CGO_list()])
        cgo_string_builder.append(content)

        # ending
        cgo_string_builder.append(f"""
            ]
cmd.load_cgo({cgo_name}, "{cgo_name}"{state})
        """)
        try:
            self.transparency[0]
        except TypeError:
            transparency = self.transparency
            cgo_string_builder.append(f"""
cmd.set("cgo_transparency", {transparency}, "{cgo_name}")
    """)
        except IndexError:
            transparency = self.transparency
            cgo_string_builder.append(f"""
cmd.set("cgo_transparency", {transparency}, "{cgo_name}")
    """)
        
        return "\n".join(cgo_string_builder)
    
    def load(self):
        from pymol import cgo
        from pymol import cmd
        cgo_name = sanitize_pymol_string(self.name)
        content = [e for e in self._create_CGO_list()]
        map_cgo_keys = {"POINTS": cgo.POINTS, "SPHERE":cgo.SPHERE, "COLOR":cgo.COLOR, "VERTEX": cgo.VERTEX, "NORMAL":cgo.NORMAL, "CYLINDER": cgo.CYLINDER, 
                        "CONE": cgo.CONE, "BEGIN": cgo.BEGIN, "END": cgo.END, "LINEWIDTH": cgo.LINEWIDTH, "LINES": cgo.LINES, "TRIANGLES": cgo.TRIANGLES}
        for idx, entry in enumerate(content):
            try: 
                content[idx] = float(entry)
            except ValueError:
                content[idx] = map_cgo_keys[entry]
        state = str(self.state)
        cmd.load_cgo(content, cgo_name, state)
        try:
            self.transparency[0]
        except TypeError:
            cmd.set("cgo_transparency", self.transparency, cgo_name)
        except IndexError:
            cmd.set("cgo_transparency", self.transparency, cgo_name)
