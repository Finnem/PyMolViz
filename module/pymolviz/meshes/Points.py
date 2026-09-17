from __future__ import annotations

import copy
import logging
import uuid

import numpy as np

from ..ColorMap import ColorMap
from ..Displayable import Displayable
from ..points import as_point_source, point_sources_from_sequence, resolve_xyz
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

    def __init__(self, vertices, color = None, name = None, state = 1, transparency = 0, colormap = "RdYlBu_r", render_as = "Spheres", radius = .3, bypass_colormap = False, vertex_sources = None, *args, **kwargs) -> None:
        global pmv_default_color_counter
        global pmv_default_color_palette
        obj_id = kwargs.pop("obj_id", None) or kwargs.pop("id", None)
        super().__init__(name, obj_id=obj_id)
        if vertex_sources is not None:
            self.vertex_sources = point_sources_from_sequence(vertex_sources)
            vertices = np.array([resolve_xyz(v) for v in self.vertex_sources])
        else:
            self.vertex_sources = None
            vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
        if color is None:
            color = pmv_default_color_palette[pmv_default_color_counter]
            kwargs["values_are_single_color"] = True
            pmv_default_color_counter += 1
            if pmv_default_color_counter >= len(pmv_default_color_palette):
                pmv_default_color_palette = get_distinct_colors(pmv_default_color_counter * 2)

        self.vertices = np.array(vertices, dtype=float).reshape(-1, 3)
        self.bypass_colormap = bypass_colormap
        self.field_id = kwargs.pop("field_id", None) or None
        if self.field_id:
            self.field_id = str(self.field_id)
        self.field_colormap = kwargs.pop("field_colormap", None) or None
        from ..util.field_sample import normalize_clims
        self.field_clims = normalize_clims(kwargs.pop("field_clims", None))
        self.field_clim_mode = kwargs.pop("field_clim_mode", None) or None
        self.field_colormap_spec = kwargs.pop("field_colormap_spec", None) or None
        if bypass_colormap:
            self.color = np.array(color)
        else:
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

    def _cgo_vertex_rgb(self):
        """Per-vertex RGB for CGO, honoring ``bypass_colormap``."""
        n = int(np.asarray(self.vertices, dtype=float).reshape(-1, 3).shape[0])
        if getattr(self, "bypass_colormap", False):
            arr = np.asarray(self.color, dtype=float)
            if arr.ndim <= 1:
                colors = arr.reshape(-1)[:3].reshape(1, 3)
            else:
                colors = arr.reshape(arr.shape[0], -1)[:, :3]
            if colors.shape[0] == 1:
                return np.repeat(colors, n, axis=0) if n else colors
            if colors.shape[0] != n:
                return np.repeat(colors[:1], n, axis=0) if n else colors[:0]
            return colors
        return np.asarray(self.colormap.get_color(self.color)[:, :3], dtype=float)

    def invalidate_cgo_cache(self) -> None:
        self._cached_cgo = None
        self._cached_resolved = None
        self._geom_serial = getattr(self, "_geom_serial", 0) + 1

    def shift_vertices(self, delta) -> None:
        """Translate baked vertices and cached CGO without remeshing."""
        self._geom_serial = getattr(self, "_geom_serial", 0) + 1
        d = np.asarray(delta, dtype=float).reshape(3)
        self.vertices = np.asarray(self.vertices, dtype=float).reshape(-1, 3) + d
        cached = getattr(self, "_cached_cgo", None)
        resolved = getattr(self, "_cached_resolved", None)
        if cached is None and resolved is None:
            return
        from ..util.cgo import offset_cgo_vertices
        if cached is not None:
            offset_cgo_vertices(cached, d)
        if resolved is not None:
            offset_cgo_vertices(resolved, d)

    def clone_baked(self):
        """Shallow-copy this mesh, sharing topology and copying vertex/CGO buffers.

        Does not remesh. Faces stay the same object; vertices and cached CGO
        lists are copied so later shifts do not mutate the source.
        """
        cloned = copy.copy(self)
        cloned._id = uuid.uuid4().hex
        verts = getattr(self, "vertices", None)
        if verts is not None:
            cloned.vertices = np.array(verts, copy=True, dtype=float)
        normals = getattr(self, "normals", None)
        if normals is not None:
            cloned.normals = np.array(normals, copy=True, dtype=float)
        color = getattr(self, "color", None)
        if color is not None and not np.isscalar(color):
            cloned.color = np.array(color, copy=True)
        cached = getattr(self, "_cached_cgo", None)
        cloned._cached_cgo = list(cached) if cached is not None else None
        resolved = getattr(self, "_cached_resolved", None)
        cloned._cached_resolved = list(resolved) if resolved is not None else None
        pair_spans = getattr(self, "_pair_spans", None)
        if pair_spans is not None:
            cloned._pair_spans = list(pair_spans)
        for attr in ("pair_radii", "pair_heads", "pair_styles"):
            val = getattr(self, attr, None)
            if isinstance(val, list):
                setattr(cloned, attr, list(val))
        return cloned

    def rebuild(self, context=None) -> None:
        self.invalidate_cgo_cache()
        if self.vertex_sources is None:
            return
        self.vertices = np.array([resolve_xyz(v, context) for v in self.vertex_sources], dtype=float)


    def as_pseudoatoms(self) -> PseudoAtoms:
        return PseudoAtoms(self.vertices, self.color, name = self.name, state = self.state, colormap = self.colormap)


    def interpolate_to_grid_data(self, grid_spacing = None, method = "linear", margin = .1, *args, **kwargs):
        """ Converts the points to GridData.
        
        Parameters:
        grid_spacing (array-like or float): Spacing between grid points in the 3 directions. Defaults to 1 in all directions.
        method (str): Interpolation method, one of 'linear' or 'nearest'. Defaults to 'linear'.

        Returns:
            pymolviz.Field: A Field with a regular voxel lattice.
        """
        from ..fields.field import Field
        
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
        

        return Field(grid_values, grid_points, *args, **kwargs)

    def to_surface(self, distance = 1, grid_spacing = 1, name = None, *args, **kwargs):
        """ Converts the points to a surface.
        
        Returns:
            IsoSurface: An IsoSurface object.
        """
        from ..volumetric.IsoSurface import IsoSurface
        from ..fields.field import Field

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
            gdata = Field(values, grid_data)
            return IsoSurface(gdata, distance, *args, **kwargs)
        else:
            gdata = Field(values, grid_data, name=f"{name}_grid")
            return IsoSurface(gdata, distance, name = f"{name}_surface", *args, **kwargs)


    def _create_CGO_list(self) -> list:
        """ Creates a CGO list from the mesh information. Points can be displayed as spheres or as points.

        Returns:
            List of str: The CGO list.
        """

        cgo_points = self.vertices
        if self.bypass_colormap:
            cgo_colors = self.color
        else:
            cgo_colors = self.colormap.get_color(self.color)
        print(len(np.unique(cgo_colors, axis = 0)))
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
        self._try_rebuild()
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
            if type(transparency) == float:
                cgo_string_builder.append(f"""
cmd.set("cgo_transparency", {transparency}, "{cgo_name}")
        """)
        except IndexError:
            try:
                if len(self.transparency) == 0:
                    transparency = 0
                else:
                    transparency = self.transparency
            except TypeError:
                transparency = self.transparency
            if type(transparency) == float:
                cgo_string_builder.append(f"""
cmd.set("cgo_transparency", {transparency}, "{cgo_name}")
        """)
        
        return "\n".join(cgo_string_builder)
    
    def load(self, cmd=None):
        self._try_rebuild()
        if cmd is None:
            from pymol import cmd

        from ..util.cgo import resolve_cgo_tokens

        cgo_name = sanitize_pymol_string(self.name)
        content = resolve_cgo_tokens([e for e in self._create_CGO_list()])
        state = str(self.state)
        cmd.load_cgo(content, cgo_name, state)
        try:
            self.transparency[0]
        except TypeError:
            cmd.set("cgo_transparency", self.transparency, cgo_name)
        except IndexError:
            cmd.set("cgo_transparency", self.transparency, cgo_name)
