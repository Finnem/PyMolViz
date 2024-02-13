from scipy.spatial.transform import Rotation
from ...util.math import get_perp
from ..Lines import Lines
import numpy as np

class Rotation_Indicator(Lines):
    def __init__(self, position, axis, radius = 1, starting_normal = None, color ="red", name = None, state = 1, transparency = 0, colormap ="RdYlBu_r", linewidth=0.05, *args, **kwargs):
        """Creates a rotation indicator mesh.

        Args:
            position (np.ndarray): The position of the rotation indicator.
            axis (np.ndarray): The axis of rotation.
            radius (float): The radius of the rotation indicator.
            starting_normal (np.ndarray): The starting normal of the rotation indicator.
            color (np.ndarray): The color of the rotation indicator.
            name (str): The name of the rotation indicator.
            state (int): The state of the rotation indicator.
            transparency (float): The transparency of the rotation indicator.
            colormap (Colormap): The colormap of the rotation indicator.
            *args: Additional arguments for the Lines class.
            **kwargs: Additional keyword arguments for the Lines class.
        """
        self.position = position
        self.axis = np.array(axis); self.axis /= np.linalg.norm(self.axis)
        self.starting_normal = starting_normal if starting_normal is not None else get_perp(self.axis)
        self.color = color
        self.name = name
        self.state = state
        self.transparency = transparency
        self.colormap = colormap

        # Create the rotation indicator mesh as set of lines
        points = []
        for i in range(1, 100):
            points.append(Rotation.from_rotvec(self.axis * (i * 1.5* np.pi / 100)).apply(self.starting_normal * (1 + linewidth)))
        

        self.vertices = np.array(points)
        self.vertices *= radius
        self.vertices = np.hstack([self.vertices[:-1], self.vertices[1:]]).reshape(-1, 2, 3)
        self.vertices -= np.mean(self.vertices, axis = 1, keepdims = True) * linewidth
        self.vertices += self.position


        # Create the Lines object
        super().__init__(self.vertices, self.color, self.name, self.state, self.transparency, self.colormap, linewidth= linewidth, *args, **kwargs)

    def _create_CGO_list(self) -> str:
        """ Creates a CGO list from the mesh information. A rotation indicator mesh will be created.
            CGO constants are kept as strings to avoid importing the pymol module.
        
        Returns:
            None
        """
        if self.render_as == "lines":
            raise NotImplementedError("Lines are not supported for rotation indicators.")
           
        elif self.render_as == "cylinders":
            cgo_list = []
            cgo_vertices = self.vertices.reshape(-1, 6)
            cgo_colors = self.colormap.get_color(self.color)[:,:3].reshape(-1, 6)
            # use final 20th fraction for cone
            cone_start = np.array(cgo_vertices[-cgo_vertices.shape[0] // 10, :3])
            cone_end = np.array(cgo_vertices[-1, :3])
            cgo_vertices = cgo_vertices[:int(-cgo_vertices.shape[0] * 0.8 // 10)]
            cgo_colors = cgo_colors[:int(-cgo_colors.shape[0] * 0.8 // 10)]
            cylinders = np.hstack([
                np.full(cgo_vertices.shape[0], "CYLINDER")[:,None], cgo_vertices, np.full(cgo_vertices.shape[0], self.linewidth)[:,None], cgo_colors \
            ]).flatten()
            cgo_list.extend(cylinders)
            cones = ["CONE", *cone_start, *cone_end, self.linewidth * 2, 0.0, *cgo_colors[-1], 1.0, 1.0]
            cgo_list.extend(cones)
        return cgo_list