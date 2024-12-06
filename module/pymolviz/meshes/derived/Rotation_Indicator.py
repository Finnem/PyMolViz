from scipy.spatial.transform import Rotation
from ...util.math import get_perp
from ..Arrows import Arrows
import numpy as np


class Rotation_Indicator(Arrows):
    def __init__(self, center_position, outer_start, rotation_axis, angle, color = None, name = None, state = 1, transparency = 0, colormap = "RdYlBu_r", linewidth = 0.05, resolution = 20, show_arrow = True, *args, **kwargs):
        """Creates a rotation indicator mesh. Center_position (s) positions the indicator, outer_start (x) is the starting point of the drawn circle:
         __x__
        /  |  \\
        |  s   |
        \\____/
        Args:
            center_position (np.ndarray): The position of the rotation indicator.
            outer_start (np.ndarray): The starting point of the rotation indicator.
            rotation_axis (np.ndarray): The axis of rotation.
            angle (float): The angle of rotation in radians.
            color (np.ndarray): The color of the rotation indicator.
            name (str): The name of the rotation indicator.
            state (int): The state of the rotation indicator.
            transparency (float): The transparency of the rotation indicator.
            colormap (Colormap): The colormap of the rotation indicator.
            linewidth (float): The linewidth of the rotation indicator.
            *args: Additional arguments for the Lines class.
            **kwargs: Additional keyword arguments for the Lines class.
        """
        self.center_position = center_position
        self.outer_start = outer_start
        self.rotation_axis = np.array(rotation_axis); self.rotation_axis = self.rotation_axis / np.linalg.norm(self.rotation_axis)
        self.angle = angle
        self.color = color
        self.name = name
        self.state = state
        self.transparency = transparency
        self.colormap = colormap


        points = []
        rotation_start = self.outer_start - self.center_position
        for i in range(0, resolution + 1):
            points.append(Rotation.from_rotvec(self.rotation_axis * (i * angle / resolution)).apply(rotation_start * (1 + linewidth)))
        self.vertices = np.array(points)
        self.vertices = np.hstack([self.vertices[:-1], self.vertices[1:]]).reshape(-1, 2, 3)
        self.vertices -= np.mean(self.vertices, axis = 1, keepdims = True) * linewidth
        self.vertices += self.center_position

        # dealing with final arrow could be done better, will look strange if resolution is to high

        self.arrow_mask = np.zeros(int(len(self.vertices)), dtype=bool)
        if show_arrow:
            self.arrow_mask[-1] = True
        #print(self.arrow_mask)


        # Create the Lines object
        super().__init__(self.vertices, self.color, self.name, self.state, self.transparency, self.colormap, linewidth= linewidth, arrow_mask = self.arrow_mask, head_length = 0.9, *args, **kwargs)
