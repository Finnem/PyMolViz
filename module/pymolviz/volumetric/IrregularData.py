import numpy as np
from ..util.math import tanh_distance_weighting
from scipy.spatial import KDTree
import logging
from .GridData import GridData
from ..meshes.Points import Points

class IrregularData():
    def __init__(self, positions, values, remove_duplicates=False):
        """ Represents irregular 3-dimensional data.

        Args:
            positions (np.array): The positions of the data.
            values (np.array or {name : np.array}): The values of the data. If a dictionary is passed, the keys will be used as names for the values.
            remove_duplicates (bool, optional): If True, duplicate positions will be removed. Defaults to False.
        """

        self.positions = positions
        self.values = values
        if remove_duplicates:
            unique_indices = np.unique(positions, axis=0, return_index=True)[1]
            self.positions = self.positions[unique_indices]
            if len(self.positions) != len(positions):
                logging.warning("IrregularData: Duplicate positions were removed.")

            for key, value in self._values.items():
                self._values[key] = value[unique_indices]
        
    @property
    def values(self):
        """Values of the data.
        
        If only a single set of values was passed, this will be a numpy array.
        Otherwise it will return a dictionary, mapping names of values to numpy arrays.
        """

        if len(self._values) == 1:
            return self._values.__iter__().__next__()
        else:
            return self._values

    @values.setter 
    def values(self, values):
        if isinstance(values, dict):
            self._values = values
        else:
            self._values = {"values" : values}


    def interpolate(self, step_size:np.array = None, step_count:np.array = None, radius:float = 1, weighting=None, fall_off=None):
        """ Interpolates the data onto a regular grid.

        Args:
            step_size (np.array, optional): The size of the steps. Defaults to None.
            step_count (np.array, optional): The number of steps. Defaults to None.
            radius (float, optional): The radius of the interpolation. Defaults to 1.
            weighting (function, optional): The weighting function. Defaults to offest tanh.
            fall_off (function, optional): The fall-off function. Defaults to offset tanh.

        Returns:
            RegularData: The interpolated data.
        """

        if step_size is None and step_count is None:
            raise ValueError("Either step_size or step_count must be specified.")
        if (step_size is not None) and (step_count is not None):
            raise ValueError("Either step_size or step_count must be specified, not both.")
        
        if step_count is None:
            if isinstance(step_size, float) or isinstance(step_size, int):
                step_size = np.array([step_size, step_size, step_size])
            else:
                step_size = np.array(step_size)


        if step_size is None:
            raise NotImplementedError("step_count is not implemented yet.")
        
        if weighting is None:
            weighting = tanh_distance_weighting(0, 1.0)
        
        if fall_off is None:
            fall_off = tanh_distance_weighting(3/16, 1.0)

        to_start_count = np.floor(np.min(self.positions, axis = 0)/step_size)
        new_min = to_start_count * step_size
        step_count = np.ceil((np.max(self.positions, axis = 0) - new_min)/step_size)
        new_max = new_min + step_count * step_size
        grid = np.meshgrid(*[np.arange(new_min[i], new_max[i], step_size[i]) for i in range(3)])
        regular_positions = np.array([g.flatten() for g in grid]).T
        
        # KDTree of regular positions and irregular positions
        regular_tree = KDTree(regular_positions)
        irregular_tree = KDTree(self.positions)

        # Find positions within radius of regular positions
        irregular_indices = regular_tree.query_ball_tree(irregular_tree, radius)
        
        # interpolate based on these
        interpolated_values = np.zeros((len(regular_positions), len(self.values)))
        for i, irregular_index in enumerate(irregular_indices):
            irregular_positions = self.positions[irregular_index]
            irregular_values = np.array([self.values[key][irregular_index] for key in self.values.keys()]).T
            distances = np.linalg.norm(irregular_positions - regular_positions[i], axis=1)
            weights = weighting(distances)
            fall_off_weighting = fall_off(distances)
            interpolated_values[i] = np.sum(irregular_values * (weights * fall_off_weighting)[:,None], axis=0) / np.sum(weights)
        new_values = {key : interpolated_values[:, i] for i, key in enumerate(self.values.keys())}

        return RegularData(regular_positions, new_values, step_size, step_count, to_start_count * step_size)

    def to_point_cloud(self, filter, value_label = None, *args, **kwargs):
        """
        Returns a point cloud representation of the data, filtered by the passed function.

        Args:
            filter (function): A function that takes a numpy array and returns a boolean array.

        Returns:
            pymolviz.Points: A point cloud representation of the data.
        """

        if value_label is None:
            values = self.values.__iter__().__next__()
        else:
            values = self.values[value_label]

        # filter
        filtered_positions = self.positions[filter(values)]
        filtered_values = values[filter(values)]

        return Points(filtered_positions, filtered_values, *args, **kwargs)