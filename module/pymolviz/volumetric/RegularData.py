import numpy as np
from ..util.math import tanh_distance_weighting
import logging
from ..meshes.Points import Points

_pmv_regular_data_counter = 0
class RegularData():
    def __init__(self, positions, values, name = None, step_size = None, step_count = None, offset = None):
        """ Represents regular 3-dimensional data. If no step_size or step_count is passed, they will be inferred.


        Args:
            positions (np.array): The positions of the regular data. Will be resorted to be in ascending order.
            values (np.array or {name : np.array}): The values of the data. If a dictionary is passed, the keys will be used as names for the values.
            step_size (np.array, optional): The size of the steps. Defaults to None.
            step_count (np.array, optional): The number of steps. Defaults to None.
            offset (np.array, optional): The number of steps to the start. Defaults to None.
        """

        # sort positions
        positions = np.array(positions)
        sorted_indices = np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0]))
        self.positions = positions[sorted_indices]
        if isinstance(values, dict):
            for key, value in values.items():
                values[key] = value[sorted_indices]
                self.values = values
        elif isinstance(values, np.ndarray):
            self.values = values[sorted_indices]
        else:
            raise ValueError("Values must be a numpy array or a dictionary of numpy arrays.")  
        self.step_size = step_size
        if name is None:
            global _pmv_regular_data_counter
            self.name = "RegularData_{}".format(_pmv_regular_data_counter)
            _pmv_regular_data_counter += 1
        else:
            self.name = name
        if isinstance(step_size, float) or isinstance(step_size, int):
            self.step_size = np.array([step_size, step_size, step_size])
        self.step_count = step_count
        if isinstance(step_count, float) or isinstance(step_count, int):
            self.step_count = np.array([step_count, step_count, step_count])

        if self.step_count is None and self.step_size is None:
            # inferring step_size
            differences = np.absolute(self.positions[0] - self.positions)
            self.step_size = np.array([np.min(differences[:, 0][differences[:,0] != 0]), np.min(differences[:, 1][differences[:,1] != 0]), np.min(differences[:, 2][differences[:,2] != 0])])
        if self.step_count is None:
            self.step_count = np.floor((self.positions.max(axis=0) - self.positions.min(axis=0)) / self.step_size) + 1

        else:
            self.step_size = np.max(positions, axis=0) - np.min(positions, axis=0)
            self.step_size = self.step_size / (self.step_count - 1)
        
        if offset is None:
            self.offset = np.min(self.positions, axis=0)
        else:
            self.offset = offset
        
    @property
    def values(self):
        """Values of the data.
        
        If only a single set of values was passed, this will be a numpy array.
        Otherwise it will return a dictionary, mapping names of values to numpy arrays.
        """

        if len(self._values) == 1:
            return self._values[self._values.__iter__().__next__()]
        else:
            return self._values

    @values.setter 
    def values(self, values):
        if isinstance(values, dict):
            self._values = values
        else:
            self._values = {"values" : values}



    def to_point_cloud(self, filter = None, value_label = None, *args, **kwargs):
        """
        Returns a point cloud representation of the data, filtered by the passed function.

        Args:
            filter (function, optional): A function that takes a numpy array and returns a boolean array.
            value_label (str, optional): The name of the value to use for the point cloud. Defaults to None.


        Returns:
            pymolviz.Points: A point cloud representation of the data.
        """

        if value_label is None:
            values = self._values[self._values.__iter__().__next__()]
        else:
            values = self.values[value_label]

        # filter
        if filter is None:
            filtered_positions = self.positions
            filtered_values = values
        else:
            filtered_positions = self.positions[filter(values)]
            filtered_values = values[filter(values)]

        return Points(filtered_positions, filtered_values, *args, **kwargs)
    
    def _create_script(self, value_label = None):
        if value_label is None:
            value_name = ""
            values = self._values[self._values.__iter__().__next__()]
        else:
            value_name = "_" + value_label
            values = self.values[value_label]
        values = values.reshape(self.step_count.astype(int))
        #values = np.swapaxes(values, 0, 2) # why is this only sometimes necessary?
        result = f"""
{self.name}{value_name}_data = np.array({np.array2string(values, threshold=1e15, separator=",")})
{self.name}{value_name} = Brick.from_numpy({self.name}{value_name}_data, {np.array2string(self.step_size, separator = ",")}, origin={np.array2string(self.offset, separator=",")})
cmd.load_brick({self.name}{value_name}, "{self.name}{value_name}")
"""
        return result

    def from_ccp4(path):
        """
        Creates a RegularData object from a CCP4 file.

        Args:
            path (str): The path to the CCP4 file.

        Returns:
            RegularData: The created RegularData object.
        """

        import gemmi


    def from_mtz(path):
        """
        Creates a RegularData object from an MTZ file.
        
        Args:
            path (str): The path to the MTZ file.
            
        Returns:
            RegularData: The created RegularData object.
        """

        import gemmi