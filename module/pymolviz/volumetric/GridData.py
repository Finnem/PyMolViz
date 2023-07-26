import numpy as np
from ..Displayable import Displayable
from ..meshes.Points import Points

class GridData(Displayable):
    def __init__(self, values, positions = None, step_sizes = None, step_counts = None, origin = None, name = None):
        """ Represents regular 3-dimensional data. Either positions or step_size and step_count must be given.

        Args:
            values (np.array): The values of the data.
            positions (np.array): Optional. The positions of the regular data. Will be resorted to be in ascending order.
            step_sizes (np.array): Optional. The size of the steps in each direction. Defaults to None.
            step_counts (np.array): Optional. The number of steps in each direction. Defaults to None.
            origin (np.array): Optional. Starting point of the grid. Defaults to None.
            name (str): Optional. The name of the data. Defaults to None.

        """


        self.step_sizes = step_sizes
        self.step_counts = step_counts
        self.origin = origin

        values = np.array(values).flatten()

        # no grid points given: infer grid from parameters
        if positions is None:
            if (self.step_counts is None) or (self.step_sizes is None):
                raise ValueError("Either positions or step_sizes and step_counts must be given.")
            else:
                sorted_indices = np.arange(len(values))
            if origin is None:
                self.origin = np.array([0, 0, 0])
        else:
            positions = np.array(positions)
            sorted_indices = np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0]))
            # determine step_size
            if self.step_sizes is None:
                self.step_sizes = np.zeros(3)
                for i in range(3):
                    unique = np.unique(positions[:, i])
                    sorted = np.sort(unique)
                    self.step_sizes[i] = np.max(np.diff(sorted))
            # determine step_count
            if self.step_counts is None:
                self.step_counts = np.zeros(3)
                for i in range(3):
                    length = np.max(positions[:, i]) - np.min(positions[:, i])
                    self.step_counts[i] = np.round(length / self.step_sizes[i])
            # determine origin
            if self.origin is None:
                self.origin = np.min(positions, axis=0)


        # sort values
        self.step_counts = np.array(self.step_counts, dtype=int)
        if len(values) != np.product(self.step_counts + 1):
            raise ValueError(f"Number of values ({len(values)}) does not match number of grid points ({np.product(self.step_counts + 1)}).")
        self.values = values[sorted_indices]
        self.step_sizes = np.array(self.step_sizes)

        super().__init__(name = name)
          

    def get_positions(self):
        x = np.linspace(self.origin[0], self.origin[0] + self.step_sizes[0] * self.step_counts[0], self.step_counts[0] + 1)
        y = np.linspace(self.origin[1], self.origin[1] + self.step_sizes[1] * self.step_counts[1], self.step_counts[1] + 1)
        z = np.linspace(self.origin[2], self.origin[2] + self.step_sizes[2] * self.step_counts[2], self.step_counts[2] + 1)
        xx, yy, zz = np.meshgrid(x, y, z)
        positions = np.array([yy.flatten(), xx.flatten(), zz.flatten()]).T
        return positions

    def to_points(self, filter = None, *args, **kwargs):
        """
        Returns a point cloud representation of the data, filtered by the passed function.

        Args:
            filter (function, optional): A function that takes a numpy array and returns a boolean array. Applied to values to determine positions to show.

        Returns:
            pymolviz.Points: A point cloud representation of the data.
        """

        # create grid
        positions = self.get_positions()
        # filter
        if filter is None:
            filtered_positions = positions
            filtered_values = self.values
        else:
            filtered_positions = positions[filter(self.values)]
            filtered_values = self.values[filter(self.values)]

        return Points(filtered_positions, filtered_values, *args, **kwargs)
    

    def _script_string(self):
        values = self.values.reshape(self.step_counts.astype(int) + 1)
        result = f"""
{self.name}_data = np.array({np.array2string(values, threshold=1e15, separator=",")})
{self.name} = Brick.from_numpy({self.name}_data, {np.array2string(self.step_sizes, separator = ",")}, origin={np.array2string(self.origin, separator=",")})
cmd.load_brick({self.name}, "{self.name}")
{self.name}_data = None

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


    def from_mtz(path, factor_column = "FWT", phase_column = "PHWT", sample_rate = 2.6, min_pos = [0, 0, 0], max_pos = [1, 1, 1], step_sizes = [1., 1., 1.], *args, **kwargs):
        """
        Creates a RegularData object from an MTZ file.
        
        Args:
            path (str): The path to the MTZ file.
            
        Returns:
            RegularData: The created RegularData object.
        """
        from ..util.io import grid_from_mtz
        return grid_from_mtz(path, factor_column, phase_column, sample_rate, min_pos, max_pos, step_sizes, *args, **kwargs)



    def from_xyz(path, in_bohr = True, *args, **kwargs):
        from ..util.io import grid_from_xyz
        return grid_from_xyz(path, in_bohr, *args, **kwargs)