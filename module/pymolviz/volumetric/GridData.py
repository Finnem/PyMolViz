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
        self.A_to = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

        values = np.array(values).flatten()

        # no grid points given: infer grid from parameters
        if positions is None:
            if (self.step_counts is None) or (self.step_sizes is None):
                raise ValueError("Either positions or step_sizes and step_counts must be given.")
            else:
                sorted_indices = np.arange(len(values))
            if origin is None:
                self.origin = np.array([0, 0, 0], dtype=np.float64)
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
                self.origin = np.min(positions, axis=0).astype(np.float64)


        # sort values
        self.step_counts = np.array(self.step_counts, dtype=int)
        if len(values) != np.product(self.step_counts + 1):
            raise ValueError(f"Number of values ({len(values)}) does not match number of grid points ({np.product(self.step_counts + 1)}).")
        self.values = values[sorted_indices]
        self.step_sizes = np.array(self.step_sizes)

        super().__init__(name = name)
          
    def _get_plane_intersections(self,  point, normal):
        
        """Computes the intersection points of the edges of the grid box with a plane. 
        
        Args:
            normal(np.array): normal that describes the orientation of the plane in relation to the coordinate system
            point(np.array): point on the plane
        """
        
        # For each edge of the grid box it is checked whether it intersects with the plane.
        # The plane is the set of points p fulfilling 
        # (p - p0) * n = 0
        # and parametrized by its normal vector n (normal) and a point on the plane p0 (point)
        
        # A point p is on an line if it fulfills
        # p = l0 + l*d
        # where l0 is another point in the line, l is a unit vector in the direction of the line and d is element of the real numbers.
        # As the edges of the bounding box have a start and and end point we define the start with d=0 and the end with d=d.
        
        # The edges of the grid box follow the axes of the coordinate system, so they are unit vectors in original coordinate system 
        # and correspond to l.
        x_dir = np.array([1, 0, 0]) 
        y_dir = np.array([0, 1, 0])
        z_dir = np.array([0, 0, 1])
         
        # Since the bounding box is a rectangular cuboid, it has 12 edges that can be grouped by their direction along the axes of the coordinate system.
        # To get all starting points l0 of these edges the corners of the cuboid with the minimum value for each axis are extracted. 
        # Note that corners that contain multiple minima (e.g. the origin) appear multiple times as they are the start point for edges with different directions.          
        min_pos = self.origin
        max_pos = self.origin + self.step_sizes * self.step_counts
        
        x_corners = np.array([[min_pos[0], min_pos[1], min_pos[2]], [min_pos[0], max_pos[1], min_pos[2]], [min_pos[0], min_pos[1], max_pos[2]], [min_pos[0], max_pos[1], max_pos[2]]])
        y_corners = np.array([[min_pos[0], min_pos[1], min_pos[2]], [max_pos[0], min_pos[1], min_pos[2]], [min_pos[0], min_pos[1], max_pos[2]], [max_pos[0], min_pos[1], max_pos[2]]])
        z_corners = np.array([[min_pos[0], min_pos[1], min_pos[2]], [max_pos[0], min_pos[1], min_pos[2]], [min_pos[0], max_pos[1], min_pos[2]], [max_pos[0], max_pos[1], min_pos[2]]])

        
        # For all possible intersection between the plane and bounding box, the scalar d is computed via
        # d = ((p0 - l0) * n)/(l * n)
        # With the computed scalars, the intersections can be computed via
        # p = l0 + l*d
        # With the previous definition of start and end points of the edges of the bounding box, it can be checked whether the computed intersections are valid.
        
        intersections = []
        if np.dot(x_dir,normal) != 0:
            d_x = np.dot(point - x_corners, normal) / np.dot(x_dir, normal)
            x_intersections = x_corners + d_x[:, np.newaxis] * x_dir
            [intersections.append(x_intersection) for x_intersection in x_intersections[(d_x >= 0) & (d_x <= max_pos[0] - min_pos[0])]]
        if np.dot(y_dir, normal) != 0:
            d_y = np.dot(point - y_corners, normal) / np.dot(y_dir, normal)
            y_intersections = y_corners + d_y[:, np.newaxis] * y_dir
            [intersections.append(y_intersection) for y_intersection in y_intersections[(d_y >= 0) & (d_y <= max_pos[1] - min_pos[1])]]
        if np.dot(z_dir, normal) != 0:
            d_z = np.dot(point - z_corners, normal) / np.dot(z_dir, normal)
            z_intersections = z_corners + d_z[:, np.newaxis] * z_dir
            [intersections.append(z_intersection) for z_intersection in z_intersections[(d_z >= 0) & (d_z <= max_pos[2] - min_pos[2])]]
        return intersections
    
    def _get_projections_on_plane(self,  point, normal):
        min_pos = self.origin
        max_pos = self.origin + self.step_sizes * self.step_counts
        
        # create lists with those corners that have a minimum value for each dimension
        x_corners = np.array([[min_pos[0], min_pos[1], min_pos[2]], [min_pos[0], max_pos[1], min_pos[2]], [min_pos[0], min_pos[1], max_pos[2]], [min_pos[0], max_pos[1], max_pos[2]]])
        y_corners = np.array([[min_pos[0], min_pos[1], min_pos[2]], [max_pos[0], min_pos[1], min_pos[2]], [min_pos[0], min_pos[1], max_pos[2]], [max_pos[0], min_pos[1], max_pos[2]]])
        z_corners = np.array([[min_pos[0], min_pos[1], min_pos[2]], [max_pos[0], min_pos[1], min_pos[2]], [min_pos[0], max_pos[1], min_pos[2]], [max_pos[0], max_pos[1], min_pos[2]]])

        projected_corners = []
        # iterate over all corners
        for corner in np.unique(np.concatenate((x_corners, y_corners, z_corners, np.array([max_pos, min_pos]))), axis=0):
        # determine distance between corner and plane
        # we explicitly do not take the absolute, since we are only interested in the distance on the remaining side
            d = np.dot((corner - point), normal)
            # if the distance is greater zero then the corner lies on the remaining side
            if d > 0:
                projection = corner - d * normal
                projected_corners.append(projection)
        return np.array(projected_corners).reshape(-1,3)
    
    def cut(self,  point, normal, interpolation = "NN"):
        """
        Cut the grid data along a plane defined by a point on the plane and a normal. 
        
        Args:
            point (np.array): point on the plane
            normal (np.array): normal of the plane
            interpolation (np.array, optional): Defaults to "NN". The interpolation method to use. Can be "NN" (nearest neighbor) or "Lin/NN" linear and nearest neighbor. 
        """
        normal = np.array(normal/np.linalg.norm(normal)) # normalize normal vector
        point = np.array(point)
        
        # compute intersection with plane and projection of corners on plane
        intersections = self._get_plane_intersections( point, normal)
        projected_corners = self._get_projections_on_plane( point, normal)
        plane_coords = np.concatenate((intersections, projected_corners))
        
        from ..util.vector_functions import get_new_basis_vectors
        # Create a new 3 dimensional space. The basis contains the original normal and two orthogonal vectors to plane normal c1 and c2 
        # that minimize the area of the parallelogram containing all plane coordinates.The origin of the new space is t. 
        c1,t,c2 = get_new_basis_vectors(plane_coords, point, normal)
        # M_to converts a point from the new space to the original space and M_from converts a point from the original space to the new space
        M_to = np.stack((c1,c2,normal), axis = 1)
        M_from = np.linalg.inv(M_to)
        
        x = np.linspace(self.origin[0], self.origin[0] + self.step_sizes[0] * self.step_counts[0], self.step_counts[0] + 1)
        y = np.linspace(self.origin[1], self.origin[1] + self.step_sizes[1] * self.step_counts[1], self.step_counts[1] + 1)
        z = np.linspace(self.origin[2], self.origin[2] + self.step_sizes[2] * self.step_counts[2], self.step_counts[2] + 1)
        xx, yy, zz = np.meshgrid(x, y, z)
        old_positions_original_space = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
        # replace the previous line with the following line to run the two examples with .xyz
        #old_positions_original_space = np.array([yy.flatten(), xx.flatten(), zz.flatten()]).T
        
        # The original grid positions are converted into the new space and a point with maximal x,y, and z coordinate is constructed. This is the point
        # that defines the end of the new grid.
        old_positions = np.matmul(M_from, (np.array(old_positions_original_space.T) - np.array(t).reshape(-1,1))).T
        max_point = [np.max(old_positions[:,0]), np.max(old_positions[:,1]), np.max(old_positions[:,2])]
        old_values = self.values 
        
        # the new origin and step sizes are computed
        new_origin = [np.min(old_positions[:,0]), np.min(old_positions[:,1]), 0]
        new_step_count = np.ceil(np.array(np.array(max_point) - np.array(new_origin))/np.array(self.step_sizes))
        new_step_sizes = np.abs(np.array(np.array(max_point) - np.array(new_origin))/np.array(new_step_count))

        # with the new parameters the new grid points can be defined
        x = np.arange(new_origin[0], max_point[0] + new_step_sizes[0], step = new_step_sizes[0])
        y = np.arange(new_origin[1], max_point[1] + new_step_sizes[1], step = new_step_sizes[1])
        z = np.arange(new_origin[2], max_point[2] + new_step_sizes[2], step = new_step_sizes[2])
        xx, yy, zz = np.meshgrid(x, y, z)
        new_positions = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
        
        # check which new positions are inside the original grid and which are outside
        new_positions_original_space = np.array(np.matmul(M_to, new_positions.T) + np.array(t).reshape(-1,1)).T
        inner_positions_index = np.where((new_positions_original_space[:,0] < np.max(old_positions_original_space[:,0])) & (new_positions_original_space[:,0] > np.min(old_positions_original_space[:,0])) & 
                                         (new_positions_original_space[:,1] < np.max(old_positions_original_space[:,1])) & (new_positions_original_space[:,1] > np.min(old_positions_original_space[:,1])) & 
                                         (new_positions_original_space[:,2] < np.max(old_positions_original_space[:,2])) & (new_positions_original_space[:,2] > np.min(old_positions_original_space[:,2])))
        inner_positions = new_positions_original_space[inner_positions_index]
        outer_position_index = np.where((new_positions_original_space[:,0] > np.max(old_positions_original_space[:,0])) | (new_positions_original_space[:,0] < np.min(old_positions_original_space[:,0])) |
                                        (new_positions_original_space[:,1] > np.max(old_positions_original_space[:,1])) | (new_positions_original_space[:,1] < np.min(old_positions_original_space[:,1])) |
                                        (new_positions_original_space[:,2] > np.max(old_positions_original_space[:,2])) | (new_positions_original_space[:,2] < np.min(old_positions_original_space[:,2])))
        outer_positions = new_positions_original_space[outer_position_index]       
        
        # the new grid values are obtained by interpolating the old values
        new_values = np.zeros(len(new_positions))
        if interpolation == "Lin/NN":
            from scipy.interpolate import LinearNDInterpolator,NearestNDInterpolator
            interp = LinearNDInterpolator(old_positions_original_space, old_values)
            new_values_inner = interp(inner_positions)
            new_values[inner_positions_index] = new_values_inner
            interp = NearestNDInterpolator(old_positions_original_space, old_values)
            new_values_outer = interp(outer_positions)
            new_values[outer_position_index] = new_values_outer
        else:
            from scipy.interpolate import NearestNDInterpolator
            interp = NearestNDInterpolator(old_positions_original_space, old_values)
            new_values = interp(new_positions_original_space)
        
        # new_positions are the positions in unit vector space, new_values are the corresponding values which are obtained in the original space
        new_grid_data = GridData(new_values, new_positions, name="%s_cut"%self.name) 
        new_grid_data.A_to = np.stack((np.concatenate((c1, [0])),np.concatenate((c2, [0])),np.concatenate((normal,[0])), np.concatenate((t,[1]))), axis = 1)
        
        if self.is_loaded:
            new_grid_data.load()
        
        return new_grid_data

    def get_positions(self):
        x = np.linspace(self.origin[0], self.origin[0] + self.step_sizes[0] * self.step_counts[0], self.step_counts[0] + 1)
        y = np.linspace(self.origin[1], self.origin[1] + self.step_sizes[1] * self.step_counts[1], self.step_counts[1] + 1)
        z = np.linspace(self.origin[2], self.origin[2] + self.step_sizes[2] * self.step_counts[2], self.step_counts[2] + 1)
        xx, yy, zz = np.meshgrid(x, y, z)
        positions = np.array([yy.flatten(), xx.flatten(), zz.flatten()]).T
        return positions
    
    @property    
    def values(self):
        return self._values
    
    @values.setter
    def values(self, values):
        self._values = values
        self.is_loaded = False

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
cmd.set("volume_mode", 0)
cmd.set_object_ttt("{self.name}", {list(self.A_to.flatten())})
{self.name}_data = None

"""
        return result

    def load(self):
        from pymol import cmd
        from chempy.brick import Brick
        if not self.is_loaded:
            cmd.delete(self.name)
            values = self.values.reshape(self.step_counts.astype(int) + 1)
            brick = Brick.from_numpy(values, self.step_sizes, origin=self.origin)
            cmd.load_brick(brick, self.name)
            cmd.set("volume_mode", 0)
            cmd.set_object_ttt(self.name, list(self.A_to.flatten()))
            self.is_loaded = True

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