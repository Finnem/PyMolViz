def get_orthogonal_vectors(x):
    import numpy as np
    """Computes two orthogonal vectors to a given vector.
    
    Args:
        x (np.array): input vector
    """
    
    # get basis for 2d plane coordinate system (using Gram-Schmidt procedure)
    u = np.random.randn(3)  # random vector that lies on the plane
    u -= u.dot(x) * x       # make it orthogonal to n0
    u /= np.linalg.norm(u)  # normalize it

    v = np.cross(x, u) # get third orthogonal vector
    # u and v now span the plane, so all points on the plane can be expressed using u and v

    return u,v

def transform_2d_to_3d(coord_list, point, u,v):
    """Transforms 2D coordinates to 3D coordinates given plane parameters.
    
    Args:
        coord_list (np.array): list of 2D coordinates that should be transformed
        point (np.array): support vector of the plane
        u (np.array): first vector spanning the plane
        v (np.array): second vector spanning the plane
    """

    return [point + coord[0] * u + coord[1] * v for coord in coord_list]

    
def transform_3d_to_2d(coord_list, point, u,v):
    """Transforms 3D coordinates to 2D coordinates given plane parameters.
    
    Args:
        coord_list(np.array): list of 3D coordinates that should be transformed
        point(np.array): support vector of the plane
        u(np.array): first vector spanning the plane
        v(np.array): second vector spanning the plane
    """
    import numpy as np

    return np.array([np.linalg.lstsq(np.array([u,v]).T,coord-point, rcond=None)[0] for coord in coord_list])


def minimum_bounding_parallelogram(points):
    """Finds the smallest bounding parallelogram for a set of 2D points.
    
    Args:
        points (np.array): an Nx2 matrix of coordinates
    
    Returns a set of points representing three corners of the bounding box.
    """
    
    import numpy as np
    from scipy.spatial import ConvexHull
    
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]
    
    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]
    edges = np.append(np.array(edges), [hull_points[0]-hull_points[len(hull_points)-1]], axis = 0)
    
    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    
    # get angle to x-axis
    angles = np.abs(np.mod(angles, np.pi))
    
    # find rotation matrices for each angle
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),  # = sin(angles)
        np.cos(angles+pi2),  # =-sin(angles)
        np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2)) # first dimension: rotations, second dimension: x/y dimension, third dimension: points
    
    # apply rotations to the hull points
    rot_points = np.dot(rotations, hull_points.T)
 
    list_spanning_points = []
    areas = []
    for i,rotation in enumerate(rot_points):
        rotation = rotation.T
        if i == len(rotation) - 2:
            end1 = rotation[-2]
            vertex = rotation[-1]
            end2 = rotation[0]
        elif i == len(rotation) - 1:
            end1 = rotation[-1]
            vertex = rotation[0]
            end2 = rotation[1]
        else:
            end1 = rotation[i]
            vertex = rotation[i+1]
            end2 = rotation[i+2]
        e1 = end1 - vertex
        e2 = end2 - vertex
        max_e2 = 0
        max_e1 = 0
        for edge in rotation:
            a,b = np.linalg.solve(np.array([e1,e2]).T,(edge - vertex))
            if b > max_e2:
                max_e2 = b
            if a > max_e1:
                max_e1 = a
        
        h = end2[1] - vertex[1]
        areas.append(max_e1 * h)
        list_spanning_points.append([vertex + np.dot(e1, max_e1), vertex, vertex + np.dot(e2, max_e2)])
    
    best_spanning_points = np.array(list_spanning_points[np.argmin(areas)])
    best_rotation = rotations[np.argmin(areas)]
    
    original_spanning_points = np.dot(best_spanning_points, best_rotation)
    
    return original_spanning_points

def get_new_basis_vectors(plane_coords,  point, normal):
    import numpy as np
    u,v = get_orthogonal_vectors(normal)
    plane_coords_2d = transform_3d_to_2d(plane_coords, point, u,v)
    
    bounding_rectangle_2d = minimum_bounding_parallelogram(plane_coords_2d)
    bounding_rectangle_3d = transform_2d_to_3d(bounding_rectangle_2d, point, u,v)
    t = bounding_rectangle_3d[1]
    max_c1 = bounding_rectangle_3d[0] - bounding_rectangle_3d[1]
    max_c2 = bounding_rectangle_3d[2] - bounding_rectangle_3d[1]
    distance_c1 = np.linalg.norm(max_c1)
    distance_c2 = np.linalg.norm(max_c2)
    c1 = max_c1/distance_c1
    c2 = max_c2/distance_c2
    
    return c1,t,c2