import numpy as np

def get_perp(v):
    if np.allclose(v, 0):
        return np.array([0, 1, 0])
    v = v / np.linalg.norm(v)
    if np.allclose(np.abs(v), [0, 0, 1]):
        return np.array([0, 1, 0])
    cross = np.cross(v, [0, 0, 1])
    return cross / np.linalg.norm(cross)


def tanh_distance_weighting(x_offset, scale):
    """Returns a function, that can be used as a weighting function for IrregularData.interpolate.
    The function is negative tanh between x values of 0 and 1 rescaled between approximately 0 and 1.
    """
    def distance_weigting(distances):
        distances = (distances - x_offset) * scale
        return (-np.tanh((distances * 4  - 2)) + 1) / 2
    return distance_weigting