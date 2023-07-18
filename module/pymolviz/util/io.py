import numpy as np


def grid_from_xyz(path, name = None, in_bohr = True):
    """
    Reads in a grid from an xyz file as output by turbomole.
    Assumes lines starting with # are comments and that the value is in the last column.
    """

    from ..volumetric.GridData import GridData
    with open(path, "r") as f:
        coords = []
        values = []
        for line in f.readlines():
            if line.startswith("#"):
                continue
            line = line.split()
            if len(line) == 4:
                coords.append([float(line[0]), float(line[1]), float(line[2])])
                values.append(float(line[-1]))

        coords = np.array(coords)
        values = np.array(values)
        if in_bohr:
            coords /= 1.89

        return GridData(values, coords, name = name)


def grids_from_mtz(path):
        pass
            
