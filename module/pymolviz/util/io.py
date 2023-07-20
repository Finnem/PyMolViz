import numpy as np


def grid_from_xyz(path, in_bohr = True, *args, **kwargs):
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

        return GridData(values, coords, *args, **kwargs)


def grid_from_mtz(path, factor_column = "FWT", phase_column = "PHWT", sample_rate = 2.6, min_pos = [0, 0, 0], max_pos = [1, 1, 1], step_size = [1., 1., 1.], *args, **kwargs):
    import gemmi
    from ..volumetric.GridData import GridData
    mtz = gemmi.read_mtz_file(path)
    map = mtz.transform_f_phi_to_map(factor_column, phase_column, sample_rate = sample_rate)
    step_size = np.array(step_size)
    m = gemmi.Mat33()
    m.fromlist([[step_size[0],0.,0.],[0.,step_size[1],0.],[0.,0.,step_size[2]]])
    transform = gemmi.Transform(m, gemmi.Vec3(*min_pos))
    values = np.zeros(np.ceil((max_pos - min_pos) / step_size).astype(int), dtype = np.float32)
    map.interpolate_values(values, transform)
    positions = (np.array(m.tolist()).reshape(3,3) @ np.indices(values.shape).reshape(3,-1)).T + np.array(min_pos)
    return GridData(values.flatten(), positions, *args, **kwargs)
            
