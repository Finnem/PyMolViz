
from ..meshes import CGOMolecule
from ..PyMOLobjects import Expressions
import numpy as np
from ..util.sanitize import sanitize_pymol_string

def ranked_highlighting(poses, scores, name = "", by_rank = False, top_percentile = 0.1, as_molecules=False, *args, **kwargs):
    """
    Generates molecule visualizations where the opacity is proportional to the score.

    Args:
        poses (list): A list of poses.
        scores (list): A list of scores.
        by_rank (bool): If True, instead of using the score, the rank of the score is used.

    Returns:
        list: A list of molecule visualizations.
    """

    scores = - np.array(scores)
    if by_rank:
        scores = np.argsort(scores)


    # get top percentile
    if top_percentile < 1:
        top_percentile = int(top_percentile * len(poses))
    else:
        top_percentile = int(top_percentile)

    poses = poses[:top_percentile]
    scores = scores[:top_percentile]

    max_score = np.max(scores)
    min_score = np.min(scores)

    molecule_visualizations = []
    transparencies = []
    for i, pose in enumerate(poses):
        transparency = (scores[i] - min_score) / (max_score - min_score)
        transparencies.append(transparency)
        if as_molecules:
            molecule_visualizations.append(pose)
        else:
            cgo_mol = CGOMolecule.from_rdkit_molecule(pose, name = f"{name}_{i}", linewidth = 1, *args, **kwargs)
            molecule_visualizations.append(cgo_mol)
    
    transparencies = np.array(transparencies)
    # exp scale transparency
    transparencies = 20**transparencies

    # normalize transparency
    max_transparency = np.max(transparencies)
    transparencies /= max_transparency
    transparencies = 1 - transparencies
    if as_molecules:
        from xbpy import rdutil
        rdutil.write_molecules(molecule_visualizations, f"{name}.sdf")
        molecule_visualizations = Expressions([f"{sanitize_pymol_string(name)} and state {i + 1}" for i in range(len(molecule_visualizations))], transparencies = transparencies)
    else:
        for mol, transparency in zip(molecule_visualizations, transparencies):
            mol.transparency = transparency


    return molecule_visualizations
        