from __future__ import annotations

import logging
import numpy as np
from .Lines import Lines

class CGOMolecule(Lines):
    """ Class representing a molecule in CGO Format. Can be used to create a CGO object in PyMol.
      Representing Molecules as CGO objects is significantly faster than as complete molecules, however their usability is limited.
      They can be used for coloring, but not for selection or other operations.
    """
    def __init__(self, atom_coordinates, atom_types, adjacency_matrix, name = None, state = 1, transparency = 0, linewidth = 1, render_as = "lines", ho_spread = 0.1, element_colors = None, scale = 1, scale_center = "origin",*args, **kwargs):
        self.atom_coordinates = atom_coordinates
        self.adjacency_matrix = adjacency_matrix
        self.atom_types = atom_types
        if element_colors is not None:
            for element, color in element_colors.items():
                self.atom_types = [color if atom_type == element else atom_type for atom_type in self.atom_types]
        self.ho_spread = ho_spread
        self.atom_types = np.array(self.atom_types, dtype=object)

        bond_coordinates = []
        colors = []
        for bond_index in np.argwhere(np.tril(adjacency_matrix) != 0):
            order = int(np.ceil(adjacency_matrix[bond_index[0], bond_index[1]]))
            bond_coordinates.extend(self.draw_bond(bond_index, order))
            colors.extend([self.atom_types[bond_index[0]], self.atom_types[bond_index[1]]] * order)
        self.coordinates = np.array(bond_coordinates)

        self.scale(scale, scale_center)

        super().__init__(self.coordinates, color = colors, name = name, state = 1, transparency = 0, colormap = None, linewidth=linewidth, render_as=render_as, *args, **kwargs)


    def draw_bond(self, bind_index, bond_order):
        """ Returns starting and end points for the given bond.

        Args:
            bind_index (int): Index of the bond.
            bond_order (int): Order of the bond.

        Returns:
            tuple: A tuple containing the starting and end points of the bond.
        
        """

        start_atom_coordinates = self.atom_coordinates[bind_index[0]]
        end_atom_coordinates = self.atom_coordinates[bind_index[1]]

        if bond_order == 1:
            return np.array([start_atom_coordinates, end_atom_coordinates])

        bond_direction = end_atom_coordinates - start_atom_coordinates; bond_direction /= np.linalg.norm(bond_direction)


        start_mask = np.ones(self.atom_coordinates.shape[0], dtype=bool); start_mask[bind_index[1]] = False
        start_neighbors_directions = self.atom_coordinates[(self.adjacency_matrix[bind_index[0]] != 0) & start_mask] - start_atom_coordinates; start_neighbors_directions /= np.linalg.norm(start_neighbors_directions, axis=1)[:, np.newaxis]
        end_mask = np.ones(self.atom_coordinates.shape[0], dtype=bool); end_mask[bind_index[0]] = False
        end_neighbors_directions = self.atom_coordinates[(self.adjacency_matrix[bind_index[1]] != 0) & end_mask] - end_atom_coordinates; end_neighbors_directions /= np.linalg.norm(end_neighbors_directions, axis=1)[:, np.newaxis]

        if (len(start_neighbors_directions) == 2) or ((len(end_neighbors_directions) != 2) and len(start_neighbors_directions) > 1):
            used_neighbors_plane_projection = start_neighbors_directions - np.sum(start_neighbors_directions * bond_direction, axis=1)[:, np.newaxis] * bond_direction
            # get largest principal component of the plane projection
            used_neighbors_plane_projection = np.linalg.svd(used_neighbors_plane_projection)[2][0]
            used_neighbors_plane_projection /= np.linalg.norm(used_neighbors_plane_projection)
        else:
            used_neighbors_plane_projection = end_neighbors_directions - np.sum(end_neighbors_directions * bond_direction, axis=1)[:, np.newaxis] * bond_direction
            # get largest principal component of the plane projection
            used_neighbors_plane_projection = np.linalg.svd(used_neighbors_plane_projection)[2][0]
            used_neighbors_plane_projection /= np.linalg.norm(used_neighbors_plane_projection)
        
        coordinates = []
        for offset in np.linspace(-self.ho_spread, self.ho_spread, bond_order):
            coordinates.append(start_atom_coordinates + used_neighbors_plane_projection * offset)
            coordinates.append(end_atom_coordinates + used_neighbors_plane_projection * offset)
        return np.array(coordinates)

    def scale(self, scale, scale_center):
        """ Scales the molecule.

        Args:
            scale (float): The scale factor.
            scale_center (str): The center of the scaling. Can be "origin" or "center".
        
        """
        if scale_center == "center":
            center = np.mean(self.atom_coordinates, axis=0)
        elif scale_center == "origin":
            center = np.zeros(3)
        else:
            center = scale_center
        self.atom_coordinates = (self.atom_coordinates - center) * scale + center
        self.coordinates = (self.coordinates - center) * scale + center

    def from_rdkit_molecule(rdkit_mol, *args, **kwargs):
        """ Creates a CGOMolecule from an rdkit molecule.

        Args:
            rdkit_mol (rdkit.Chem.rdchem.Mol): An rdkit molecule.

        Returns:
            CGOMolecule: A CGOMolecule object.
        
        """
        from rdkit import Chem
        atom_coordinates = np.array(rdkit_mol.GetConformer().GetPositions())
        atom_types = np.array([atom.GetSymbol() for atom in rdkit_mol.GetAtoms()])
        adjacency_matrix = np.array(Chem.GetAdjacencyMatrix(rdkit_mol, useBO=True))
        return CGOMolecule(atom_coordinates, atom_types, adjacency_matrix, *args, **kwargs)