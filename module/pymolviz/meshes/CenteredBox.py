from . import Mesh
import numpy as np
class CenteredBox(Mesh):
    def __init__(self, center, extent, color = None, *args, **kwargs) -> None:
        """
            Create a rectangular box around given center with given extent.
            Use from_corners to create a box from two corners.

            Parameters:
                center : tuple, center of the box (x, y, z)
                extent : tuple, extent of the box (dx, dy, dz)
                color : str, color of the box
                *args, **kwargs : additional arguments to pass to Mesh

            Returns:
                Mesh
        """
        cx, cy, cz = center
        dx, dy, dz = extent
            # Define the eight vertices of the box
        vertices = np.array([
            [cx - dx / 2, cy - dy / 2, cz - dz / 2],
            [cx + dx / 2, cy - dy / 2, cz - dz / 2],
            [cx + dx / 2, cy + dy / 2, cz - dz / 2],
            [cx - dx / 2, cy + dy / 2, cz - dz / 2],
            [cx - dx / 2, cy - dy / 2, cz + dz / 2],
            [cx + dx / 2, cy - dy / 2, cz + dz / 2],
            [cx + dx / 2, cy + dy / 2, cz + dz / 2],
            [cx - dx / 2, cy + dy / 2, cz + dz / 2]
        ])

        # Define the faces by the indices of the vertices
        faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom face
        [4, 5, 6], [4, 6, 7],  # Top face
        [0, 1, 5], [0, 5, 4],  # Front face
        [2, 3, 7], [2, 7, 6],  # Back face
        [0, 3, 7], [0, 7, 4],  # Left face
        [1, 2, 6], [1, 6, 5]   # Right face
        ])

        normals = vertices / np.linalg.norm(vertices, axis=1)[:, np.newaxis]
        

        super().__init__(vertices, color, normals, faces, *args, **kwargs)


    def from_corners(corner1, corner2, color = "red", *args, **kwargs):
        """
            Create a box from two corners.

            Parameters:
                corner1 : tuple, first corner of the box (x, y, z) 
                corner2 : tuple, second corner of the box (x, y, z)
                color : str, color of the box
                *args, **kwargs : additional arguments to pass to Mesh

            Returns:
                Mesh
        """
        cx1, cy1, cz1 = corner1
        cx2, cy2, cz2 = corner2
        dx, dy, dz = (cx2 - cx1, cy2 - cy1, cz2 - cz1)
        #print(dx, dy, dz)
        center = ((cx1 + cx2) / 2, (cy1 + cy2) / 2, (cz1 + cz2) / 2)
        #print(center)
        return CenteredBox(center, (dx, dy, dz), color, *args, **kwargs)