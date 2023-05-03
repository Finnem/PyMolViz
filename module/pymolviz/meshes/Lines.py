from __future__ import annotations

import numpy as np
from .Mesh import Mesh

class Lines(Mesh):
    """ Class to store all relevant information required to create a CGO Line object.
    
    
    Attributes:
        vertices (np.array): A 3xN array of vertices. Each line is defined by two vertices. Each beginning and end of a line should directly follow each other: [start1, end1, start2, end2, start3, end3] 
        color (np.array): A 3xN array of colors. May have a color for each line or each vertex.
        transformation (np.array): A 4x4 transformation matrix.
    """

    def __init__(self, lines : np.array, color : np.array = None, transformation : np.array = None, linewidth = 1, **kwargs) -> None:
        if type(lines) == list:
            lines = np.array(lines)
        if not color is None:
            if type(color) == list:
                color = np.array(color)
            if (type(color) is np.ndarray) and color.shape == () and color.dtype.kind in ["U", "S"]:
                color = str(color)
            if (not type(color) in [str, np.str_]) and (type(color) in [list, np.ndarray]) and  (len(color) == (len(lines.reshape(-1, 3)) / 2)):
                color = np.repeat(color, 2, axis = 0)
            else:
                color = color
        super().__init__(lines.reshape(-1, 3), color, None, None, transformation, **kwargs)
        self.linewidth = linewidth

    def as_dashed(self, visible_portion = .5, dash_length = .25, width = .05, resolution = 100):
        from .Cylinder import Cylinder
        lines = self.vertices.reshape(-1, 6)
        colors = self.color[::2]
        cylinders = []
        for line, color in zip(lines, colors):
            start = line[:3]
            end = line[3:]
            direction = end - start
            length = np.linalg.norm(direction)
            direction = direction / length
            num_cylinders = int(length * visible_portion / dash_length)
            spacing = ((1 - visible_portion) * dash_length) * 2
            cur_start_point = start + direction * spacing / 2
            if num_cylinders <= 1:
                cylinders.append(Cylinder(cur_start_point, end - direction * spacing / 2, width, color, resolution=resolution ))
            else:
                while (cur_start_point @ direction) < ((end - direction * spacing / 2) @ direction):
                    cur_end_point = cur_start_point + direction * dash_length
                    cylinders.append(Cylinder(cur_start_point, cur_end_point, width, color, .05,resolution=resolution ))
                    cur_start_point += direction * (dash_length + spacing)
        return Mesh.combine(cylinders)

    def angle_mesh(self, name = None, visible_portion = .5, dash_length = 0.174533, width = .05, resolution = 100, colors = None, line_color = None, **kwargs):
        """ Creates a mesh collection that shows the angle between pairwise lines.
        
        Args:
            name (str, optional): The name of the mesh. Defaults to None.
            visible_portion (float, optional): The portion of the line that is visible. Defaults to .5.
            dash_length (float, optional): The length of the dashes in radians. Defaults to 30°.
            width (float, optional): The width of the dashes. Defaults to .05.
            resolution (int, optional): The resolution of the cylinders. Defaults to 100.
            colors (np.array, optional): The colors of the cylinders. Defaults to color of first point of each line pair.
            line_color (np.array, optional): The color of the lines. Defaults to color of the lines.

            Returns:
                Mesh: A mesh that shows the angle between pairwise lines.
        
        """
        from .Cylinder import Cylinder
        from ..Collection import Collection

        lines = self.vertices.reshape(-1, 12)
        if colors is None:
            colors = self.color[::4]
        if line_color is None:
            line_color = self.color

        meshes = []
        for i, (line_pair, color) in enumerate(zip(lines, colors)):
            c1 = line_color[i*4]; c2 = line_color[i*4+1]; c3 = line_color[i*4+2]; c4 = line_color[i*4+3]
            start1 = line_pair[:3]
            end1 = line_pair[3:6]
            start2 = line_pair[6:9]
            end2 = line_pair[9:]
            print(line_pair)
            
            display_position = (start1 + start2) / 2
            direction1 = end1 - start1; unit1 = direction1 / np.linalg.norm(direction1)
            direction2 = end2 - start2; unit2 = direction2 / np.linalg.norm(direction2)
            ortho = np.cross(unit1, unit2); ortho = ortho / np.linalg.norm(ortho)
            ortho = -np.cross(unit1, ortho); ortho = ortho / np.linalg.norm(ortho)
            angle = np.arccos(unit1 @ unit2)
        
            spacing = ((1 - visible_portion) * dash_length) * 2
            cur_start_angle = 0 + spacing/2
            num_cylinders = int(angle * visible_portion / dash_length)
            cylinders = []
            if num_cylinders <= 1:
                end_angle = angle - spacing/2
                start_point = display_position + np.cos(cur_start_angle) * unit1 + np.sin(cur_start_angle) * ortho
                end_point = display_position + np.cos(end_angle) * unit1 + np.sin(end_angle) * ortho
                cylinders.append(Cylinder(start_point, end_point, width, color, resolution=resolution, **kwargs))
            else:
                while cur_start_angle < (angle - spacing/2):
                    end_angle = cur_start_angle + dash_length
                    start_point = display_position + np.cos(cur_start_angle) * unit1 + np.sin(cur_start_angle) * ortho
                    end_point = display_position + np.cos(end_angle) * unit1 + np.sin(end_angle) * ortho
                    cylinders.append(Cylinder(start_point, end_point, width, color, resolution=resolution, **kwargs))
                    cur_start_angle += dash_length + spacing
            
            angle_display = Mesh.combine(cylinders)
            if np.allclose(start1, start2):
                lines_display = Mesh.combine([Cylinder(display_position, direction1 + display_position, width, [c2, c1], resolution=resolution),
                                                Cylinder(display_position, direction2 + display_position, width, [c4, c3], resolution=resolution)])
                meshes.extend([angle_display, lines_display])
            else:
                line_display = Cylinder(start1, start2, width/2, color, resolution=resolution, **kwargs)
                proj_display = Lines([end1, direction1 + display_position, end2, direction2 + display_position], color, width/2, **kwargs)
                proj_display = proj_display.as_dashed(width = width/2)
                lines_display = Mesh.combine([Cylinder(display_position, direction1 + display_position, width, [c2, c1], resolution=resolution),
                                                Cylinder(display_position, direction2 + display_position, width, [c4, c3], resolution=resolution)])
                meshes.extend([angle_display, line_display, lines_display, proj_display])

        return Collection(meshes, name=name)
            



    def _create_CGO(self) -> str:
        """ Creates a CGO list from the mesh information. A line mesh will be created.
            CGO constants are kept as strings to avoid importing the pymol module.
        
        Returns:
            None
        """

        vertices = self.vertices @ self.transformation[:3,:3].T + self.transformation[:3,3]

        cgo_list = []
        
        cgo_list.extend(["LINEWIDTH", self.linewidth])

        cgo_list.extend(["BEGIN", "LINES"])


        cgo_vertices = vertices
        cgo_colors = self.color

        #vertices
        triangles = np.hstack([
            np.full(cgo_colors.shape[0], "COLOR")[:,None], cgo_colors, \
            np.full(cgo_vertices.shape[0], "VERTEX")[:,None], cgo_vertices, \
            ]).flatten()
        cgo_list.extend(triangles)

        # ending
        cgo_list.append("END")

        return cgo_list

    def from_o3d_lineset_mesh(o3d_mesh) -> Lines:
        """ Creates a Mesh object from an Open3D triangle mesh.
        
        Args:
            o3d_mesh (o3d.geometry.LineSet): An Open3D lineset.
        
        Returns:
            Mesh: A Mesh object.
        """
        vertices = np.asarray(o3d_mesh.vertices)
        color = np.asarray(o3d_mesh.vertex_colors)
        lines = np.asarray(o3d_mesh.lines_normals)
        vertices = vertices[lines].reshape(-1, 3)

        return Lines(vertices, color)

    

    def combine(lines):
        """ Combines multiple lines into one.
        
        Args:
            lines (list): A list of lines.
        
        Returns:
            Lines: A Lines object.
        """
        vertices = np.vstack([line.vertices for line in lines])
        colors = np.vstack([line.color for line in lines])
        return Lines(vertices, colors)

