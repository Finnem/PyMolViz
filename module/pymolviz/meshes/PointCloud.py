from __future__ import annotations

import numpy as np
import logging
from .Mesh import Mesh

class PointCloud(Mesh):
    """ Class to store points and associated colors which can be displayed as a point cloud.
    
    
    Attributes:
        vertices (np.array): A 3xN array of vertices.
        color (np.array): A 3xN array of colors.
        transformation (np.array): A 4x4 transformation matrix.
    """

    def __init__(self, vertices : np.array, color : np.array = None, normals = None, transformation : np.array = None, render_as_spheres = False, sphere_radius = .1, **kwargs) -> None:
        super().__init__(vertices, color, normals, None, transformation, **kwargs)
        self.render_as_spheres = render_as_spheres
        self.sphere_radius = sphere_radius

    def create_CGO(self) -> str:
        """ Creates a CGO list from the mesh information. Points can be displayed as spheres or as points.

        Args:
            as_spheres (bool, optional): If True, the points are displayed as spheres. Defaults to False.
            radius (float, optional): Only relevant is as_spheres is True. The radius of the spheres. Defaults to .1.
        
        Returns:
            None
        """

        vertices = self.vertices @ self.transformation[:3,:3].T + self.transformation[:3,3]

        cgo_points = vertices
        cgo_colors = self.color

        cgo_list = []
        
        
        if self.render_as_spheres:
            #vertices
            point_meshes = np.hstack([
                np.full(cgo_points.shape[0], "COLOR")[:,None], cgo_colors, \
                np.full(cgo_points.shape[0], "SPHERE")[:,None], cgo_points, \
                np.full(cgo_points.shape[0], self.sphere_radius)[:,None], \
                ]).flatten()
            cgo_list.extend(point_meshes)

        else:
            cgo_list.extend(["BEGIN", "POINTS"])
            #vertices
            points = np.hstack([
                np.full(cgo_points.shape[0], "COLOR")[:,None], cgo_colors, \
                np.full(cgo_points.shape[0], "VERTEX")[:,None], cgo_points, \
                ]).flatten()
            cgo_list.extend(points)
            # ending
            cgo_list.append("END")

        return cgo_list

    def from_o3d_point_cloud(o3d_mesh) -> PointCloud:
        """ Creates a Mesh object from an Open3D triangle mesh.
        
        Args:
            o3d_mesh (o3d.geometry.TriangleMesh): An Open3D triangle mesh.
        
        Returns:
            Mesh: A Mesh object.
        """
        vertices = np.asarray(o3d_mesh.points)
        color = np.asarray(o3d_mesh.colors)
        if not o3d_mesh.has_normals():
            logging.info("No normals found in point cloud. Computing normals.")
            o3d_mesh.estimate_normals()
        normals = np.asarray(o3d_mesh.normals)

        return PointCloud(vertices, color, normals)

    
