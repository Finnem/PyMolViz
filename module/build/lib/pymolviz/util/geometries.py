import numpy as np
from scipy.spatial.transform import Rotation

# Create a sphere mesh
def get_sphere_mesh(position, radius = 1, resolution = 10):
    # Create a sphere mesh
    u, v = np.mgrid[0:2*np.pi:resolution*1j, 0:np.pi:resolution*1j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    vertices = np.vstack((x, y, z)).T * radius
    normals = -vertices / np.linalg.norm(vertices, axis=1)[:, None]
    faces = []
    for i in range(1, resolution):
        for j in range(1, resolution):
            faces.append([i*resolution+j-1, i*resolution+j, (i-1)*resolution+j-1])
            faces.append([i*resolution+j, (i-1)*resolution+j, (i-1)*resolution+j-1])
    faces = np.array(faces)
    return {"positions" : vertices + position, "faces" : faces, "normals" : normals}


def generate_circle_points(samples):
    angles = np.pi * 2 * np.arange(samples) / samples
    return np.array([np.cos(angles), np.sin(angles)]).T


def generate_cone(length, resolution, thickness):
    top_vertex = np.array([0, 0, length])
    bottom = generate_circle_points(resolution) * thickness
    bottom = np.hstack([bottom, np.full(bottom.shape[0], 0)[:,None]])
    faces = []
    for i in range(resolution):
        faces.append([(i % resolution) + 1, ((i + 1) % resolution) + 1, 0])
    faces = np.array(faces)
    vertices = np.vstack([top_vertex[None, :], bottom])
    return {"positions" : vertices, "faces" : faces}


def generate_cylinder(length, resolution, thickness, curvature):
    top_vertices = generate_circle_points(resolution) * thickness
    top_center = np.array([0, 0, length], dtype=np.float32)
    bottom_vertices = generate_circle_points(resolution) * thickness
    bottom_center = np.array([0, 0, 0], dtype=np.float32)
    top_vertices = np.hstack([top_vertices, np.full(top_vertices.shape[0], length)[:,None]])
    bottom_vertices = np.hstack([bottom_vertices, np.full(bottom_vertices.shape[0], 0)[:,None]])
    bottom_center[2] -= float(curvature * length)
    top_center[2] += float(curvature * length)
    faces = []
    for i in range(resolution):
        offset = resolution + 2
        faces.append([(i % resolution) + 1, 0,  ((i + 1) % resolution) + 1])
        faces.append([((i + 1) % resolution) + offset,  resolution + 1, (i % resolution) + offset])
        faces.append([(i % resolution) + 1, ((i + 1) % resolution) + 1, (i % resolution) + offset])
        faces.append([((i + 1) % resolution) + 1, ((i + 1) % resolution) + offset, (i % resolution) + offset])
    faces = np.array(faces)
    vertices = np.vstack([top_center[None,:], top_vertices, bottom_center[None,:], bottom_vertices])
    center = np.array([0, 0, length/2])
    #normals = np.vstack([np.array([0, 0, 1])[None, :],  top_vertices - top_center, np.array([0, 0, -1])[None, :], bottom_vertices - bottom_center])
    normals = vertices - center
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]
    return {"vertices" : vertices, "normals" : normals, "faces" : faces}

def get_arrow_mesh(position, direction = [0, 0, 1], length = 2, resolution = 10, thickness = .25):
    # Create a mesh of an 3d arrow

    # Create a cone
    cone = generate_cone(length * 0.4, resolution, thickness)
    cone["positions"][:, 2] += length * 0.6
    cylinder = generate_cylinder(length * 0.6, resolution, thickness/1.5)
    vertices = np.vstack([cone["positions"], cylinder["positions"]])
    faces = np.vstack([cone["faces"], cylinder["faces"] + len(cone["positions"])])
    # Rotate the arrow
    direction = direction / np.linalg.norm(direction)
    if not np.allclose(direction, [0, 0, 1]):
        rotation_vector = np.array([direction[1], -direction[0], 0])
        rotation_vector /= np.linalg.norm(rotation_vector)
        new_length = -np.arccos(direction[2])
        rotation_vector = rotation_vector * new_length
        rotation = Rotation.from_rotvec(rotation_vector)
        vertices = rotation.apply(vertices)

    # Translate the arrow
    vertices += position
    return {"positions" : vertices, "faces" : faces}
    

# Reconstructs surface from a set of points using poisson reconstruction from open3d
def get_surface_from_points(points, normals, colors):
    from pymolviz import Mesh
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    mesh.compute_vertex_normals()
    return Mesh(np.asarray(mesh.vertices), faces = np.asarray(mesh.triangles),\
         normals = np.asarray(mesh.vertex_normals), color = np.asarray(mesh.vertex_colors))

