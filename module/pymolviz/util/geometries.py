import math

import numpy as np
from scipy.spatial.transform import Rotation


def point_on_sphere(center, radius, theta, phi):
    """Spherical coordinates → Cartesian point on a sphere."""
    cx, cy, cz = center
    st = math.sin(theta)
    return (
        cx + radius * st * math.cos(phi),
        cy + radius * st * math.sin(phi),
        cz + radius * math.cos(theta),
    )


def _normalize3(vec):
    x, y, z = (float(vec[0]), float(vec[1]), float(vec[2]))
    length = math.sqrt(x * x + y * y + z * z)
    if length < 1e-12:
        return (0.0, 0.0, 1.0)
    return (x / length, y / length, z / length)


def _cross3(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _outward_face(verts, i, j, k):
    v0, v1, v2 = verts[i], verts[j], verts[k]
    normal = _cross3(
        (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]),
        (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]),
    )
    center = (
        (v0[0] + v1[0] + v2[0]) / 3.0,
        (v0[1] + v1[1] + v2[1]) / 3.0,
        (v0[2] + v1[2] + v2[2]) / 3.0,
    )
    if _dot3(normal, center) < 0.0:
        return (i, k, j)
    return (i, j, k)


_GEODESIC_CACHE = {}

_ICOSAHEDRON_VERTS = None
_ICOSAHEDRON_FACES = None


def _icosahedron():
    global _ICOSAHEDRON_VERTS, _ICOSAHEDRON_FACES
    if _ICOSAHEDRON_VERTS is not None:
        return _ICOSAHEDRON_VERTS, _ICOSAHEDRON_FACES
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = [
        _normalize3(v) for v in (
            (-1.0, phi, 0.0), (1.0, phi, 0.0), (-1.0, -phi, 0.0), (1.0, -phi, 0.0),
            (0.0, -1.0, phi), (0.0, 1.0, phi), (0.0, -1.0, -phi), (0.0, 1.0, -phi),
            (phi, 0.0, -1.0), (phi, 0.0, 1.0), (-phi, 0.0, -1.0), (-phi, 0.0, 1.0),
        )
    ]
    faces = [
        _outward_face(verts, *face) for face in (
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        )
    ]
    _ICOSAHEDRON_VERTS = verts
    _ICOSAHEDRON_FACES = faces
    return verts, faces


def geodesic_icosphere(frequency=4):
    """Class-I geodesic sphere. Faces = 20 * frequency².

    frequency 2/3/4/6/8 → 80 / 180 / 320 / 720 / 1280 triangles.
    Returns (vertices, faces, edges) on the unit sphere, outward winding.
    """
    freq = max(1, int(frequency))
    cached = _GEODESIC_CACHE.get(freq)
    if cached is not None:
        return cached

    base_verts, base_faces = _icosahedron()
    verts = []
    index_of = {}

    def add_vert(vec):
        point = _normalize3(vec)
        key = (round(point[0], 7), round(point[1], 7), round(point[2], 7))
        existing = index_of.get(key)
        if existing is not None:
            return existing
        idx = len(verts)
        verts.append(point)
        index_of[key] = idx
        return idx

    faces = []
    inv = 1.0 / float(freq)
    for ia, ib, ic in base_faces:
        a, b, c = base_verts[ia], base_verts[ib], base_verts[ic]
        grid = {}
        for p in range(freq + 1):
            for q in range(freq + 1 - p):
                t = p * inv
                s = q * inv
                r = 1.0 - t - s
                grid[(p, q)] = add_vert((
                    r * a[0] + t * b[0] + s * c[0],
                    r * a[1] + t * b[1] + s * c[1],
                    r * a[2] + t * b[2] + s * c[2],
                ))
        for p in range(freq):
            for q in range(freq - p):
                v00 = grid[(p, q)]
                v10 = grid[(p + 1, q)]
                v01 = grid[(p, q + 1)]
                faces.append(_outward_face(verts, v00, v10, v01))
                if p + q + 1 < freq:
                    v11 = grid[(p + 1, q + 1)]
                    faces.append(_outward_face(verts, v10, v11, v01))

    edges = set()
    for i, j, k in faces:
        edges.add((i, j) if i < j else (j, i))
        edges.add((j, k) if j < k else (k, j))
        edges.add((k, i) if k < i else (i, k))

    result = (
        np.asarray(verts, dtype=float),
        np.asarray(faces, dtype=int),
        np.asarray(sorted(edges), dtype=int),
    )
    _GEODESIC_CACHE[freq] = result
    return result


def icosphere(subdivisions=2, frequency=None):
    """Unit geodesic sphere. ``frequency`` wins; else frequency = 2**subdivisions."""
    if frequency is None:
        frequency = 2 ** max(0, int(subdivisions))
    return geodesic_icosphere(frequency)


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

