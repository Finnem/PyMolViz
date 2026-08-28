"""CGO construction helpers (requires pymol.cgo)."""

import math

from pymol.cgo import CYLINDER

from .geometries import icosphere, point_on_sphere

_CGO_TOKEN_NAMES = (
    "POINTS", "SPHERE", "COLOR", "VERTEX", "NORMAL", "CYLINDER", "CONE",
    "BEGIN", "END", "LINEWIDTH", "LINES", "TRIANGLES", "ALPHA",
)


def resolve_cgo_tokens(content: list) -> list:
    """Convert string CGO opcodes to pymol.cgo integer constants."""
    from pymol import cgo

    map_cgo_keys = {name: getattr(cgo, name) for name in _CGO_TOKEN_NAMES if hasattr(cgo, name)}
    out = []
    for entry in content:
        if isinstance(entry, str):
            if entry not in map_cgo_keys:
                raise KeyError("Unknown CGO token %r" % entry)
            out.append(map_cgo_keys[entry])
            continue
        if isinstance(entry, bool):
            raise TypeError("Unexpected bool in CGO list: %r" % entry)
        if isinstance(entry, int) and entry in map_cgo_keys.values():
            out.append(entry)
            continue
        try:
            out.append(float(entry))
        except (TypeError, ValueError):
            if entry not in map_cgo_keys and hasattr(cgo, entry):
                map_cgo_keys[entry] = getattr(cgo, entry)
            if entry not in map_cgo_keys:
                raise KeyError("Unknown CGO token %r" % entry)
            out.append(map_cgo_keys[entry])
    return out

_BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

_BOX_FACES = (
    ((0, 2, 1), (0.0, 0.0, -1.0)),
    ((0, 3, 2), (0.0, 0.0, -1.0)),
    ((4, 5, 6), (0.0, 0.0, 1.0)),
    ((4, 6, 7), (0.0, 0.0, 1.0)),
    ((0, 1, 5), (0.0, -1.0, 0.0)),
    ((0, 5, 4), (0.0, -1.0, 0.0)),
    ((2, 3, 7), (0.0, 1.0, 0.0)),
    ((2, 7, 6), (0.0, 1.0, 0.0)),
    ((0, 3, 7), (-1.0, 0.0, 0.0)),
    ((0, 7, 4), (-1.0, 0.0, 0.0)),
    ((1, 2, 6), (1.0, 0.0, 0.0)),
    ((1, 6, 5), (1.0, 0.0, 0.0)),
)


def _color_alpha_prefix(color, alpha=1.0):
    red, green, blue = [float(c) for c in color]
    a = max(0.0, min(1.0, float(alpha)))
    if a >= 1.0 - 1e-6:
        return ["COLOR", red, green, blue]
    return ["ALPHA", a, "COLOR", red, green, blue]


def _box_corners(center, extent):
    cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))
    hx = float(extent[0]) / 2.0
    hy = float(extent[1]) / 2.0
    hz = float(extent[2]) / 2.0
    return [
        (cx - hx, cy - hy, cz - hz),
        (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz),
        (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz),
        (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz),
        (cx - hx, cy + hy, cz + hz),
    ]


def wireframe_box_cgo(center, extent, color, line_r=0.012, alpha=1.0):
    """Wireframe box as thin cylinders along the twelve edges."""
    red, green, blue = [float(c) for c in color]
    a = max(0.0, min(1.0, float(alpha)))
    corners = _box_corners(center, extent)
    obj = []
    if a < 1.0 - 1e-6:
        obj.extend(["ALPHA", a])
    for i0, i1 in _BOX_EDGES:
        p0 = corners[i0]
        p1 = corners[i1]
        obj.extend([
            CYLINDER,
            p0[0], p0[1], p0[2],
            p1[0], p1[1], p1[2],
            line_r, red, green, blue, red, green, blue,
        ])
    return obj


def solid_box_cgo(center, extent, color, alpha=1.0):
    """Filled box as CGO triangles with outward face normals."""
    corners = _box_corners(center, extent)
    obj = ["BEGIN", "TRIANGLES"]
    obj.extend(_color_alpha_prefix(color, alpha))
    for face, normal in _BOX_FACES:
        nx, ny, nz = normal
        for idx in face:
            x, y, z = corners[idx]
            obj.extend(["NORMAL", nx, ny, nz, "VERTEX", x, y, z])
    obj.append("END")
    return obj


def unit_wireframe_box_cgo(color, line_r=0.012, alpha=1.0):
    """Unit box centered at the origin with full extent (2, 2, 2)."""
    return wireframe_box_cgo((0.0, 0.0, 0.0), (2.0, 2.0, 2.0), color, line_r=line_r, alpha=alpha)


def unit_solid_box_cgo(color, alpha=1.0):
    """Unit box centered at the origin with full extent (2, 2, 2)."""
    return solid_box_cgo((0.0, 0.0, 0.0), (2.0, 2.0, 2.0), color, alpha=alpha)


def solid_sphere_cgo(center, radius, color, subdivisions=2, alpha=1.0, frequency=None):
    """Filled sphere as a geodesic triangle mesh with outward vertex normals."""
    unit_verts, faces, _edges = icosphere(subdivisions, frequency=frequency)
    cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))
    r = float(radius)
    obj = ["BEGIN", "TRIANGLES"]
    obj.extend(_color_alpha_prefix(color, alpha))
    for i, j, k in faces:
        for idx in (int(i), int(j), int(k)):
            nx, ny, nz = unit_verts[idx]
            obj.extend([
                "NORMAL", float(nx), float(ny), float(nz),
                "VERTEX", cx + r * float(nx), cy + r * float(ny), cz + r * float(nz),
            ])
    obj.append("END")
    return obj


def wireframe_sphere_mesh_cgo(center, radius, color, subdivisions=2, line_r=0.012, alpha=1.0, frequency=None):
    """Wireframe sphere from geodesic edges (even triangle density, no polar pinch)."""
    unit_verts, _faces, edges = icosphere(subdivisions, frequency=frequency)
    cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))
    r = float(radius)
    red, green, blue = [float(c) for c in color]
    a = max(0.0, min(1.0, float(alpha)))
    obj = []
    if a < 1.0 - 1e-6:
        obj.extend(["ALPHA", a])

    def point(idx):
        nx, ny, nz = unit_verts[int(idx)]
        return (cx + r * float(nx), cy + r * float(ny), cz + r * float(nz))

    for i0, i1 in edges:
        p0 = point(i0)
        p1 = point(i1)
        obj.extend([CYLINDER, *p0, *p1, line_r, red, green, blue, red, green, blue])
    return obj


def wireframe_sphere_cgo(center, radius, color, n_lon=6, n_lat=4, n_seg=10, line_r=0.012, alpha=1.0):
    """Thin cylinder cage approximating a wireframe sphere."""
    red, green, blue = [float(c) for c in color]
    a = max(0.0, min(1.0, float(alpha)))
    obj = []
    if a < 1.0 - 1e-6:
        obj.extend(["ALPHA", a])

    def add_edge(p0, p1):
        obj.extend([CYLINDER, *p0, *p1, line_r, red, green, blue, red, green, blue])

    for i in range(n_lon):
        phi = 2.0 * math.pi * i / n_lon
        for j in range(n_seg):
            add_edge(
                point_on_sphere(center, radius, math.pi * j / n_seg, phi),
                point_on_sphere(center, radius, math.pi * (j + 1) / n_seg, phi),
            )
    for i in range(1, n_lat + 1):
        theta = math.pi * i / (n_lat + 1)
        for j in range(n_seg):
            add_edge(
                point_on_sphere(center, radius, theta, 2.0 * math.pi * j / n_seg),
                point_on_sphere(center, radius, theta, 2.0 * math.pi * (j + 1) / n_seg),
            )
    return obj


def native_spheres_cgo(positions, radius, color=(1.0, 1.0, 1.0), alpha=1.0):
    """CGO list of native SPHERE primitives (same tokens as Points render_as='Spheres')."""
    red, green, blue = [float(c) for c in color]
    a = max(0.0, min(1.0, float(alpha)))
    r = float(radius)
    obj = []
    for pos in positions:
        x, y, z = (float(pos[0]), float(pos[1]), float(pos[2]))
        obj.extend(_color_alpha_prefix(color, a))
        obj.extend(["SPHERE", x, y, z, r])
    return obj
