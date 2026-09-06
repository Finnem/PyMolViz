"""CGO construction helpers (requires pymol.cgo)."""

import math

from pymol.cgo import CYLINDER

from .geometries import icosphere, point_on_sphere

_CGO_TOKEN_NAMES = (
    "POINTS", "SPHERE", "COLOR", "VERTEX", "NORMAL", "CYLINDER", "CONE",
    "BEGIN", "END", "LINEWIDTH", "LINES", "TRIANGLES", "ALPHA",
    "ENABLE", "DISABLE", "LIGHTING",
)


def resolve_cgo_tokens(content: list) -> list:
    """Convert string CGO opcodes to pymol.cgo integer constants."""
    from pymol import cgo

    map_cgo_keys = {name: getattr(cgo, name) for name in _CGO_TOKEN_NAMES if hasattr(cgo, name)}
    out = []
    for entry in content:
        if isinstance(entry, bool):
            raise TypeError("Unexpected bool in CGO list: %r" % entry)
        if isinstance(entry, int) and entry in map_cgo_keys.values():
            out.append(entry)
            continue
        if isinstance(entry, str) and entry in map_cgo_keys:
            out.append(map_cgo_keys[entry])
            continue
        # Numeric strings (e.g. numpy-hstack '0.86') are values, not opcodes.
        try:
            out.append(float(entry))
            continue
        except (TypeError, ValueError):
            pass
        token = entry if isinstance(entry, str) else repr(entry)
        if token not in map_cgo_keys and hasattr(cgo, token):
            map_cgo_keys[token] = getattr(cgo, token)
        if token not in map_cgo_keys:
            raise KeyError("Unknown CGO token %r" % entry)
        out.append(map_cgo_keys[token])
    return out


def _py_cgo_token(token):
    """Unwrap numpy scalars; leave Python scalars and strings as-is."""
    if isinstance(token, bool):
        return token
    if isinstance(token, (str, int, float)):
        return token
    item = getattr(token, "item", None)
    if callable(item):
        try:
            return token.item()
        except Exception:
            return token
    return token


def _cgo_opcode_kind(token, opcode_ints):
    """Return a named opcode, or None.

    Real PyMOL stores opcodes as floats (VERTEX is 4.0, and so is TRIANGLES).
    Coordinate payloads can hold the same values, so callers must only invoke
    this at opcode positions and then skip each opcode's payload.
    """
    py = _py_cgo_token(token)
    if isinstance(py, bool):
        return None
    if isinstance(py, str):
        return py if py in opcode_ints else None
    try:
        value = float(py)
    except (TypeError, ValueError):
        return None
    for name, code in opcode_ints.items():
        if code is None:
            continue
        try:
            if value == float(code):
                return name
        except (TypeError, ValueError):
            continue
    return None


# Tokens after the opcode. Primitive modes (TRIANGLES/LINES/…) are not listed:
# VERTEX aliases TRIANGLES (both 4.0), so a mode must be skipped via BEGIN, not
# matched as VERTEX.
_CGO_SKIP_AFTER = {
    "COLOR": 3,
    "NORMAL": 3,
    "ALPHA": 1,
    "LINEWIDTH": 1,
    "ENABLE": 1,
    "DISABLE": 1,
    "END": 0,
}


def offset_cgo_vertices(content, delta):
    """Add ``delta`` to VERTEX/SPHERE/CYLINDER/CONE positions in a CGO list.

    The token after BEGIN is a primitive mode (TRIANGLES/LINES/…). In real
    PyMOL that mode value aliases VERTEX (both 4.0), so it must not be shifted.
    Payloads are skipped so a coordinate of 4.0 is not treated as VERTEX.
    """
    from pymol import cgo

    dx, dy, dz = (float(delta[0]), float(delta[1]), float(delta[2]))
    opcode_ints = {
        "BEGIN": getattr(cgo, "BEGIN", None),
        "END": getattr(cgo, "END", None),
        "COLOR": getattr(cgo, "COLOR", None),
        "NORMAL": getattr(cgo, "NORMAL", None),
        "VERTEX": getattr(cgo, "VERTEX", None),
        "SPHERE": getattr(cgo, "SPHERE", None),
        "CYLINDER": getattr(cgo, "CYLINDER", None),
        "CONE": getattr(cgo, "CONE", None),
        "ALPHA": getattr(cgo, "ALPHA", None),
        "LINEWIDTH": getattr(cgo, "LINEWIDTH", None),
        "ENABLE": getattr(cgo, "ENABLE", None),
        "DISABLE": getattr(cgo, "DISABLE", None),
    }
    i = 0
    n = len(content)
    after_begin = False
    while i < n:
        if after_begin:
            after_begin = False
            i += 1
            continue
        kind = _cgo_opcode_kind(content[i], opcode_ints)
        if kind == "BEGIN":
            after_begin = True
            i += 1
            continue
        if kind == "VERTEX" and i + 3 < n:
            content[i + 1] = float(content[i + 1]) + dx
            content[i + 2] = float(content[i + 2]) + dy
            content[i + 3] = float(content[i + 3]) + dz
            i += 4
            continue
        if kind == "SPHERE" and i + 4 < n:
            content[i + 1] = float(content[i + 1]) + dx
            content[i + 2] = float(content[i + 2]) + dy
            content[i + 3] = float(content[i + 3]) + dz
            i += 5
            continue
        if kind == "CYLINDER" and i + 6 < n:
            content[i + 1] = float(content[i + 1]) + dx
            content[i + 2] = float(content[i + 2]) + dy
            content[i + 3] = float(content[i + 3]) + dz
            content[i + 4] = float(content[i + 4]) + dx
            content[i + 5] = float(content[i + 5]) + dy
            content[i + 6] = float(content[i + 6]) + dz
            i += 14
            continue
        if kind == "CONE" and i + 6 < n:
            content[i + 1] = float(content[i + 1]) + dx
            content[i + 2] = float(content[i + 2]) + dy
            content[i + 3] = float(content[i + 3]) + dz
            content[i + 4] = float(content[i + 4]) + dx
            content[i + 5] = float(content[i + 5]) + dy
            content[i + 6] = float(content[i + 6]) + dz
            i += 17
            continue
        if kind in _CGO_SKIP_AFTER:
            i += 1 + _CGO_SKIP_AFTER[kind]
            continue
        i += 1
    return content

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


def _perp_frame(direction):
    dx, dy, dz = (float(direction[0]), float(direction[1]), float(direction[2]))
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-8:
        return (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    ax, ay, az = (dx / length, dy / length, dz / length)
    if abs(az) < 0.9:
        px, py, pz = (-ay, ax, 0.0)
    else:
        px, py, pz = (0.0, -az, ay)
    plen = math.sqrt(px * px + py * py + pz * pz)
    px, py, pz = (px / plen, py / plen, pz / plen)
    bx = ay * pz - az * py
    by = az * px - ax * pz
    bz = ax * py - ay * px
    return (px, py, pz), (bx, by, bz)


def _normalize3(v):
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    length = math.sqrt(x * x + y * y + z * z)
    if length < 1e-8:
        return (0.0, 0.0, 1.0)
    inv = 1.0 / length
    return (x * inv, y * inv, z * inv)


def _azimuth_dirs(n_seg, perp, bitan):
    dirs = []
    n_seg = max(int(n_seg), 3)
    for i in range(n_seg):
        ang = 2.0 * math.pi * i / n_seg
        ca, sa = math.cos(ang), math.sin(ang)
        dirs.append((
            perp[0] * ca + bitan[0] * sa,
            perp[1] * ca + bitan[1] * sa,
            perp[2] * ca + bitan[2] * sa,
        ))
    return dirs


def _ring_points(center, axis, radius, n_seg, perp, bitan):
    cx, cy, cz = center
    return [
        (cx + radius * d[0], cy + radius * d[1], cz + radius * d[2])
        for d in _azimuth_dirs(n_seg, perp, bitan)
    ]


def _emit_triangle(obj, n0, p0, n1, p1, n2, p2):
    obj.extend(["NORMAL", n0[0], n0[1], n0[2], "VERTEX", p0[0], p0[1], p0[2]])
    obj.extend(["NORMAL", n1[0], n1[1], n1[2], "VERTEX", p1[0], p1[1], p1[2]])
    obj.extend(["NORMAL", n2[0], n2[1], n2[2], "VERTEX", p2[0], p2[1], p2[2]])


def lines_cgo(segments, color, width=2.0, alpha=1.0):
    """2D CGO line segments."""
    obj = ["LINEWIDTH", float(width)]
    obj.extend(_color_alpha_prefix(color, alpha))
    obj.extend(["BEGIN", "LINES"])
    for p0, p1 in segments:
        obj.extend(["VERTEX", float(p0[0]), float(p0[1]), float(p0[2])])
        obj.extend(["VERTEX", float(p1[0]), float(p1[1]), float(p1[2])])
    obj.append("END")
    return obj


def mesh_cylinder_cgo(p0, p1, radius, color, n_seg=8, alpha=1.0, caps=True):
    """Cylinder as triangles along p0→p1. ``caps`` closes both ends with disks."""
    axis = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
    length = math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)
    if length < 1e-8 or radius <= 0.0:
        return []
    inv = 1.0 / length
    nrm = (axis[0] * inv, axis[1] * inv, axis[2] * inv)
    perp, bitan = _perp_frame(axis)
    a = _ring_points(p0, nrm, radius, n_seg, perp, bitan)
    b = _ring_points(p1, nrm, radius, n_seg, perp, bitan)
    obj = ["BEGIN", "TRIANGLES"]
    obj.extend(_color_alpha_prefix(color, alpha))
    for i in range(n_seg):
        j = (i + 1) % n_seg
        n_i = ((a[i][0] - p0[0]) / radius, (a[i][1] - p0[1]) / radius, (a[i][2] - p0[2]) / radius)
        n_j = ((a[j][0] - p0[0]) / radius, (a[j][1] - p0[1]) / radius, (a[j][2] - p0[2]) / radius)
        _emit_triangle(obj, n_i, a[i], n_j, a[j], n_j, b[j])
        _emit_triangle(obj, n_i, a[i], n_j, b[j], n_i, b[i])
    if caps:
        n_back = (-nrm[0], -nrm[1], -nrm[2])
        for i in range(n_seg):
            j = (i + 1) % n_seg
            _emit_triangle(obj, n_back, p0, n_back, a[j], n_back, a[i])
            _emit_triangle(obj, nrm, p1, nrm, b[i], nrm, b[j])
    obj.append("END")
    return obj


def mesh_cone_cgo(base, tip, radius, color, n_seg=8, alpha=1.0, cap_base=True, n_rings=None):
    """Cone from base ring to tip with slant normals.

    ``n_rings`` is the number of circumference rings before the tip (1 = a
    triangle fan). Default scales with ``n_seg`` so long heads shade smoothly.
    ``cap_base`` closes the open back of the cone.
    """
    axis = (tip[0] - base[0], tip[1] - base[1], tip[2] - base[2])
    length = math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)
    if length < 1e-8 or radius <= 0.0:
        return []
    n_seg = max(int(n_seg), 3)
    if n_rings is None:
        n_rings = max(1, min(4, n_seg // 4))
    else:
        n_rings = max(1, int(n_rings))
    inv = 1.0 / length
    nrm = (axis[0] * inv, axis[1] * inv, axis[2] * inv)
    perp, bitan = _perp_frame(axis)
    radials = _azimuth_dirs(n_seg, perp, bitan)
    slope = radius / length
    normals = [
        _normalize3((d[0] + slope * nrm[0], d[1] + slope * nrm[1], d[2] + slope * nrm[2]))
        for d in radials
    ]
    rings = []
    for k in range(n_rings):
        t = float(k) / float(n_rings)
        rr = radius * (1.0 - t)
        cx = base[0] + (tip[0] - base[0]) * t
        cy = base[1] + (tip[1] - base[1]) * t
        cz = base[2] + (tip[2] - base[2]) * t
        rings.append([
            (cx + rr * d[0], cy + rr * d[1], cz + rr * d[2])
            for d in radials
        ])
    obj = ["BEGIN", "TRIANGLES"]
    obj.extend(_color_alpha_prefix(color, alpha))
    n_back = (-nrm[0], -nrm[1], -nrm[2])
    for k in range(len(rings) - 1):
        a, b = rings[k], rings[k + 1]
        for i in range(n_seg):
            j = (i + 1) % n_seg
            _emit_triangle(obj, normals[i], a[i], normals[j], a[j], normals[j], b[j])
            _emit_triangle(obj, normals[i], a[i], normals[j], b[j], normals[i], b[i])
    last = rings[-1]
    for i in range(n_seg):
        j = (i + 1) % n_seg
        n_tip = _normalize3((
            normals[i][0] + normals[j][0],
            normals[i][1] + normals[j][1],
            normals[i][2] + normals[j][2],
        ))
        _emit_triangle(obj, normals[i], last[i], normals[j], last[j], n_tip, tip)
        if cap_base:
            _emit_triangle(obj, n_back, base, n_back, rings[0][j], n_back, rings[0][i])
    obj.append("END")
    return obj
