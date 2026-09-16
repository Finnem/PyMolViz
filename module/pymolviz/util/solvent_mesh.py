"""Mesh welding, orientation, boundary repair, and shared isosurface helpers."""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from scipy.spatial import cKDTree

from .marching_cubes import march_cubes_mesh

def _orient(tri, hint):
    a, b, c = tri
    n = np.cross(b - a, c - a)
    if np.dot(n, hint) < 0.0:
        return (a, c, b)
    return tri
def _weld(tris, ndigits=5):
    if not tris:
        return (
            np.zeros((0, 3), dtype=float),
            np.zeros((0, 3), dtype=int),
        )
    index = {}
    verts = []
    faces = []
    for a, b, c in tris:
        ids = []
        for p in (a, b, c):
            key = (round(float(p[0]), ndigits), round(float(p[1]), ndigits), round(float(p[2]), ndigits))
            vid = index.get(key)
            if vid is None:
                vid = len(verts)
                index[key] = vid
                verts.append((float(p[0]), float(p[1]), float(p[2])))
            ids.append(vid)
        if ids[0] != ids[1] and ids[1] != ids[2] and ids[2] != ids[0]:
            faces.append(ids)
    if not faces:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int)
    return np.asarray(verts, dtype=float), _unique_faces(np.asarray(faces, dtype=int))
def _unique_faces(faces):
    """Keep the first winding of each unordered vertex triple."""
    if faces.shape[0] == 0:
        return faces
    faces = np.asarray(faces, dtype=int)
    degenerates = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    )
    if np.any(degenerates):
        faces = faces[~degenerates]
        if faces.shape[0] == 0:
            return faces
    keys = np.sort(faces, axis=1)
    _uniq, index = np.unique(keys, axis=0, return_index=True)
    return faces[np.sort(index)]
def _drop_degenerate_faces(vertices, faces, min_cross=1e-12):
    if faces.shape[0] == 0:
        return faces
    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]
    area2 = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    keep = area2 > float(min_cross)
    if np.all(keep):
        return faces
    return faces[keep]
def _drop_overcovered_edge_faces(vertices, faces):
    """Drop extra slivers on edges that already have two faces.

    A Delaunay cap plus torus share the contact polyline (manifold). Tiny
    leftover clip/fan triangles stacked on those edges z-fight as a dark
    stitch. Keep the two largest faces per over-covered edge.
    """
    faces = np.asarray(faces, dtype=int).reshape(-1, 3)
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    if faces.shape[0] == 0:
        return faces
    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]
    area2 = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    keep = np.ones((int(faces.shape[0]),), dtype=bool)
    changed = True
    while changed:
        changed = False
        edge_faces = {}
        for fi, (a, b, c) in enumerate(faces):
            if not keep[fi]:
                continue
            for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
                key = (u, v) if u < v else (v, u)
                bucket = edge_faces.get(key)
                if bucket is None:
                    edge_faces[key] = [fi]
                else:
                    bucket.append(fi)
        for fis in edge_faces.values():
            if len(fis) <= 2:
                continue
            order = sorted(fis, key=lambda i: (float(area2[i]), int(i)))
            for fi in order[: len(fis) - 2]:
                if keep[fi]:
                    keep[fi] = False
                    changed = True
    if np.all(keep):
        return faces
    return faces[keep]
def _compact_mesh(vertices, faces):
    if faces.shape[0] == 0:
        return vertices, faces
    used = np.unique(np.asarray(faces, dtype=int).ravel())
    if used.size == vertices.shape[0]:
        return vertices, faces
    remap = np.full(int(vertices.shape[0]), -1, dtype=int)
    remap[used] = np.arange(used.size, dtype=int)
    return vertices[used], remap[faces]
def _face_normals(vertices, faces):
    """Area-weighted vertex normals from triangle cross products."""
    if vertices.shape[0] == 0:
        return np.zeros((0, 3), dtype=float)
    normals = np.zeros_like(vertices, dtype=float)
    for face in faces:
        p0, p1, p2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        n = np.cross(p1 - p0, p2 - p0)
        for idx in face:
            normals[idx] += n
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths < 1e-12] = 1.0
    return normals / lengths
def _reweld(vertices, faces, ndigits=5):
    """Merge vertices that round to the same grid point, preserving faces."""
    if faces.shape[0] == 0:
        return vertices, faces
    tris = [
        (vertices[int(a)], vertices[int(b)], vertices[int(c)])
        for a, b, c in faces
    ]
    return _weld(tris, ndigits=ndigits)
def _subdivide_triangle(a, b, c, ab, bc, ca, out):
    """Subdivide one triangle after inserting shared edge midpoints.

    Never emits a split original edge: that would leave a T-junction.
    """
    if ab is None and bc is None and ca is None:
        out.append((a, b, c))
        return
    if ab is not None and bc is None and ca is None:
        out.append((a, ab, c))
        out.append((ab, b, c))
        return
    if bc is not None and ab is None and ca is None:
        out.append((a, b, bc))
        out.append((a, bc, c))
        return
    if ca is not None and ab is None and bc is None:
        out.append((a, b, ca))
        out.append((b, c, ca))
        return
    if ab is not None and bc is not None and ca is None:
        out.append((ab, b, bc))
        out.append((a, ab, c))
        out.append((ab, bc, c))
        return
    if bc is not None and ca is not None and ab is None:
        out.append((bc, c, ca))
        out.append((a, b, bc))
        out.append((a, bc, ca))
        return
    if ca is not None and ab is not None and bc is None:
        out.append((ca, a, ab))
        out.append((b, c, ca))
        out.append((ab, b, ca))
        return
    out.append((ab, b, bc))
    out.append((ab, bc, c))
    out.append((ab, c, ca))
    out.append((ab, ca, a))
def _split_t_junctions(vertices, faces, max_dist=0.04, t_pad=0.02, max_passes=8, cleanup=True):
    """Split a boundary edge when a hanging rim vertex lies on it.

    Cap/torus T-junctions show up as a dense-patch vertex sitting in the
    interior of a coarser boundary edge. Interior chords are left alone:
    on a curved SES those look close in Euclidean space without being
    T-junctions.
    """
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    faces = np.asarray(faces, dtype=int).reshape(-1, 3)
    if faces.shape[0] == 0 or vertices.shape[0] < 4:
        return vertices, faces
    max_dist = float(max_dist)
    t_pad = float(t_pad)

    def edge_key(i, j):
        i, j = int(i), int(j)
        return (i, j) if i < j else (j, i)

    did_split = False
    for _ in range(max(0, int(max_passes))):
        faces = np.asarray(faces, dtype=int).reshape(-1, 3)
        if faces.shape[0] == 0:
            break
        adj = _boundary_adjacency(faces)
        if len(adj) < 3:
            break
        bverts = np.array(list(adj.keys()), dtype=int)
        tree = cKDTree(vertices[bverts])
        boundary_edges = {}
        for u, nbrs in adj.items():
            u = int(u)
            for v in nbrs:
                key = edge_key(u, v)
                if key not in boundary_edges:
                    boundary_edges[key] = True
        incident = {}
        for a, b, c in faces:
            a, b, c = int(a), int(b), int(c)
            for u, v in ((a, b), (b, c), (c, a)):
                key = edge_key(u, v)
                if key not in boundary_edges:
                    continue
                bucket = incident.get(key)
                if bucket is None:
                    incident[key] = {a, b, c}
                else:
                    bucket.update((a, b, c))
        splits = {}
        for (u, v), used in incident.items():
            pu = vertices[u]
            pv = vertices[v]
            span = pv - pu
            length2 = float(np.dot(span, span))
            # Dense clip hits put ~2–3 hanging verts on one torus chord.
            # Skipping anything shorter than 2*max_dist left the last hit
            # on a ~0.05 Å stub and a slit in the contact seam.
            if length2 < 0.02 * 0.02:
                continue
            length = math.sqrt(length2)
            mid = 0.5 * (pu + pv)
            cand = tree.query_ball_point(mid, r=0.5 * length + max_dist)
            best = None
            for ci in cand:
                w = int(bverts[int(ci)])
                if w in used:
                    continue
                t = float(np.dot(vertices[w] - pu, span) / length2)
                if t <= t_pad or t >= 1.0 - t_pad:
                    continue
                dist = float(np.linalg.norm(vertices[w] - (pu + t * span)))
                if dist >= max_dist:
                    continue
                if best is None or dist < best[0]:
                    best = (dist, w)
            if best is not None:
                splits[edge_key(u, v)] = best[1]
        if not splits:
            break
        did_split = True
        new_faces = []
        for a, b, c in faces:
            a, b, c = int(a), int(b), int(c)
            _subdivide_triangle(
                a, b, c,
                splits.get(edge_key(a, b)),
                splits.get(edge_key(b, c)),
                splits.get(edge_key(c, a)),
                new_faces,
            )
        faces = np.asarray(new_faces, dtype=int)
    if did_split and cleanup:
        faces = _unique_faces(faces)
        faces = _drop_degenerate_faces(vertices, faces)
    return vertices, faces
def _weld_contact_seam_stubs(vertices, faces, centers, vdw, tol=0.02):
    """Merge near-duplicate cap/torus rim vertices on the contact circle.

    T-split leaves unmatched 3–4 cycles whose endpoints miss by ~5e-4 Å. Those
    slits line up around the join and render as a dark stitch. Only the
    two-atom contact band is welded so a tight reentrant is not chorded.
    """
    faces = np.asarray(faces, dtype=int).reshape(-1, 3)
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    if faces.shape[0] == 0 or vertices.shape[0] < 2 or centers.shape[0] < 2:
        return vertices, faces
    adj = _boundary_adjacency(faces)
    if len(adj) < 2:
        return vertices, faces
    bverts = np.array(list(adj.keys()), dtype=int)
    sdf = signed_distance(vertices[bverts], centers, vdw)
    keep = np.abs(sdf) < 0.08
    if int(np.count_nonzero(keep)) < 2:
        return vertices, faces
    sel = bverts[keep]
    tree = cKDTree(vertices[sel])
    pairs = tree.query_pairs(r=float(tol))
    if not pairs:
        return vertices, faces
    parent = np.arange(int(vertices.shape[0]), dtype=int)

    def find(i):
        i = int(i)
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in pairs:
        union(int(sel[int(i)]), int(sel[int(j)]))
    clusters = {}
    for i in sel:
        clusters.setdefault(find(int(i)), []).append(int(i))
    out = np.array(vertices, copy=True)
    for root, members in clusters.items():
        if len(members) < 2:
            continue
        out[root] = np.mean(out[np.asarray(members, dtype=int)], axis=0)
    new_faces = []
    for a, b, c in faces:
        a, b, c = find(int(a)), find(int(b)), find(int(c))
        if a == b or b == c or c == a:
            continue
        new_faces.append((a, b, c))
    if not new_faces:
        return vertices, faces
    faces = _unique_faces(np.asarray(new_faces, dtype=int))
    return _compact_mesh(out, faces)
def _split_seam_edges(vertices, faces, centers, radii):
    """Insert shared circle / triple-point vertices on mixed-owner faces."""
    if faces.shape[0] == 0 or centers.shape[0] < 2:
        return vertices, faces
    owner = _sas_owners(vertices, centers, radii)
    verts = [np.asarray(vertices[i], dtype=float) for i in range(vertices.shape[0])]
    mid = {}
    triples = {}

    def edge_mid(a, b):
        if owner[a] == owner[b]:
            return None
        key = (a, b) if a < b else (b, a)
        vid = mid.get(key)
        if vid is None:
            ia, ib = int(owner[key[0]]), int(owner[key[1]])
            point = _snap_to_intersection_circle(
                0.5 * (verts[key[0]] + verts[key[1]]),
                centers[ia], float(radii[ia]),
                centers[ib], float(radii[ib]),
            )
            vid = len(verts)
            verts.append(np.asarray(point, dtype=float))
            mid[key] = vid
        return vid

    def triple_vid(ia, ib, ic, hint):
        pts = _sphere_triple_points(
            centers[ia], float(radii[ia]),
            centers[ib], float(radii[ib]),
            centers[ic], float(radii[ic]),
        )
        if not pts:
            return None
        chosen = min(pts, key=lambda point: float(np.linalg.norm(point - hint)))
        key = (
            tuple(sorted((int(ia), int(ib), int(ic)))),
            tuple(np.round(np.asarray(chosen, dtype=float), 4)),
        )
        vid = triples.get(key)
        if vid is None:
            vid = len(verts)
            verts.append(np.asarray(chosen, dtype=float))
            triples[key] = vid
        return vid

    new_faces = []
    for a, b, c in faces:
        a, b, c = int(a), int(b), int(c)
        ab, bc, ca = edge_mid(a, b), edge_mid(b, c), edge_mid(c, a)
        kinds = {int(owner[a]), int(owner[b]), int(owner[c])}
        if len(kinds) == 3 and ab is not None and bc is not None and ca is not None:
            ia, ib, ic = sorted(kinds)
            centroid = (verts[a] + verts[b] + verts[c]) / 3.0
            tid = triple_vid(ia, ib, ic, centroid)
            if tid is not None:
                span = max(
                    float(np.linalg.norm(verts[a] - verts[b])),
                    float(np.linalg.norm(verts[b] - verts[c])),
                    float(np.linalg.norm(verts[c] - verts[a])),
                    1e-12,
                )
                if float(np.linalg.norm(verts[tid] - centroid)) <= 2.5 * span:
                    ring = (a, ab, b, bc, c, ca)
                    for i, q in enumerate(ring):
                        r = ring[(i + 1) % 6]
                        if q != r and tid != q and tid != r:
                            new_faces.append((tid, q, r))
                    continue
        _subdivide_triangle(a, b, c, ab, bc, ca, new_faces)
    if not new_faces:
        return vertices, faces
    return np.asarray(verts, dtype=float), _unique_faces(np.asarray(new_faces, dtype=int))
def _smooth_on_sas(vertices, faces, centers, radii, pin=None, iterations=8, lam=0.35):
    """Umbrella smooth in the surface, reprojecting onto the SAS each step."""
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return vertices
    n = int(vertices.shape[0])
    e0 = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    e1 = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    src = np.concatenate([e0, e1])
    dst = np.concatenate([e1, e0])
    verts = np.asarray(vertices, dtype=float)
    for _ in range(int(iterations)):
        acc = np.zeros_like(verts)
        np.add.at(acc, src, verts[dst])
        cnt = np.bincount(src, minlength=n).astype(float)
        cnt = np.maximum(cnt, 1.0)
        verts = (1.0 - lam) * verts + lam * (acc / cnt[:, None])
        verts = _apply_sas_projection(verts, centers, radii, pin=pin)
    return verts
def _canonical_cycle(loop):
    n = len(loop)
    seq = list(loop)
    forward = min(tuple(seq[i:] + seq[:i]) for i in range(n))
    rev = list(reversed(seq))
    backward = min(tuple(rev[i:] + rev[:i]) for i in range(n))
    return forward if forward <= backward else backward
def _boundary_adjacency(faces):
    count = {}
    for a, b, c in faces:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (u, v) if u < v else (v, u)
            count[key] = count.get(key, 0) + 1
    adj = {}
    for (u, v), n in count.items():
        if n != 1:
            continue
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    for key, nbrs in adj.items():
        adj[key] = list(dict.fromkeys(nbrs))
    return adj
def _simple_boundary_cycles(adj, max_loop=16):
    """Simple cycles in the boundary graph, including those that share a hub vertex."""
    cycles = []
    seen = set()
    nodes = list(adj.keys())

    def dfs(start, prev, cur, path, in_path):
        if len(path) > max_loop:
            return
        for nxt in adj.get(cur, ()):
            if nxt == prev:
                continue
            if nxt == start:
                if len(path) >= 3:
                    key = _canonical_cycle(path)
                    if key not in seen:
                        seen.add(key)
                        cycles.append(list(path))
                continue
            if nxt in in_path:
                continue
            path.append(nxt)
            in_path.add(nxt)
            dfs(start, cur, nxt, path, in_path)
            path.pop()
            in_path.remove(nxt)

    for start in nodes:
        dfs(start, -1, start, [start], {start})
    return cycles
def _loop_fan_origin(points, loop):
    """Index into *loop* whose fan has the largest minimum triangle area."""
    n = len(loop)
    best_i = 0
    best_score = -1.0
    for i in range(n):
        origin = np.asarray(points[loop[i]], dtype=float)
        min_area = None
        ok = True
        for k in range(1, n - 1):
            a = np.asarray(points[loop[(i + k) % n]], dtype=float)
            b = np.asarray(points[loop[(i + k + 1) % n]], dtype=float)
            area = float(np.linalg.norm(np.cross(a - origin, b - origin)))
            if area <= 1e-16:
                ok = False
                break
            if min_area is None or area < min_area:
                min_area = area
        if ok and min_area is not None and min_area > best_score:
            best_score = min_area
            best_i = i
    return best_i
def _fill_boundary_holes(vertices, faces, max_loop=8, max_edge=0.85):
    """Fan-fill small boundary loops left by cap/torus T-junctions.

    Only short edges are filled so a missing contact cap is not papered over
    with a chord through the groove.
    """
    if faces.shape[0] == 0:
        return vertices, faces
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    extra = []
    used = set()
    max_loop = int(max_loop)
    max_edge = float(max_edge)
    occupancy = {}
    for a, b, c in faces:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (u, v) if u < v else (v, u)
            occupancy[key] = occupancy.get(key, 0) + 1

    def edge_key(u, v):
        u, v = int(u), int(v)
        return (u, v) if u < v else (v, u)

    def try_add_loop(loop):
        n = len(loop)
        if n < 3 or n > max_loop:
            return
        edges = []
        for i, vertex in enumerate(loop):
            nxt = loop[(i + 1) % n]
            key = edge_key(vertex, nxt)
            if key in used:
                return
            dist = float(np.linalg.norm(vertices[int(vertex)] - vertices[int(nxt)]))
            if dist > max_edge:
                return
            edges.append(key)
        origin_i = 0 if n == 3 else _loop_fan_origin(vertices, loop)
        rotated = loop[origin_i:] + loop[:origin_i]
        origin = rotated[0]
        added = []
        for i in range(1, n - 1):
            a, b = rotated[i], rotated[i + 1]
            if origin == a or a == b or b == origin:
                continue
            p0 = vertices[int(origin)]
            p1 = vertices[int(a)]
            p2 = vertices[int(b)]
            area2 = float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)))
            if area2 <= 1e-12:
                continue
            tri_edges = (edge_key(origin, a), edge_key(a, b), edge_key(b, origin))
            if any(occupancy.get(key, 0) >= 2 for key in tri_edges):
                return
            added.append((origin, a, b, tri_edges))
        if not added:
            return
        for origin, a, b, tri_edges in added:
            extra.append((origin, a, b))
            for key in tri_edges:
                occupancy[key] = occupancy.get(key, 0) + 1
        used.update(edges)

    adj = _boundary_adjacency(faces)
    if not adj:
        return vertices, faces
    for v, nbrs in adj.items():
        if len(nbrs) < 2:
            continue
        nbrs_v = list(nbrs)
        for i, a in enumerate(nbrs_v):
            nbrs_a = adj.get(a, ())
            for b in nbrs_v[i + 1:]:
                if b in nbrs_a:
                    try_add_loop([int(v), int(a), int(b)])

    combined = faces
    if extra:
        combined = np.vstack((faces, np.asarray(extra, dtype=int)))
    adj = _boundary_adjacency(combined)
    seen = set()
    for start in adj:
        if start in seen or len(adj.get(start, ())) != 2:
            continue
        cycle = [int(start)]
        prev = None
        cur = int(start)
        ok = True
        for _ in range(max_loop + 1):
            nxt = None
            for cand in adj.get(cur, ()):
                if cand != prev:
                    nxt = int(cand)
                    break
            if nxt is None:
                ok = False
                break
            if nxt == int(start):
                break
            if nxt in seen or nxt in cycle:
                ok = False
                break
            cycle.append(nxt)
            prev, cur = cur, nxt
        else:
            ok = False
        if ok and len(cycle) >= 3:
            try_add_loop(cycle)
        seen.update(cycle)

    if not extra:
        return vertices, faces
    return vertices, np.vstack((faces, np.asarray(extra, dtype=int)))
def _orient_faces_to_vertex_normals(vertices, faces, normals):
    """Flip a triangle when most of its vertex normals oppose the geometric normal."""
    if faces.shape[0] == 0:
        return faces
    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]
    geom = np.cross(p1 - p0, p2 - p0)
    d0 = np.sum(geom * normals[faces[:, 0]], axis=1)
    d1 = np.sum(geom * normals[faces[:, 1]], axis=1)
    d2 = np.sum(geom * normals[faces[:, 2]], axis=1)
    votes = (d0 < 0.0).astype(np.int32) + (d1 < 0.0).astype(np.int32) + (d2 < 0.0).astype(np.int32)
    area2 = np.linalg.norm(geom, axis=1)
    flip = (votes >= 2) & (area2 > 1e-10)
    if not np.any(flip):
        return faces
    out = np.array(faces, copy=True, dtype=int)
    out[flip] = out[flip][:, (0, 2, 1)]
    return out
def _orient_faces_outward(vertices, faces, centers, radii):
    """Flip each triangle so its geometric normal points out of the union.

    PyMOL lights CGO triangles independently. A mesh-wide winding walk inverts
    whole patches that do not share vertex indices; each face is oriented here
    on its own.
    """
    if faces.shape[0] == 0:
        return faces
    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-18)
    unit = normals / lengths
    centroid = (p0 + p1 + p2) / 3.0
    eps = 0.05
    sdf_pos = signed_distance(centroid + eps * unit, centers, radii)
    sdf_neg = signed_distance(centroid - eps * unit, centers, radii)
    owners = _sas_owners(centroid, centers, radii)
    hint = centroid - centers[owners]
    owner_inward = np.sum(normals * hint, axis=1) < 0.0
    both_inside = (sdf_pos < 0.0) & (sdf_neg < 0.0)
    flip = np.where(both_inside, owner_inward, sdf_pos < sdf_neg)
    if not np.any(flip):
        return faces
    out = np.array(faces, copy=True, dtype=int)
    out[flip] = out[flip][:, (0, 2, 1)]
    return out
def _trilinear_sample(grid, origin, h, xyz):
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    n = int(xyz.shape[0])
    vector = grid.ndim == 4
    nx, ny, nz = grid.shape[:3]
    if n == 0:
        return np.zeros((0, 3), dtype=float) if vector else np.zeros((0,), dtype=float)
    p = (xyz - np.asarray(origin, dtype=float)) / float(h)
    p[:, 0] = np.clip(p[:, 0], 0.0, max(nx - 1.000001, 0.0))
    p[:, 1] = np.clip(p[:, 1], 0.0, max(ny - 1.000001, 0.0))
    p[:, 2] = np.clip(p[:, 2], 0.0, max(nz - 1.000001, 0.0))
    i0 = np.floor(p).astype(np.int32)
    i1 = np.stack(
        (
            np.minimum(i0[:, 0] + 1, nx - 1),
            np.minimum(i0[:, 1] + 1, ny - 1),
            np.minimum(i0[:, 2] + 1, nz - 1),
        ),
        axis=1,
    )
    f = p - i0.astype(float)
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    if vector:
        fx, fy, fz = fx[:, None], fy[:, None], fz[:, None]

    def corner(ix, iy, iz):
        return grid[ix, iy, iz]

    c000 = corner(i0[:, 0], i0[:, 1], i0[:, 2])
    c100 = corner(i1[:, 0], i0[:, 1], i0[:, 2])
    c010 = corner(i0[:, 0], i1[:, 1], i0[:, 2])
    c110 = corner(i1[:, 0], i1[:, 1], i0[:, 2])
    c001 = corner(i0[:, 0], i0[:, 1], i1[:, 2])
    c101 = corner(i1[:, 0], i0[:, 1], i1[:, 2])
    c011 = corner(i0[:, 0], i1[:, 1], i1[:, 2])
    c111 = corner(i1[:, 0], i1[:, 1], i1[:, 2])
    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz
def _cubes_with_sign_change(field):
    nx, ny, nz = field.shape
    if nx < 2 or ny < 2 or nz < 2:
        return np.zeros((0, 3), dtype=np.int32)
    corners = (
        field[0:nx - 1, 0:ny - 1, 0:nz - 1],
        field[1:nx, 0:ny - 1, 0:nz - 1],
        field[0:nx - 1, 1:ny, 0:nz - 1],
        field[1:nx, 1:ny, 0:nz - 1],
        field[0:nx - 1, 0:ny - 1, 1:nz],
        field[1:nx, 0:ny - 1, 1:nz],
        field[0:nx - 1, 1:ny, 1:nz],
        field[1:nx, 1:ny, 1:nz],
    )
    vmin = corners[0]
    vmax = corners[0]
    for values in corners[1:]:
        vmin = np.minimum(vmin, values)
        vmax = np.maximum(vmax, values)
    keep = (vmin < 0.0) & (vmax >= 0.0)
    ii, jj, kk = np.nonzero(keep)
    if ii.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    return np.stack((ii, jj, kk), axis=1).astype(np.int32)
def _isosurface_from_signed_field(origin, h, field, grad):
    empty = (
        np.zeros((0, 3), dtype=float),
        np.zeros((0, 3), dtype=float),
        np.zeros((0, 3), dtype=int),
    )
    cubes = _cubes_with_sign_change(field)
    if cubes.shape[0] == 0:
        return empty
    vertices, faces = march_cubes_mesh(origin, h, field, cubes)
    if faces.shape[0] == 0:
        return empty
    faces = _unique_faces(faces)
    faces = _drop_degenerate_faces(vertices, faces)
    vertices, faces = _compact_mesh(vertices, faces)
    if faces.shape[0] == 0:
        return empty
    normals = _trilinear_sample(grad, origin, h, vertices)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    tiny = lengths[:, 0] < 1e-12
    lengths = np.maximum(lengths, 1e-12)
    normals = normals / lengths
    if np.any(tiny):
        geom = _face_normals(vertices, faces)
        normals = np.array(normals, copy=True)
        normals[tiny] = geom[tiny]
    faces = _orient_faces_to_vertex_normals(vertices, faces, normals)
    return vertices, normals, faces
