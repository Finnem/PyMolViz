"""CGO tokens for a clip-plane rectangle and a symbolized eye.

String opcodes only — no pymol import. Keep-side is along ``normal``; the eye
sits on that side and looks at the plane.
"""

from __future__ import annotations

import math

import numpy as np

from .math import get_perp

_DEFAULT_GIZMO_AXES = (True, True, True)


def normalize_clip_gizmo_state(raw=None) -> dict:
    """Editor-only visibility: ``{shown, axes}`` with three per-axis flags."""
    shown = True
    axes = [True, True, True]
    if isinstance(raw, dict):
        if "shown" in raw:
            shown = bool(raw.get("shown"))
        elif "visible" in raw:
            shown = bool(raw.get("visible"))
        src = raw.get("axes")
        if src is not None:
            for i, value in enumerate(list(src)[:3]):
                axes[i] = bool(value)
    elif raw is not None:
        shown = bool(raw)
    return {"shown": shown, "axes": axes}


def _clip_gizmo_children(obj):
    if obj is None:
        return ()
    if isinstance(obj, (list, tuple)):
        return list(obj)
    for name in ("objects", "children"):
        kids = getattr(obj, name, None)
        if kids:
            return list(kids)
    return ()


def read_clip_gizmo_state(obj):
    if obj is None:
        return normalize_clip_gizmo_state(None)
    raw = getattr(obj, "clip_gizmos", None)
    if raw is None:
        for child in _clip_gizmo_children(obj):
            raw = getattr(child, "clip_gizmos", None)
            if raw is not None:
                break
    return normalize_clip_gizmo_state(raw)


def stamp_clip_gizmo_state(obj, state) -> None:
    if obj is None:
        return
    normalized = normalize_clip_gizmo_state(state)

    def _set(target) -> None:
        if target is None:
            return
        try:
            target.clip_gizmos = normalized
        except Exception:
            pass

    _set(obj)
    for child in _clip_gizmo_children(obj):
        _set(child)

_PLANE_FILL = (0.30, 0.82, 1.00)
_PLANE_FILL_SEL = (1.00, 0.72, 0.18)
_PLANE_EDGE = (0.12, 0.55, 0.78)
_PLANE_EDGE_SEL = (0.85, 0.45, 0.05)
_SCLERA = (0.97, 0.97, 0.99)
_IRIS = (0.18, 0.42, 0.72)
_PUPIL = (0.06, 0.06, 0.07)
_HIGHLIGHT = (1.00, 1.00, 1.00)
_LID = (0.12, 0.12, 0.14)
_GAZE = (0.20, 0.20, 0.22)


def _unit(v):
    arr = np.asarray(v, dtype=float).reshape(3)
    ln = float(np.linalg.norm(arr))
    if ln < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return arr / ln


def _frame(normal):
    n = _unit(normal)
    u = get_perp(n)
    u = u / float(np.linalg.norm(u))
    v = np.cross(n, u)
    v = v / float(np.linalg.norm(v))
    return n, u, v


_FIT_MARGIN = 1.12
_MIN_HALF = 1.0


def fit_plane_rectangle(origin, normal, points=None, scale=5.0, margin=_FIT_MARGIN):
    """In-plane half-extents that cover ``points`` (or a square of ``scale``).

    Returns ``(center, n, u_ext, v_ext)`` with ``center`` on the plane. The clip
    origin is unchanged; the quad is recentered over the projected mesh.
    """
    origin = np.asarray(origin, dtype=float).reshape(3)
    n, u, v = _frame(normal)
    pts = None if points is None else np.asarray(points, dtype=float).reshape(-1, 3)
    if pts is None or pts.size == 0:
        half = 0.5 * max(float(scale), 0.5)
        return origin, n, u * half, v * half
    rel = pts - origin
    pu = rel @ u
    pv = rel @ v
    umin, umax = float(np.min(pu)), float(np.max(pu))
    vmin, vmax = float(np.min(pv)), float(np.max(pv))
    cu = 0.5 * (umin + umax)
    cv = 0.5 * (vmin + vmax)
    hu = max(0.5 * (umax - umin) * float(margin), _MIN_HALF)
    hv = max(0.5 * (vmax - vmin) * float(margin), _MIN_HALF)
    center = origin + u * cu + v * cv
    return center, n, u * hu, v * hv


def rectangle_corners(origin, normal, scale=5.0, points=None, margin=_FIT_MARGIN):
    """Four corners of a rectangle in the clip plane covering the surface."""
    center, n, u_ext, v_ext = fit_plane_rectangle(
        origin, normal, points=points, scale=scale, margin=margin,
    )
    return (
        center - u_ext - v_ext,
        center + u_ext - v_ext,
        center + u_ext + v_ext,
        center - u_ext + v_ext,
    ), n, u_ext, v_ext, center


def _color(obj, rgb, alpha=1.0):
    a = max(0.0, min(1.0, float(alpha)))
    if a < 1.0 - 1e-6:
        obj.extend(["ALPHA", a])
    obj.extend(["COLOR", float(rgb[0]), float(rgb[1]), float(rgb[2])])


def _cylinder(obj, p0, p1, radius, rgb):
    obj.extend([
        "CYLINDER",
        float(p0[0]), float(p0[1]), float(p0[2]),
        float(p1[0]), float(p1[1]), float(p1[2]),
        float(radius),
        float(rgb[0]), float(rgb[1]), float(rgb[2]),
        float(rgb[0]), float(rgb[1]), float(rgb[2]),
    ])


def _sphere(obj, center, radius, rgb):
    _color(obj, rgb, 1.0)
    obj.extend([
        "SPHERE",
        float(center[0]), float(center[1]), float(center[2]),
        float(radius),
    ])


def _poly_chain(obj, points, radius, rgb):
    for i in range(len(points) - 1):
        _cylinder(obj, points[i], points[i + 1], radius, rgb)


def build_clip_gizmo_cgo(
    origin,
    normal,
    scale=5.0,
    *,
    points=None,
    selected=False,
    draft=False,
    axis_arrow=False,
):
    """Rectangle fill + edges, plus a 3D eye on the keep side."""
    origin = np.asarray(origin, dtype=float).reshape(3)
    corners, n, u_ext, v_ext, center = rectangle_corners(
        origin, normal, scale=scale, points=points,
    )
    span = float(math.hypot(float(np.linalg.norm(u_ext)), float(np.linalg.norm(v_ext))))
    fill = _PLANE_FILL_SEL if selected else _PLANE_FILL
    edge = _PLANE_EDGE_SEL if selected else _PLANE_EDGE
    fill_alpha = 0.38 if draft else 0.22
    if selected:
        fill_alpha = min(0.5, fill_alpha + 0.12)
    edge_r = max(0.018, min(0.06, 0.008 * span))

    c0, c1, c2, c3 = corners
    obj = ["BEGIN", "TRIANGLES"]
    _color(obj, fill, fill_alpha)
    for tri in ((c0, c1, c2), (c0, c2, c3), (c0, c2, c1), (c0, c3, c2)):
        for p in tri:
            obj.extend(["VERTEX", float(p[0]), float(p[1]), float(p[2])])
    obj.append("END")
    obj.extend(["ALPHA", 1.0])

    for a, b in ((c0, c1), (c1, c2), (c2, c3), (c3, c0)):
        _cylinder(obj, a, b, edge_r, edge)

    eye_dist = min(max(1.15, 0.22 * span), 8.0)
    sclera_r = min(max(0.28, 0.07 * span), 1.8)
    eye = center + n * eye_dist
    toward = -n
    iris_r = 0.52 * sclera_r
    pupil_r = 0.24 * sclera_r
    _, u_hat, v_hat = _frame(normal)
    iris = eye + toward * (0.62 * sclera_r)
    pupil = eye + toward * (0.82 * sclera_r)
    highlight = eye + toward * (0.55 * sclera_r) + u_hat * (0.28 * sclera_r)
    highlight = highlight + v_hat * (0.22 * sclera_r)

    _sphere(obj, eye, sclera_r, _SCLERA)
    _sphere(obj, iris, iris_r, _IRIS)
    _sphere(obj, pupil, pupil_r, _PUPIL)
    _sphere(obj, highlight, 0.11 * sclera_r, _HIGHLIGHT)

    lid_center = eye + toward * (0.08 * sclera_r)
    ru, rv = 1.18 * sclera_r, 0.58 * sclera_r
    lid_r = max(0.016, 0.06 * sclera_r)

    def arc(t0, t1, steps=7):
        pts = []
        for i in range(steps):
            t = t0 + (t1 - t0) * (float(i) / float(steps - 1))
            pts.append(lid_center + ru * math.cos(t) * u_hat + rv * math.sin(t) * v_hat)
        return pts

    _poly_chain(obj, arc(0.22, math.pi - 0.22), lid_r, _LID)
    _poly_chain(obj, arc(math.pi + 0.22, 2.0 * math.pi - 0.22), lid_r, _LID)

    gaze_end = center + n * min(0.15 * span, 1.2)
    _cylinder(obj, pupil, gaze_end, max(0.012, 0.035 * sclera_r), _GAZE)
    if axis_arrow:
        obj.extend(_axis_move_arrow(center, n, span, selected=selected))
    return obj


def _axis_move_arrow(center, normal, span, selected=False):
    """One translation arrow along the keep-side normal (no XYZ rings)."""
    n = _unit(normal)
    length = min(max(1.6, 0.28 * float(span)), 7.0)
    shaft_r = max(0.03, 0.012 * float(span))
    head_len = min(max(0.45, 0.35 * length), 1.6)
    head_r = shaft_r * 2.4
    rgb = _PLANE_EDGE_SEL if selected else _PLANE_EDGE
    base = center
    tip = center + n * length
    shaft_end = tip - n * head_len
    obj = []
    _cylinder(obj, base, shaft_end, shaft_r, rgb)
    obj.extend([
        "CONE",
        float(shaft_end[0]), float(shaft_end[1]), float(shaft_end[2]),
        float(tip[0]), float(tip[1]), float(tip[2]),
        float(head_r), 0.02,
        float(rgb[0]), float(rgb[1]), float(rgb[2]),
        float(rgb[0]), float(rgb[1]), float(rgb[2]),
        1.0, 1.0,
    ])
    return obj
