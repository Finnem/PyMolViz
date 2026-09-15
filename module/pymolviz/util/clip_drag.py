"""Hitchhike on PyMOL's native ``cmd.drag`` manipulator for clip planes.

A small dummy sphere sits at the plane's visual center. ``cmd.drag(..., mode=1)``
shows the same arrows/rings as right-click → Drag, and updates the object TTT.
``wizard=0`` keeps the PyMolViz wizard on the stack.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from .pymol_helpers import restore_view

CLIP_DRAG_NAME = "_pmv_prev_clip_drag"
_IDENTITY = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


def _as_matrix16(matrix) -> Tuple[float, ...]:
    if matrix is None:
        return _IDENTITY
    flat = []
    try:
        for item in matrix:
            try:
                flat.extend(float(v) for v in item)
            except TypeError:
                flat.append(float(item))
    except TypeError:
        return _IDENTITY
    if len(flat) < 16:
        return _IDENTITY
    return tuple(flat[:16])


def homogeneous_from_ttt(ttt) -> Tuple[float, ...]:
    """Convert a PyMOL object TTT (post-translation in the last row) to 4x4."""
    r00, r01, r02, p0, r10, r11, r12, p1, r20, r21, r22, p2, t0, t1, t2, _w = _as_matrix16(ttt)
    tx = r00 * t0 + r01 * t1 + r02 * t2 + p0
    ty = r10 * t0 + r11 * t1 + r12 * t2 + p1
    tz = r20 * t0 + r21 * t1 + r22 * t2 + p2
    return (
        r00, r01, r02, tx,
        r10, r11, r12, ty,
        r20, r21, r22, tz,
        0.0, 0.0, 0.0, 1.0,
    )


def rotation_and_translation(matrix) -> Tuple[np.ndarray, np.ndarray]:
    """``(R, t)`` from a homogenous ``get_object_matrix`` 4x4 (row-major)."""
    m = _as_matrix16(matrix)
    rot = np.array(
        (
            (m[0], m[1], m[2]),
            (m[4], m[5], m[6]),
            (m[8], m[9], m[10]),
        ),
        dtype=float,
    )
    trans = np.array((m[3], m[7], m[11]), dtype=float)
    return rot, trans


def apply_drag_matrix(rest_origin, rest_normal, rest_center, matrix):
    """Map a drag TTT onto the clip pose stored when dragging started.

    Rotation is about the visual center (where the dummy atom sits).
    """
    rot, trans = rotation_and_translation(matrix)
    origin = np.asarray(rest_origin, dtype=float).reshape(3)
    normal = np.asarray(rest_normal, dtype=float).reshape(3)
    center = np.asarray(rest_center, dtype=float).reshape(3)
    new_center = rot @ center + trans
    new_origin = new_center + rot @ (origin - center)
    new_normal = rot @ normal
    ln = float(np.linalg.norm(new_normal))
    if ln > 1e-12:
        new_normal = new_normal / ln
    else:
        new_normal = np.array([0.0, 0.0, 1.0], dtype=float)
    return (
        [float(new_origin[0]), float(new_origin[1]), float(new_origin[2])],
        [float(new_normal[0]), float(new_normal[1]), float(new_normal[2])],
        [float(new_center[0]), float(new_center[1]), float(new_center[2])],
    )


def apply_axis_drag_matrix(rest_origin, rest_normal, rest_center, matrix):
    """Like :func:`apply_drag_matrix`, but only the translation along ``rest_normal``.

    Rings on PyMOL's manipulator are ignored so field-crop faces stay on X/Y/Z.
    """
    rot, trans = rotation_and_translation(matrix)
    origin = np.asarray(rest_origin, dtype=float).reshape(3)
    normal = np.asarray(rest_normal, dtype=float).reshape(3)
    center = np.asarray(rest_center, dtype=float).reshape(3)
    ln = float(np.linalg.norm(normal))
    if ln > 1e-12:
        normal = normal / ln
    else:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
    moved = rot @ center + trans
    shift = float(np.dot(moved - center, normal))
    new_origin = origin + normal * shift
    new_center = center + normal * shift
    return (
        [float(new_origin[0]), float(new_origin[1]), float(new_origin[2])],
        [float(normal[0]), float(normal[1]), float(normal[2])],
        [float(new_center[0]), float(new_center[1]), float(new_center[2])],
    )


def matrix_changed(previous, current, atol: float = 1e-5) -> bool:
    old = _as_matrix16(previous)
    new = _as_matrix16(current)
    return any(abs(a - b) > atol for a, b in zip(old, new))


def read_object_matrix(cmd_, name: str = CLIP_DRAG_NAME) -> Tuple[float, ...]:
    """Object transform as a homogenous 4x4 (translation in the last column).

    Prefer ``get_object_ttt``: ``get_object_matrix`` returns ``None`` until a
    state matrix exists, and that ``None`` must not be treated as identity
    while a TTT is already in play.
    """
    ttt = None
    try:
        ttt = cmd_.get_object_ttt(name)
    except Exception:
        ttt = None
    if ttt is not None:
        return homogeneous_from_ttt(ttt)
    try:
        matrix = cmd_.get_object_matrix(name)
    except Exception:
        matrix = None
    if matrix is None:
        return _IDENTITY
    return _as_matrix16(matrix)


def stop_clip_drag(cmd_, name: str = CLIP_DRAG_NAME, button_mode=None) -> None:
    """Leave drag mode and drop the dummy object."""
    try:
        cmd_.drag()
    except Exception:
        pass
    try:
        cmd_.delete(name)
    except Exception:
        pass
    if button_mode is not None:
        try:
            cmd_.set("button_mode", button_mode, quiet=1)
        except TypeError:
            try:
                cmd_.set("button_mode", button_mode)
            except Exception:
                pass
        except Exception:
            pass
        try:
            cmd_.mouse()
        except Exception:
            pass


def start_clip_drag(cmd_, center: Sequence[float], name: str = CLIP_DRAG_NAME, *, native_widget=True):
    """Place a dummy sphere at ``center`` and enable the native drag widget.

    ``native_widget=False`` keeps object TTT dragging without PyMOL's 3-axis
    rings so a uniaxial CGO arrow can be the only handle.

    Returns the previous ``button_mode`` so the caller can restore it.
    """
    old_mode = None
    try:
        old_mode = cmd_.get("button_mode")
    except Exception:
        pass
    stop_clip_drag(cmd_, name=name)
    view = None
    try:
        view = tuple(cmd_.get_view())
    except Exception:
        pass
    pos = [float(center[0]), float(center[1]), float(center[2])]
    try:
        cmd_.pseudoatom(name, pos=pos)
    except Exception:
        return old_mode
    try:
        cmd_.hide("everything", name)
    except Exception:
        pass
    if native_widget:
        try:
            cmd_.show("spheres", name)
        except Exception:
            pass
        try:
            cmd_.set("sphere_scale", 0.35, name)
        except Exception:
            pass
    try:
        cmd_.set("matrix_mode", 1, name)
    except TypeError:
        try:
            cmd_.set("matrix_mode", 1, name, quiet=1)
        except Exception:
            pass
    except Exception:
        pass
    try:
        cmd_.drag(name, wizard=0, edit=1 if native_widget else 0, mode=1)
    except TypeError:
        try:
            cmd_.drag(name, wizard=0, edit=1 if native_widget else 0)
        except TypeError:
            try:
                cmd_.drag(name)
            except Exception:
                pass
    except Exception:
        pass
    if not native_widget:
        try:
            cmd_.edit_mode(0)
        except Exception:
            pass
    restore_view(cmd_, view)
    return old_mode
