"""Pure helpers for PyMOL's 18-float get_view() matrix.

No pymol import — safe for script generation and non-PyMOL tooling.
View layout (column-major 3x3 model→camera in [0:9]):
  [9:12]  origin in camera space
  [12:15] origin in model space
  [15:16] near/far clip
  [17]    ortho flag / FOV-related
"""

import math


def camera_to_model_offset(view, cam_offset):
    """Map a camera-space offset through R^T into model space."""
    dx, dy, dz = (float(cam_offset[0]), float(cam_offset[1]), float(cam_offset[2]))
    return (
        view[0] * dx + view[1] * dy + view[2] * dz,
        view[3] * dx + view[4] * dy + view[5] * dz,
        view[6] * dx + view[7] * dy + view[8] * dz,
    )


def model_to_camera(view, pos):
    """OpenGL camera space: +X right, +Y up, look down -Z."""
    rel = (pos[0] - view[12], pos[1] - view[13], pos[2] - view[14])
    return (
        view[0] * rel[0] + view[3] * rel[1] + view[6] * rel[2] + view[9],
        view[1] * rel[0] + view[4] * rel[1] + view[7] * rel[2] + view[10],
        view[2] * rel[0] + view[5] * rel[1] + view[8] * rel[2] + view[11],
    )


def screen_center(view):
    """Model-space point currently in the middle of the viewer."""
    # Origin in camera space is view[9:12]. Screen center at that depth is (0, 0, z).
    rx, ry, rz = camera_to_model_offset(view, (-float(view[9]), -float(view[10]), 0.0))
    return (float(view[12]) + rx, float(view[13]) + ry, float(view[14]) + rz)


def click_ray_points(view, sx, sy, width, height, fov, n=8):
    """Model-space samples along the pick ray through a framebuffer pixel.

    ``sx, sy`` are PyMOL framebuffer coordinates (Y from the bottom).
    """
    width = max(float(width), 1.0)
    height = max(float(height), 1.0)
    origin_depth = max(abs(float(view[11])), 1e-6)
    fov_width = 2.0 * math.tan(math.radians(max(abs(float(fov)), 1.0)) / 2.0)
    angstrom_per_px = origin_depth * fov_width / height
    cam_x = (float(sx) - width * 0.5) * angstrom_per_px
    cam_y = (float(sy) - height * 0.5) * angstrom_per_px
    plane = camera_to_model_offset(
        view,
        (cam_x - float(view[9]), cam_y - float(view[10]), 0.0),
    )
    origin = (
        float(view[12]) + plane[0],
        float(view[13]) + plane[1],
        float(view[14]) + plane[2],
    )
    look = camera_to_model_offset(view, (0.0, 0.0, -1.0))
    near = max(float(view[15]), 0.5)
    far = max(float(view[16]), near + 1.0)
    start = near - origin_depth
    span = far - near
    if n < 2:
        n = 2
    points = []
    for i in range(n):
        t = start + span * (float(i) / float(n - 1))
        points.append((
            origin[0] + look[0] * t,
            origin[1] + look[1] * t,
            origin[2] + look[2] * t,
        ))
    return points


def translation_ttt(center):
    """Identity rotation + translation as a 16-float object TTT matrix."""
    return [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        float(center[0]), float(center[1]), float(center[2]), 1.0,
    ]


def scale_translation_ttt(center, scale):
    """Uniform scale + translate for PyMOL object TTT.

    PyMOL applies the bottom row before the diagonal scale (see ViewportCallback),
    so world position = scale * local + scale * t_row  =>  t_row = center / scale.
    """
    s = float(scale)
    if abs(s) < 1e-12:
        s = 1e-12
    cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))
    return [
        s, 0.0, 0.0, 0.0,
        0.0, s, 0.0, 0.0,
        0.0, 0.0, s, 0.0,
        cx / s, cy / s, cz / s, 1.0,
    ]


def box_object_ttt(center, extent):
    """Non-uniform scale + translate for a unit box (-1..1 on each axis)."""
    sx = float(extent[0]) / 2.0
    sy = float(extent[1]) / 2.0
    sz = float(extent[2]) / 2.0
    if abs(sx) < 1e-12:
        sx = 1e-12
    if abs(sy) < 1e-12:
        sy = 1e-12
    if abs(sz) < 1e-12:
        sz = 1e-12
    cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))
    return [
        sx, 0.0, 0.0, 0.0,
        0.0, sy, 0.0, 0.0,
        0.0, 0.0, sz, 0.0,
        cx / sx, cy / sy, cz / sz, 1.0,
    ]
