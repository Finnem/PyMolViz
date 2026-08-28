"""Pure helpers for PyMOL's 18-float get_view() matrix.

No pymol import — safe for script generation and non-PyMOL tooling.
View layout (column-major 3x3 model→camera in [0:9]):
  [9:12]  origin in camera space
  [12:15] origin in model space
  [15:16] near/far clip
  [17]    ortho flag / FOV-related
"""


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


def translation_ttt(center):
    """Identity rotation + translation as a 16-float object TTT matrix."""
    return [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        float(center[0]), float(center[1]), float(center[2]), 1.0,
    ]
