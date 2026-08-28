"""Build styled arrow CGOs from point pairs."""

from __future__ import annotations

from typing import Sequence

from ...meshes.Points import Points
from ...util.cgo import lines_cgo, mesh_cone_cgo, mesh_cylinder_cgo, native_spheres_cgo
from .line_style import ARROW_QUALITY_SEGMENTS, LineStyle, apply_margin, dash_on_segments
from .pairs import VisualPair

DEFAULT_SHAFT_RADIUS = 0.045
HEAD_LENGTH = 0.28
HEAD_WIDTH = 2.4


def _direction(p0, p1):
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    dz = p1[2] - p0[2]
    length = (dx * dx + dy * dy + dz * dz) ** 0.5
    if length < 1e-8:
        return (0.0, 0.0, 1.0), 0.0
    return (dx / length, dy / length, dz / length), length


def _offset(point, direction, distance):
    return (
        point[0] + direction[0] * distance,
        point[1] + direction[1] * distance,
        point[2] + direction[2] * distance,
    )


def _arrow_heads(style: LineStyle):
    ends = style.ends
    return ends in ("Arrow", "Double arrow"), ends == "Double arrow", ends == "Circles"


def arrow_cgo(
    start: Sequence[float],
    end: Sequence[float],
    color,
    quality: int,
    style: LineStyle,
    alpha: float = 1.0,
    radius: float = DEFAULT_SHAFT_RADIUS,
) -> list:
    p0, p1 = apply_margin(start, end, style.margin)
    direction, length = _direction(p0, p1)
    if length < 1e-8:
        return []
    head_end, head_start, circles = _arrow_heads(style)
    head_len = min(length * HEAD_LENGTH, length * 0.45) if (head_end or head_start) else 0.0
    shaft0 = _offset(p0, direction, head_len if head_start else 0.0)
    shaft1 = _offset(p1, direction, -head_len if head_end else 0.0)
    if _direction(shaft0, shaft1)[1] < 1e-8:
        shaft0, shaft1 = p0, p1
        head_len = 0.0
        head_end = head_start = False

    quality = max(0, min(5, int(quality)))
    obj = []
    if quality == 0:
        segs = dash_on_segments(shaft0, shaft1, style.pattern(), style.dash_scale)
        obj.extend(lines_cgo(segs, color, width=2.4, alpha=alpha))
        if head_end:
            obj.extend(_line_arrowhead(p1, direction, color, alpha))
        if head_start:
            obj.extend(_line_arrowhead(p0, (-direction[0], -direction[1], -direction[2]), color, alpha))
        if circles:
            obj.extend(native_spheres_cgo([p0, p1], radius * 1.6, color, alpha=alpha))
        return obj

    n_seg = ARROW_QUALITY_SEGMENTS[quality]
    segs = dash_on_segments(shaft0, shaft1, style.pattern(), style.dash_scale)
    for a, b in segs:
        obj.extend(mesh_cylinder_cgo(a, b, radius, color, n_seg=n_seg, alpha=alpha))
    head_r = radius * HEAD_WIDTH
    if head_end:
        obj.extend(mesh_cone_cgo(shaft1, p1, head_r, color, n_seg=n_seg, alpha=alpha))
    if head_start:
        obj.extend(mesh_cone_cgo(shaft0, p0, head_r, color, n_seg=n_seg, alpha=alpha))
    if circles:
        obj.extend(native_spheres_cgo([p0, p1], radius * 1.8, color, alpha=alpha))
    return obj


def _line_arrowhead(tip, direction, color, alpha):
    size = 0.18
    from ...util.cgo import _perp_frame

    perp, bitan = _perp_frame(direction)
    base = _offset(tip, direction, -size)
    left = (
        base[0] + perp[0] * size * 0.45,
        base[1] + perp[1] * size * 0.45,
        base[2] + perp[2] * size * 0.45,
    )
    right = (
        base[0] - perp[0] * size * 0.45,
        base[1] - perp[1] * size * 0.45,
        base[2] - perp[2] * size * 0.45,
    )
    return lines_cgo([(left, tip), (right, tip)], color, width=2.4, alpha=alpha)


def build_arrow_cgo_list(pairs: Sequence[VisualPair], quality: int, style: LineStyle) -> list:
    merged = []
    for pair in pairs:
        merged.extend(arrow_cgo(pair.start.xyz(), pair.end.xyz(), pair.color, quality, style, pair.alpha))
    return merged


class TokenCGO(Points):
    """Displayable wrapper around a prebuilt CGO token list."""

    def __init__(self, tokens, name):
        import numpy as np

        Points.__init__(
            self, np.zeros((1, 3)), color=(1, 1, 1), name=name, bypass_colormap=True,
        )
        self._tokens = tokens

    def _create_CGO_list(self):
        return self._tokens
