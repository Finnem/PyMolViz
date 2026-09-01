from __future__ import annotations

import numpy as np
from .Lines import Lines
from ..points import as_point_source, point_sources_from_sequence, resolve_xyz
from ..util.cgo import lines_cgo, mesh_cone_cgo, mesh_cylinder_cgo, native_spheres_cgo
from ..util.line_style import (
    ARROW_QUALITY_SEGMENTS,
    LineStyle,
    apply_margin,
    dash_on_segments,
)
from ..util.math import get_perp

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


def _line_arrowhead(tip, direction, color, alpha):
    size = 0.18
    from ..util.cgo import _perp_frame

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


def build_styled_arrow_cgo(
    start,
    end,
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


class Arrows(Lines):
    """Arrows with wizard-compatible dash / margin / end-cap styling."""

    def __init__(
        self,
        lines=None,
        color=None,
        name=None,
        state=1,
        transparency=0,
        colormap="RdYlBu_r",
        linewidth=0.05,
        head_length=0.25,
        head_width=1.618,
        render_as="cylinders",
        starts=None,
        ends=None,
        arrow_mask=None,
        quality=3,
        line_style=None,
        dash="Solid",
        dash_scale=1.0,
        margin=0.0,
        ends_style="Arrow",
        shaft_radius=DEFAULT_SHAFT_RADIUS,
        use_styled_cgo=False,
        *args,
        **kwargs,
    ) -> None:
        self.original_color = color
        self.head_length = head_length
        self.head_width = head_width
        self.quality = int(quality)
        self.shaft_radius = float(shaft_radius)
        self.use_styled_cgo = bool(use_styled_cgo)
        if line_style is not None:
            self.line_style = line_style
        else:
            self.line_style = LineStyle(dash=dash, dash_scale=dash_scale, margin=margin, ends=ends_style)

        if lines is None:
            if starts is None or ends is None:
                raise ValueError("Either lines or start and end must be given.")
            self._start_sources = point_sources_from_sequence(starts)
            self._end_sources = point_sources_from_sequence(ends)
            starts_arr = np.array([resolve_xyz(s) for s in self._start_sources])
            ends_arr = np.array([resolve_xyz(s) for s in self._end_sources])
            lines = np.hstack([starts_arr, ends_arr])
        else:
            lines = np.array(lines)
            if starts is not None and ends is not None:
                self._start_sources = point_sources_from_sequence(starts)
                self._end_sources = point_sources_from_sequence(ends)
            else:
                pairs = np.asarray(lines, dtype=float).reshape(-1, 6)
                self._start_sources = [as_point_source(row[:3]) for row in pairs]
                self._end_sources = [as_point_source(row[3:]) for row in pairs]

        self.arrow_mask = arrow_mask if arrow_mask is not None else np.ones(int(len(lines)), dtype=bool)

        if use_styled_cgo:
            super().__init__(
                lines.reshape(-1, 3), color, name, state, transparency, colormap,
                linewidth, "lines", *args, **kwargs,
            )
            self.render_as = render_as
            self.linewidth = linewidth
            return

        try:
            if (not np.issubdtype(type(color), np.str_)) and (not (color is None)):
                if (len(color) == (len(lines.reshape(-1, 3)) / 2)):
                    self.original_color = np.repeat(color, 2, axis=0)
                    color = np.repeat(color, 10, axis=0)
                elif (len(color) == len(lines.reshape(-1, 3))):
                    color = np.hstack([
                        color[::2, None],
                        np.repeat(color[1::2], 9, axis=0).reshape(-1, 9),
                    ]).flatten()
        except TypeError:
            pass
        original_lines = lines.reshape(-1, 6)
        new_lines = np.zeros((original_lines.shape[0] * 4, 6))
        for i, line in enumerate(original_lines):
            start = line[:3]
            end = line[3:]
            vector = end - start
            head_start = (vector) * (1 - head_length) + start
            perp = get_perp(vector)
            x1 = head_start + perp * head_width
            x2 = head_start - perp * head_width
            ortho = np.cross(vector, perp)
            ortho /= np.linalg.norm(ortho)
            y1 = head_start + ortho * head_width
            y2 = head_start - ortho * head_width
            new_lines[i * 4] = np.hstack([end, x1])
            new_lines[i * 4 + 1] = np.hstack([end, x2])
            new_lines[i * 4 + 2] = np.hstack([end, y1])
            new_lines[i * 4 + 3] = np.hstack([end, y2])
        lines = np.hstack([original_lines, new_lines.reshape(-1, 24)])
        self.transparency = transparency
        try:
            self.transparency[0]
        except TypeError:
            self.transparency = np.full(int(original_lines.shape[0]), self.transparency)
        super().__init__(
            lines.reshape(-1, 3), color, name, state, self.transparency, colormap,
            linewidth, render_as, starts=starts, ends=ends, *args, **kwargs,
        )
        self.linewidth = linewidth

    def _expand_heads(self, starts_arr, ends_arr):
        original_lines = np.hstack([starts_arr, ends_arr])
        new_lines = np.zeros((original_lines.shape[0] * 4, 6))
        for i, line in enumerate(original_lines):
            start = line[:3]
            end = line[3:]
            vector = end - start
            head_start = (vector) * (1 - self.head_length) + start
            perp = get_perp(vector)
            x1 = head_start + perp * self.head_width
            x2 = head_start - perp * self.head_width
            ortho = np.cross(vector, perp)
            ortho /= np.linalg.norm(ortho) + 1e-15
            y1 = head_start + ortho * self.head_width
            y2 = head_start - ortho * self.head_width
            new_lines[i * 4] = np.hstack([end, x1])
            new_lines[i * 4 + 1] = np.hstack([end, x2])
            new_lines[i * 4 + 2] = np.hstack([end, y1])
            new_lines[i * 4 + 3] = np.hstack([end, y2])
        return np.hstack([original_lines, new_lines.reshape(-1, 24)]).reshape(-1, 3)

    def rebuild(self, context=None) -> None:
        self.invalidate_cgo_cache()
        if self._start_sources is None or self._end_sources is None:
            return
        starts_arr = np.array([resolve_xyz(s, context) for s in self._start_sources])
        ends_arr = np.array([resolve_xyz(s, context) for s in self._end_sources])
        if self.use_styled_cgo:
            self.vertices = np.hstack([starts_arr, ends_arr]).reshape(-1, 3)
            return
        self.vertices = self._expand_heads(starts_arr, ends_arr)

    def from_start_end(
        starts,
        ends,
        color=None,
        name=None,
        state=1,
        transparency=0,
        colormap="RdYlBu_r",
        linewidth=0.05,
        head_length=0.25,
        head_width=1.618,
        render_as="cylinders",
        quality=3,
        line_style=None,
        *args,
        **kwargs,
    ):
        starts = np.array(starts).reshape(-1, 3)
        ends = np.array(ends).reshape(-1, 3)
        return Arrows(
            np.hstack([starts, ends]), color, name, state, transparency, colormap,
            linewidth, head_length, head_width, render_as, starts, ends,
            quality=quality, line_style=line_style, *args, **kwargs,
        )

    def _create_CGO_list(self) -> list:
        if self.use_styled_cgo:
            merged = []
            n_pairs = self.vertices.reshape(-1, 2, 3).shape[0]
            colors = self.color
            if getattr(self, "bypass_colormap", False):
                cgo_colors = np.asarray(colors, dtype=float).reshape(-1, 3)
            else:
                cgo_colors = self.colormap.get_color(colors)[:, :3]
            if cgo_colors.shape[0] == n_pairs:
                per_pair_colors = cgo_colors
            else:
                per_pair_colors = cgo_colors.reshape(-1, 2, 3)[:, 0, :]
            transparency = self.transparency
            try:
                transparency[0]
            except (TypeError, IndexError):
                transparency = np.full(n_pairs, float(self.transparency) if np.isscalar(self.transparency) else 0.0)
            for i in range(n_pairs):
                start = tuple(self.vertices[i * 2])
                end = tuple(self.vertices[i * 2 + 1])
                color = tuple(per_pair_colors[i])
                alpha = 1.0 - float(transparency[i] if i < len(transparency) else 0.0)
                merged.extend(
                    build_styled_arrow_cgo(
                        start, end, color, self.quality, self.line_style,
                        alpha=alpha, radius=self.shaft_radius,
                    )
                )
            return merged

        if self.render_as == "lines":
            return super()._create_CGO_list()

        cgo_list = []
        starts = self.vertices[::10]
        ends = self.vertices[1::10]
        cylinder_ends = starts + ((ends - starts) * (1 - self.head_length))
        cylinder_ends[~self.arrow_mask] = ends[~self.arrow_mask]
        cgo_colors = self.colormap.get_color(self.color)[:, :3].reshape(-1, 3)
        start_colors = cgo_colors[::10]
        end_colors = cgo_colors[1::10]
        transparency = 1 - self.transparency
        cylinders = np.hstack([
            np.full(starts.shape[0], "ALPHA")[:, None], transparency[:, None],
            np.full(starts.shape[0], "CONE")[:, None], starts, cylinder_ends,
            np.full(starts.shape[0], self.linewidth)[:, None],
            np.full(starts.shape[0], self.linewidth)[:, None],
            start_colors, end_colors, np.full((starts.shape[0], 2), (1.0, 0.0)),
        ]).flatten()
        cgo_list.extend(cylinders)
        cones = np.hstack([
            np.full(starts.shape[0], "ALPHA")[:, None], transparency[:, None],
            np.full(starts.shape[0], "CONE")[:, None], cylinder_ends, ends,
            np.full(starts.shape[0], self.linewidth * self.head_width)[:, None],
            np.full(starts.shape[0], 0.0)[:, None],
            end_colors, end_colors, np.full((starts.shape[0], 2), 0.0),
        ])
        cones = cones[self.arrow_mask]
        cgo_list.extend(cones.flatten())
        return cgo_list
