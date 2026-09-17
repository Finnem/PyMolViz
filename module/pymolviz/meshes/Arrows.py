from __future__ import annotations

import numpy as np
from .Points import Points
from ..points import as_point_source, point_sources_from_sequence, resolve_xyz
from ..util.cgo import lines_cgo, mesh_cone_cgo, native_spheres_cgo
from ..util.line_style import (
    ARROW_QUALITY_SEGMENTS,
    LineStyle,
    absolute_head_length,
    apply_margin,
    arrow_cone_segments,
    dash_on_segments,
    default_head_length,
    style_margins,
)
from ..util.math import get_perp
from ..util.mesh_clip import clip_segment_by_planes, normalize_clip_planes

DEFAULT_SHAFT_RADIUS = 0.045
HEAD_WIDTH = 2.4
HEAD_LENGTH = default_head_length(DEFAULT_SHAFT_RADIUS)
# Quality-0 LINEWIDTH is pixels; keep the default shaft looking like width 2.4.
_LINE_WIDTH_PER_ANGSTROM = 2.4 / DEFAULT_SHAFT_RADIUS


def _native_cylinder_cgo(p0, p1, radius, color, alpha=1.0, color_end=None):
    """Native CGO CYLINDER (ray-traces; equal-radius CONE does not)."""
    c0 = [float(c) for c in color[:3]]
    c1 = [float(c) for c in (color if color_end is None else color_end)[:3]]
    r = float(radius)
    a = max(0.0, min(1.0, float(alpha)))
    return [
        "ALPHA", a,
        "CYLINDER",
        float(p0[0]), float(p0[1]), float(p0[2]),
        float(p1[0]), float(p1[1]), float(p1[2]),
        r,
        c0[0], c0[1], c0[2], c1[0], c1[1], c1[2],
    ]


def _coerce_line_style(line_style, dash, dash_scale, margin, ends_style, render_ends):
    if line_style is not None:
        return line_style.copy() if hasattr(line_style, "copy") else line_style
    if render_ends:
        return LineStyle(dash=dash, dash_scale=dash_scale, margin=margin, ends="Circles")
    return LineStyle(dash=dash, dash_scale=dash_scale, margin=margin, ends=ends_style)


def _pair_vertices(lines, starts, ends):
    existing_starts = None
    existing_ends = None
    provided = lines
    if starts is not None and ends is not None:
        start_sources = point_sources_from_sequence(starts)
        end_sources = point_sources_from_sequence(ends)
        if provided is None:
            starts_arr = np.array([resolve_xyz(s) for s in start_sources], dtype=float).reshape(-1, 3)
            ends_arr = np.array([resolve_xyz(s) for s in end_sources], dtype=float).reshape(-1, 3)
            lines = np.hstack([starts_arr, ends_arr])
        else:
            lines = np.asarray(lines, dtype=float)
        return np.asarray(lines, dtype=float), start_sources, end_sources
    lines = np.asarray(lines, dtype=float) if lines is not None else None
    if lines is None or lines.size == 0:
        return np.zeros((0, 3), dtype=float), [], []
    pairs = np.asarray(lines, dtype=float).reshape(-1, 6)
    start_sources = [as_point_source(row[:3]) for row in pairs]
    end_sources = [as_point_source(row[3:]) for row in pairs]
    return pairs.reshape(-1, 3), start_sources, end_sources


def default_head_radius(shaft_radius: float) -> float:
    return float(shaft_radius) * HEAD_WIDTH


def head_radius_follows_shaft(shaft_radius, head_radius, *, tol=1e-4) -> bool:
    if head_radius is None:
        return True
    return abs(float(head_radius) - default_head_radius(shaft_radius)) < tol


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
    start = getattr(style, "start_head", None)
    end = getattr(style, "end_head", None)
    if start is None and end is None:
        from ..util.line_style import split_ends

        start, end = split_ends(getattr(style, "ends", "Arrow"))
    return end == "Arrow", start == "Arrow", start == "Circles", end == "Circles"


def _line_arrowhead(tip, direction, color, alpha, size=0.18):
    size = max(float(size), 0.02)
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


def _lerp_rgb(a, b, t):
    t = max(0.0, min(1.0, float(t)))
    return tuple((1.0 - t) * float(a[i]) + t * float(b[i]) for i in range(3))


def _lerp_xyz(p0, p1, t):
    t = max(0.0, min(1.0, float(t)))
    return tuple((1.0 - t) * float(p0[i]) + t * float(p1[i]) for i in range(3))


def _colors_close(a, b, tol=1e-4) -> bool:
    return all(abs(float(a[i]) - float(b[i])) < tol for i in range(3))


def _project_t(p0, p1, p) -> float:
    dx = float(p1[0]) - float(p0[0])
    dy = float(p1[1]) - float(p0[1])
    dz = float(p1[2]) - float(p0[2])
    length2 = dx * dx + dy * dy + dz * dz
    if length2 < 1e-16:
        return 0.0
    t = (
        (float(p[0]) - float(p0[0])) * dx
        + (float(p[1]) - float(p0[1])) * dy
        + (float(p[2]) - float(p0[2])) * dz
    ) / length2
    return max(0.0, min(1.0, t))


GRADIENT_SHAFT_SLICES = 12


def _gradient_shaft_cgo(p0, p1, radius, c0, c1, alpha, quality, segs):
    """Shaft as a start→end color gradient (native CYLINDER, or colored 2D lines)."""
    if not segs:
        return []
    if quality == 0:
        obj = []
        line_w = max(1.0, float(radius) * _LINE_WIDTH_PER_ANGSTROM)
        for a, b in segs:
            ta = _project_t(p0, p1, a)
            tb = _project_t(p0, p1, b)
            obj.extend(lines_cgo([(a, b)], _lerp_rgb(c0, c1, 0.5 * (ta + tb)), width=line_w, alpha=alpha))
        return obj
    if len(segs) == 1 and not _colors_close(c0, c1):
        obj = []
        n = GRADIENT_SHAFT_SLICES
        for i in range(n):
            t0 = i / float(n)
            t1 = (i + 1) / float(n)
            obj.extend(_native_cylinder_cgo(
                _lerp_xyz(p0, p1, t0),
                _lerp_xyz(p0, p1, t1),
                radius,
                _lerp_rgb(c0, c1, t0),
                alpha=alpha,
                color_end=_lerp_rgb(c0, c1, t1),
            ))
        return obj
    obj = []
    for a, b in segs:
        ta = _project_t(p0, p1, a)
        tb = _project_t(p0, p1, b)
        obj.extend(_native_cylinder_cgo(
            a, b, radius, _lerp_rgb(c0, c1, ta), alpha=alpha,
            color_end=_lerp_rgb(c0, c1, tb),
        ))
    return obj


def build_styled_arrow_cgo(
    start,
    end,
    color,
    quality: int,
    style: LineStyle,
    alpha: float = 1.0,
    radius: float = DEFAULT_SHAFT_RADIUS,
    head_length: float = None,
    head_radius: float = None,
    color_end=None,
) -> list:
    head_end, head_start, circle_start, circle_end = _arrow_heads(style)
    c0 = color
    c1 = color if color_end is None else color_end
    n_heads = int(head_end) + int(head_start)
    wanted_head = (
        default_head_length(radius) if head_length is None else max(float(head_length), 0.0)
    )
    start_pad, end_pad = style_margins(style)
    p0, p1 = apply_margin(
        start,
        end,
        head_length=wanted_head if n_heads else 0.0,
        double_head=bool(head_start and head_end),
        start_margin=start_pad,
        end_margin=end_pad,
    )
    direction, length = _direction(p0, p1)
    if length < 1e-8:
        return []
    head_len = (
        absolute_head_length(radius, length, wanted_head, n_heads=n_heads)
        if n_heads
        else 0.0
    )
    shaft0 = _offset(p0, direction, head_len if head_start else 0.0)
    shaft1 = _offset(p1, direction, -head_len if head_end else 0.0)
    shaft_len = _direction(shaft0, shaft1)[1]

    quality = max(0, min(5, int(quality)))
    obj = []
    if shaft_len >= 1e-8:
        segs = dash_on_segments(shaft0, shaft1, style.pattern(), style.dash_scale)
        obj.extend(_gradient_shaft_cgo(
            shaft0, shaft1, radius, c0, c1, alpha, quality, segs,
        ))
    if quality == 0:
        if head_end:
            obj.extend(_line_arrowhead(p1, direction, c1, alpha, size=max(head_len, 0.02)))
        if head_start:
            obj.extend(_line_arrowhead(
                p0, (-direction[0], -direction[1], -direction[2]), c0, alpha, size=max(head_len, 0.02),
            ))
        if circle_start:
            obj.extend(native_spheres_cgo([p0], radius * 1.6, c0, alpha=alpha))
        if circle_end:
            obj.extend(native_spheres_cgo([p1], radius * 1.6, c1, alpha=alpha))
        return obj

    n_seg = ARROW_QUALITY_SEGMENTS[quality]
    head_r = float(head_radius) if head_radius is not None else radius * HEAD_WIDTH
    cone_seg = arrow_cone_segments(n_seg, radius, head_r)
    if head_end and head_len > 1e-8:
        obj.extend(mesh_cone_cgo(shaft1, p1, head_r, c1, n_seg=cone_seg, alpha=alpha, cap_base=True))
    if head_start and head_len > 1e-8:
        obj.extend(mesh_cone_cgo(shaft0, p0, head_r, c0, n_seg=cone_seg, alpha=alpha, cap_base=True))
    if circle_start:
        obj.extend(native_spheres_cgo([p0], radius * 1.8, c0, alpha=alpha))
    if circle_end:
        obj.extend(native_spheres_cgo([p1], radius * 1.8, c1, alpha=alpha))
    return obj


class Arrows(Points):
    """Segment visual: arrows or lines, with Options stored as ``line_style``."""

    def __init__(
        self,
        lines=None,
        color=None,
        name=None,
        state=1,
        transparency=0,
        colormap="RdYlBu_r",
        linewidth=None,
        head_length=0.25,
        head_width=1.618,
        render_as="cylinders",
        starts=None,
        ends=None,
        arrow_mask=None,
        quality=None,
        line_style=None,
        dash="Solid",
        dash_scale=1.0,
        margin=0.0,
        ends_style="Arrow",
        shaft_radius=None,
        use_styled_cgo=True,
        head_radius=None,
        clip_planes=None,
        render_ends=False,
        *args,
        **kwargs,
    ) -> None:
        self.original_color = color
        self.head_length = head_length
        self.head_width = head_width
        self.head_radius = None if head_radius is None else float(head_radius)
        if shaft_radius is None:
            shaft_radius = DEFAULT_SHAFT_RADIUS if linewidth is None else float(linewidth)
        self.shaft_radius = float(shaft_radius)
        render_as = "lines" if render_as == "line" else str(render_as or "cylinders")
        if render_as == "cylinder":
            render_as = "cylinders"
        if quality is None:
            quality = 0 if render_as in ("lines",) else 3
        self.quality = int(quality)
        self.pair_radii = None
        self.pair_heads = None
        self.pair_styles = None
        self.use_styled_cgo = bool(use_styled_cgo)
        self.clip_planes = normalize_clip_planes(clip_planes)
        self.line_style = _coerce_line_style(
            line_style, dash, dash_scale, margin, ends_style, bool(render_ends),
        )
        kwargs.pop("render_ends", None)
        vertices, start_sources, end_sources = _pair_vertices(lines, starts, ends)
        self._start_sources = start_sources
        self._end_sources = end_sources
        n_pairs = int(np.asarray(vertices, dtype=float).reshape(-1, 3).shape[0] // 2)
        self.arrow_mask = arrow_mask if arrow_mask is not None else np.ones(max(n_pairs, 0), dtype=bool)

        if self.use_styled_cgo:
            super().__init__(
                vertices, color, name, state, transparency, colormap,
                *args, **kwargs,
            )
            return

        try:
            if (not np.issubdtype(type(color), np.str_)) and (not (color is None)):
                if (len(color) == (len(vertices.reshape(-1, 3)) / 2)):
                    self.original_color = np.repeat(color, 2, axis=0)
                    color = np.repeat(color, 10, axis=0)
                elif (len(color) == len(vertices.reshape(-1, 3))):
                    color = np.hstack([
                        color[::2, None],
                        np.repeat(color[1::2], 9, axis=0).reshape(-1, 9),
                    ]).flatten()
        except TypeError:
            pass
        original_lines = vertices.reshape(-1, 6)
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
            *args, **kwargs,
        )

    @property
    def linewidth(self):
        return float(self.shaft_radius)

    @linewidth.setter
    def linewidth(self, value):
        self.shaft_radius = float(value)

    @property
    def render_as(self):
        return "lines" if int(getattr(self, "quality", 3)) == 0 else "cylinders"

    @render_as.setter
    def render_as(self, value):
        # Points.__init__ assigns render_as="Spheres"; that is not an Arrows mode.
        if value in (None, "Spheres", "Dots"):
            return
        value = "lines" if value == "line" else str(value or "cylinders")
        if value == "cylinder":
            value = "cylinders"
        if value == "lines":
            self.quality = 0
        elif value == "cylinders" and int(getattr(self, "quality", 0)) == 0:
            self.quality = 3

    def options(self) -> dict:
        """Options-section fields: dash, scale, margins, and end caps."""
        style = self.line_style if self.line_style is not None else LineStyle()
        return {
            "dash": style.dash,
            "dash_scale": float(style.dash_scale),
            "start_margin": float(style.start_margin),
            "end_margin": float(style.end_margin),
            "start_head": style.start_head,
            "end_head": style.end_head,
            "line_style": style,
        }

    @property
    def starts(self):
        if self._start_sources is not None:
            return self._start_sources
        verts = self.vertices.reshape(-1, 2, 3)
        return [as_point_source(v[0]) for v in verts]

    @starts.setter
    def starts(self, value):
        self._start_sources = point_sources_from_sequence(value)

    @property
    def ends(self):
        if self._end_sources is not None:
            return self._end_sources
        verts = self.vertices.reshape(-1, 2, 3)
        return [as_point_source(v[1]) for v in verts]

    @ends.setter
    def ends(self, value):
        self._end_sources = point_sources_from_sequence(value)

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
            from ..util.field_sample import paint_mesh_by_field
            paint_mesh_by_field(self)
            return
        self.vertices = self._expand_heads(starts_arr, ends_arr)
        from ..util.field_sample import paint_mesh_by_field
        paint_mesh_by_field(self)

    @staticmethod
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
            np.hstack([starts, ends]),
            color=color,
            name=name,
            state=state,
            transparency=transparency,
            colormap=colormap,
            linewidth=linewidth,
            head_length=head_length,
            head_width=head_width,
            render_as=render_as,
            starts=starts,
            ends=ends,
            quality=quality,
            line_style=line_style,
            *args,
            **kwargs,
        )

    def _create_CGO_list(self) -> list:
        if self.use_styled_cgo or int(self.quality) == 0:
            verts = np.asarray(self.vertices, dtype=float).reshape(-1, 3)
            if verts.shape[0] < 2:
                self._pair_spans = []
                return []
            merged = []
            n_pairs = verts.reshape(-1, 2, 3).shape[0]
            colors = self.color
            if getattr(self, "bypass_colormap", False):
                cgo_colors = np.asarray(colors, dtype=float).reshape(-1, 3)
            else:
                cgo_colors = self.colormap.get_color(colors)[:, :3]
            n_color = cgo_colors.shape[0]
            if n_color == n_pairs * 2:
                start_colors = cgo_colors[::2]
                end_colors = cgo_colors[1::2]
            elif n_color == n_pairs:
                start_colors = cgo_colors
                end_colors = cgo_colors
            elif n_color == 1:
                start_colors = np.repeat(cgo_colors, n_pairs, axis=0)
                end_colors = start_colors
            elif n_color >= n_pairs:
                start_colors = cgo_colors[:n_pairs]
                end_colors = start_colors
            else:
                start_colors = np.resize(cgo_colors, (n_pairs, 3))
                end_colors = start_colors
            transparency = self.transparency
            try:
                transparency[0]
            except (TypeError, IndexError):
                transparency = np.full(n_pairs, float(self.transparency) if np.isscalar(self.transparency) else 0.0)
            spans = []
            radii = list(self.pair_radii or [])
            heads = list(self.pair_heads or [])
            styles = list(getattr(self, "pair_styles", None) or [])
            for i in range(n_pairs):
                start = tuple(self.vertices[i * 2])
                end = tuple(self.vertices[i * 2 + 1])
                color = tuple(start_colors[i])
                color_end = tuple(end_colors[i])
                alpha = 1.0 - float(transparency[i] if i < len(transparency) else 0.0)
                radius = float(radii[i]) if i < len(radii) else self.shaft_radius
                style = styles[i] if i < len(styles) else self.line_style
                head = float(heads[i]) if i < len(heads) else default_head_length(radius)
                clipped = clip_segment_by_planes(
                    start, end, getattr(self, "clip_planes", None),
                )
                if clipped is None:
                    continue
                start, end = clipped
                chunk_start = len(merged)
                merged.extend(
                    build_styled_arrow_cgo(
                        start, end, color, self.quality, style,
                        alpha=alpha, radius=radius, head_length=head,
                        head_radius=getattr(self, "head_radius", None),
                        color_end=color_end,
                    )
                )
                spans.append((chunk_start, len(merged)))
            self._pair_spans = spans
            return merged

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
            np.full(starts.shape[0], "CYLINDER")[:, None], starts, cylinder_ends,
            np.full(starts.shape[0], self.linewidth)[:, None],
            start_colors, end_colors,
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
