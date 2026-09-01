from __future__ import annotations

import logging
import numpy as np
from .Points import Points
from ..points import as_point_source, point_sources_from_sequence, resolve_xyz


class Lines(Points):
    """Class to store all relevant information required to create a CGO Line object."""

    def __init__(
        self,
        lines=None,
        color=None,
        name=None,
        state=1,
        transparency=0,
        colormap="RdYlBu_r",
        linewidth=0.05,
        render_as="cylinders",
        starts=None,
        ends=None,
        render_ends=False,
        *args,
        **kwargs,
    ) -> None:
        existing_starts = getattr(self, "_start_sources", None)
        existing_ends = getattr(self, "_end_sources", None)
        provided_lines = lines
        if starts is not None and ends is not None:
            self._start_sources = point_sources_from_sequence(starts)
            self._end_sources = point_sources_from_sequence(ends)
            if provided_lines is None:
                starts_arr = np.array([resolve_xyz(s) for s in self._start_sources])
                ends_arr = np.array([resolve_xyz(s) for s in self._end_sources])
                lines = np.hstack([starts_arr, ends_arr])
        elif existing_starts is not None and existing_ends is not None:
            self._start_sources = existing_starts
            self._end_sources = existing_ends
        else:
            self._start_sources = None
            self._end_sources = None

        lines = np.array(lines) if lines is not None else None
        if self._start_sources is None and lines is not None and lines.size and lines.size % 6 == 0:
            pairs = np.asarray(lines, dtype=float).reshape(-1, 6)
            self._start_sources = [as_point_source(row[:3]) for row in pairs]
            self._end_sources = [as_point_source(row[3:]) for row in pairs]
        try:
            if (not np.issubdtype(type(color), np.str_)) and (not (color is None)):
                if lines is not None and (len(color) == (len(lines.reshape(-1, 3)) / 2)):
                    color = np.repeat(color, 2, axis=0)
        except TypeError:
            pass

        if lines is None:
            if self._start_sources is None or self._end_sources is None:
                raise ValueError("Either lines or start and end must be given.")
            starts_arr = np.array([resolve_xyz(s) for s in self._start_sources])
            ends_arr = np.array([resolve_xyz(s) for s in self._end_sources])
            lines = np.hstack([starts_arr, ends_arr])

        super().__init__(lines.reshape(-1, 3), color, name, state, transparency, colormap, *args, **kwargs)
        self.linewidth = linewidth
        self.render_as = render_as
        self.render_ends = render_ends
        self.transparency = transparency
        try:
            self.transparency[0]
        except (TypeError, IndexError):
            self.transparency = np.full(int(lines.reshape(-1, 6).shape[0]), self.transparency)
        if self.render_as == "line":
            self.render_as = "lines"
        if self.render_as == "cylinder":
            self.render_as = "cylinders"

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

    def rebuild(self, context=None) -> None:
        self.invalidate_cgo_cache()
        if self._start_sources is None or self._end_sources is None:
            return
        starts_arr = np.array([resolve_xyz(s, context) for s in self._start_sources])
        ends_arr = np.array([resolve_xyz(s, context) for s in self._end_sources])
        self.vertices = np.hstack([starts_arr, ends_arr]).reshape(-1, 3)

    def from_start_end(
        starts,
        ends,
        color=None,
        name=None,
        state=1,
        transparency=0,
        colormap="RdYlBu_r",
        linewidth=1,
        render_as="cylinders",
        *args,
        **kwargs,
    ):
        starts = np.array(starts).reshape(-1, 3)
        ends = np.array(ends).reshape(-1, 3)
        return Lines(
            np.hstack([starts, ends]),
            color,
            name,
            state,
            transparency,
            colormap,
            linewidth,
            render_as,
            *args,
            **kwargs,
        )

    def _create_CGO_list(self) -> str:
        cgo_list = []
        if self.render_as == "lines":
            cgo_list.extend(["LINEWIDTH", self.linewidth])
            cgo_list.extend(["BEGIN", "LINES"])
            cgo_vertices = self.vertices
            cgo_colors = self.colormap.get_color(self.color)[:, :3]
            triangles = np.hstack([
                np.full(cgo_colors.shape[0], "COLOR")[:, None], cgo_colors,
                np.full(cgo_vertices.shape[0], "VERTEX")[:, None], cgo_vertices,
            ]).flatten()
            cgo_list.extend(triangles)
            cgo_list.append("END")
        elif self.render_as == "cylinders":
            cgo_vertices = self.vertices.reshape(-1, 6)
            cgo_colors = self.colormap.get_color(self.color)[:, :3].reshape(-1, 6)
            transparency = 1 - self.transparency
            try:
                transparency[0]
            except (TypeError, IndexError):
                if len(cgo_vertices) > 0:
                    transparency = np.full(int(cgo_vertices.shape[0]), transparency)
            triangles = np.hstack([
                np.full(cgo_vertices.shape[0], "ALPHA")[:, None], transparency[:, None],
                np.full(cgo_vertices.shape[0], "CONE")[:, None], cgo_vertices,
                np.full(cgo_vertices.shape[0], self.linewidth)[:, None],
                np.full(cgo_vertices.shape[0], self.linewidth)[:, None],
                cgo_colors, np.full((cgo_vertices.shape[0], 2), (1.0, 1.0)),
            ]).flatten()
            cgo_list.extend(triangles)
        if self.render_ends:
            point_meshes = np.hstack([
                np.full(cgo_vertices.shape[0], "COLOR")[:, None], cgo_colors,
                np.full(cgo_vertices.shape[0], "SPHERE")[:, None], cgo_vertices,
                np.full(cgo_vertices.shape[0], self.linewidth / 4)[:, None],
            ]).flatten()
            cgo_list.extend(point_meshes)
        return cgo_list

    def combine(lines):
        vertices = np.vstack([line.vertices for line in lines])
        colors = np.vstack([line.color for line in lines])
        return Lines(vertices, colors)

    def as_dotted(self, max_length=0.1):
        vertices = self.vertices.reshape(-1, 2, 3)
        lines = vertices[:, 1] - vertices[:, 0]
        lengths = np.linalg.norm(lines, axis=1)
        directions = lines / lengths[:, None]
        segments = np.ceil(lengths / max_length).astype(int)
        new_vertices = []
        new_colors = []
        new_transparency = []
        for i, (segment, direction) in enumerate(zip(segments, directions)):
            for j in range(segment):
                if j % 2 == 1:
                    continue
                new_vertices.append(vertices[i, 0] + direction * j * max_length)
                new_vertices.append(vertices[i, 0] + direction * (j + 1) * max_length)
                new_colors.append(self.color[i])
                new_colors.append(self.color[i])
                new_transparency.append(self.transparency[i])
        vertices = np.array(new_vertices).reshape(-1, 3)
        return Lines(
            vertices, new_colors, self.name, self.state,
            np.array(new_transparency), self.colormap, self.linewidth, self.render_as,
        )
