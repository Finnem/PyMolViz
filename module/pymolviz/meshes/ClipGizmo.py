from __future__ import annotations

import numpy as np

from .Points import Points
from ..util.clip_gizmo import build_clip_gizmo_cgo


class ClipGizmo(Points):
    """Preview-only rectangle + eye for a surface clip plane."""

    def __init__(
        self,
        origin,
        normal,
        scale=5.0,
        *,
        points=None,
        selected=False,
        draft=False,
        axis_arrow=False,
        name=None,
        **kwargs,
    ) -> None:
        origin = np.asarray(origin, dtype=float).reshape(3)
        kwargs.setdefault("bypass_colormap", True)
        kwargs.setdefault("render_as", "dots")
        super().__init__(origin.reshape(1, 3), color=(0.3, 0.8, 1.0), name=name, **kwargs)
        self.origin = origin
        self.normal = np.asarray(normal, dtype=float).reshape(3)
        ln = float(np.linalg.norm(self.normal))
        self.normal = self.normal / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0])
        self.scale = float(scale)
        self.points = None if points is None else np.asarray(points, dtype=float).reshape(-1, 3)
        self.selected = bool(selected)
        self.draft = bool(draft)
        self.axis_arrow = bool(axis_arrow)

    def _create_CGO_list(self):
        cached = getattr(self, "_cached_cgo", None)
        if cached is not None:
            return cached
        cgo_list = build_clip_gizmo_cgo(
            self.origin, self.normal, self.scale,
            points=self.points,
            selected=self.selected, draft=self.draft,
            axis_arrow=self.axis_arrow,
        )
        self._cached_cgo = cgo_list
        return cgo_list
