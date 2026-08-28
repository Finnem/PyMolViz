"""Start/end point pairs for two-point meshes (arrows, lines)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .points import VisualPoint, selection_points

RGB = Tuple[float, float, float]


@dataclass
class VisualPair:
    start: VisualPoint
    end: VisualPoint

    @property
    def color(self) -> RGB:
        return self.start.color

    @property
    def alpha(self) -> float:
        return float(self.start.alpha)

    def rgba(self) -> Tuple[float, float, float, float]:
        return self.start.rgba()

    def label(self) -> str:
        return "%s → %s" % (self.start.name, self.end.name)

    def with_color(self, color: Sequence[float]) -> "VisualPair":
        return VisualPair(self.start.with_color(color), self.end.with_color(color))

    def with_start(self, start: VisualPoint) -> "VisualPair":
        return VisualPair(start, self.end)

    def with_end(self, end: VisualPoint) -> "VisualPair":
        return VisualPair(self.start, end)


def flatten_pair_points(pairs: Sequence[VisualPair]) -> List[VisualPoint]:
    out = []
    for pair in pairs:
        out.append(pair.start)
        out.append(pair.end)
    return out


def take_single_selection_point(
    cmd_,
    existing: Sequence[VisualPoint] = (),
    interactive_only: bool = False,
) -> Tuple[Optional[VisualPoint], str]:
    """Return (point, status) where status is empty / one / multiple."""
    pts = selection_points(cmd_, existing, interactive_only=interactive_only)
    if not pts:
        return None, "empty"
    if len(pts) > 1:
        return None, "multiple"
    return pts[0], "one"
