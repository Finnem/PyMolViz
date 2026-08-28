"""Build styled arrow CGOs from point pairs (delegates to Arrows mesh)."""

from __future__ import annotations

from typing import Sequence

from ...meshes.Arrows import Arrows, build_styled_arrow_cgo
from ...meshes.CGOCollection import CGOCollection
from .line_style import LineStyle
from .pairs import VisualPair
from .runtime_helper import build_arrow_collection

DEFAULT_SHAFT_RADIUS = 0.045


def arrow_cgo(start, end, color, quality, style, alpha=1.0, radius=DEFAULT_SHAFT_RADIUS):
    return build_styled_arrow_cgo(start, end, color, quality, style, alpha=alpha, radius=radius)


def build_arrow_cgo_list(pairs: Sequence[VisualPair], quality: int, style: LineStyle) -> list:
    merged = []
    for pair in pairs:
        merged.extend(arrow_cgo(pair.start.xyz(), pair.end.xyz(), pair.color, quality, style, pair.alpha))
    return merged


def build_arrow_cgo_collection(
    pairs: Sequence[VisualPair],
    quality: int,
    style: LineStyle,
    name: str,
) -> CGOCollection:
    return build_arrow_collection(pairs, quality, style, name, draft=False)
