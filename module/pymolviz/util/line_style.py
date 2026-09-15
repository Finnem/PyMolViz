"""Dash / margin / end-cap line style helpers (shared by Arrows and wizard)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence, Tuple

DASH_PRESETS = (
    ("Solid", (1.0, 0.0)),
    ("Dashed", (0.45, 0.28)),
    ("Dotted", (0.08, 0.18)),
    ("Dash-dot", (0.45, 0.16, 0.08, 0.16)),
    ("Dash-dot-dot", (0.45, 0.14, 0.08, 0.14, 0.08, 0.14)),
)

HEAD_STYLES = ("None", "Arrow", "Circles")
# Combined "Double arrow" is start_head=Arrow and end_head=Arrow.
END_STYLES = HEAD_STYLES

ARROW_QUALITY_SEGMENTS = {0: 0, 1: 6, 2: 8, 3: 10, 4: 14, 5: 18}
# Head azimuth samples may follow a wider cone, but must not explode when the
# shaft slider is far below a fixed head radius during live preview.
MAX_ARROW_CONE_SEGMENTS = 24

# Cone length is a multiple of shaft radius, not of pair length.
HEAD_LENGTH_OVER_RADIUS = 8.0
MAX_ARROW_MARGIN = 8.0
MIN_ARROW_SHAFT = 0.05


def default_head_length(width: float) -> float:
    return max(float(width), 0.0) * HEAD_LENGTH_OVER_RADIUS


def arrow_cone_segments(n_seg: int, radius: float, head_radius: float) -> int:
    """Azimuth count for the arrow head, capped for live preview.

    A wide head on a thin shaft used to scale ``n_seg * head/shaft`` without
    bound, so dragging the shaft slider toward zero built huge cones.
    """
    n_seg = max(int(n_seg), 3)
    shaft = max(float(radius), 1e-8)
    ratio = max(float(head_radius), 0.0) / shaft
    wanted = int(round(n_seg * min(max(ratio, 1.0), 3.0)))
    return max(n_seg, min(wanted, MAX_ARROW_CONE_SEGMENTS))


def absolute_head_length(
    width: float,
    shaft_length: float,
    head_length: Optional[float] = None,
    n_heads: int = 1,
    min_shaft: float = MIN_ARROW_SHAFT,
) -> float:
    """Head length in Å. Defaults from width; shrinks so heads and a sliver of shaft fit."""
    wanted = default_head_length(width) if head_length is None else max(float(head_length), 0.0)
    n = max(int(n_heads), 1)
    budget = max(float(shaft_length) - float(min_shaft), 0.0)
    if budget <= 1e-8:
        return 0.0
    return min(wanted, budget / n)


def max_margin_for_length(
    length: float,
    head_length: float = 0.0,
    double_head: bool = False,
    min_shaft: float = MIN_ARROW_SHAFT,
) -> float:
    """Largest per-end inset that still leaves room for heads and a minimum shaft."""
    occupied = max(float(head_length), 0.0) * (2.0 if double_head else 1.0)
    leftover = float(length) - occupied - float(min_shaft)
    if leftover <= 0.0:
        return 0.0
    return leftover / 2.0


def _normalize_head(name) -> str:
    if name in HEAD_STYLES:
        return name
    if name == "Double arrow":
        return "Arrow"
    return "None"


def split_ends(ends) -> Tuple[str, str]:
    """Map a legacy combined ``ends`` value to ``(start_head, end_head)``."""
    if ends is None:
        return "None", "Arrow"
    text = str(ends)
    if text == "Double arrow":
        return "Arrow", "Arrow"
    if text == "Circles":
        return "Circles", "Circles"
    if text == "Arrow":
        return "None", "Arrow"
    if text == "None":
        return "None", "None"
    if "/" in text:
        left, right = text.split("/", 1)
        return _normalize_head(left), _normalize_head(right)
    return "None", "Arrow"


def combined_ends(start_head, end_head) -> str:
    """Legacy combined name, lossless for mixed pairs via ``start/end``."""
    start_head = _normalize_head(start_head)
    end_head = _normalize_head(end_head)
    if start_head == "None" and end_head == "Arrow":
        return "Arrow"
    if start_head == "Arrow" and end_head == "Arrow":
        return "Double arrow"
    if start_head == "Circles" and end_head == "Circles":
        return "Circles"
    if start_head == "None" and end_head == "None":
        return "None"
    return "%s/%s" % (start_head, end_head)


@dataclass(init=False)
class LineStyle:
    dash: str = "Solid"
    dash_scale: float = 1.0
    start_margin: float = 0.0
    end_margin: float = 0.0
    start_head: str = "None"
    end_head: str = "Arrow"

    def __init__(
        self,
        dash: str = "Solid",
        dash_scale: float = 1.0,
        margin: Optional[float] = None,
        start_head: str = "None",
        end_head: str = "Arrow",
        ends: Optional[str] = None,
        start_margin: Optional[float] = None,
        end_margin: Optional[float] = None,
    ):
        if ends is not None:
            start_head, end_head = split_ends(ends)
        self.dash = str(dash)
        self.dash_scale = float(dash_scale)
        fallback = 0.0 if margin is None else float(margin)
        self.start_margin = fallback if start_margin is None else float(start_margin)
        self.end_margin = fallback if end_margin is None else float(end_margin)
        self.start_head = _normalize_head(start_head)
        self.end_head = _normalize_head(end_head)

    @property
    def margin(self) -> float:
        """Symmetric inset when both ends match; otherwise the start inset."""
        if abs(self.start_margin - self.end_margin) < 1e-9:
            return float(self.start_margin)
        return float(self.start_margin)

    @margin.setter
    def margin(self, value: float):
        pad = float(value)
        self.start_margin = pad
        self.end_margin = pad

    @property
    def ends(self) -> str:
        return combined_ends(self.start_head, self.end_head)

    def n_arrow_heads(self) -> int:
        return int(self.start_head == "Arrow") + int(self.end_head == "Arrow")

    def pattern(self) -> Tuple[float, ...]:
        for name, pattern in DASH_PRESETS:
            if name == self.dash:
                return pattern
        return (1.0, 0.0)

    def copy(self) -> "LineStyle":
        return replace(self)

    def updated(self, **kwargs) -> "LineStyle":
        if "ends" in kwargs and "start_head" not in kwargs and "end_head" not in kwargs:
            start, end = split_ends(kwargs.pop("ends"))
            kwargs["start_head"] = start
            kwargs["end_head"] = end
        if "margin" in kwargs:
            pad = float(kwargs.pop("margin"))
            kwargs.setdefault("start_margin", pad)
            kwargs.setdefault("end_margin", pad)
        return replace(self, **kwargs)

    def to_dict(self) -> dict:
        return {
            "dash": self.dash,
            "dash_scale": float(self.dash_scale),
            "margin": float(self.margin),
            "start_margin": float(self.start_margin),
            "end_margin": float(self.end_margin),
            "start_head": self.start_head,
            "end_head": self.end_head,
            "ends": self.ends,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LineStyle":
        data = data or {}
        kwargs = {
            "dash": data.get("dash", "Solid"),
            "dash_scale": float(data.get("dash_scale", 1.0)),
        }
        if "start_margin" in data or "end_margin" in data:
            fallback = float(data.get("margin", 0.0))
            kwargs["start_margin"] = float(data.get("start_margin", fallback))
            kwargs["end_margin"] = float(data.get("end_margin", fallback))
        else:
            kwargs["margin"] = float(data.get("margin", 0.0))
        if "start_head" in data or "end_head" in data:
            kwargs["start_head"] = data.get("start_head", "None")
            kwargs["end_head"] = data.get("end_head", "Arrow")
        else:
            kwargs["ends"] = data.get("ends", "Arrow")
        return cls(**kwargs)


def dash_on_segments(
    p0: Sequence[float],
    p1: Sequence[float],
    pattern: Sequence[float],
    scale: float,
) -> list:
    x0, y0, z0 = (float(p0[0]), float(p0[1]), float(p0[2]))
    x1, y1, z1 = (float(p1[0]), float(p1[1]), float(p1[2]))
    dx, dy, dz = (x1 - x0, y1 - y0, z1 - z0)
    length = (dx * dx + dy * dy + dz * dz) ** 0.5
    if length < 1e-8:
        return []
    inv = 1.0 / length
    nx, ny, nz = (dx * inv, dy * inv, dz * inv)
    raw = [max(float(v), 0.0) for v in pattern]
    if not raw:
        raw = [1.0, 0.0]
    if len(raw) == 1:
        raw = [raw[0], 0.0]
    scale = max(float(scale), 1e-6)
    units = [v * scale for v in raw]
    off_spans = units[1::2]
    if not off_spans or max(off_spans) <= 1e-8:
        return [((x0, y0, z0), (x1, y1, z1))]
    period = sum(units)
    if period < 1e-8:
        return [((x0, y0, z0), (x1, y1, z1))]

    segments = []
    travelled = 0.0
    cycle = 0
    while travelled < length - 1e-8:
        span = units[cycle % len(units)]
        on = (cycle % 2) == 0
        if span <= 1e-8:
            cycle += 1
            continue
        start_t = travelled
        end_t = min(length, travelled + span)
        if on and end_t > start_t + 1e-8:
            segments.append((
                (x0 + nx * start_t, y0 + ny * start_t, z0 + nz * start_t),
                (x0 + nx * end_t, y0 + ny * end_t, z0 + nz * end_t),
            ))
        travelled = end_t
        cycle += 1
    return segments


def style_margins(style) -> Tuple[float, float]:
    """Start and end insets in Å, falling back to a legacy symmetric ``margin``."""
    fallback = float(getattr(style, "margin", 0.0) or 0.0)
    start = getattr(style, "start_margin", None)
    end = getattr(style, "end_margin", None)
    return (
        fallback if start is None else float(start),
        fallback if end is None else float(end),
    )


def apply_margin(
    p0,
    p1,
    margin: float = 0.0,
    head_length: float = 0.0,
    double_head: bool = False,
    min_shaft: float = MIN_ARROW_SHAFT,
    start_margin: Optional[float] = None,
    end_margin: Optional[float] = None,
):
    x0, y0, z0 = (float(p0[0]), float(p0[1]), float(p0[2]))
    x1, y1, z1 = (float(p1[0]), float(p1[1]), float(p1[2]))
    dx, dy, dz = (x1 - x0, y1 - y0, z1 - z0)
    length = (dx * dx + dy * dy + dz * dz) ** 0.5
    if length < 1e-8:
        return (x0, y0, z0), (x1, y1, z1)
    occupied = max(float(head_length), 0.0) * (2.0 if double_head else 1.0)
    budget = max(length - occupied - float(min_shaft), 0.0)
    fallback = max(float(margin), 0.0)
    pad0 = fallback if start_margin is None else max(float(start_margin), 0.0)
    pad1 = fallback if end_margin is None else max(float(end_margin), 0.0)
    total = pad0 + pad1
    if budget <= 0.0 or total <= 0.0:
        return (x0, y0, z0), (x1, y1, z1)
    if total > budget:
        scale = budget / total
        pad0 *= scale
        pad1 *= scale
    inv = 1.0 / length
    return (
        (x0 + dx * inv * pad0, y0 + dy * inv * pad0, z0 + dz * inv * pad0),
        (x1 - dx * inv * pad1, y1 - dy * inv * pad1, z1 - dz * inv * pad1),
    )
