"""Dash / margin / end-cap line style helpers (shared by Arrows and wizard)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

DASH_PRESETS = (
    ("Solid", (1.0, 0.0)),
    ("Dashed", (0.45, 0.28)),
    ("Dotted", (0.08, 0.18)),
    ("Dash-dot", (0.45, 0.16, 0.08, 0.16)),
    ("Dash-dot-dot", (0.45, 0.14, 0.08, 0.14, 0.08, 0.14)),
)

END_STYLES = ("None", "Arrow", "Double arrow", "Circles")

ARROW_QUALITY_SEGMENTS = {0: 0, 1: 6, 2: 8, 3: 10, 4: 14, 5: 18}


@dataclass
class LineStyle:
    dash: str = "Solid"
    dash_scale: float = 1.0
    margin: float = 0.0
    ends: str = "Arrow"

    def pattern(self) -> Tuple[float, ...]:
        for name, pattern in DASH_PRESETS:
            if name == self.dash:
                return pattern
        return (1.0, 0.0)

    def to_dict(self) -> dict:
        return {
            "dash": self.dash,
            "dash_scale": float(self.dash_scale),
            "margin": float(self.margin),
            "ends": self.ends,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LineStyle":
        return cls(
            dash=data.get("dash", "Solid"),
            dash_scale=float(data.get("dash_scale", 1.0)),
            margin=float(data.get("margin", 0.0)),
            ends=data.get("ends", "Arrow"),
        )


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


def apply_margin(p0, p1, margin: float):
    x0, y0, z0 = (float(p0[0]), float(p0[1]), float(p0[2]))
    x1, y1, z1 = (float(p1[0]), float(p1[1]), float(p1[2]))
    dx, dy, dz = (x1 - x0, y1 - y0, z1 - z0)
    length = (dx * dx + dy * dy + dz * dz) ** 0.5
    pad = max(float(margin), 0.0)
    if length < 1e-8 or pad <= 0.0:
        return (x0, y0, z0), (x1, y1, z1)
    if pad * 2.0 >= length:
        pad = length * 0.25
    inv = 1.0 / length
    return (
        (x0 + dx * inv * pad, y0 + dy * inv * pad, z0 + dz * inv * pad),
        (x1 - dx * inv * pad, y1 - dy * inv * pad, z1 - dz * inv * pad),
    )
