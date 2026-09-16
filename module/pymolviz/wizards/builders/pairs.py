"""Start/end point pairs for two-point meshes (arrows, lines)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

from ...points import FixedPoint, PointUnresolvedError
from ...util.line_style import LineStyle, default_head_length
from .points import (
    VisualPoint,
    atom_anchor_label,
    insertion_selection_count,
    manual_fallback_name,
    selection_points,
)

RGB = Tuple[float, float, float]

DEFAULT_ARROW_WIDTH = 0.045
DEFAULT_ARROW_HEAD = default_head_length(DEFAULT_ARROW_WIDTH)

STATUS_OK = "ok"
STATUS_PICKING = "picking"
STATUS_MISSING = "missing"

STATUS_GLYPH = {
    STATUS_OK: "✓",
    STATUS_PICKING: "●",
    STATUS_MISSING: "!",
}

PENDING_START = "[pick start…]"
PENDING_END = "[pick end…]"

MULTI_CLICKED = "clicked"
MULTI_CENTER = "center"
POLL_SELECTION_MAX_ATOMS = 32
STATUS_TOO_LARGE = "too_large"
SELECTION_TOO_LARGE_TITLE = "Selection too large"


def selection_too_large_message(limit: Optional[int] = None) -> str:
    n = int(POLL_SELECTION_MAX_ATOMS if limit is None else limit)
    return (
        "The current selection has more than %d atoms, so it was not scanned. "
        "Narrow the selection and try again."
    ) % n


def new_pair_id() -> str:
    return "arrow-" + uuid.uuid4().hex[:8]


@dataclass
class VisualPair:
    start: VisualPoint
    end: Optional[VisualPoint] = None
    pair_id: str = field(default_factory=new_pair_id)
    title: str = ""
    width: float = DEFAULT_ARROW_WIDTH
    head: float = DEFAULT_ARROW_HEAD
    style: LineStyle = field(default_factory=LineStyle)

    def is_complete(self) -> bool:
        return self.end is not None

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.start, "enabled", True))

    def with_enabled(self, enabled: bool) -> "VisualPair":
        start = self.start.with_enabled(enabled)
        end = self.end.with_enabled(enabled) if self.end is not None else None
        return replace(self, start=start, end=end)

    @property
    def color(self) -> RGB:
        return self.start.color

    @property
    def alpha(self) -> float:
        return float(self.start.alpha)

    def rgba(self) -> Tuple[float, float, float, float]:
        return self.start.rgba()

    def label(self) -> str:
        return pair_row_text(self)

    def with_color(self, color: Sequence[float]) -> "VisualPair":
        start = self.start.with_color(color)
        end = self.end.with_color(color) if self.end is not None else None
        return replace(self, start=start, end=end)

    def with_color_choice(self, choice) -> "VisualPair":
        start = self.start.with_color_choice(choice)
        end = self.end.with_color_choice(choice) if self.end is not None else None
        return replace(self, start=start, end=end)

    def color_choice(self):
        return self.start.color_choice()

    def with_start(self, start: VisualPoint) -> "VisualPair":
        return replace(self, start=start)

    def with_end(self, end: Optional[VisualPoint]) -> "VisualPair":
        return replace(self, end=end)

    def with_title(self, title: str) -> "VisualPair":
        return replace(self, title=str(title))

    def with_width(self, width: float) -> "VisualPair":
        return replace(self, width=float(width))

    def with_head(self, head: float) -> "VisualPair":
        return replace(self, head=float(head))

    def with_style(self, style: LineStyle) -> "VisualPair":
        copied = style.copy() if style is not None else LineStyle()
        return replace(self, style=copied)

    def swapped(self) -> "VisualPair":
        if self.end is None:
            return self
        return replace(self, start=self.end, end=self.start)


def complete_pairs(pairs: Sequence[VisualPair]) -> List[VisualPair]:
    return [pair for pair in pairs if pair.is_complete() and pair.enabled]


def pair_index(pairs: Sequence[VisualPair], pair_id: str) -> int:
    for index, pair in enumerate(pairs):
        if pair.pair_id == pair_id:
            return index
    return -1


def flatten_pair_points(pairs: Sequence[VisualPair]) -> List[VisualPoint]:
    out = []
    for pair in pairs:
        out.append(pair.start)
        if pair.end is not None:
            out.append(pair.end)
    return out


def commit_pair_anchors(pairs: Sequence[VisualPair]) -> List[VisualPair]:
    out = []
    for pair in pairs:
        if not pair.is_complete() or not pair.enabled:
            continue
        out.append(replace(
            pair,
            start=pair.start.commit_anchor(),
            end=pair.end.commit_anchor(),
        ))
    return out


def selection_center_point(
    pts: Sequence[VisualPoint],
    existing: Sequence[VisualPoint] = (),
) -> VisualPoint:
    """One free point at the mean of ``pts`` (not attached to an atom)."""
    n = max(len(pts), 1)
    cx = sum(float(pt.x) for pt in pts) / n
    cy = sum(float(pt.y) for pt in pts) / n
    cz = sum(float(pt.z) for pt in pts) / n
    pos = (cx, cy, cz)
    name = manual_fallback_name("center", existing)
    return VisualPoint(name, "manual", cx, cy, cz, point_source=FixedPoint(pos))


def take_single_selection_point(
    cmd_,
    existing: Sequence[VisualPoint] = (),
    interactive_only: bool = False,
    hook_to_selection: bool = True,
    multi_atom: str = MULTI_CENTER,
    max_expand: Optional[int] = None,
) -> Tuple[Optional[VisualPoint], str]:
    """Return (point, status) where status is empty / one / multiple / too_large."""
    start, end, status = take_selection_endpoints(
        cmd_,
        existing,
        interactive_only=interactive_only,
        hook_to_selection=hook_to_selection,
        multi_atom=multi_atom,
        pair_on_two=False,
        max_expand=max_expand,
    )
    if status == "pair":
        return None, "multiple"
    if status == "one":
        return start, "one"
    return None, status


def take_selection_endpoints(
    cmd_,
    existing: Sequence[VisualPoint] = (),
    interactive_only: bool = False,
    hook_to_selection: bool = True,
    multi_atom: str = MULTI_CENTER,
    pair_on_two: bool = True,
    max_expand: Optional[int] = None,
) -> Tuple[Optional[VisualPoint], Optional[VisualPoint], str]:
    """Return (start, end, status) for the current selection.

    Status is ``empty``, ``one``, ``pair``, ``multiple``, or ``too_large``.

    ``multi_atom`` is ``clicked`` (one or two scanned atoms) or ``center``
    (one endpoint at the mean of all selected atoms).

    ``max_expand`` caps how many atoms may be iterated. Larger selections
    return ``too_large`` without scanning.
    """
    use_center = str(multi_atom) == MULTI_CENTER
    if max_expand is not None:
        count = insertion_selection_count(cmd_, interactive_only=interactive_only)
        if count <= 0:
            return None, None, "empty"
        if count > int(max_expand):
            return None, None, STATUS_TOO_LARGE
    pts = selection_points(
        cmd_, existing,
        interactive_only=interactive_only,
        hook_to_selection=hook_to_selection,
    )
    if not pts:
        return None, None, "empty"
    if len(pts) == 1:
        return pts[0], None, "one"
    if use_center:
        return selection_center_point(pts, existing), None, "one"
    if pair_on_two and len(pts) == 2:
        return pts[0], pts[1], "pair"
    return None, None, "multiple"


def free_point_display_names(pairs: Sequence[VisualPair]) -> Dict[int, str]:
    """Assign ``Point 1``, ``Point 2``, … to endpoints that are not atom anchors."""
    names: Dict[int, str] = {}
    n = 0
    for pair in pairs:
        for pt in (pair.start, pair.end):
            if pt is None or pt.can_anchor():
                continue
            ident = id(pt)
            if ident in names:
                continue
            n += 1
            names[ident] = "Point %d" % n
    return names


def endpoint_label(
    pt: Optional[VisualPoint],
    *,
    pending: str = PENDING_END,
    free_names: Optional[Dict[int, str]] = None,
) -> str:
    if pt is None:
        return pending
    if pt.can_anchor():
        label = atom_anchor_label(pt.atom_ref)
        if label:
            return label
    if free_names is not None:
        named = free_names.get(id(pt))
        if named:
            return named
    return pt.name or "Point"


def endpoint_xyz_text(pt: Optional[VisualPoint]) -> str:
    if pt is None:
        return ""
    return "%.1f, %.1f, %.1f" % (float(pt.x), float(pt.y), float(pt.z))


def pair_row_text(
    pair: VisualPair,
    free_names: Optional[Dict[int, str]] = None,
) -> str:
    title = (pair.title or "").strip()
    if title:
        return title
    start = endpoint_label(pair.start, pending=PENDING_START, free_names=free_names)
    end = endpoint_label(pair.end, pending=PENDING_END, free_names=free_names)
    return "%s  →  %s" % (start, end)


def endpoint_is_missing(pt: Optional[VisualPoint], context=None) -> bool:
    if pt is None or context is None:
        return False
    src = pt.point_source
    if src is None or not src.has_dynamic_source():
        return False
    try:
        src.resolve(context)
        return False
    except PointUnresolvedError:
        return True
    except Exception:
        return False


def pair_status(pair: VisualPair, context=None) -> str:
    if not pair.is_complete():
        return STATUS_PICKING
    if endpoint_is_missing(pair.start, context) or endpoint_is_missing(pair.end, context):
        return STATUS_MISSING
    return STATUS_OK


def pair_status_glyph(status: str) -> str:
    return STATUS_GLYPH.get(status, "●")
