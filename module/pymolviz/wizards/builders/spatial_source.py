"""Shared source/status vocabulary for Point and Arrow editors."""

from __future__ import annotations

from typing import Optional

KIND_ATOM = "Atom"
KIND_CAMERA = "Camera center"
KIND_SELECTION = "Selection"
KIND_FIXED = "Fixed position"
STATUS_ATTACHED = "Anchored"

KIND_PENDING = "—"


def point_kind_label(pt) -> str:
    """Collapsed/summary kind: Atom, Camera center, Selection, or Fixed position."""
    if pt is None:
        return KIND_PENDING
    if getattr(pt, "atom_ref", None) is not None:
        return KIND_ATOM
    source = str(getattr(pt, "source", "") or "")
    name = str(getattr(pt, "name", "") or "")
    if source == "selection":
        return KIND_SELECTION
    if source == "camera" or name.startswith("cam"):
        return KIND_CAMERA
    if source == "manual":
        return KIND_FIXED
    return KIND_FIXED


def point_attach_status(pt) -> str:
    """Second status line: Anchored, or the same kind when not anchored."""
    if pt is None:
        return KIND_PENDING
    if getattr(pt, "can_anchor", lambda: False)() and pt.wants_anchor():
        return STATUS_ATTACHED
    kind = point_kind_label(pt)
    if kind == KIND_ATOM:
        return KIND_ATOM
    return kind


def point_source_summary(pt) -> str:
    """Single-line source for a collapsed Point row."""
    return point_kind_label(pt)


def _short_kind(kind: str) -> str:
    if kind == KIND_FIXED:
        return "Fixed"
    return kind


def arrow_source_summary(start, end=None) -> str:
    """Collapsed Arrow summary: ``Atom → Atom``, ``Fixed → Fixed``, or ``Mixed``."""
    left = point_kind_label(start)
    right = point_kind_label(end)
    if left == KIND_PENDING or right == KIND_PENDING:
        return "%s → %s" % (_short_kind(left), _short_kind(right))
    if left == right:
        return "%s → %s" % (_short_kind(left), _short_kind(right))
    known = {KIND_ATOM, KIND_CAMERA, KIND_FIXED, KIND_SELECTION}
    if left in known and right in known:
        return "%s → %s" % (_short_kind(left), _short_kind(right))
    return "Mixed"


def point_primary_label(pt, *, pending: str = "[pick…]") -> str:
    if pt is None:
        return pending
    from .points import atom_anchor_label

    label = atom_anchor_label(getattr(pt, "atom_ref", None))
    if label:
        return label
    name = str(getattr(pt, "name", "") or "").strip()
    return name or pending
