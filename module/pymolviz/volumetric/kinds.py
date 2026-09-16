"""Canonical names for native field visuals (Volume / Iso*)."""

from __future__ import annotations

VOLUME_KINDS = frozenset({"Volume", "IsoVolume"})
ISO_KINDS = frozenset({"IsoSurface", "IsoMesh"})
FIELD_VISUAL_TYPES = frozenset(VOLUME_KINDS | ISO_KINDS)
FIELD_VISUAL_KINDS = tuple(sorted(FIELD_VISUAL_TYPES))


def is_field_visual(obj) -> bool:
    return type(obj).__name__ in FIELD_VISUAL_TYPES
