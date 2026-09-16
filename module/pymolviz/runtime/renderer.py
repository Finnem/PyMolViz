"""Semantic object → CGO tokens (reuses mesh _create_CGO_list)."""

from __future__ import annotations

from ..volumetric.kinds import FIELD_VISUAL_TYPES

_NON_CGO_TYPES = frozenset({"Field", "GridData"}) | FIELD_VISUAL_TYPES


def renders_cgo(obj) -> bool:
    """True when ``obj`` should be loaded into PyMOL as CGO.

    Fields (and other non-visual session data) are Displayables for identity
    and recipes, but they do not produce CGO. Prefer the ``renders_cgo`` flag
    over catching a missing ``_create_CGO_list``.
    """
    if getattr(obj, "renders_cgo", None) is False:
        return False
    if type(obj).__name__ in _NON_CGO_TYPES:
        return False
    return callable(getattr(obj, "_create_CGO_list", None))


def cgo_tokens(obj, context=None):
    if not renders_cgo(obj):
        return []
    if context is not None and hasattr(obj, "rebuild"):
        obj.rebuild(context)
    merged = getattr(obj, "_merged_cgo_list", None)
    if callable(merged):
        return merged()
    return obj._create_CGO_list()


def _resolved_one(obj, resolve_cgo_tokens):
    cached = getattr(obj, "_cached_resolved", None)
    if cached is not None:
        return cached
    raw = obj._create_CGO_list()
    resolved = resolve_cgo_tokens(raw)
    obj._cached_resolved = resolved
    return resolved


def _child_serials(children):
    return tuple(getattr(child, "_geom_serial", 0) for child in children)


def resolved_cgo_tokens(obj, context=None):
    """Integer CGO tokens, reusing per-mesh caches when geometry was only shifted."""
    from ..util.cgo import resolve_cgo_tokens

    if not renders_cgo(obj):
        return []
    if context is not None and hasattr(obj, "rebuild"):
        obj.rebuild(context)
    prepare = getattr(obj, "prepare_child_look", None)
    if callable(prepare):
        prepare()
    if type(obj).__name__ != "CGOCollection":
        return _resolved_one(obj, resolve_cgo_tokens)
    from ..meshes.CGOCollection import cgo_children

    children = cgo_children(obj)
    serials = _child_serials(children)
    cached = getattr(obj, "_cached_merged_resolved", None)
    spans = getattr(obj, "_child_spans", None)
    stored = getattr(obj, "_child_serials", None)
    if stored is not None:
        stored = tuple(stored)
    if (
        cached is not None
        and spans is not None
        and stored == serials
        and len(spans) == len(children)
        and context is None
    ):
        return cached
    out = []
    spans = []
    for child in children:
        start = len(out)
        out.extend(_resolved_one(child, resolve_cgo_tokens))
        spans.append((start, len(out)))
    obj._cached_merged_resolved = out
    obj._child_spans = spans
    obj._child_serials = serials
    return out
