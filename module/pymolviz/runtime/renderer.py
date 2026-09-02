"""Semantic object → CGO tokens (reuses mesh _create_CGO_list)."""

from __future__ import annotations


def cgo_tokens(obj, context=None):
    if context is not None and hasattr(obj, "rebuild"):
        obj.rebuild(context)
    merged = getattr(obj, "_merged_cgo_list", None)
    if callable(merged):
        return merged()
    return obj._create_CGO_list()


def _cgo_children(obj):
    if type(obj).__name__ == "CGOCollection":
        return list(obj)
    return None


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

    if context is not None and hasattr(obj, "rebuild"):
        obj.rebuild(context)
    children = _cgo_children(obj)
    if children is None:
        return _resolved_one(obj, resolve_cgo_tokens)
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
