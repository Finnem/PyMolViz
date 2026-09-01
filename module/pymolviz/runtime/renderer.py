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


def resolved_cgo_tokens(obj, context=None):
    """Integer CGO tokens, reusing per-mesh caches when geometry was only shifted."""
    from ..util.cgo import resolve_cgo_tokens

    if context is not None and hasattr(obj, "rebuild"):
        obj.rebuild(context)
    children = _cgo_children(obj)
    if children is None:
        return _resolved_one(obj, resolve_cgo_tokens)
    out = []
    for child in children:
        out.extend(_resolved_one(child, resolve_cgo_tokens))
    return out
