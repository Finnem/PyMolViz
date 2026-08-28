"""Semantic object → CGO tokens (reuses mesh _create_CGO_list)."""

from __future__ import annotations


def cgo_tokens(obj, context=None):
    if context is not None and hasattr(obj, "rebuild"):
        obj.rebuild(context)
    merged = getattr(obj, "_merged_cgo_list", None)
    if callable(merged):
        return merged()
    return obj._create_CGO_list()
