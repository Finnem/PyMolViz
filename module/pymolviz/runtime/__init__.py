"""Disposable PyMOL binding layer (does not shadow the real ``pymol`` module)."""

from .runtime import PyMOLRuntime, get_runtime

__all__ = ["PyMOLRuntime", "get_runtime"]
