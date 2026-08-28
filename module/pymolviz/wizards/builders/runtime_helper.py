"""Thin wrappers around preview collection builders (wizard Create / arrow_geom)."""

from __future__ import annotations

from typing import Optional, Sequence

from .preview import (
    build_arrow_collection as _build_arrows,
    build_box_cgo_collection,
    build_cgo_collection,
)


def build_sphere_collection(
    points,
    radius,
    wireframe,
    name,
    wireframe_quality=3,
    *,
    draft=False,
    obj_id: Optional[str] = None,
):
    collection = build_cgo_collection(
        points, radius, wireframe, name, wireframe_quality=wireframe_quality,
    )
    if obj_id:
        collection.id = obj_id
    return collection


def build_box_collection(
    points,
    extent: Sequence[float],
    wireframe,
    name,
    *,
    draft=False,
    obj_id: Optional[str] = None,
):
    collection = build_box_cgo_collection(points, extent, wireframe, name)
    if obj_id:
        collection.id = obj_id
    return collection


def build_arrow_collection(pairs, quality, style, name, *, draft=False, obj_id=None):
    collection = _build_arrows(pairs, quality, style, name)
    if obj_id:
        collection.id = obj_id
    return collection
