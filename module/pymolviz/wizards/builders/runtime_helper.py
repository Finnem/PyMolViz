"""Thin wrappers around preview collection builders (wizard Create / arrow_geom)."""

from __future__ import annotations

from typing import Optional, Sequence

from ...util.solvent_surface import DEFAULT_RADIUS_MODE, DEFAULT_VDW_SCALE
from .preview import (
    build_arrow_collection as _build_arrows,
    build_box_cgo_collection,
    build_cgo_collection,
    build_surface_collection as _build_surface,
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


def build_surface_collection(
    points,
    atom_radius,
    probe_radius,
    algorithm,
    quality,
    wireframe,
    name,
    *,
    draft=False,
    obj_id=None,
    radius_mode=DEFAULT_RADIUS_MODE,
    vdw_scale=DEFAULT_VDW_SCALE,
    clip_planes=None,
):
    collection = _build_surface(
        points, atom_radius, probe_radius, algorithm, quality, wireframe, name,
        radius_mode=radius_mode, vdw_scale=vdw_scale, clip_planes=clip_planes,
    )
    if obj_id:
        collection.id = obj_id
    return collection
