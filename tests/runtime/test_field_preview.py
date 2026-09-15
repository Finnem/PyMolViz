"""Runtime preview for From Selection / Field Visual native iso and volume."""

from __future__ import annotations

import numpy as np
import pytest

from pymolviz.fields.domain import Domain
from pymolviz.fields.identity import GEN_DISTANCE
from pymolviz.points import FixedPoint
from pymolviz.runtime import session as pmv_session
from pymolviz.wizards.builders.field_preview import (
    PREVIEW_FIELD_ISO_NAME,
    PREVIEW_FIELD_VISUAL_NAME,
    PREVIEW_GEOM_MAP_NAME,
    build_field_iso_preview_visual,
    build_grid_preview_visual,
    iso_preview_key,
    load_preview_field_visual,
)
from pymolviz.wizards.builders.points import AtomRef, VisualPoint
from pymolviz.runtime.runtime import get_runtime, reset_runtime
from pymolviz.wizards.builders.preview import FieldVisualPreview, FromSelectionFieldPreview


def _points():
    return [
        VisualPoint(
            "C1", "manual", 0.0, 0.0, 0.0,
            point_source=FixedPoint((0.0, 0.0, 0.0)),
            atom_ref=AtomRef("m", 1, elem="C"),
        ),
    ]


def _blob_field():
    from pymolviz.fields.field import as_field
    from pymolviz.volumetric.GridData import GridData

    n = 8
    xs = np.linspace(-2.0, 2.0, n)
    xx, yy, zz = np.meshgrid(xs, xs, xs, indexing="ij")
    values = np.exp(-(xx * xx + yy * yy + zz * zz))
    h = float(xs[1] - xs[0])
    grid = GridData(
        values.reshape(-1),
        step_sizes=(h, h, h),
        step_counts=(n - 1, n - 1, n - 1),
        origin=(-2.0, -2.0, -2.0),
        name="blob",
    )
    return as_field(grid)


def _cgo_names(cmd):
    return [name for name, typ in cmd.object_types.items() if typ == "object:cgo"]


def test_from_selection_field_preview_switches_marker_and_iso(fake_cmd):
    reset_runtime()
    cmd = fake_cmd
    get_runtime(cmd)
    pts = _points()
    domain = Domain(padding=2.0, spacing=1.0)
    preview = FromSelectionFieldPreview(cmd)

    preview.update(pts, 0.4, live_iso=False)
    assert preview.collection is not None
    assert type(preview.collection[0]).__name__ == "Sphere"

    iso_key = iso_preview_key(
        pts,
        algorithm="gaussian_atoms",
        domain=domain,
        quality=1,
        resolution=2.0,
        property_key="b_factor",
        iso_level=0.5,
    )
    iso_visual = build_field_iso_preview_visual(
        cmd,
        pts,
        algorithm="gaussian_atoms",
        domain=domain,
        quality=1,
        resolution=2.0,
        iso_level=0.5,
        name=PREVIEW_FIELD_ISO_NAME,
    )
    preview.update(
        pts, 0.4, live_iso=True, iso_visual=iso_visual, iso_key=iso_key,
    )
    assert preview.visual is not None
    assert type(preview.visual).__name__ == "IsoSurface"
    assert cmd.object_types.get(PREVIEW_FIELD_ISO_NAME) == "object:isosurface"
    assert cmd.object_types.get(PREVIEW_GEOM_MAP_NAME) == "object:map"
    assert PREVIEW_FIELD_ISO_NAME not in _cgo_names(cmd)

    preview.cleanup()
    reset_runtime()


def test_distance_preview_toggle_does_not_persist_visual(fake_cmd):
    reset_runtime()
    cmd = fake_cmd
    get_runtime(cmd)
    pts = _points()
    domain = Domain(padding=2.0, spacing=1.0)
    preview = FromSelectionFieldPreview(cmd)

    preview.update(pts, 0.4, live_iso=False)
    iso_visual = build_field_iso_preview_visual(
        cmd,
        pts,
        algorithm=GEN_DISTANCE,
        domain=domain,
        quality=1,
        resolution=2.0,
        iso_level=1.0,
        name=PREVIEW_FIELD_ISO_NAME,
    )
    preview.update(pts, 0.4, live_iso=True, iso_visual=iso_visual, iso_key=("d",))
    assert preview.visual is not None
    assert type(preview.visual).__name__ == "IsoSurface"
    kinds = {type(obj).__name__ for obj in pmv_session.all_objects()}
    assert "IsoSurface" not in kinds
    assert "IsoMesh" not in kinds
    assert "Volume" not in kinds
    assert "Field" not in kinds
    assert cmd.object_types.get(PREVIEW_FIELD_ISO_NAME) == "object:isosurface"

    preview.update(pts, 0.4, live_iso=False)
    assert preview.visual is None
    assert type(preview.collection[0]).__name__ == "Sphere"
    preview.cleanup()
    names = [str(n) for n in cmd.get_names("objects")]
    assert not any(name.startswith("_pmv_prev_") for name in names)
    reset_runtime()


def test_field_visual_preview_loads_native_isosurface(fake_cmd):
    reset_runtime()
    cmd = fake_cmd
    get_runtime(cmd)
    field = _blob_field()
    before = list(pmv_session.all_objects())
    visual = build_grid_preview_visual(
        field, kind="IsoSurface", iso_level=0.4, name=PREVIEW_FIELD_VISUAL_NAME,
    )
    preview = FieldVisualPreview(cmd)
    preview.update(visual, iso_key=("iso", 0.4))
    assert preview.visual is not None
    assert type(preview.visual).__name__ == "IsoSurface"
    assert pmv_session.all_objects() == before
    assert cmd.object_types.get(PREVIEW_FIELD_VISUAL_NAME) == "object:isosurface"
    assert cmd.object_types.get(PREVIEW_GEOM_MAP_NAME) == "object:map"
    assert PREVIEW_FIELD_VISUAL_NAME not in _cgo_names(cmd)
    names = [str(n) for n in cmd.get_names("objects")]
    assert any(str(n).startswith("_pmv_prev_") for n in names)
    preview.cleanup()
    names = [str(n) for n in cmd.get_names("objects")]
    assert not any(str(n).startswith("_pmv_prev_") for n in names)
    reset_runtime()


def test_field_visual_preview_loads_isomesh_and_volume(fake_cmd):
    reset_runtime()
    cmd = fake_cmd
    get_runtime(cmd)
    field = _blob_field()
    mesh = build_grid_preview_visual(
        field, kind="IsoMesh", iso_level=0.4, name=PREVIEW_FIELD_VISUAL_NAME,
    )
    preview = FieldVisualPreview(cmd)
    preview.update(mesh, iso_key=("mesh", 0.4))
    assert cmd.object_types.get(PREVIEW_FIELD_VISUAL_NAME) == "object:mesh"
    preview.cleanup()

    vol = build_grid_preview_visual(
        field, kind="Volume", name=PREVIEW_FIELD_VISUAL_NAME,
    )
    preview.update(vol, iso_key=("vol",))
    assert cmd.object_types.get(PREVIEW_FIELD_VISUAL_NAME) == "object:volume"
    assert PREVIEW_FIELD_VISUAL_NAME not in _cgo_names(cmd)
    preview.cleanup()
    names = [str(n) for n in cmd.get_names("objects")]
    assert not any(str(n).startswith("_pmv_prev_") for n in names)
    reset_runtime()


def test_load_preview_field_visual_uses_native_cmds(fake_cmd):
    reset_runtime()
    cmd = fake_cmd
    field = _blob_field()
    visual = build_grid_preview_visual(
        field, kind="IsoSurface", iso_level=0.4, name=PREVIEW_FIELD_VISUAL_NAME,
    )
    load_preview_field_visual(cmd, visual)
    assert cmd.object_types.get(PREVIEW_FIELD_VISUAL_NAME) == "object:isosurface"
    assert cmd.object_types.get(PREVIEW_GEOM_MAP_NAME) == "object:map"
    stored = cmd.objects.get(PREVIEW_FIELD_VISUAL_NAME) or {}
    assert stored.get("map") == PREVIEW_GEOM_MAP_NAME
    assert stored.get("level") == pytest.approx(0.4)
    reset_runtime()


def test_from_selection_domain_box_follows_preview_flag(fake_cmd):
    reset_runtime()
    cmd = fake_cmd
    get_runtime(cmd)
    pts = _points()
    domain = Domain(padding=2.0, spacing=1.0)
    aabb = domain.resolve_aabb([(0.0, 0.0, 0.0)])
    preview = FromSelectionFieldPreview(cmd)

    preview.update(pts, 0.4, live_iso=False, domain_aabb=aabb, show_domain=False)
    assert preview._domain._obj is None

    preview.update(pts, 0.4, live_iso=False, domain_aabb=aabb, show_domain=True)
    assert preview._domain._obj is not None
    assert type(preview._domain._obj[0]).__name__ == "CenteredBox"
    assert preview._domain._obj[0].wireframe is True

    preview.update(pts, 0.4, live_iso=False, domain_aabb=aabb, show_domain=False)
    assert preview._domain._obj is None
    preview.cleanup()
    names = [str(n) for n in cmd.get_names("objects")]
    assert not any(name.startswith("_pmv_prev_") for name in names)
    reset_runtime()


def test_field_visual_preview_draws_domain_box(fake_cmd):
    reset_runtime()
    cmd = fake_cmd
    get_runtime(cmd)
    field = _blob_field()
    visual = build_grid_preview_visual(
        field, kind="IsoSurface", iso_level=0.4, name=PREVIEW_FIELD_VISUAL_NAME,
    )
    preview = FieldVisualPreview(cmd)
    aabb = [[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]]
    preview.update(visual, iso_key=("iso", 0.4), domain_aabb=aabb, show_domain=True)
    assert preview.visual is not None
    assert preview._domain._obj is not None
    assert type(preview._domain._obj[0]).__name__ == "CenteredBox"
    preview.cleanup()
    names = [str(n) for n in cmd.get_names("objects")]
    assert not any(str(n).startswith("_pmv_prev_") for n in names)
    reset_runtime()
