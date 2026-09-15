"""Field isosurface preview helpers (no wizard import)."""

from __future__ import annotations

import numpy as np
import pytest

from pymolviz.fields.identity import GEN_DISTANCE, GEN_GAUSSIAN, GEN_SIGNED_VDW
from pymolviz.points import FixedPoint
from pymolviz.wizards.builders.field_params import field_model_shows
from pymolviz.wizards.builders.field_preview import (
    PREVIEW_COLOR_MAP_NAME,
    PREVIEW_GEOM_MAP_NAME,
    VOLUME_PREVIEW_IS_ISO_PROXY,
    build_field_iso_preview_visual,
    build_grid_preview_visual,
    default_field_iso_level,
    field_supports_iso_preview,
    field_visual_preview_key,
    iso_preview_key,
    make_unregistered_preview_visual,
    volume_preview_is_iso_proxy,
)
from pymolviz.wizards.builders.points import AtomRef, VisualPoint
from tests.fakes.cmd import FakeCmd


def _gaussian_points():
    return [
        VisualPoint(
            "C1", "manual", 0.0, 0.0, 0.0,
            point_source=FixedPoint((0.0, 0.0, 0.0)),
            atom_ref=AtomRef("m", 1, elem="C"),
        ),
        VisualPoint(
            "O1", "manual", 1.5, 0.0, 0.0,
            point_source=FixedPoint((1.5, 0.0, 0.0)),
            atom_ref=AtomRef("m", 2, elem="O"),
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


def test_field_model_shows_iso_and_live_preview_for_all_algorithms():
    assert field_model_shows(GEN_GAUSSIAN, "iso_value") is True
    assert field_model_shows(GEN_GAUSSIAN, "live_preview") is True
    assert field_model_shows(GEN_DISTANCE, "iso_value") is True
    assert field_model_shows(GEN_DISTANCE, "live_preview") is True
    assert field_model_shows(GEN_SIGNED_VDW, "iso_value") is True
    assert field_model_shows(GEN_SIGNED_VDW, "live_preview") is True
    assert field_model_shows(GEN_DISTANCE, "quality") is False


def test_field_supports_iso_preview_and_default_level():
    assert field_supports_iso_preview(GEN_GAUSSIAN) is True
    assert field_supports_iso_preview(GEN_DISTANCE) is True
    assert field_supports_iso_preview(GEN_SIGNED_VDW) is True
    assert default_field_iso_level(GEN_GAUSSIAN) == pytest.approx(1.0)
    assert default_field_iso_level(GEN_DISTANCE) == pytest.approx(1.5)
    assert default_field_iso_level(GEN_SIGNED_VDW) == pytest.approx(0.0)


def test_iso_preview_key_tracks_iso_level():
    from pymolviz.fields.domain import Domain

    pts = _gaussian_points()
    domain = Domain(padding=1.0, spacing=1.0)
    key_a = iso_preview_key(
        pts,
        algorithm=GEN_GAUSSIAN,
        domain=domain,
        quality=2,
        resolution=2.0,
        property_key="b_factor",
        iso_level=1.0,
    )
    key_b = iso_preview_key(
        pts,
        algorithm=GEN_GAUSSIAN,
        domain=domain,
        quality=2,
        resolution=2.0,
        property_key="b_factor",
        iso_level=0.5,
    )
    assert key_a is not None
    assert key_a != key_b


def test_make_unregistered_preview_visual_is_native_iso():
    from pymolviz.runtime import session as session_mod

    session_mod.clear()
    field = _blob_field()
    visual = make_unregistered_preview_visual(
        field, kind="IsoSurface", iso_level=0.4, name="_pmv_prev_field_visual",
    )
    assert visual is not None
    assert type(visual).__name__ == "IsoSurface"
    assert str(visual.name).startswith("_pmv_prev_")
    assert visual.grid_data.name == PREVIEW_GEOM_MAP_NAME
    assert float(visual.level) == pytest.approx(0.4)
    assert str(visual.id).startswith("preview_")
    assert session_mod.get(str(visual.id)) is None
    assert session_mod.get(str(field.id)) is None
    assert not any(type(obj).__name__ in ("IsoSurface", "IsoMesh", "Volume") for obj in session_mod.all_objects())
    session_mod.clear()


def test_make_unregistered_preview_visual_isomesh_and_volume():
    from pymolviz.runtime import session as session_mod

    session_mod.clear()
    field = _blob_field()
    mesh = make_unregistered_preview_visual(field, kind="IsoMesh", iso_level=0.4, name="_pmv_prev_mesh")
    vol = make_unregistered_preview_visual(field, kind="Volume", name="_pmv_prev_vol")
    assert type(mesh).__name__ == "IsoMesh"
    assert type(vol).__name__ == "Volume"
    assert mesh.grid_data.name == PREVIEW_GEOM_MAP_NAME
    assert vol.grid_data.name == PREVIEW_GEOM_MAP_NAME
    assert session_mod.get(str(mesh.id)) is None
    assert session_mod.get(str(vol.id)) is None
    session_mod.clear()


def test_build_field_iso_preview_visual_unregistered():
    from pymolviz.fields.domain import Domain
    from pymolviz.runtime import session as session_mod

    session_mod.clear()
    cmd = FakeCmd()
    pts = _gaussian_points()
    domain = Domain(padding=2.0, spacing=1.0)
    visual = build_field_iso_preview_visual(
        cmd,
        pts,
        algorithm=GEN_GAUSSIAN,
        domain=domain,
        quality=1,
        resolution=2.0,
        iso_level=0.5,
        name="_pmv_prev_field_iso",
    )
    assert visual is not None
    assert type(visual).__name__ == "IsoSurface"
    assert visual.grid_data.name == PREVIEW_GEOM_MAP_NAME
    assert not any(type(obj).__name__ in ("Field", "IsoSurface", "IsoMesh", "Volume") for obj in session_mod.all_objects())
    session_mod.clear()


def test_iso_preview_uses_uniform_color_not_point_ramp():
    from pymolviz.fields.domain import Domain
    from pymolviz.runtime import session as session_mod
    from pymolviz.volumetric.ColorRamp import ColorRamp
    from pymolviz.wizards.builders.surface_params import COLOR_MODE_PER_POINT, COLOR_MODE_UNIFORM

    session_mod.clear()
    cmd = FakeCmd()
    pts = [
        VisualPoint(
            "C1", "manual", 0.0, 0.0, 0.0,
            color=(1.0, 0.0, 0.0),
            point_source=FixedPoint((0.0, 0.0, 0.0)),
            atom_ref=AtomRef("m", 1, elem="C"),
        ),
        VisualPoint(
            "O1", "manual", 1.5, 0.0, 0.0,
            color=(0.0, 0.0, 1.0),
            point_source=FixedPoint((1.5, 0.0, 0.0)),
            atom_ref=AtomRef("m", 2, elem="O"),
        ),
    ]
    domain = Domain(padding=2.0, spacing=1.0)
    visual = build_field_iso_preview_visual(
        cmd,
        pts,
        algorithm=GEN_GAUSSIAN,
        domain=domain,
        quality=1,
        resolution=2.0,
        iso_level=0.5,
        color_mode=COLOR_MODE_UNIFORM,
        name="_pmv_prev_field_iso",
    )
    assert visual is not None
    assert not issubclass(type(visual.color), ColorRamp)
    assert tuple(float(c) for c in visual.color[:3]) == pytest.approx((1.0, 0.0, 0.0))
    legacy = build_field_iso_preview_visual(
        cmd,
        pts,
        algorithm=GEN_GAUSSIAN,
        domain=domain,
        quality=1,
        resolution=2.0,
        iso_level=0.5,
        color_mode=COLOR_MODE_PER_POINT,
        name="_pmv_prev_field_iso",
    )
    assert legacy is not None
    assert not issubclass(type(legacy.color), ColorRamp)
    assert not any(type(obj).__name__ in ("Field", "IsoSurface") for obj in session_mod.all_objects())
    session_mod.clear()


def test_iso_preview_from_field_attaches_scalar_ramp():
    from pymolviz.fields.domain import Domain
    from pymolviz.runtime import session as session_mod
    from pymolviz.volumetric.ColorRamp import ColorRamp
    from pymolviz.wizards.builders.load_field import field_from_selection
    from pymolviz.wizards.builders.surface_params import COLOR_MODE_FIELD

    session_mod.clear()
    cmd = FakeCmd()
    pts = _gaussian_points()
    domain = Domain(padding=2.0, spacing=1.0)
    tint = field_from_selection(
        cmd, pts, "tint", algorithm=GEN_DISTANCE, domain=domain,
    )
    visual = build_field_iso_preview_visual(
        cmd,
        pts,
        algorithm=GEN_GAUSSIAN,
        domain=domain,
        quality=1,
        resolution=2.0,
        iso_level=0.5,
        color_mode=COLOR_MODE_FIELD,
        color_field_id=str(tint.id),
        name="_pmv_prev_field_iso",
    )
    assert visual is not None
    assert issubclass(type(visual.color), ColorRamp)
    assert visual.color.data.name == PREVIEW_COLOR_MAP_NAME
    session_mod.clear()


def test_distance_iso_preview_register_false():
    from pymolviz.fields.domain import Domain
    from pymolviz.runtime import session as session_mod

    session_mod.clear()
    cmd = FakeCmd()
    pts = _gaussian_points()
    domain = Domain(padding=2.0, spacing=1.0)
    visual = build_field_iso_preview_visual(
        cmd,
        pts,
        algorithm=GEN_DISTANCE,
        domain=domain,
        quality=1,
        resolution=2.0,
        iso_level=1.0,
        name="_pmv_prev_field_iso",
    )
    assert visual is not None
    assert type(visual).__name__ == "IsoSurface"
    assert float(visual.level) == pytest.approx(1.0)
    assert not any(type(obj).__name__ in ("Field", "IsoSurface", "IsoMesh", "Volume") for obj in session_mod.all_objects())
    session_mod.clear()


def test_signed_vdw_iso_preview_at_zero():
    from pymolviz.fields.domain import Domain

    cmd = FakeCmd()
    pts = _gaussian_points()
    domain = Domain(padding=3.0, spacing=0.75)
    visual = build_field_iso_preview_visual(
        cmd,
        pts,
        algorithm=GEN_SIGNED_VDW,
        domain=domain,
        quality=1,
        resolution=2.0,
        iso_level=0.0,
        name="_pmv_prev_field_iso",
    )
    assert visual is not None
    assert type(visual).__name__ == "IsoSurface"
    assert float(visual.level) == pytest.approx(0.0)


def test_grid_preview_visual_kinds_not_interned():
    from pymolviz.runtime import session as session_mod

    session_mod.clear()
    field = _blob_field()
    solid = build_grid_preview_visual(field, kind="IsoSurface", iso_level=0.4, name="_pmv_prev_iso")
    wire = build_grid_preview_visual(field, kind="IsoMesh", iso_level=0.4, name="_pmv_prev_mesh")
    assert type(solid).__name__ == "IsoSurface"
    assert type(wire).__name__ == "IsoMesh"
    assert solid.grid_data.name == PREVIEW_GEOM_MAP_NAME
    assert session_mod.get(str(field.id)) is None
    session_mod.clear()


def test_volume_preview_is_native_not_iso_proxy():
    assert VOLUME_PREVIEW_IS_ISO_PROXY is False
    assert volume_preview_is_iso_proxy("Volume") is False
    field = _blob_field()
    visual = build_grid_preview_visual(field, kind="Volume", iso_level=0.4, name="_pmv_prev_vol")
    assert type(visual).__name__ == "Volume"
    assert visual.grid_data.name == PREVIEW_GEOM_MAP_NAME


def test_preview_volume_clip_shrinks_the_preview_brick():
    field = _blob_field()
    full = build_grid_preview_visual(field, kind="Volume", name="_pmv_prev_vol")
    cropped = build_grid_preview_visual(
        field,
        kind="Volume",
        name="_pmv_prev_vol",
        clip_aabb=[[-0.5, -2.0, -2.0], [0.5, 2.0, 2.0]],
    )
    full_span = float(np.asarray(full.grid_data.step_counts)[0])
    crop_span = float(np.asarray(cropped.grid_data.step_counts)[0])
    assert crop_span < full_span
    assert cropped.clip_aabb is None


def test_field_visual_preview_key_tracks_iso_and_clip():
    key_a = field_visual_preview_key(
        kind="IsoSurface", field_id="f1", iso_level=0.4, side="positive",
    )
    key_b = field_visual_preview_key(
        kind="IsoSurface", field_id="f1", iso_level=0.8, side="positive",
    )
    key_c = field_visual_preview_key(
        kind="IsoSurface", field_id="f1", iso_level=0.4, side="positive",
        clip_aabb=[[0, 0, 0], [1, 1, 1]],
    )
    assert key_a != key_b
    assert key_a != key_c


def test_field_visual_preview_key_tracks_map_origin():
    key_a = field_visual_preview_key(
        kind="Volume", field_id="f1", iso_level=0.4, origin=(0.0, 0.0, 0.0),
    )
    key_b = field_visual_preview_key(
        kind="Volume", field_id="f1", iso_level=0.4, origin=(10.0, 0.0, 0.0),
    )
    assert key_a != key_b


def test_field_visual_preview_key_tracks_colormap_spec_and_clims():
    spec_a = {"preset": "Custom 2", "customized": True, "stops": [{"position": 0.0, "rgba": [0, 0, 1, 1]}]}
    spec_b = {"preset": "Custom 2", "customized": True, "stops": [{"position": 0.0, "rgba": [1, 0, 0, 1]}]}
    key_a = field_visual_preview_key(
        kind="Volume", field_id="f1", iso_level=0.4, colormap="Custom 2", colormap_spec=spec_a,
    )
    key_b = field_visual_preview_key(
        kind="Volume", field_id="f1", iso_level=0.4, colormap="Custom 2", colormap_spec=spec_b,
    )
    key_c = field_visual_preview_key(
        kind="Volume", field_id="f1", iso_level=0.4, colormap="Custom 2", clims=(-1.0, 2.0),
    )
    key_d = field_visual_preview_key(
        kind="Volume", field_id="f1", iso_level=0.4, colormap="Custom 2", clims=(0.0, 1.0),
    )
    assert key_a != key_b
    assert key_c != key_a
    assert key_c != key_d


def test_build_domain_box_collection_is_wire_centered_box():
    from pymolviz.wizards.builders.field_preview import build_domain_box_collection

    collection = build_domain_box_collection(
        [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], name="_test_domain",
    )
    assert len(collection) == 1
    mesh = collection[0]
    assert type(mesh).__name__ == "CenteredBox"
    assert mesh.wireframe is True
    tokens = mesh._create_CGO_list()
    kinds = {tok for tok in tokens if isinstance(tok, str)}
    assert "CONE" in kinds or "CYLINDER" in kinds or "LINES" in kinds or "CONE" in kinds
    empty = build_domain_box_collection([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    assert len(empty) == 0


def test_build_domain_box_collection_sheared_grid_is_line_parallelepiped():
    from pymolviz.volumetric.GridData import GridData
    from pymolviz.wizards.builders.field_preview import build_domain_box_collection

    grid = GridData(
        np.zeros(8, dtype=float),
        step_sizes=(1.0, 1.0, 1.0),
        step_counts=(1, 1, 1),
        origin=(0.0, 0.0, 0.0),
        name="cell",
    )
    A = np.eye(4)
    A[:3, 0] = [10.0, 0.0, 0.0]
    A[:3, 1] = [-5.0, 8.66, 0.0]
    A[:3, 2] = [0.0, 0.0, 10.0]
    grid.A_to = A
    grid._crystal_O = A[:3, :3].tolist()
    grid._crystal_nmin = [0, 0, 0]
    grid._crystal_nmax = [1, 1, 1]
    grid._crystal_origin = [0.0, 0.0, 0.0]
    collection = build_domain_box_collection(
        [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]], name="_test_domain", grid=grid,
    )
    assert len(collection) == 1
    mesh = collection[0]
    assert type(mesh).__name__ == "Lines"
    assert mesh.vertices.reshape(-1, 6).shape[0] == 12
