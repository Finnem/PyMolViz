"""Preview fidelity helpers (no Qt)."""

import numpy as np
import pytest

from pymolviz.fields.domain import Domain
from pymolviz.volumetric.GridData import GridData
from pymolviz.meshes.CGOCollection import CGOCollection
from pymolviz.meshes.Sphere import Sphere
from pymolviz.serialization import displayable_from_dict, displayable_to_dict
from pymolviz.wizards.builders.preview_mode import (
    PREVIEW_FULL,
    PREVIEW_FULL_LABEL,
    PREVIEW_OFF,
    PREVIEW_OFF_LABEL,
    PREVIEW_SIMPLE,
    PREVIEW_SIMPLE_LABEL,
    read_preview_mode,
    stamp_preview_mode,
    normalize_preview_mode,
    preview_arrow_quality,
    preview_can_promote,
    preview_domain,
    preview_grid,
    preview_iso_kind,
    preview_is_full,
    preview_is_on,
    preview_mesh_quality,
    preview_mode_choices,
    preview_wireframe,
    stride_sample_grid,
)


def test_read_and_stamp_preview_mode_on_objects():
    mesh = Sphere((0.0, 0.0, 0.0), 1.0, name="s")
    assert read_preview_mode(mesh) == PREVIEW_SIMPLE
    stamp_preview_mode(mesh, PREVIEW_FULL)
    assert mesh.preview_mode == PREVIEW_FULL
    assert read_preview_mode(mesh) == PREVIEW_FULL
    collection = CGOCollection([mesh], name="coll")
    stamp_preview_mode(collection, PREVIEW_OFF)
    data = displayable_to_dict(collection)
    assert data["preview_mode"] == PREVIEW_OFF
    restored = displayable_from_dict(data)
    assert read_preview_mode(restored) == PREVIEW_OFF


def test_preview_mode_labels_and_aliases():
    labels = {value: label for value, label, _tip in preview_mode_choices()}
    assert labels[PREVIEW_OFF] == PREVIEW_OFF_LABEL == "No preview"
    assert labels[PREVIEW_SIMPLE] == PREVIEW_SIMPLE_LABEL == "Simple preview"
    assert labels[PREVIEW_FULL] == PREVIEW_FULL_LABEL == "Full preview"
    assert normalize_preview_mode("none") == PREVIEW_OFF
    assert normalize_preview_mode("draft") == PREVIEW_SIMPLE
    assert normalize_preview_mode("live") == PREVIEW_FULL
    assert preview_is_on(PREVIEW_SIMPLE) is True
    assert preview_is_on(PREVIEW_OFF) is False
    assert preview_is_full(PREVIEW_FULL) is True
    assert preview_can_promote(PREVIEW_SIMPLE) is False
    assert preview_can_promote(PREVIEW_FULL) is True


def test_simple_preview_uses_draft_meshes_and_isomesh():
    assert preview_mesh_quality(5, PREVIEW_SIMPLE) == 1
    assert preview_mesh_quality(5, PREVIEW_FULL) == 5
    assert preview_arrow_quality(3, PREVIEW_SIMPLE) == 0
    assert preview_arrow_quality(3, PREVIEW_FULL) == 3
    assert preview_wireframe(False, PREVIEW_SIMPLE) is False
    assert preview_wireframe(True, PREVIEW_SIMPLE) is True
    assert preview_wireframe(False, PREVIEW_FULL) is False
    assert preview_iso_kind("IsoSurface", PREVIEW_SIMPLE) == "IsoMesh"
    assert preview_iso_kind("IsoSurface", PREVIEW_FULL) == "IsoSurface"
    assert preview_iso_kind("Volume", PREVIEW_SIMPLE) == "Volume"


def test_simple_preview_coarsens_domain_spacing_and_grid():
    domain = Domain(spacing=0.16)
    simple = preview_domain(domain, PREVIEW_SIMPLE)
    assert simple.spacing == pytest.approx(0.32)
    assert preview_domain(domain, PREVIEW_FULL).spacing == pytest.approx(0.16)

    counts = (3, 3, 3)
    values = np.arange(int(np.prod(np.array(counts) + 1)), dtype=float)
    grid = GridData(
        values, step_sizes=(0.5, 0.5, 0.5), step_counts=counts, origin=(1.0, 2.0, 3.0),
    )
    sampled = stride_sample_grid(grid, 2)
    assert tuple(int(v) for v in sampled.step_counts) == (1, 1, 1)
    assert sampled.values.size == 8
    assert tuple(float(v) for v in sampled.step_sizes) == (1.0, 1.0, 1.0)
    assert tuple(float(v) for v in sampled.origin) == (1.0, 2.0, 3.0)
    assert preview_grid(grid, PREVIEW_FULL) is grid
    assert preview_grid(grid, PREVIEW_SIMPLE).values.size == 8
