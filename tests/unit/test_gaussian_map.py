"""PyMOL map_new gaussian density (Cromer–Mann splat)."""

from __future__ import annotations

import numpy as np
import pytest

from pymolviz.util.gaussian_map import (
    DEFAULT_GAUSSIAN_B_FLOOR,
    gaussian_density_at_origin,
    gaussian_density_brick,
    gaussian_pad_radius,
    normalize_gaussian_map,
    paint_gaussian_density,
)


def test_carbon_b20_peak_matches_pymol_formula():
    peak = gaussian_density_at_origin("C", b_factor=20.0, occupancy=1.0, resolution=2.0)
    assert peak == pytest.approx(1.59742, abs=1e-4)


def test_paint_gaussian_density_peaks_at_atom():
    origin = np.array([-2.0, -2.0, -2.0], dtype=float)
    h = 0.5
    shape = (9, 9, 9)
    field = paint_gaussian_density(
        shape, origin, h, [(0.0, 0.0, 0.0)], ["C"],
        b_factors=[DEFAULT_GAUSSIAN_B_FLOOR],
    )
    assert float(field.max()) == pytest.approx(
        gaussian_density_at_origin("C"), abs=0.08,
    )
    ijk = np.unravel_index(int(np.argmax(field)), field.shape)
    xyz = origin + np.array(ijk, dtype=float) * h
    assert np.linalg.norm(xyz) < h * 1.5


def test_normalize_gaussian_map_is_zero_mean_unit_stdev():
    raw = np.array([0.0, 0.0, 1.0, 3.0], dtype=float)
    out = normalize_gaussian_map(raw)
    assert float(out.mean()) == pytest.approx(0.0, abs=1e-12)
    assert float(out.std(ddof=1)) == pytest.approx(1.0, abs=1e-12)


def test_paint_gaussian_matches_cromer_mann_at_grid_points():
    from pymolviz.util.gaussian_map import atom_gaussian_terms, gaussian_blur_factor

    origin = np.array([-1.5, -1.5, -1.5], dtype=float)
    h = 0.5
    shape = (7, 7, 7)
    centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    field = paint_gaussian_density(shape, origin, h, centers, ["C", "C"])
    blur = gaussian_blur_factor(2.0)
    amps, kappas, rcut = atom_gaussian_terms("C", DEFAULT_GAUSSIAN_B_FLOOR, 1.0, blur)
    ijk = np.array((3, 2, 4), dtype=int)
    xyz = origin + ijk.astype(float) * h
    expected = 0.0
    for center in centers:
        delta = xyz - center
        dist = float(np.linalg.norm(delta)) * blur
        if dist >= rcut:
            continue
        d2 = max(dist, 1e-8) ** 2
        expected += float(np.sum(amps * np.exp(-kappas * d2)) * blur)
    assert float(field[tuple(ijk)]) == pytest.approx(expected, rel=1e-10, abs=1e-12)


def test_array_backend_probe_returns_numpy_or_working_cupy():
    from pymolviz.util.array_backend import array_backend_name, reset_array_module

    reset_array_module()
    assert array_backend_name() in ("numpy", "cupy")
    assert array_backend_name(prefer="numpy") == "numpy"


def test_cuda_startup_message_only_when_cupy_works():
    from pymolviz.util.array_backend import (
        array_backend_name,
        cuda_startup_message,
        reset_array_module,
    )

    reset_array_module()
    msg = cuda_startup_message()
    if array_backend_name() == "cupy":
        assert msg is not None
        assert "CUDA" in msg
        assert "PyMOLViz" in msg
    else:
        assert msg is None


def test_announce_cuda_is_silent_without_device(capsys):
    from pymolviz.util.array_backend import (
        announce_cuda,
        array_backend_name,
        reset_array_module,
    )

    reset_array_module()
    if array_backend_name() == "cupy":
        pytest.skip("CUDA is available in this environment")
    assert announce_cuda() is False
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_announce_cuda_prints_once_when_cupy_works(capsys):
    from pymolviz.util.array_backend import (
        announce_cuda,
        array_backend_name,
        reset_array_module,
    )

    reset_array_module()
    if array_backend_name() != "cupy":
        pytest.skip("CUDA is not available in this environment")
    assert announce_cuda() is True
    first = capsys.readouterr()
    assert "CUDA" in first.out
    assert announce_cuda() is False
    second = capsys.readouterr()
    assert second.out == ""


def test_paint_auto_backend_matches_forced_numpy():
    origin = np.array([-1.5, -1.5, -1.5], dtype=float)
    centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    kwargs = dict(shape=(7, 7, 7), origin=origin, h=0.5, centers=centers, elements=["C", "C"])
    numpy_field = paint_gaussian_density(backend="numpy", **kwargs)
    auto_field = paint_gaussian_density(**kwargs)
    assert numpy_field == pytest.approx(auto_field, rel=1e-6, abs=1e-9)


def test_gaussian_density_brick_is_normalized_and_covers_atom():
    brick = gaussian_density_brick(
        [(0.0, 0.0, 0.0)], ["C"], 0.5, resolution=2.0, b_floor=DEFAULT_GAUSSIAN_B_FLOOR,
    )
    assert brick is not None
    origin, step, density = brick
    assert float(step) == pytest.approx(0.5)
    assert density.ndim == 3
    assert float(density.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(density.std(ddof=1)) == pytest.approx(1.0, abs=1e-5)
    ijk = np.unravel_index(int(np.argmax(density)), density.shape)
    xyz = np.asarray(origin, dtype=float) + np.array(ijk, dtype=float) * step
    assert np.linalg.norm(xyz) < 1.5
    assert gaussian_density_brick([], ["C"], 0.5) is None
    assert gaussian_pad_radius(["C"]) >= 2.0

