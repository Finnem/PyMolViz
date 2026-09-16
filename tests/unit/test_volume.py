"""Volume ramp construction stays O(N) in memory."""

from __future__ import annotations

import numpy as np

from pymolviz.volumetric.GridData import GridData
from pymolviz.volumetric.Volume import Volume, _expand_range_to_volume_clims, _pair_bin_counts


def test_pair_bin_counts_matches_broadcast_and_avoids_outer_bool():
    values = np.array([-1.0, 0.0, 0.1, 0.5, 0.9, 2.0])
    clims = _expand_range_to_volume_clims(0.0, 1.0, 5)
    used = len(clims) - (len(clims) % 2)
    bins = np.reshape(clims[:used], (-1, 2))
    expected = np.sum((bins[:, 0] < values[:, None]) & (bins[:, 1] >= values[:, None]), axis=0)
    got = _pair_bin_counts(values, bins)
    assert np.array_equal(got, expected)


def test_volume_alphas_on_large_flat_grid():
    values = np.zeros(200000, dtype=float)
    values[::10] = 0.02
    values[::1000] = 1.5
    grid = GridData(values, origin=(0, 0, 0), step_sizes=(1, 1, 1), step_counts=(199999, 0, 0), name="big")
    vol = Volume(grid, name="pmv_vol", colormap="RdYlBu_r")
    assert len(vol.alphas) == len(vol.clims)
    assert np.all(np.isfinite(vol.alphas))
