"""Iso-surface (and iso-mesh) on a tiny 2×2×2 field."""

from pathlib import Path

import numpy as np
import pymolviz as pmv

out = Path(__file__).resolve().parent / "out"
out.mkdir(exist_ok=True)

positions = np.array(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ],
    dtype=float,
)
values = np.sum(positions, axis=1)
field = pmv.Field(values, positions, name="grid_data")
surface = pmv.IsoSurface(field, 1.5, name="basic_surface")
labels = pmv.Labels(positions, labels=[f"{v:.0f}" for v in values], name="basic_labels")
pmv.Script([field, surface, labels]).write(out / "basic_surface.py")
pmv.IsoMesh(field, 1.5, name="basic_isomesh").write(out / "basic_isomesh.py")
