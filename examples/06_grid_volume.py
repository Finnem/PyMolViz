"""Synthetic field displayed as a PyMOL volume."""

from pathlib import Path

import numpy as np
import pymolviz as pmv
from pymolviz.io import save

out = Path(__file__).resolve().parent / "out"
out.mkdir(exist_ok=True)

n = 16
xs = np.linspace(0.0, 1.0, n)
xx, yy, zz = np.meshgrid(xs, xs, xs, indexing="ij")
values = xx + yy + zz  # Manhattan-like distance from origin
h = float(xs[1] - xs[0])
field = pmv.Field(
    values.reshape(-1),
    step_sizes=(h, h, h),
    step_counts=(n - 1, n - 1, n - 1),
    origin=(0.0, 0.0, 0.0),
    name="manhattan",
)
volume = pmv.Volume(field, name="manhattan_volume")

field.write(out / "manhattan_field.py")
field.to_points(name="manhattan_points", radius=0.04).write(out / "manhattan_points.py")
volume.write(out / "manhattan_volume.py")
save([field, volume], out / "manhattan.pmv")
