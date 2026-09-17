"""Line segments and arrows (there is no separate Lines class)."""

from pathlib import Path

import numpy as np
import pymolviz as pmv

out = Path(__file__).resolve().parent / "out"
out.mkdir(exist_ok=True)

np.random.seed(2)
starts = np.random.rand(10, 3) * 1
ends = np.random.rand(10, 3) * 10
values = np.linalg.norm(ends, axis=1)

# Shafts only — same role as the old Lines type.
pmv.Arrows(
    starts=starts,
    ends=ends,
    name="basic_lines",
    color=values,
    ends_style="None",
).write(out / "basic_lines.py")

pmv.Arrows(
    starts=starts,
    ends=ends,
    name="basic_arrows",
    color=values,
).write(out / "basic_arrows.py")

origin = np.zeros((3, 3))
axes = np.eye(3)
pmv.Arrows(
    starts=origin,
    ends=axes,
    name="coordinate_axes_arrows",
    color=axes,
).write(out / "coordinate_axes_arrows.py")
