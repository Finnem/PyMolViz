"""Basic headless point cloud — run this, then load ``out/points.py`` in PyMOL."""

from pathlib import Path

import numpy as np
import pymolviz as pmv

out = Path(__file__).resolve().parent / "out"
out.mkdir(exist_ok=True)

np.random.seed(0)
points = np.random.rand(10, 3) * 10
pmv.Points(points, color="red", name="basic_points").write(out / "points.py")
