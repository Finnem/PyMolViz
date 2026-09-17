"""Points colored by a scalar, as spheres or dots."""

from pathlib import Path

import numpy as np
import pymolviz as pmv

out = Path(__file__).resolve().parent / "out"
out.mkdir(exist_ok=True)

np.random.seed(1)
vertices = np.random.rand(10, 3) * 10
values = np.linalg.norm(vertices, axis=1)

pmv.Points(vertices, color="red", name="basic_points").write(out / "basic_points.py")
pmv.Points(vertices, render_as="Dots", name="dot_points").write(out / "dot_points.py")
pmv.Points(vertices, radius=1, name="larger_points").write(out / "larger_points.py")
pmv.Points(vertices, color=values, name="colored_points").write(out / "colored_points.py")
pmv.Points(
    vertices,
    color=values,
    colormap="viridis",
    name="viridis_points",
).write(out / "viridis_points.py")
