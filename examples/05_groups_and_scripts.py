"""Bundle several displayables into one PyMOL script or a native .pmv pack."""

from pathlib import Path

import numpy as np
import pymolviz as pmv

out = Path(__file__).resolve().parent / "out"
out.mkdir(exist_ok=True)

origin = np.zeros((3, 3))
axes = np.eye(3)
arrows = pmv.Arrows(
    starts=origin,
    ends=axes,
    name="coordinate_axes_arrows",
    color=axes,
)
labels = pmv.Labels(axes * 1.1, labels=["x", "y", "z"], name="coordinate_axes_labels")
group = pmv.Group([arrows, labels], name="coordinate_axes")
group.write(out / "coordinate_axes.py")

points = pmv.Points(np.random.default_rng(3).random((8, 3)) * 2, name="cloud", color="gray")
pmv.Script([group, points]).write(out / "scene.py")
# Native pack can be loaded back into the wizard; .py is export-only.
points.write(out / "cloud.pmv")
