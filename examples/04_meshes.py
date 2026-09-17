"""Triangle mesh and a sphere, exported without PyMOL."""

from pathlib import Path

import numpy as np
import pymolviz as pmv

out = Path(__file__).resolve().parent / "out"
out.mkdir(exist_ok=True)

verts = np.array(
    [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
    ]
)
faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
pmv.Mesh(verts, faces=faces, color="skyblue", name="tet").write(out / "tet.py")
pmv.Sphere([1.0, 1.0, 1.0], 0.4, color="orange", name="sphere").write(out / "sphere.py")
