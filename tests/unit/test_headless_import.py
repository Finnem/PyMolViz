"""Ensure the public API works without PyMOL installed."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_import_pymolviz_without_pymol_subprocess():
    module_root = Path(__file__).resolve().parents[2] / "module"
    code = """
import sys
for key in list(sys.modules):
    if key == "pymol" or key.startswith("pymol."):
        del sys.modules[key]
sys.path.insert(0, %r)
import pymolviz
from pymolviz import Mesh, Script, Sphere
import numpy as np
verts = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
mesh = Mesh(verts, faces=[[0, 1, 2]], name="headless_tri")
script = str(Script([mesh]))
assert "load_cgo" in script
assert "headless_tri" in script
sphere = Sphere([0., 0., 0.], 1.0, name="headless_sphere")
assert sphere.name == "headless_sphere"
""" % (str(module_root),)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
