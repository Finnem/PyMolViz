"""Electron-density-style volume from a Turbomole xyz grid (test/data/td.xyz)."""

from pathlib import Path

import pymolviz as pmv

root = Path(__file__).resolve().parents[1]
out = Path(__file__).resolve().parent / "out"
out.mkdir(exist_ok=True)

xyz = root / "test" / "data" / "td.xyz"
field = pmv.Field.from_xyz(xyz, name="td")
cmap = pmv.ColorMap([0, 0.5])
pmv.Volume(field, name="td_volume", colormap=cmap).write(out / "td_volume.py")
pmv.IsoVolume(
    field,
    name="td_isovolume",
    colormap=pmv.ColorMap([0, 0.5], colormap="viridis"),
).write(out / "td_isovolume.py")
