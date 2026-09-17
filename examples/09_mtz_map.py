"""FFT a reflection column from an MTZ file into a map (gemmi is a core dependency)."""

from pathlib import Path

import pymolviz as pmv

root = Path(__file__).resolve().parents[1]
out = Path(__file__).resolve().parent / "out"
out.mkdir(exist_ok=True)

mtz = root / "test" / "data" / "4de3_phases.mtz"
# Full unit cell by default. Pass min_pos / max_pos to crop around a region.
field = pmv.Field.from_mtz(mtz, name="mtz_4de3")
cmap = pmv.ColorMap([-1, 1], range_mode="custom")
pmv.IsoVolume(field, name="mtz_4de3_volume", colormap=cmap).write(out / "mtz_4de3_isovolume.py")
