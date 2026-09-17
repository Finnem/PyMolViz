# Headless examples

These scripts build PyMOLViz displayables **without PyMOL**. Each writes a `.py` script (and sometimes a `.pmv` pack) under `out/`. Open the `.py` file in PyMOL (`File → Run script…`, or `run path/to/out/….py` in the command line).

```bash
pip install pymolviz
python examples/01_points.py
```

| Script | What it shows |
|--------|----------------|
| `01_points.py` | Named point cloud (same snippet as the README) |
| `02_colored_points.py` | Value-mapped colors, dots vs spheres |
| `03_arrows.py` | Segments (`ends_style="None"`) and arrows |
| `04_meshes.py` | Triangle mesh and a sphere |
| `05_groups_and_scripts.py` | `Group`, `Script`, coordinate axes, `.pmv` pack |
| `06_grid_volume.py` | `Field` from arrays + volume |
| `07_isosurface.py` | Iso-surface at a level, with labels |
| `08_xyz_volume.py` | `Field.from_xyz` on `test/data/td.xyz` |
| `09_mtz_map.py` | `Field.from_mtz` |

`test/` still holds the original notebooks and exported CGO scripts; these files are the v2 headless equivalents.
