"""MTZ column pairing and multi-map field load (no gemmi FFT)."""

from __future__ import annotations

import numpy as np
import pytest

from pymolviz.fields.field import Field


class _FakeMtzCol:
    def __init__(self, label, ctype):
        self.label = label
        self.type = ctype


class _FakeMtz:
    def __init__(self, columns):
        self.columns = [_FakeMtzCol(label, ctype) for label, ctype in columns]


def _tiny_field(name="g", obj_id=None):
    return Field(
        np.arange(8, dtype=float),
        step_sizes=(1.0, 1.0, 1.0),
        step_counts=(1, 1, 1),
        origin=(0.0, 0.0, 0.0),
        name=name,
        obj_id=obj_id,
    )


def test_choose_mtz_map_columns_falls_back_when_fwt_missing():
    from pymolviz.util.io import choose_mtz_map_columns, iter_mtz_map_column_pairs

    mtz = _FakeMtz([
        ("H", "H"), ("K", "H"), ("L", "H"),
        ("FOBS", "F"), ("SIGFOBS", "Q"), ("PHIB", "P"), ("FOM", "W"),
        ("FreeR_flag", "I"),
    ])
    factor, phase = choose_mtz_map_columns(mtz, factor_column="FWT", phase_column="PHWT")
    assert factor == "FOBS"
    assert phase == "PHIB"
    missing = _FakeMtz([("H", "H"), ("FTZ", "F"), ("PHIB", "P")])
    factor, phase = choose_mtz_map_columns(missing, factor_column="FTZ", phase_column="PHWT")
    assert factor == "FTZ"
    assert phase == "PHIB"
    pairs = list(iter_mtz_map_column_pairs(mtz, "FWT", "PHWT"))
    assert ("FOBS", "PHIB") in pairs
    assert ("FreeR_flag", "PHIB") not in pairs


def test_choose_mtz_map_columns_prefers_fwt_when_present():
    from pymolviz.util.io import choose_mtz_map_columns

    mtz = _FakeMtz([
        ("FWT", "F"), ("PHWT", "P"), ("DELFWT", "F"), ("PHDELWT", "P"),
    ])
    assert choose_mtz_map_columns(mtz) == ("FWT", "PHWT")


def test_choose_mtz_map_columns_errors_without_phase():
    from pymolviz.util.io import choose_mtz_map_columns

    mtz = _FakeMtz([("H", "H"), ("FOBS", "F"), ("SIGFOBS", "Q")])
    with pytest.raises(ValueError, match="amplitude/phase"):
        choose_mtz_map_columns(mtz)


def test_mtz_crop_shape_defaults_to_full_unit_cell():
    from pymolviz.util.io import mtz_crop_shape

    assert mtz_crop_shape(None, None, None) is None
    assert mtz_crop_shape([0, 0, 0], [1, 1, 1], [1, 1, 1]) is None
    assert mtz_crop_shape([0, 0, 0], [10, 10, 10], [1, 1, 1]) == (10, 10, 10)


def test_griddata_from_gemmi_map_uses_cell_spacing():
    from pymolviz.util.io import griddata_from_gemmi_map

    class _Vec:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class _Map:
        def __init__(self):
            self._arr = np.arange(24, dtype=float).reshape(2, 3, 4)
            self.unit_cell = type("Cell", (), {
                "a": 10.0, "b": 20.0, "c": 30.0,
                "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
            })()

        def __array__(self, dtype=None):
            return np.asarray(self._arr, dtype=dtype)

        def get_position(self, u, v, w):
            nx, ny, nz = self._arr.shape
            return _Vec(
                u * 10.0 / (nx - 1),
                v * 20.0 / (ny - 1),
                w * 30.0 / (nz - 1),
            )

    grid = griddata_from_gemmi_map(_Map(), name="cell")
    assert tuple(int(n) for n in grid.step_counts) == (1, 2, 3)
    assert grid.step_sizes[0] == pytest.approx(10.0)
    assert grid.step_sizes[1] == pytest.approx(10.0)
    assert grid.step_sizes[2] == pytest.approx(10.0)
    assert grid.values.size == 24
    assert getattr(grid, "_crystal_cell", None) is not None
    assert getattr(grid, "_crystal_axis_grid", False) is False


def test_griddata_from_gemmi_map_keeps_sheared_cell():
    from pymolviz.fields.crystal import grid_is_crystal_axis
    from pymolviz.util.io import griddata_from_gemmi_map

    class _Vec:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class _Map:
        def __init__(self):
            self._arr = np.arange(27, dtype=float).reshape(3, 3, 3)
            self.unit_cell = type("Cell", (), {
                "a": 10.0, "b": 10.0, "c": 10.0,
                "alpha": 90.0, "beta": 90.0, "gamma": 120.0,
            })()
            self.spacegroup = type("SG", (), {"hm": "P 1 21 1"})()

        def __array__(self, dtype=None):
            return np.asarray(self._arr, dtype=dtype)

        def get_position(self, u, v, w):
            return _Vec(float(u), float(v), float(w))

    grid = griddata_from_gemmi_map(_Map(), name="sheared")
    assert grid_is_crystal_axis(grid) is True
    assert grid._crystal_cell[6] == "P 1 21 1"


def test_lattice_single_sample_axis_does_not_crash():
    grid = Field(
        [1.0],
        positions=np.array([[0.0, 0.0, 0.0]]),
        name="one",
    )
    assert tuple(int(n) for n in grid.step_counts) == (0, 0, 0)
    assert grid.values.size == 1


def test_mtz_inventory_lists_maps_and_columns():
    from pymolviz.util.io import mtz_field_basename, mtz_inventory, selected_mtz_maps

    mtz = _FakeMtz([
        ("H", "H"), ("K", "H"), ("L", "H"),
        ("FWT", "F"), ("PHWT", "P"),
        ("DELFWT", "F"), ("PHDELWT", "P"),
        ("FOBS", "F"), ("SIGFOBS", "Q"), ("PHIB", "P"),
        ("FreeR_flag", "I"),
    ])
    inv = mtz_inventory(mtz)
    ids = [row["id"] for row in inv["maps"]]
    assert "FWT|PHWT" in ids
    assert "DELFWT|PHDELWT" in ids
    assert "FOBS|PHIB" in ids
    defaults = [row["id"] for row in inv["maps"] if row.get("default")]
    assert "FWT|PHWT" in defaults
    assert "DELFWT|PHDELWT" in defaults
    labels = [row["label"] for row in inv["columns"]]
    assert labels[:3] == ["H", "K", "L"]
    assert "SIGFOBS" in labels
    types = {row["label"]: row["type"] for row in inv["columns"]}
    assert types["PHWT"] == "P"
    assert types["FWT"] == "F"
    picked = selected_mtz_maps(inv, ["DELFWT|PHDELWT", "FC|PHIC"])
    assert [row["factor"] for row in picked] == ["DELFWT", "FC"]
    assert picked[0]["title"] == "Fo-Fc"
    assert mtz_field_basename("1abc", picked[0]) == "1abc_Fo-Fc"


def test_load_mtz_fields_one_field_per_map(monkeypatch):
    from pymolviz.util import io as io_mod
    from pymolviz.wizards.builders.load_field import load_mtz_fields
    from tests.fakes.cmd import FakeCmd

    names = []

    def fake_grid(path, **kwargs):
        names.append(kwargs.get("name"))
        return _tiny_field(name=kwargs.get("name") or "g", obj_id="mtz_%s" % kwargs.get("name"))

    monkeypatch.setattr(io_mod, "grid_from_mtz", fake_grid)
    fields = load_mtz_fields(
        FakeCmd(),
        "/tmp/1abc.mtz",
        [
            {"factor": "FWT", "phase": "PHWT", "title": "2Fo-Fc"},
            {"factor": "DELFWT", "phase": "PHDELWT", "title": "Fo-Fc"},
        ],
        stem="1abc",
    )
    assert [field.generator.get("factor_column") for field in fields] == ["FWT", "DELFWT"]
    assert [field.generator.get("phase_column") for field in fields] == ["PHWT", "PHDELWT"]
    assert names == ["1abc_2Fo-Fc", "1abc_Fo-Fc"]
