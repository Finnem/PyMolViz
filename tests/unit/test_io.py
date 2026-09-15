"""Native ``.pmv`` pack and multi-format save/load."""

from __future__ import annotations

import json
import zipfile
from io import BytesIO

import numpy as np
import pytest

from pymolviz.fields.field import Field
from pymolviz.fields.identity import GEN_DISTANCE, GEN_IMPORTED
from pymolviz.io import (
    FORMAT_NATIVE,
    FORMAT_SCRIPT,
    OPEN_NATIVE_FILTER,
    complete_save_path,
    detect_format,
    format_from_path,
    intern_loaded,
    load,
    save,
)
from pymolviz.io.native import NATIVE_FORMAT, load_native, save_native
from pymolviz.meshes.CGOCollection import CGOCollection
from pymolviz.meshes.Sphere import Sphere
from pymolviz.serialization import ARRAY_REF_KEY, SerializationError, session_document
from pymolviz.volumetric.GridData import GridData


def _tiny_grid(name="density"):
    values = np.arange(8, dtype=np.float32)
    return GridData(
        values,
        step_sizes=(1.0, 1.0, 1.0),
        step_counts=(1, 1, 1),
        origin=(0.0, 0.0, 0.0),
        name=name,
    )


def test_complete_save_path_uses_filter_when_suffix_missing():
    path, fmt = complete_save_path("out", "PyMolViz (*.pmv)")
    assert path.endswith(".pmv")
    assert fmt == FORMAT_NATIVE
    path, fmt = complete_save_path("out", "Python script (*.py)")
    assert path.endswith(".py")
    assert fmt == FORMAT_SCRIPT
    path, fmt = complete_save_path("kept.pmv", "Python script (*.py)")
    assert path == "kept.pmv"
    assert fmt == FORMAT_NATIVE


def test_native_pack_roundtrips_collection(tmp_path):
    sphere = Sphere(
        (1.0, 2.0, 3.0), 0.5, color=(1.0, 0.0, 0.0),
        bypass_colormap=True, obj_id="sph1",
    )
    coll = CGOCollection([sphere], name="pmv_spheres", obj_id="coll1")
    path = tmp_path / "spheres.pmv"
    save(coll, path)
    assert detect_format(path) == FORMAT_NATIVE
    restored = load(path)
    assert len(restored) == 1
    got = restored[0]
    assert isinstance(got, CGOCollection)
    assert got.id == "coll1"
    assert len(got) == 1
    assert got[0].id == "sph1"


def test_native_pack_stores_field_brick_as_npy_not_json_list(tmp_path):
    grid = _tiny_grid()
    field = Field(
        name="dens",
        generator={"type": GEN_IMPORTED},
        grid_data=grid,
        obj_id="fld-imp",
    )
    path = tmp_path / "field.pmv"
    save_native(field, path)
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        assert "manifest.json" in names
        arrays = [n for n in names if n.startswith("arrays/") and n.endswith(".npy")]
        assert arrays
        doc = json.loads(zf.read("manifest.json"))
        assert doc["format"] == NATIVE_FORMAT
        brick = doc["objects"][0]["brick"]
        assert isinstance(brick["values"], dict)
        assert ARRAY_REF_KEY in brick["values"]
        assert "1.0" not in json.dumps(brick["values"])
        raw = zf.read(arrays[0])
        arr = np.load(BytesIO(raw), allow_pickle=False)
    restored = load_native(path)
    assert len(restored) == 1
    got = restored[0]
    assert isinstance(got, Field)
    assert got.id == "fld-imp"
    assert np.allclose(np.asarray(got.grid_data.values).reshape(-1), np.arange(8))


def test_native_pack_embeds_generated_brick_but_session_json_does_not():
    from pymolviz.fields.domain import BOUNDS_AROUND_SELECTION, Domain
    from pymolviz.fields.field import ensure_brick

    field = Field(
        name="dist",
        generator={
            "type": GEN_DISTANCE,
            "atoms": [{"xyz": [0.0, 0.0, 0.0], "elem": "C"}],
        },
        domain=Domain(bounds_mode=BOUNDS_AROUND_SELECTION, padding=1.0, spacing=1.0),
        obj_id="fld-dist",
    )
    ensure_brick(field)
    doc = session_document([field])
    assert "brick" not in doc["objects"][0]
    store_doc = []

    from pymolviz.serialization import ArrayStore, using_array_store

    store = ArrayStore()
    with using_array_store(store):
        packed = session_document([field])
    assert "brick" in packed["objects"][0]
    assert ARRAY_REF_KEY in packed["objects"][0]["brick"]["values"]
    assert store.arrays
    del store_doc


def test_displayable_write_dispatches_by_extension(tmp_path):
    sphere = Sphere((0.0, 0.0, 0.0), 1.0, bypass_colormap=True, obj_id="s")
    py_path = tmp_path / "s.py"
    pmv_path = tmp_path / "s.pmv"
    sphere.write(py_path)
    sphere.write(pmv_path)
    text = py_path.read_text()
    assert "from pymol" in text
    assert "import" in text
    restored = load(pmv_path)
    assert restored[0].id == "s"


def test_load_python_script_is_export_only(tmp_path):
    path = tmp_path / "x.py"
    path.write_text("print('hi')\n")
    with pytest.raises(SerializationError, match="export-only"):
        load(path)


def test_format_from_path_aliases():
    assert format_from_path("a.pmv") == FORMAT_NATIVE
    assert format_from_path("a.py") == FORMAT_SCRIPT
    assert format_from_path("a.bin", "pmv") == FORMAT_NATIVE
    assert format_from_path("a.bin") == FORMAT_SCRIPT


def test_open_native_filter_excludes_python_scripts():
    assert "*.pmv" in OPEN_NATIVE_FILTER
    assert "*.py" not in OPEN_NATIVE_FILTER


def test_session_export_items_omits_previews():
    from pymolviz.runtime import session as sess
    from pymolviz.runtime.session import PREVIEW_ID_PREFIX
    from pymolviz.wizards.session_io import session_export_items

    live = Sphere((0.0, 0.0, 0.0), 1.0, bypass_colormap=True, obj_id="live")
    preview = Sphere(
        (0.0, 0.0, 0.0), 1.0, bypass_colormap=True,
        obj_id="%stmp" % PREVIEW_ID_PREFIX,
    )
    sess._live[live.id] = live
    sess._live[preview.id] = preview
    assert [obj.id for obj in session_export_items()] == ["live"]


def test_import_session_path_interns(tmp_path, fake_cmd):
    from pymolviz.runtime.session import get
    from pymolviz.wizards.session_io import import_session_path

    sphere = Sphere(
        (0.0, 0.0, 0.0), 1.0, bypass_colormap=True, obj_id="imp1", name="imp1",
    )
    path = tmp_path / "one.pmv"
    save(sphere, path)
    interned = import_session_path(fake_cmd, path)
    assert len(interned) == 1
    assert interned[0].id == "imp1"
    assert get("imp1") is interned[0]


def test_intern_loaded_accepts_sequence(tmp_path, fake_cmd):
    sphere = Sphere(
        (1.0, 0.0, 0.0), 0.4, bypass_colormap=True, obj_id="seq1", name="seq1",
    )
    path = tmp_path / "seq.pmv"
    save(sphere, path)
    interned = intern_loaded(fake_cmd, load(path))
    assert interned[0].id == "seq1"


def test_prompt_import_session_cancelled(monkeypatch):
    monkeypatch.setattr(
        "pymolviz.wizards.pick.overlay_get_open_file_name",
        lambda *args, **kwargs: ("", ""),
    )
    from pymolviz.wizards.session_io import prompt_import_session

    assert prompt_import_session(None, None) is None


def test_prompt_export_session_empty_informs(monkeypatch):
    seen = []
    monkeypatch.setattr(
        "pymolviz.wizards.pick.overlay_information",
        lambda parent, title, text: seen.append((title, text)),
    )
    from pymolviz.wizards.session_io import (
        EMPTY_EXPORT_MSG,
        EXPORT_TITLE,
        prompt_export_session,
    )

    assert prompt_export_session(None) is False
    assert seen == [(EXPORT_TITLE, EMPTY_EXPORT_MSG)]


def test_prompt_export_session_writes_pack(tmp_path, monkeypatch):
    from pymolviz.runtime.session import add
    from pymolviz.wizards.session_io import DEFAULT_EXPORT_NAME, prompt_export_session

    sphere = Sphere(
        (2.0, 0.0, 0.0), 0.3, bypass_colormap=True, obj_id="exp1", name="exp1",
    )
    add(sphere)
    dest = tmp_path / ("%s.pmv" % DEFAULT_EXPORT_NAME)
    monkeypatch.setattr(
        "pymolviz.wizards.builders.export.overlay_get_save_file_name",
        lambda *args, **kwargs: (str(dest), "PyMolViz (*.pmv)"),
    )
    written = prompt_export_session(None)
    assert written == str(dest)
    restored = load(dest)
    assert restored[0].id == "exp1"
