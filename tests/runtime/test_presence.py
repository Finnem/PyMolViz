"""Session objects stay in lockstep with the PyMOL object list."""

from __future__ import annotations

import numpy as np

from pymolviz.meshes.CGOCollection import CGOCollection
from pymolviz.meshes.Sphere import Sphere
from pymolviz.runtime import session as pmv_session
from pymolviz.runtime.presence import (
    after_pymol_delete,
    install_delete_hook,
    object_pymol_name,
    pause_presence_sync,
    sync_session_with_pymol,
    tracks_pymol_object,
)
from pymolviz.volumetric.GridData import GridData
from pymolviz.wizards.builders.field_visual import make_field_visual, persist_field_visual
from pymolviz.wizards.catalog import field_library_rows
from tests.fakes.cmd import FakeCmd


def _tiny_grid(name="density"):
    return GridData(
        np.arange(8, dtype=float),
        step_sizes=(1.0, 1.0, 1.0),
        step_counts=(1, 1, 1),
        origin=(0.0, 0.0, 0.0),
        name=name,
    )


def test_volume_persist_creates_named_pymol_volume(runtime, fake_cmd):
    grid = _tiny_grid()
    grid.is_loaded = True
    visual = make_field_visual("Volume", grid, "pmv_volume", obj_id="vol1")
    persist_field_visual(fake_cmd, visual)
    assert "pmv_volume" in fake_cmd.objects
    assert fake_cmd.object_types.get("pmv_volume") == "object:volume"
    assert pmv_session.get("vol1") is visual
    from pymolviz.runtime.runtime import get_runtime

    binding = get_runtime(fake_cmd).bindings.get("vol1")
    assert binding is not None
    assert binding.pymol_name == "pmv_volume"
    assert tracks_pymol_object(visual) is True
    assert object_pymol_name(visual, get_runtime(fake_cmd)) == "pmv_volume"


def test_deleting_volume_in_pymol_drops_menu_row(runtime, fake_cmd):
    grid = _tiny_grid()
    grid.is_loaded = True
    visual = make_field_visual("Volume", grid, "pmv_volume", obj_id="vol2")
    persist_field_visual(fake_cmd, visual)
    fake_cmd.delete("pmv_volume")
    dropped = sync_session_with_pymol(fake_cmd)
    assert dropped == [visual]
    assert pmv_session.get("vol2") is None
    rows = field_library_rows(pmv_session.all_objects(), cmd=fake_cmd)
    assert not any(row.get("id") == "vol2" for row in rows)


def test_delete_hook_prunes_session(runtime, fake_cmd):
    grid = _tiny_grid()
    grid.is_loaded = True
    visual = make_field_visual("IsoSurface", grid, "iso", level=0.5, obj_id="iso-hook")
    persist_field_visual(fake_cmd, visual)
    install_delete_hook(fake_cmd)
    fake_cmd.delete("iso")
    assert pmv_session.get("iso-hook") is None


def test_paused_delete_does_not_prune(runtime, fake_cmd):
    grid = _tiny_grid()
    grid.is_loaded = True
    visual = make_field_visual("IsoMesh", grid, "mesh", level=0.5, obj_id="mesh1")
    persist_field_visual(fake_cmd, visual)
    with pause_presence_sync():
        fake_cmd.delete("mesh")
        after_pymol_delete(fake_cmd, "mesh")
        assert pmv_session.get("mesh1") is visual
    dropped = sync_session_with_pymol(fake_cmd)
    assert [obj.id for obj in dropped] == ["mesh1"]


def test_generated_field_recipe_is_kept_without_pymol_object(runtime, fake_cmd):
    from pymolviz.fields import Domain
    from pymolviz.fields.field import Field
    from pymolviz.fields.identity import GEN_DISTANCE

    field = Field(
        name="dist",
        generator={"type": GEN_DISTANCE, "atoms": [{"xyz": [0.0, 0.0, 0.0], "elem": "C"}]},
        domain=Domain(padding=1.0, spacing=1.0),
        obj_id="fld-keep",
    )
    pmv_session.add(field)
    assert tracks_pymol_object(field) is False
    dropped = sync_session_with_pymol(fake_cmd)
    assert dropped == []
    assert pmv_session.get("fld-keep") is field


def test_deleted_cgo_is_dropped_from_session(runtime, fake_cmd):
    sphere = Sphere((0.0, 0.0, 0.0), 0.5, bypass_colormap=True, obj_id="sph")
    coll = CGOCollection([sphere], name="pmv_spheres", obj_id="coll-drop")
    pmv_session.add(coll)
    runtime.materialize(coll)
    fake_cmd.delete("pmv_spheres")
    dropped = sync_session_with_pymol(fake_cmd)
    assert [obj.id for obj in dropped] == ["coll-drop"]
    assert pmv_session.get("coll-drop") is None
