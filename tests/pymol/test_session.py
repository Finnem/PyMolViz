"""Optional PyMOL integration tests (run with ``pytest -m pymol``)."""

from __future__ import annotations

import pytest

pymol = pytest.importorskip("pymol", reason="PyMOL not installed")
from pymol import cmd  # noqa: E402

pytestmark = pytest.mark.skipif(
    not hasattr(cmd, "load_cgo"),
    reason="Real PyMOL cmd API not available",
)


@pytest.mark.pymol
def test_pymol_load_cgo_roundtrip():
    from pymolviz.util.cgo import resolve_cgo_tokens

    tokens = resolve_cgo_tokens(["BEGIN", "LINES", "VERTEX", 0.0, 0.0, 0.0, "VERTEX", 1.0, 0.0, 0.0, "END"])
    name = "_pmv_test_line"
    try:
        cmd.delete(name)
    except Exception:
        pass
    cmd.load_cgo(tokens, name)
    assert name in cmd.get_names("objects")
    cmd.delete(name)


@pytest.mark.pymol
def test_pymol_session_namespace():
    import pymolviz.runtime.session as pmv_session
    from pymolviz.meshes.CGOCollection import CGOCollection
    from pymolviz.meshes.Sphere import Sphere

    pmv_session.clear()
    sphere = Sphere((0.0, 0.0, 0.0), 1.0, bypass_colormap=True, obj_id="live_s1")
    coll = CGOCollection([sphere], name="_pmv_live_test", obj_id="live_c1")
    pmv_session.add(coll)
    blob = pmv_session.persist()
    assert blob.get("schema") == 1
    assert len(blob.get("objects", [])) == 1
    pmv_session.clear()
