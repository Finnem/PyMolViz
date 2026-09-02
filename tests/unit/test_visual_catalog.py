"""Object-list catalog rows for the wizard library (no Qt)."""

from __future__ import annotations

from pymolviz.meshes.CenteredBox import CenteredBox
from pymolviz.meshes.CGOCollection import CGOCollection
from pymolviz.meshes.Sphere import Sphere
from pymolviz.points import AtomPoint, FixedPoint
from pymolviz.wizards.catalog import (
    editor_kind,
    object_row,
    object_rows,
    point_count,
    type_label,
)


def _sphere_collection():
    sphere = Sphere(
        AtomPoint("prot", 10, chain="A", resi="1", name="CA", last_xyz=(1.0, 2.0, 3.0)),
        0.5,
        bypass_colormap=True,
        obj_id="sph1",
    )
    return CGOCollection([sphere], name="pmv_spheres", obj_id="coll_sph")


def test_object_row_sphere_collection():
    coll = _sphere_collection()
    row = object_row(coll)
    assert row["id"] == "coll_sph"
    assert row["name"] == "pmv_spheres"
    assert row["type"] == "Sphere"
    assert row["n_points"] == 1
    assert row["com"] == "1.00, 2.00, 3.00"
    assert row["editor"] == "Sphere"


def test_object_rows_skips_fields():
    coll = _sphere_collection()

    class Volume:
        def __init__(self):
            self.id = "vol1"
            self._name = "density"

    rows = object_rows([coll, Volume()])
    assert [row["id"] for row in rows] == ["coll_sph"]


def test_type_label_box_and_point_count():
    box = CenteredBox(
        FixedPoint((0.0, 0.0, 0.0)),
        (2.0, 2.0, 2.0),
        bypass_colormap=True,
    )
    coll = CGOCollection([box, box], name="pmv_boxes", obj_id="coll_box")
    # second box is the same instance; point_count uses len(children)
    assert type_label(coll) == "Box"
    assert point_count(coll) == 2
    assert editor_kind(coll) == "Box"
