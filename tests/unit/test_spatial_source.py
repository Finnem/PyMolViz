"""Shared Point / Arrow source vocabulary."""

from pymolviz.wizards.builders.spatial_source import (
    KIND_ATOM,
    KIND_CAMERA,
    KIND_FIXED,
    KIND_SELECTION,
    STATUS_ATTACHED,
    arrow_source_summary,
    point_kind_label,
    point_primary_label,
    point_source_summary,
)


class _Pt:
    def __init__(self, name="", source="", atom_ref=None, can=False, attached=False):
        self.name = name
        self.source = source
        self.atom_ref = atom_ref
        self._can = can
        self._attached = attached

    def can_anchor(self):
        return self._can

    def wants_anchor(self):
        return self._attached


def test_point_kind_labels():
    atom = _Pt(atom_ref=object(), can=True, attached=True)
    assert point_kind_label(atom) == KIND_ATOM
    assert point_source_summary(atom) == KIND_ATOM
    cam = _Pt(name="cam_4", source="manual")
    assert point_kind_label(cam) == KIND_CAMERA
    sel = _Pt(source="selection")
    assert point_kind_label(sel) == KIND_SELECTION
    fixed = _Pt(name="custom", source="manual")
    assert point_kind_label(fixed) == KIND_FIXED


def test_arrow_source_summary_combinations():
    atom = _Pt(atom_ref=object())
    cam = _Pt(name="cam_1", source="manual")
    fixed = _Pt(name="p1", source="manual")
    assert arrow_source_summary(atom, atom) == "Atom → Atom"
    assert arrow_source_summary(cam, atom) == "Camera center → Atom"
    assert arrow_source_summary(fixed, fixed) == "Fixed → Fixed"
    mixed = arrow_source_summary(atom, cam)
    assert mixed == "Atom → Camera center"


def test_point_primary_label_prefers_atom_anchor():
    from pymolviz.wizards.builders.points import AtomRef

    pt = _Pt(name="ignored", atom_ref=AtomRef("prot", 1, "A", "61", "OG"))
    assert point_primary_label(pt) == "A/61/OG"
    assert point_primary_label(_Pt(name="cam_4")) == "cam_4"
    assert STATUS_ATTACHED == "Anchored"
