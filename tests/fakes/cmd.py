"""Minimal in-memory PyMOL ``cmd`` for runtime and PointSource tests."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class FakeAtom:
    model: str
    atom_id: int
    x: float
    y: float
    z: float
    chain: str = ""
    resn: str = ""
    resi: str = ""
    name: str = ""
    elem: str = "C"
    index: int = 0

    def __post_init__(self):
        if not self.index:
            self.index = self.atom_id
        if not self.name and self.elem:
            self.name = self.elem


class FakeCmd:
    """Subset of ``cmd`` used by PointSource resolution and PyMOLRuntime."""

    def __init__(self) -> None:
        self.atoms: List[FakeAtom] = []
        self.objects: Dict[str, list] = {}
        self.selections: Dict[str, List[FakeAtom]] = {}
        self.state: int = 1
        self.settings: Dict[str, Dict[str, float]] = {}
        self._view = [1.0] * 18

    def add_atom(self, atom: FakeAtom) -> None:
        self.atoms.append(atom)

    def select(self, name: str, sele_expr: str) -> None:
        if sele_expr.strip().lower() in ("none", ""):
            self.selections[name] = []
            return
        self.selections[name] = self._match(sele_expr)

    def count_atoms(self, sele_expr: str, state: int = 0) -> int:
        return len(self._resolve_selection(sele_expr))

    def iterate(self, sele_expr: str, expr: str, space: Optional[dict] = None) -> None:
        space = space if space is not None else {}
        atoms_out = space.setdefault("atoms", [])
        for atom in self._resolve_selection(sele_expr):
            self._append_atom(expr, atom, atoms_out)

    def iterate_state(self, state: int, sele_expr: str, expr: str, space: Optional[dict] = None) -> None:
        self.iterate(sele_expr, expr, space)

    def _append_atom(self, expr: str, atom: FakeAtom, atoms_out: list) -> None:
        if "model" in expr and "resn" in expr:
            atoms_out.append([
                atom.model,
                atom.chain,
                atom.elem,
                atom.resn,
                atom.resi,
                atom.atom_id,
                atom.x,
                atom.y,
                atom.z,
            ])
        else:
            atoms_out.append([atom.x, atom.y, atom.z])

    def _resolve_selection(self, sele_expr: str) -> List[FakeAtom]:
        expr = str(sele_expr).strip()
        if expr in ("(sele)", "sele"):
            return list(self.selections.get("sele", []))
        if expr in ("(selextended)",):
            return list(self.selections.get("sele", []))
        if expr in ("(pk1)",):
            pk = self.selections.get("pk1", [])
            return list(pk[:1])
        if expr.startswith("(") and expr.endswith(")"):
            name = expr[1:-1]
            if name in self.selections:
                return list(self.selections[name])
        return self._match(expr)

    def _match(self, sele_expr: str) -> List[FakeAtom]:
        expr = str(sele_expr).strip()
        obj_match = re.search(r'object\s+"([^"]+)"', expr)
        id_match = re.search(r'\bid\s+(\d+)', expr)
        chain_match = re.search(r'chain\s+"([^"]+)"', expr)
        resi_match = re.search(r'\bresi\s+(\S+)', expr)
        name_match = re.search(r'name\s+"([^"]+)"', expr)

        out = []
        for atom in self.atoms:
            if obj_match and atom.model != obj_match.group(1):
                continue
            if id_match and atom.atom_id != int(id_match.group(1)):
                continue
            if chain_match and atom.chain != chain_match.group(1):
                continue
            if resi_match and str(atom.resi) != resi_match.group(1):
                continue
            if name_match and atom.name != name_match.group(1):
                continue
            out.append(atom)
        return out

    def load_cgo(self, cgo: Sequence, name: str, state: int = 1, zoom: int = 0) -> None:
        self.objects[str(name)] = list(cgo)

    def load_object(self, loadable, cgo, name, zoom=0) -> None:
        self.load_cgo(cgo, name)

    def delete(self, name: str) -> None:
        self.objects.pop(str(name), None)

    def get_names(self, typ: str) -> List[str]:
        if typ == "objects":
            return list(self.objects.keys())
        if typ == "selections":
            return list(self.selections.keys())
        return []

    def get_state(self) -> int:
        return self.state

    def set(self, key: str, value, name: str = "") -> None:
        if name:
            self.settings.setdefault(str(name), {})[str(key)] = float(value)

    def get_view(self) -> list:
        return list(self._view)

    def set_view(self, view, animate=0) -> None:
        self._view = list(view)

    def unpick(self) -> None:
        pass

    def group(self, name, members, action="add") -> None:
        pass

    def refresh_wizard(self) -> None:
        pass

    def translate(self, vector, sele="all") -> None:
        dx, dy, dz = vector[:3]
        for atom in self._resolve_selection(sele):
            atom.x += float(dx)
            atom.y += float(dy)
            atom.z += float(dz)
