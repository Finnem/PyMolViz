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
        self.disabled: set = set()

    def add_atom(self, atom: FakeAtom) -> None:
        self.atoms.append(atom)

    def select(self, name: str, sele_expr: str) -> None:
        if sele_expr.strip().lower() in ("none", ""):
            self.selections[name] = []
            return
        self.selections[name] = self._resolve_selection(sele_expr)

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
        if re.search(r"\s+or\s+", expr, flags=re.I):
            out = []
            seen = set()
            for part in re.split(r"\s+or\s+", expr, flags=re.I):
                for atom in self._resolve_selection(part.strip()):
                    key = (atom.model, atom.atom_id)
                    if key not in seen:
                        seen.add(key)
                        out.append(atom)
            return out
        if expr in ("(sele)", "sele"):
            return list(self.selections.get("sele", []))
        if expr in ("(selextended)",):
            return list(self.selections.get("sele", []))
        if expr in ("(pk1)",):
            pk = self.selections.get("pk1", [])
            return list(pk[:1])
        within = re.search(
            r"within\s+([0-9.eE+-]+)\s+of\s+\[\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*\]",
            expr,
            flags=re.I,
        )
        if within:
            radius = float(within.group(1))
            px, py, pz = (float(within.group(2)), float(within.group(3)), float(within.group(4)))
            base = (expr[: within.start()] + expr[within.end() :]).strip()
            base = re.sub(r"\band\s*$", "", base, flags=re.I).strip("() ").strip()
            if not base or base.lower() in ("all", "visible", "enabled", "visible and enabled"):
                atoms = list(self.atoms)
            else:
                atoms = self._resolve_selection(base)
            r2 = radius * radius
            return [
                atom
                for atom in atoms
                if (atom.x - px) ** 2 + (atom.y - py) ** 2 + (atom.z - pz) ** 2 <= r2
            ]
        gt = {axis: [] for axis in "xyz"}
        lt = {axis: [] for axis in "xyz"}
        for axis, value in re.findall(r"\b([xyz])\s*>\s*([0-9.eE+-]+)", expr, flags=re.I):
            gt[axis.lower()].append(float(value))
        for axis, value in re.findall(r"\b([xyz])\s*<\s*([0-9.eE+-]+)", expr, flags=re.I):
            lt[axis.lower()].append(float(value))
        if any(gt.values()) or any(lt.values()):
            xmin = max(gt["x"]) if gt["x"] else None
            xmax = min(lt["x"]) if lt["x"] else None
            ymin = max(gt["y"]) if gt["y"] else None
            ymax = min(lt["y"]) if lt["y"] else None
            zmin = max(gt["z"]) if gt["z"] else None
            zmax = min(lt["z"]) if lt["z"] else None
            out = []
            for atom in self.atoms:
                if xmin is not None and atom.x <= xmin:
                    continue
                if xmax is not None and atom.x >= xmax:
                    continue
                if ymin is not None and atom.y <= ymin:
                    continue
                if ymax is not None and atom.y >= ymax:
                    continue
                if zmin is not None and atom.z <= zmin:
                    continue
                if zmax is not None and atom.z >= zmax:
                    continue
                out.append(atom)
            return out
        if expr in self.selections:
            return list(self.selections[expr])
        if expr.startswith("(") and expr.endswith(")"):
            name = expr[1:-1]
            if name in self.selections:
                return list(self.selections[name])
        by_model = [atom for atom in self.atoms if atom.model == expr]
        if by_model:
            return by_model
        return self._match(expr)

    def _match(self, sele_expr: str) -> List[FakeAtom]:
        expr = str(sele_expr).strip()
        obj_match = re.search(r'object\s+"([^"]+)"', expr)
        id_match = re.search(r'\bid\s+(\d+)', expr)
        index_match = re.search(r'\bindex\s+(\d+)', expr)
        chain_match = re.search(r'chain\s+"([^"]+)"', expr)
        resi_match = re.search(r'\bresi\s+(\S+)', expr)
        name_match = re.search(r'name\s+"([^"]+)"', expr)

        out = []
        for atom in self.atoms:
            if obj_match and atom.model != obj_match.group(1):
                continue
            if id_match and atom.atom_id != int(id_match.group(1)):
                continue
            if index_match and atom.index != int(index_match.group(1)):
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

    def get_names(self, typ: str = "objects", enabled_only: int = 0, selection: str = "") -> List[str]:
        if typ == "objects":
            names = list(self.objects.keys())
        elif typ == "selections":
            names = list(self.selections.keys())
        else:
            names = []
        if enabled_only:
            names = [name for name in names if name not in self.disabled]
        return names

    def set_object_ttt(self, name: str, matrix) -> None:
        self.objects.setdefault(str(name), [])
        self.settings.setdefault(str(name), {})["_ttt"] = list(matrix)

    def get_object_ttt(self, name: str, state: int = 1):
        stored = self.settings.get(str(name), {}).get("_ttt")
        if stored:
            return list(stored)
        return [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]

    def get_object_matrix(self, name: str, state: int = 1, incl_ttt: int = 1, history: int = 1):
        """Homogenous 4x4, matching real PyMOL (translation in the last column)."""
        identity = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
        if not incl_ttt:
            return list(identity)
        ttt = self.get_object_ttt(name, state)
        if not ttt:
            return list(identity)
        r00, r01, r02, p0, r10, r11, r12, p1, r20, r21, r22, p2, t0, t1, t2, _w = [
            float(v) for v in ttt[:16]
        ]
        tx = r00 * t0 + r01 * t1 + r02 * t2 + p0
        ty = r10 * t0 + r11 * t1 + r12 * t2 + p1
        tz = r20 * t0 + r21 * t1 + r22 * t2 + p2
        return [r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz, 0.0, 0.0, 0.0, 1.0]

    def get_extent(self, sele_expr: str, state: int = 1):
        atoms = self._resolve_selection(sele_expr)
        if not atoms:
            return [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        xs = [atom.x for atom in atoms]
        ys = [atom.y for atom in atoms]
        zs = [atom.z for atom in atoms]
        return [
            [min(xs), min(ys), min(zs)],
            [max(xs), max(ys), max(zs)],
        ]

    def enable(self, name: str) -> None:
        self.disabled.discard(str(name))

    def disable(self, name: str) -> None:
        self.disabled.add(str(name))

    def get_viewport(self):
        return (640.0, 480.0)

    def get(self, key: str, name: str = ""):
        if name:
            return self.settings.get(str(name), {}).get(str(key), 0.0)
        return 0.0

    def get_coords(self, sele_expr: str, state: int = 1):
        atoms = self._resolve_selection(sele_expr)
        if not atoms:
            return None
        return [[a.x, a.y, a.z] for a in atoms]

    def get_coordset(self, name: str, state: int = 1, copy: int = 1):
        atoms = [atom for atom in self.atoms if atom.model == str(name)]
        if not atoms:
            return None
        return [[atom.x, atom.y, atom.z] for atom in atoms]

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

    def pseudoatom(self, name: str, pos=None, **_kwargs) -> None:
        pos = pos or [0.0, 0.0, 0.0]
        obj = str(name)
        self.objects[obj] = [float(pos[0]), float(pos[1]), float(pos[2])]
        self.add_atom(FakeAtom(obj, 1, float(pos[0]), float(pos[1]), float(pos[2]), name="PSD"))

    def zoom(self, sele_expr: str, animate: int = -1, buffer: float = 0) -> None:
        atoms = self._resolve_selection(sele_expr)
        if not atoms:
            return
        xs = [a.x for a in atoms]
        ys = [a.y for a in atoms]
        zs = [a.z for a in atoms]
        self._last_zoom = {
            "sele": sele_expr,
            "center": (
                (min(xs) + max(xs)) / 2.0,
                (min(ys) + max(ys)) / 2.0,
                (min(zs) + max(zs)) / 2.0,
            ),
            "animate": animate,
            "buffer": buffer,
        }
