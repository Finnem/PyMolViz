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

    vdw: Optional[float] = None

    def __post_init__(self):
        if not self.index:
            self.index = self.atom_id
        if not self.name and self.elem:
            self.name = self.elem
        if self.vdw is None:
            table = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80}
            self.vdw = float(table.get(str(self.elem or "C").upper()[:1], 1.70))


class FakeCmd:
    """Subset of ``cmd`` used by PointSource resolution and PyMOLRuntime."""

    def __init__(self) -> None:
        self.atoms: List[FakeAtom] = []
        self.objects: Dict[str, list] = {}
        self.object_types: Dict[str, str] = {}
        self.volume_fields: Dict[str, object] = {}
        self.extents: Dict[str, list] = {}
        self.symmetries: Dict[str, list] = {}
        self.selections: Dict[str, List[FakeAtom]] = {}
        self.state: int = 1
        self.settings: Dict[str, Dict[str, float]] = {}
        self._view = [1.0] * 18
        self.disabled: set = set()
        self._drag_selection = ""
        self._drag_mode = -1
        self._drag_wizard = 1
        self._edit_mode = 0
        self._hidden: set = set()
        self._shown: set = set()
        self._origin = [0.0, 0.0, 0.0]

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
        if "vdw" in expr:
            atoms_out.append(float(atom.vdw) if atom.vdw is not None else 1.7)
            return
        if "model" in expr and "resn" in expr:
            row = [
                atom.model,
                atom.chain,
                atom.elem,
                atom.resn,
                atom.resi,
                atom.atom_id,
            ]
            if "name" in expr:
                row.extend([atom.name, atom.x, atom.y, atom.z])
            else:
                row.extend([atom.x, atom.y, atom.z])
            atoms_out.append(row)
            return
        atoms_out.append([atom.x, atom.y, atom.z])

    def _resolve_selection(self, sele_expr: str) -> List[FakeAtom]:
        expr = str(sele_expr).strip()
        first_match = re.match(r"^first\s+(.+)$", expr, flags=re.I)
        if first_match:
            inner = first_match.group(1).strip()
            atoms = self._resolve_selection(inner)
            return atoms[:1]
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
        lowered = expr.lower().strip("() ")
        if lowered in ("all", "visible", "enabled", "visible and enabled"):
            return list(self.atoms)
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
        self.object_types[str(name)] = "object:cgo"

    def load_callback(self, obj, name: str, state: int = 1, finish: int = 1, discrete: int = 0, **_kwargs) -> None:
        self.objects[str(name)] = obj

    def refresh(self) -> None:
        pass

    def load_object(self, loadable, cgo, name, zoom=0) -> None:
        self.load_cgo(cgo, name)

    def delete(self, name: str) -> None:
        obj = str(name)
        self.objects.pop(obj, None)
        self.settings.pop(obj, None)
        self.object_types.pop(obj, None)
        self.volume_fields.pop(obj, None)
        self.extents.pop(obj, None)
        self.disabled.discard(obj)
        self.atoms = [atom for atom in self.atoms if atom.model != obj]
        if self._drag_selection in (obj, "", self._drag_selection) and (
            self._drag_selection == obj
        ):
            self._drag_selection = ""

    def load(self, filename, object="", state=1, format="", finish=1, discrete=0, quiet=1, **_kwargs) -> None:
        from pathlib import Path

        name = str(object) if object else Path(str(filename)).stem
        self.objects[name] = []
        ext = Path(str(filename)).suffix.lower()
        if ext in {".ccp4", ".mrc", ".map", ".dx", ".xplor", ".grd", ".mtz"}:
            self.object_types[name] = "object:map"

    def load_brick(self, brick, name: str) -> None:
        self.objects[str(name)] = brick
        self.object_types[str(name)] = "object:map"
        origin = getattr(brick, "origin", None)
        step = getattr(brick, "step_sizes", None)
        if step is None:
            step = getattr(brick, "spacing", None)
        counts = getattr(brick, "step_counts", None)
        if origin is None or step is None or counts is None:
            return
        try:
            lo = [float(origin[0]), float(origin[1]), float(origin[2])]
            st = [float(step[0]), float(step[1]), float(step[2])]
            n = [float(counts[0]), float(counts[1]), float(counts[2])]
            self.extents[str(name)] = [
                lo,
                [lo[0] + st[0] * n[0], lo[1] + st[1] * n[1], lo[2] + st[2] * n[2]],
            ]
        except (TypeError, ValueError, IndexError):
            pass

    def get_symmetry(self, object="", state: int = 1):
        stored = self.symmetries.get(str(object))
        if stored:
            return list(stored)
        return [1.0, 1.0, 1.0, 90.0, 90.0, 90.0, "P1"]

    def set_symmetry(self, selection, a, b, c, alpha=90.0, beta=90.0, gamma=90.0, spacegroup="P1"):
        self.symmetries[str(selection)] = [
            float(a), float(b), float(c), float(alpha), float(beta), float(gamma), str(spacegroup or "P1"),
        ]

    def get_object_state(self, name: str) -> int:
        return int(self.state)

    def volume_ramp_new(self, name: str, values) -> None:
        self.objects[str(name)] = list(values or [])
        self.object_types[str(name)] = "object:ramp"

    def volume_color(self, name, ramp) -> None:
        info = self.objects.get(str(name))
        if not isinstance(info, dict):
            raise KeyError(name)
        info["ramp"] = ramp if isinstance(ramp, str) else list(ramp)

    def ramp_new(self, name, map_name, range=None, color=None, state=1, **_kwargs) -> None:
        self.objects[str(name)] = {
            "map": str(map_name),
            "range": list(range or []),
            "color": color,
            "state": int(state or 1),
        }
        self.object_types[str(name)] = "object:ramp"

    def volume(self, name, map_name, ramp="", selection="", carve=None, state=1, **_kwargs) -> None:
        self.objects[str(name)] = {"map": str(map_name), "ramp": str(ramp)}
        self.object_types[str(name)] = "object:volume"

    def isosurface(self, name, map_name, level=1.0, selection="", carve=None, side=1, **_kwargs) -> None:
        self.objects[str(name)] = {"map": str(map_name), "level": float(level)}
        self.object_types[str(name)] = "object:isosurface"

    def isomesh(self, name, map_name, level=1.0, selection="", carve=None, **_kwargs) -> None:
        self.objects[str(name)] = {"map": str(map_name), "level": float(level)}
        self.object_types[str(name)] = "object:mesh"

    def set_color(self, name: str, rgb) -> None:
        self.settings.setdefault(str(name), {})["rgb"] = list(rgb)

    def color(self, color, selection="") -> None:
        self.settings.setdefault(str(selection), {})["color"] = str(color)

    def set_name(self, old: str, new: str) -> None:
        old, new = str(old), str(new)
        if old == new:
            return
        if old not in self.objects:
            raise KeyError(old)
        self.objects[new] = self.objects.pop(old)
        if old in self.settings:
            self.settings[new] = self.settings.pop(old)
        if old in self.disabled:
            self.disabled.discard(old)
            self.disabled.add(new)

    def get_unused_name(self, prefix: str = "tmp", alwaysnumber: int = 0) -> str:
        prefix = str(prefix)
        names = set(self.objects) | set(self.selections)
        if not alwaysnumber and prefix not in names:
            return prefix
        index = 1
        while True:
            candidate = "%s_%d" % (prefix, index)
            if candidate not in names:
                return candidate
            index += 1

    def get_names(self, type="objects", enabled_only=0, selection=""):
        """Match PyMOL ``cmd.get_names(type, enabled_only, selection)``.

        ``count_atoms("sele")`` is independent of this: disabling a selection
        removes it from ``type="selections", enabled_only=1`` (and from
        ``type="enabled"``) but does not clear its atoms.
        """
        typ = str(type)
        if typ == "objects":
            names = list(self.objects.keys())
        elif typ == "selections":
            names = list(self.selections.keys())
        elif typ == "all":
            names = list(self.objects.keys()) + list(self.selections.keys())
        elif typ == "enabled":
            names = [
                name
                for name in list(self.objects.keys()) + list(self.selections.keys())
                if name not in self.disabled
            ]
            return names
        else:
            names = []
        if enabled_only:
            names = [name for name in names if name not in self.disabled]
        return names

    def get_type(self, name: str) -> str:
        return str(self.object_types.get(str(name), "object:molecule"))

    def get_names_of_type(self, kind: str) -> List[str]:
        wanted = str(kind)
        return [name for name, typ in self.object_types.items() if typ == wanted]

    def get_volume_field(self, name: str):
        return self.volume_fields.get(str(name))

    def get_extent(self, sele_expr: str, state: int = 1):
        stored = self.extents.get(str(sele_expr))
        if stored:
            return stored
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

    def enable(self, name: str) -> None:
        self.disabled.discard(str(name))

    def disable(self, name: str) -> None:
        self.disabled.add(str(name))

    def get_viewport(self):
        return (640.0, 480.0)

    def get(self, key: str, name: str = ""):
        return self.settings.get(str(name), {}).get(str(key), 0.0)

    def set(self, key: str, value, name: str = "", quiet=1) -> None:
        self.settings.setdefault(str(name), {})[str(key)] = value

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

    def set(self, key: str, value, name: str = "", quiet=1) -> None:
        self.settings.setdefault(str(name), {})[str(key)] = value

    def get_view(self) -> list:
        return list(self._view)

    def set_view(self, view, animate=0) -> None:
        self._view = list(view)

    def hide(self, representation: str = "everything", selection: str = "all") -> None:
        self._hidden.add((str(representation), str(selection)))

    def show(self, representation: str = "everything", selection: str = "all") -> None:
        self._shown.add((str(representation), str(selection)))

    def drag(self, selection=None, wizard=1, edit=1, quiet=1, mode=-1) -> None:
        if selection is None or selection == "":
            self._drag_selection = ""
            self._drag_mode = -1
            return
        self._drag_selection = str(selection)
        self._drag_mode = int(mode)
        self._drag_wizard = int(wizard)
        if int(edit):
            self._edit_mode = 1
            self.set("button_mode", 1)

    def get_drag_object_name(self) -> str:
        return self._drag_selection

    def get_editor_scheme(self) -> int:
        return 3 if self._drag_selection else 0

    def get_object_list(self, sele_expr: str):
        atoms = self._resolve_selection(sele_expr)
        names = []
        seen = set()
        for atom in atoms:
            if atom.model not in seen:
                seen.add(atom.model)
                names.append(atom.model)
        if not names and str(sele_expr) in self.objects:
            names.append(str(sele_expr))
        return names

    def edit_mode(self, value=1) -> None:
        self._edit_mode = int(value)

    def mouse(self) -> None:
        pass

    def origin(self, selection="(all)", object=None, position=None, state=0, **_kwargs) -> None:
        if position is not None:
            self._origin = [float(position[0]), float(position[1]), float(position[2])]
            return
        atoms = self._resolve_selection(selection)
        if not atoms:
            return
        n = float(len(atoms))
        self._origin = [
            sum(atom.x for atom in atoms) / n,
            sum(atom.y for atom in atoms) / n,
            sum(atom.z for atom in atoms) / n,
        ]

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

    def pseudoatom(self, name: str, pos=None, **kwargs) -> None:
        pos = pos or [0.0, 0.0, 0.0]
        obj = str(name)
        self.objects[obj] = [float(pos[0]), float(pos[1]), float(pos[2])]
        label = kwargs.get("label")
        if label:
            self.settings.setdefault(obj, {})["label"] = str(label)
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
