"""Named 3D points for CGO builders (camera center, selection, manual)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from ...util.sanitize import sanitize_pymol_string
from ...util.view import screen_center
from .colors import DEFAULT_SPHERE_COLOR, colors_for_new_points

RGB = Tuple[float, float, float]

AA_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


@dataclass
class VisualPoint:
    name: str
    source: str
    x: float
    y: float
    z: float
    color: RGB = field(default_factory=lambda: DEFAULT_SPHERE_COLOR)
    alpha: float = 1.0

    def xyz(self) -> Tuple[float, float, float]:
        return (float(self.x), float(self.y), float(self.z))

    def rgba(self) -> Tuple[float, float, float, float]:
        return (
            float(self.color[0]),
            float(self.color[1]),
            float(self.color[2]),
            float(self.alpha),
        )

    def with_xyz(self, xyz: Sequence[float]) -> "VisualPoint":
        return VisualPoint(self.name, self.source, float(xyz[0]), float(xyz[1]), float(xyz[2]), self.color, self.alpha)

    def with_name(self, name: str) -> "VisualPoint":
        return VisualPoint(str(name), self.source, self.x, self.y, self.z, self.color, self.alpha)

    def with_source(self, source: str) -> "VisualPoint":
        return VisualPoint(self.name, str(source), self.x, self.y, self.z, self.color, self.alpha)

    def with_color(self, color: Sequence[float]) -> "VisualPoint":
        alpha = float(color[3]) if len(color) >= 4 else self.alpha
        return VisualPoint(
            self.name,
            self.source,
            self.x,
            self.y,
            self.z,
            (float(color[0]), float(color[1]), float(color[2])),
            alpha,
        )


def assign_distinct_colors(points: List[VisualPoint]) -> None:
    """Reassign distinct palette colors in place."""
    for i, pt in enumerate(points):
        points[i] = pt.with_color(colors_for_new_points(len(points))[i])


def apply_global_color(points: List[VisualPoint], color: RGB) -> None:
    for i, pt in enumerate(points):
        points[i] = pt.with_color(color)


def abbreviate_object_name(name: str, max_len: int = 11) -> str:
    name = str(name or "obj")
    if len(name) <= max_len:
        return name
    return "%s...%s" % (name[:5], name[-3:])


def resn_one_letter(resn: str) -> Optional[str]:
    if not resn:
        return None
    resn = str(resn).upper()
    if len(resn) == 1:
        return resn
    one = AA_ONE.get(resn)
    if one:
        return one
    if len(resn) == 3:
        return resn
    return None


def atom_point_name(
    model: str,
    elem: str = "",
    resn: str = "",
    resi: str = "",
    chain: str = "",
    index: int = 0,
) -> str:
    parts = [abbreviate_object_name(model)]
    if chain and str(chain).strip():
        parts.append(str(chain).strip())
    if elem and str(elem).strip():
        parts.append(str(elem).strip())
    one = resn_one_letter(resn)
    if one:
        parts.append(one)
    if resi not in (None, ""):
        parts.append(str(resi))
    if index:
        parts.append(str(int(index)))
    return sanitize_pymol_string("_".join(parts))


def manual_fallback_name(prefix: str, existing: Sequence[VisualPoint]) -> str:
    used = {p.name for p in existing}
    n = 1
    while True:
        candidate = sanitize_pymol_string("%s_%d" % (prefix, n))
        if candidate not in used:
            return candidate
        n += 1


def _dist2(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((float(a[i]) - float(b[i])) ** 2 for i in range(3))


def _selection_expr(name: str) -> str:
    """Wrap a selection name for use in PyMOL selection expressions."""
    name = str(name).strip()
    if name.startswith("(") and name.endswith(")"):
        return name
    return "(%s)" % name


def _current_state(cmd_) -> int:
    try:
        return int(cmd_.get_state())
    except Exception:
        return 1


def _count_selection_atoms(cmd_, sele_expr: str, state: int = 0) -> int:
    try:
        return int(cmd_.count_atoms(sele_expr))
    except Exception:
        pass
    if state:
        try:
            return int(cmd_.count_atoms(sele_expr, state))
        except TypeError:
            try:
                return int(cmd_.count_atoms(sele_expr, state=state))
            except Exception:
                pass
        except Exception:
            pass
    return 0


def _iterate_atoms(cmd_, sele_expr: str, atoms: list, state: int = 0) -> bool:
    """Fill atoms with [model, chain, elem, resn, resi, id, x, y, z] per atom."""
    expressions = (
        "atoms.append([model, chain, elem, resn, resi, ID, x, y, z])",
        "atoms.append([model, chain, elem, resn, resi, index, x, y, z])",
    )
    for expr in expressions:
        atoms.clear()
        try:
            if state:
                cmd_.iterate_state(state, sele_expr, expr, space={"atoms": atoms})
            else:
                cmd_.iterate(sele_expr, expr, space={"atoms": atoms})
            if atoms:
                return True
        except Exception:
            continue
    atoms.clear()
    return False


def _active_selection(cmd_) -> Optional[str]:
    """Return a selection expression with at least one atom, or None."""
    state = _current_state(cmd_)
    candidates = ["(sele)", "(selextended)", "(pk1)"]
    try:
        for name in cmd_.get_names("selections"):
            expr = _selection_expr(name)
            if expr not in candidates:
                candidates.append(expr)
    except Exception:
        pass
    for expr in candidates:
        if _count_selection_atoms(cmd_, expr, state) > 0:
            return expr
    return None


def nearest_atom_within(cmd_, pos: Sequence[float], radius: float = 1.0, sele: str = "all"):
    """Return atom identity and coordinates if any atom in sele is within radius of pos."""
    best = None
    best_d2 = radius * radius
    state = _current_state(cmd_)
    atoms = []
    if not _iterate_atoms(cmd_, sele, atoms, state):
        return None
    for model, chain, elem, resn, resi, atom_id, x, y, z in atoms:
        d2 = _dist2(pos, (x, y, z))
        if d2 <= best_d2:
            best_d2 = d2
            best = {
                "model": model,
                "chain": chain or "",
                "elem": elem or "",
                "resn": resn or "",
                "resi": resi or "",
                "index": int(atom_id),
                "x": float(x),
                "y": float(y),
                "z": float(z),
            }
    return best


def camera_center_point(cmd_, snap_to_atom: bool = False, existing: Sequence[VisualPoint] = ()):
    view = tuple(cmd_.get_view())
    pos = screen_center(view)
    source = "manual"
    name = manual_fallback_name("cam", existing)
    if snap_to_atom:
        atom = nearest_atom_within(cmd_, pos, radius=1.0)
        if atom is not None:
            pos = (atom["x"], atom["y"], atom["z"])
            name = atom_point_name(
                atom["model"],
                elem=atom["elem"],
                resn=atom["resn"],
                resi=atom["resi"],
                chain=atom["chain"],
                index=atom["index"],
            )
    return VisualPoint(name, source, pos[0], pos[1], pos[2])


def selection_points(cmd_, existing: Sequence[VisualPoint] = ()) -> List[VisualPoint]:
    sele = _active_selection(cmd_)
    if sele is None:
        return []
    state = _current_state(cmd_)
    atoms = []
    if not _iterate_atoms(cmd_, sele, atoms, state):
        return []
    used = {p.name for p in existing}
    out = []
    for model, chain, elem, resn, resi, atom_id, x, y, z in atoms:
        name = atom_point_name(
            model,
            elem=elem,
            resn=resn,
            resi=resi,
            chain=chain,
            index=int(atom_id),
        )
        base = name
        n = 1
        while name in used:
            name = "%s_%d" % (base, n)
            n += 1
        used.add(name)
        out.append(VisualPoint(name, "selection", float(x), float(y), float(z)))
    return out


def export_points_to_selection(cmd_, points: Sequence[VisualPoint], tmp_object: str = "_pmv_points_tmp"):
    """Create pseudoatoms and a PyMOL selection covering them."""
    purge = []
    try:
        cmd_.delete(tmp_object)
    except Exception:
        pass
    if not points:
        return None
    for i, pt in enumerate(points):
        obj = "%s_%d" % (tmp_object, i)
        purge.append(obj)
        try:
            cmd_.delete(obj)
        except Exception:
            pass
        cmd_.pseudoatom(obj, pos=[pt.x, pt.y, pt.z], label=pt.name)
    sele_name = "_pmv_points_sel"
    names = " or ".join('object "%s"' % n for n in purge)
    try:
        cmd_.select(sele_name, names)
    except Exception:
        cmd_.select(sele_name, "none")
    return sele_name
