"""Named 3D points for CGO builders (camera center, selection, manual)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from ...points import AtomPoint, FixedPoint, PointSource, PseudoAtomPoint
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
class AtomRef:
    """Atom identity preserved so a point can be re-anchored after unhooking."""
    model: str
    atom_id: int
    chain: str = ""
    resi: str = ""
    name: str = ""
    elem: str = ""


def atom_ref_from_point_source(source: Optional[PointSource]) -> Optional[AtomRef]:
    if isinstance(source, AtomPoint):
        return AtomRef(
            source.object,
            int(source.atom_id),
            source.chain or "",
            source.resi or "",
            source.name or "",
            source.elem or "",
        )
    return None


def atom_anchor_label(ref: Optional[AtomRef]) -> str:
    """Compact residue label for UI, e.g. ``A/42/CA``."""
    if ref is None:
        return ""
    parts = [
        str(ref.chain or "").strip(),
        str(ref.resi or "").strip(),
        str(ref.name or "").strip(),
    ]
    return "/".join(part for part in parts if part)


def _atom_ref(model, atom_id, chain, resi, atom_name, elem="") -> AtomRef:
    return AtomRef(
        str(model),
        int(atom_id),
        str(chain or ""),
        str(resi or ""),
        str(atom_name or ""),
        str(elem or ""),
    )


@dataclass
class VisualPoint:
    name: str
    source: str
    x: float
    y: float
    z: float
    color: RGB = field(default_factory=lambda: DEFAULT_SPHERE_COLOR)
    alpha: float = 1.0
    point_source: Optional[PointSource] = None
    atom_ref: Optional[AtomRef] = None
    anchor_intent: Optional[bool] = None
    radius: Optional[float] = None

    def __post_init__(self):
        if self.point_source is None:
            self.point_source = FixedPoint((self.x, self.y, self.z))
        if self.atom_ref is None:
            self.atom_ref = atom_ref_from_point_source(self.point_source)
        if self.anchor_intent is None:
            self.anchor_intent = isinstance(
                self.point_source, (AtomPoint, PseudoAtomPoint)
            )

    def xyz(self) -> Tuple[float, float, float]:
        return (float(self.x), float(self.y), float(self.z))

    def resolve(self, context=None) -> Tuple[float, float, float]:
        if self.point_source is not None:
            xyz = self.point_source.resolve(context)
            return (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        return self.xyz()

    def is_anchored(self) -> bool:
        return isinstance(self.point_source, (AtomPoint, PseudoAtomPoint))

    def wants_anchor(self) -> bool:
        if self.anchor_intent is None:
            return self.is_anchored()
        return bool(self.anchor_intent)

    def can_anchor(self) -> bool:
        return self.atom_ref is not None

    def with_anchor_intent(self, anchored: bool) -> "VisualPoint":
        """Record checkbox state without swapping the live PointSource."""
        if not self.can_anchor():
            return self
        return self._replace(anchor_intent=bool(anchored))

    def commit_anchor(self) -> "VisualPoint":
        """Apply pending checkbox state to the PointSource (Create / Update)."""
        return self.with_anchored(self.wants_anchor())

    def with_anchored(self, anchored: bool) -> "VisualPoint":
        ref = self.atom_ref
        if ref is None:
            return self
        xyz = self.xyz()
        if anchored:
            ps = AtomPoint(
                ref.model,
                ref.atom_id,
                chain=ref.chain,
                resi=ref.resi,
                name=ref.name,
                elem=ref.elem or "",
                last_xyz=xyz,
            )
        else:
            ps = FixedPoint(xyz)
        return self._replace(point_source=ps, atom_ref=ref, anchor_intent=bool(anchored))

    def sync_from_source(self, context=None) -> "VisualPoint":
        xyz = self.resolve(context)
        return self._replace(x=xyz[0], y=xyz[1], z=xyz[2])

    def rgba(self) -> Tuple[float, float, float, float]:
        return (
            float(self.color[0]),
            float(self.color[1]),
            float(self.color[2]),
            float(self.alpha),
        )

    def _replace(self, **kwargs) -> "VisualPoint":
        return VisualPoint(
            kwargs.get("name", self.name),
            kwargs.get("source", self.source),
            kwargs.get("x", self.x),
            kwargs.get("y", self.y),
            kwargs.get("z", self.z),
            kwargs.get("color", self.color),
            kwargs.get("alpha", self.alpha),
            kwargs.get("point_source", self.point_source),
            kwargs.get("atom_ref", self.atom_ref),
            kwargs.get("anchor_intent", self.anchor_intent),
            kwargs.get("radius", self.radius),
        )

    def with_xyz(self, xyz: Sequence[float]) -> "VisualPoint":
        fp = FixedPoint(xyz)
        return self._replace(
            x=float(xyz[0]),
            y=float(xyz[1]),
            z=float(xyz[2]),
            point_source=fp,
            anchor_intent=False,
        )

    def with_name(self, name: str) -> "VisualPoint":
        return self._replace(name=str(name))

    def with_source(self, source: str) -> "VisualPoint":
        return self._replace(source=str(source))

    def with_color(self, color: Sequence[float]) -> "VisualPoint":
        alpha = float(color[3]) if len(color) >= 4 else self.alpha
        return self._replace(
            color=(float(color[0]), float(color[1]), float(color[2])),
            alpha=alpha,
        )

    def with_radius(self, radius: Optional[float]) -> "VisualPoint":
        value = None if radius is None else float(radius)
        return self._replace(radius=value)


def assign_distinct_colors(points: List[VisualPoint]) -> None:
    """Reassign distinct palette colors in place."""
    for i, pt in enumerate(points):
        points[i] = pt.with_color(colors_for_new_points(len(points))[i])


def apply_global_color(points: List[VisualPoint], color: Sequence[float]) -> None:
    for i, pt in enumerate(points):
        points[i] = pt.with_color(color)


def commit_point_anchors(points: Sequence[VisualPoint]) -> List[VisualPoint]:
    """Swap PointSources to match checkbox state at Create / Update."""
    return [pt.commit_anchor() for pt in points]


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
        state = int(cmd_.get_state())
    except Exception:
        return 1
    return state if state > 0 else 1


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
        "atoms.append([model, chain, elem, resn, resi, ID, name, x, y, z])",
        "atoms.append([model, chain, elem, resn, resi, ID, x, y, z])",
        "atoms.append([model, chain, elem, resn, resi, index, name, x, y, z])",
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


def _active_selection(cmd_, interactive_only: bool = False) -> Optional[str]:
    """Return a selection expression with at least one atom, or None."""
    state = _current_state(cmd_)
    candidates = ["(sele)", "(selextended)", "(pk1)"]
    if not interactive_only:
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


def _atom_row_as_dict(row):
    if len(row) >= 10:
        model, chain, elem, resn, resi, atom_id, atom_name, x, y, z = row[:10]
    else:
        model, chain, elem, resn, resi, atom_id, x, y, z = row[:9]
        atom_name = elem
    return {
        "model": model,
        "chain": chain or "",
        "elem": elem or "",
        "name": atom_name or elem or "",
        "resn": resn or "",
        "resi": resi or "",
        "index": int(atom_id),
        "x": float(x),
        "y": float(y),
        "z": float(z),
    }


SNAP_TO_ATOM_RADIUS = 2.0


def nearest_atom_within(cmd_, pos: Sequence[float], radius: float = 1.0, sele: str = "visible"):
    """Return atom identity and coordinates if any atom in sele is within radius of pos."""
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    expr = "((%s) within %g of [%g,%g,%g])" % (sele, radius, x, y, z)
    best = None
    best_d2 = radius * radius
    state = _current_state(cmd_)
    atoms = []
    if not _iterate_atoms(cmd_, expr, atoms, state):
        return None
    for row in atoms:
        atom = _atom_row_as_dict(row)
        d2 = _dist2(pos, (atom["x"], atom["y"], atom["z"]))
        if d2 <= best_d2:
            best_d2 = d2
            best = atom
    return best


def _atom_dicts(cmd_, sele: str):
    rows = []
    if not _iterate_atoms(cmd_, sele, rows, _current_state(cmd_)):
        return []
    return [_atom_row_as_dict(row) for row in rows]


def nearest_atom_at_view_center(cmd_, radius=SNAP_TO_ATOM_RADIUS):
    """Closest visible atom to the camera-center marker, or None if none within radius."""
    try:
        view = tuple(cmd_.get_view())
    except Exception:
        return None
    pos = screen_center(view)
    limit = float(radius) * float(radius)
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    pad = float(radius) + 0.05
    near_box = (
        "x > %f and x < %f and y > %f and y < %f and z > %f and z < %f"
        % (x - pad, x + pad, y - pad, y + pad, z - pad, z + pad)
    )
    atoms = _atom_dicts(cmd_, near_box)
    if not atoms:
        for sele in ("visible", "visible and enabled", "all"):
            atoms = _atom_dicts(cmd_, sele)
            if atoms:
                break
    best = None
    best_d2 = limit
    for atom in atoms:
        d2 = _dist2(pos, (atom["x"], atom["y"], atom["z"]))
        if d2 <= best_d2:
            best_d2 = d2
            best = atom
    return best


def _point_source_for_atom(
    hook_to_selection, model, atom_id, chain, resi, atom_name, xyz, elem="",
):
    xyz = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    if hook_to_selection:
        return AtomPoint(
            model,
            int(atom_id),
            chain=chain or "",
            resi=resi or "",
            name=atom_name or "",
            elem=elem or "",
            last_xyz=xyz,
        )
    return FixedPoint(xyz)


def camera_center_point(
    cmd_,
    snap_to_atom: bool = False,
    existing: Sequence[VisualPoint] = (),
    hook_to_selection: bool = True,
):
    view = tuple(cmd_.get_view())
    pos = screen_center(view)
    source = "manual"
    name = manual_fallback_name("cam", existing)
    if snap_to_atom:
        atom = nearest_atom_at_view_center(cmd_)
        if atom is not None:
            pos = (atom["x"], atom["y"], atom["z"])
            source = "selection"
            name = atom_point_name(
                atom["model"],
                elem=atom["elem"],
                resn=atom["resn"],
                resi=atom["resi"],
                chain=atom["chain"],
                index=atom["index"],
            )
            pt_source = _point_source_for_atom(
                hook_to_selection,
                atom["model"],
                atom["index"],
                atom["chain"],
                atom["resi"],
                atom.get("name") or atom["elem"],
                pos,
                elem=atom.get("elem") or "",
            )
            ref = _atom_ref(
                atom["model"],
                atom["index"],
                atom["chain"],
                atom["resi"],
                atom.get("name") or atom["elem"],
                atom.get("elem") or "",
            )
            return VisualPoint(
                name, source, pos[0], pos[1], pos[2],
                point_source=pt_source, atom_ref=ref,
            )
    return VisualPoint(name, source, pos[0], pos[1], pos[2], point_source=FixedPoint(pos))


def selection_points(
    cmd_,
    existing: Sequence[VisualPoint] = (),
    interactive_only: bool = False,
    hook_to_selection: bool = True,
) -> List[VisualPoint]:
    sele = _active_selection(cmd_, interactive_only=interactive_only)
    if sele is None:
        return []
    state = _current_state(cmd_)
    atoms = []
    if not _iterate_atoms(cmd_, sele, atoms, state):
        return []
    used = {p.name for p in existing}
    out = []
    for row in atoms:
        if len(row) >= 10:
            model, chain, elem, resn, resi, atom_id, atom_name, x, y, z = row[:10]
        else:
            model, chain, elem, resn, resi, atom_id, x, y, z = row[:9]
            atom_name = elem
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
        xyz = (float(x), float(y), float(z))
        ref = _atom_ref(
            model, atom_id, chain, resi, atom_name or elem or "", elem or "",
        )
        out.append(VisualPoint(
            name, "selection", xyz[0], xyz[1], xyz[2],
            point_source=_point_source_for_atom(
                hook_to_selection, model, atom_id, chain, resi,
                atom_name or elem or "", xyz, elem=elem or "",
            ),
            atom_ref=ref,
        ))
    return out


def apply_location(dst: VisualPoint, src: VisualPoint) -> VisualPoint:
    """Copy location and atom identity from src, keep dst color, alpha, and radius."""
    return src.with_color(dst.rgba()).with_radius(dst.radius)


def _unique_named(point: VisualPoint, used) -> VisualPoint:
    name = point.name
    if name not in used:
        return point
    base = name
    n = 1
    while True:
        candidate = "%s_%d" % (base, n)
        if candidate not in used:
            return point.with_name(candidate)
        n += 1


def _valid_rows(points: Sequence[VisualPoint], rows: Sequence[int]) -> List[int]:
    n = len(points)
    return sorted({int(row) for row in rows if 0 <= int(row) < n})


def update_points_from_camera(
    cmd_,
    points: Sequence[VisualPoint],
    rows: Sequence[int],
    snap_to_atom: bool = False,
    hook_to_selection: bool = True,
) -> List[VisualPoint]:
    """Replace selected points with the current camera-center point."""
    out = list(points)
    for row in _valid_rows(out, rows):
        existing = [pt for i, pt in enumerate(out) if i != row]
        fresh = camera_center_point(
            cmd_,
            snap_to_atom,
            existing,
            hook_to_selection=hook_to_selection,
        )
        out[row] = apply_location(out[row], fresh)
    return out


def update_points_from_selection(
    cmd_,
    points: Sequence[VisualPoint],
    rows: Sequence[int],
    hook_to_selection: bool = True,
) -> Optional[List[VisualPoint]]:
    """Replace selected points from the current PyMOL selection.

    One selected atom is applied to every chosen row. Several atoms are
    zipped onto the chosen rows in order. Returns None if sele is empty.
    """
    out = list(points)
    chosen = _valid_rows(out, rows)
    if not chosen:
        return out
    atoms = selection_points(cmd_, existing=(), hook_to_selection=hook_to_selection)
    if not atoms:
        return None
    srcs = atoms if len(atoms) > 1 else [atoms[0]] * len(chosen)
    for row, src in zip(chosen, srcs):
        used = {pt.name for i, pt in enumerate(out) if i != row}
        out[row] = apply_location(out[row], _unique_named(src, used))
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
