"""Named 3D points for CGO builders (camera center, selection, manual)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from ...points import AtomPoint, FixedPoint, PointSource, PseudoAtomPoint
from ...util.sanitize import sanitize_pymol_string
from ...util.view import screen_center
from .colors import DEFAULT_SPHERE_COLOR, ColorChoice, as_color_choice, colors_for_new_points
from .surface_params import COLOR_MODE_FIELD, COLOR_MODE_PER_POINT, COLOR_MODE_UNIFORM

RGB = Tuple[float, float, float]

SOURCE_SELECTION = "selection"
SOURCE_CAMERA = "manual"
SOURCE_MANUAL = "manual"

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
class PointDefinition:
    """Intermediate point placement before Appearance stamps color/opacity."""

    x: float
    y: float
    z: float
    name: str
    source_type: str
    label: str = ""
    atom_ref: Optional[AtomRef] = None
    point_source: Optional[PointSource] = None


@dataclass
class VisualPoint:
    name: str
    source: str
    x: float
    y: float
    z: float
    enabled: bool = True
    color: RGB = field(default_factory=lambda: DEFAULT_SPHERE_COLOR)
    alpha: float = 1.0
    point_source: Optional[PointSource] = None
    atom_ref: Optional[AtomRef] = None
    anchor_intent: Optional[bool] = None
    radius: Optional[float] = None
    field_id: Optional[str] = None
    field_colormap: Optional[str] = None
    field_clims: Optional[Tuple[float, float]] = None
    field_clim_mode: Optional[str] = None
    field_colormap_spec: Optional[dict] = None

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

    def with_enabled(self, enabled: bool) -> "VisualPoint":
        return self._replace(enabled=bool(enabled))

    def _replace(self, **kwargs) -> "VisualPoint":
        return VisualPoint(
            kwargs.get("name", self.name),
            kwargs.get("source", self.source),
            kwargs.get("x", self.x),
            kwargs.get("y", self.y),
            kwargs.get("z", self.z),
            kwargs.get("enabled", self.enabled),
            kwargs.get("color", self.color),
            kwargs.get("alpha", self.alpha),
            kwargs.get("point_source", self.point_source),
            kwargs.get("atom_ref", self.atom_ref),
            kwargs.get("anchor_intent", self.anchor_intent),
            kwargs.get("radius", self.radius),
            kwargs.get("field_id", self.field_id),
            kwargs.get("field_colormap", self.field_colormap),
            kwargs.get("field_clims", self.field_clims),
            kwargs.get("field_clim_mode", self.field_clim_mode),
            kwargs.get("field_colormap_spec", self.field_colormap_spec),
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
        if isinstance(color, ColorChoice):
            return self.with_color_choice(color)
        alpha = float(color[3]) if len(color) >= 4 else self.alpha
        return self._replace(
            color=(float(color[0]), float(color[1]), float(color[2])),
            alpha=alpha,
            field_id=None,
            field_colormap=None,
            field_clims=None,
            field_clim_mode=None,
            field_colormap_spec=None,
        )

    def with_color_choice(self, choice: ColorChoice) -> "VisualPoint":
        choice = as_color_choice(choice)
        field_id = str(choice.field_id) if choice.field_id else None
        rgb = choice.rgb
        clim_mode = None
        clims = None
        if field_id:
            from .colors import effective_clim_mode, resolve_color_clims

            clim_mode = effective_clim_mode(choice)
            clims = resolve_color_clims(clim_mode, choice.clims, field_id)
            from ...util.field_sample import sample_rgb_at
            sampled = sample_rgb_at(
                self.xyz(), field_id, choice.colormap_spec or choice.colormap, clims,
            )
            if sampled is not None:
                rgb = sampled
        return self._replace(
            color=rgb,
            alpha=float(choice.rgba[3]),
            field_id=field_id,
            field_colormap=choice.colormap if field_id else None,
            field_clims=clims if field_id else None,
            field_clim_mode=clim_mode if field_id else None,
            field_colormap_spec=getattr(choice, "colormap_spec", None) if field_id else None,
        )

    def color_choice(self) -> ColorChoice:
        return ColorChoice(
            rgba=self.rgba(),
            field_id=self.field_id,
            colormap=self.field_colormap or "RdYlBu_r",
            clims=self.field_clims,
            clim_mode=self.field_clim_mode,
            colormap_spec=self.field_colormap_spec,
        )

    def with_radius(self, radius: Optional[float]) -> "VisualPoint":
        value = None if radius is None else float(radius)
        return self._replace(radius=value)


def enabled_points(points: Sequence[VisualPoint]) -> List[VisualPoint]:
    """Points included in preview, commit, export, and clip span."""
    return [pt for pt in points if getattr(pt, "enabled", True)]


def _colors_equal(a: Sequence[float], b: Sequence[float], tol: float = 1e-5) -> bool:
    return all(abs(float(a[i]) - float(b[i])) < tol for i in range(3))


def _alphas_equal(points: Sequence[VisualPoint], tol: float = 1e-5) -> bool:
    if not points:
        return True
    first = float(points[0].alpha)
    return all(abs(float(pt.alpha) - first) < tol for pt in points)


def infer_color_mode(points: Sequence[VisualPoint]) -> str:
    """Infer Uniform / Per-point / Field from enabled points."""
    active = enabled_points(points)
    if not active:
        return COLOR_MODE_UNIFORM
    field_ids = {pt.field_id for pt in active if pt.field_id}
    if field_ids:
        if len(field_ids) == 1:
            cmap = {pt.field_colormap for pt in active if pt.field_id}
            clims = {pt.field_clims for pt in active if pt.field_id}
            if len(cmap) <= 1 and len(clims) <= 1:
                return COLOR_MODE_FIELD
        return COLOR_MODE_PER_POINT
    first = active[0].color
    if all(_colors_equal(first, pt.color) for pt in active) and _alphas_equal(active):
        return COLOR_MODE_UNIFORM
    return COLOR_MODE_PER_POINT


def definition_from_visual_point(pt: VisualPoint) -> PointDefinition:
    label = atom_anchor_label(pt.atom_ref) or str(pt.name or "")
    return PointDefinition(
        float(pt.x),
        float(pt.y),
        float(pt.z),
        str(pt.name),
        str(pt.source),
        label=label,
        atom_ref=pt.atom_ref,
        point_source=pt.point_source,
    )


def visual_point_from_definition(
    defn: PointDefinition,
    *,
    color: Sequence[float] = DEFAULT_SPHERE_COLOR,
    alpha: float = 1.0,
    enabled: bool = True,
) -> VisualPoint:
    ps = defn.point_source or FixedPoint((defn.x, defn.y, defn.z))
    return VisualPoint(
        defn.name,
        defn.source_type,
        defn.x,
        defn.y,
        defn.z,
        enabled=bool(enabled),
        color=(float(color[0]), float(color[1]), float(color[2])),
        alpha=float(alpha),
        point_source=ps,
        atom_ref=defn.atom_ref,
    )


def assign_distinct_colors(points: List[VisualPoint]) -> None:
    """Reassign distinct palette colors in place."""
    for i, pt in enumerate(points):
        points[i] = pt.with_color(colors_for_new_points(len(points))[i])


def apply_global_color(
    points: List[VisualPoint],
    color: Sequence[float],
    *,
    rows: Optional[Sequence[int]] = None,
) -> None:
    targets = list(rows) if rows is not None else list(range(len(points)))
    if isinstance(color, ColorChoice):
        for row in targets:
            if 0 <= row < len(points):
                points[row] = points[row].with_color_choice(color)
        return
    for row in targets:
        if 0 <= row < len(points):
            points[row] = points[row].with_color(color)


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


_PICK_SELECTION_NAMES = frozenset({"pk1", "pk2", "pk3", "pkmol", "pkbond"})


def _can_iterate_selection(cmd_, sele_expr: str) -> bool:
    """False for pick names that are not in the selection panel (avoids pk1 errors)."""
    name = _unwrap_selection_name(sele_expr)
    if name not in _PICK_SELECTION_NAMES:
        return True
    listed = _enabled_selection_names(cmd_)
    if listed is None:
        return False
    return name in listed


def _unwrap_selection_name(sele_expr: str) -> str:
    name = str(sele_expr).strip()
    if name.startswith("(") and name.endswith(")"):
        inner = name[1:-1].strip()
        if inner and "(" not in inner and ")" not in inner:
            return inner
    return name


def _enabled_selection_names(cmd_):
    """Enabled selection names, or None if PyMOL cannot report them."""
    get_names = getattr(cmd_, "get_names", None)
    if not callable(get_names):
        return None
    try:
        names = get_names("selections", enabled_only=1)
    except TypeError:
        try:
            names = get_names("selections", 1)
        except Exception:
            return None
    except Exception:
        return None
    try:
        return {str(name) for name in names}
    except TypeError:
        return None


def _selection_is_enabled(cmd_, sele_expr: str) -> bool:
    """True if the named selection is enabled in PyMOL's object panel.

    Disabling ``sele`` (the usual deselect) does **not** clear its atoms:
    ``cmd.count_atoms("sele")`` and ``"sele and enabled"`` still return the
    old count. ``enabled`` in a selection expression means atoms in enabled
    *objects*. Panel state is ``name in cmd.get_names("selections", enabled_only=1)``.
    """
    name = _unwrap_selection_name(sele_expr)
    if not name:
        return False
    enabled = _enabled_selection_names(cmd_)
    if enabled is None:
        return name not in _PICK_SELECTION_NAMES
    return name in enabled


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
    """Return a selection expression with at least one atom, or None.

    When ``interactive_only`` is True, only the live atom selection
    ``(sele)`` / ``(selextended)`` is considered — not ``(pk1)`` or saved
    named selections that can remain populated after the user clears sele.

    A **disabled** named selection is treated as absent. PyMOL keeps ``sele``
    and its atom count after ``disable sele``; only
    ``get_names("selections", enabled_only=1)`` reports the panel state.
    """
    state = _current_state(cmd_)
    if interactive_only:
        candidates = ["(sele)", "(selextended)"]
    else:
        candidates = ["(sele)", "(selextended)", "(pk1)"]
        try:
            listed = cmd_.get_names("selections", enabled_only=1)
        except TypeError:
            try:
                listed = cmd_.get_names("selections", 1)
            except Exception:
                listed = ()
        except Exception:
            listed = ()
        try:
            for name in listed:
                expr = _selection_expr(name)
                if expr not in candidates:
                    candidates.append(expr)
        except TypeError:
            pass
    for expr in candidates:
        if not _selection_is_enabled(cmd_, expr):
            continue
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


def points_from_selection_expr(
    cmd_,
    sele_expr: str,
    existing: Sequence[VisualPoint] = (),
    hook_to_selection: bool = True,
) -> List[VisualPoint]:
    """VisualPoints for every atom in ``sele_expr`` (no active-selection filter)."""
    if cmd_ is None or not str(sele_expr).strip():
        return []
    if not _can_iterate_selection(cmd_, sele_expr):
        return []
    state = _current_state(cmd_)
    atoms = []
    if not _iterate_atoms(cmd_, sele_expr, atoms, state):
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


def selection_points(
    cmd_,
    existing: Sequence[VisualPoint] = (),
    interactive_only: bool = False,
    hook_to_selection: bool = True,
) -> List[VisualPoint]:
    sele = _active_selection(cmd_, interactive_only=interactive_only)
    if sele is None:
        return []
    return points_from_selection_expr(
        cmd_, sele, existing=existing, hook_to_selection=hook_to_selection,
    )


def apply_location(dst: VisualPoint, src: VisualPoint) -> VisualPoint:
    """Copy location and atom identity from src, keep dst color, alpha, and radius."""
    return src.with_color_choice(dst.color_choice()).with_radius(dst.radius)


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
    atoms = selection_points(
        cmd_, existing=(), interactive_only=True, hook_to_selection=hook_to_selection,
    )
    if not atoms:
        return None
    srcs = atoms if len(atoms) > 1 else [atoms[0]] * len(chosen)
    for row, src in zip(chosen, srcs):
        used = {pt.name for i, pt in enumerate(out) if i != row}
        out[row] = apply_location(out[row], _unique_named(src, used))
    return out


POINTS_EXPORT_PREFIX = "_pmv_points_tmp"
POINTS_EXPORT_SELE = "_pmv_points_sel"


def export_points_to_selection(
    cmd_,
    points: Sequence[VisualPoint],
    tmp_object: str = POINTS_EXPORT_PREFIX,
):
    """Create labeled pseudoatoms and a PyMOL selection covering enabled points."""
    purge = []
    try:
        cmd_.delete(tmp_object)
    except Exception:
        pass
    active = enabled_points(points)
    if not active:
        return None
    for i, pt in enumerate(active):
        obj = "%s_%d" % (tmp_object, i)
        purge.append(obj)
        try:
            cmd_.delete(obj)
        except Exception:
            pass
        cmd_.pseudoatom(obj, pos=[pt.x, pt.y, pt.z], label=pt.name)
        try:
            cmd_.show("labels", obj)
        except Exception:
            pass
    try:
        cmd_.select(POINTS_EXPORT_SELE, " or ".join('object "%s"' % n for n in purge))
    except Exception:
        cmd_.select(POINTS_EXPORT_SELE, "none")
    return POINTS_EXPORT_SELE


def _exported_point_object_names(cmd_, prefix: str = POINTS_EXPORT_PREFIX):
    names = []
    try:
        existing = list(cmd_.get_names("objects"))
    except Exception:
        existing = []
    prefix = str(prefix)
    for name in existing:
        text = str(name)
        if text == prefix or text.startswith(prefix + "_"):
            names.append(text)
    return names


def hide_exported_point_labels(cmd_) -> None:
    """Hide labels on the Create-PyMOL-selection overlay without dropping the selection."""
    targets = _exported_point_object_names(cmd_)
    if POINTS_EXPORT_SELE not in targets:
        targets.append(POINTS_EXPORT_SELE)
    for name in targets:
        for selection in (name, 'object "%s"' % name):
            try:
                cmd_.hide("labels", selection)
            except Exception:
                continue
            break


INSERT_SOURCE_SELECTION = "selection"
INSERT_SOURCE_CAMERA = "camera"
INSERT_SOURCE_FRESH = "fresh"
INSERTION_NOTHING_SELECTED = (
    "Nothing selected. Select atoms, then Add Current Selection, or turn on "
    "Add Clicked Atoms and pick in PyMOL."
)
INSERTION_FRESH_HINT = "Turn on Add Clicked Atoms, then pick atoms in PyMOL."
INSERTION_FRESH_WAITING = "Select atoms in PyMOL to add them."
INSERTION_SELECTION_EMPTY = "Nothing selected"
INSERTION_CAPTION_ITERATE_LIMIT = 32
ADD_POINT_LABEL = "Add Current Selection"


def _selection_atom_dicts(cmd_, interactive_only: bool = True) -> List[dict]:
    """Atom rows from the active PyMOL selection without building VisualPoints."""
    sele = _active_selection(cmd_, interactive_only=interactive_only)
    if sele is None:
        return []
    return _atom_dicts(cmd_, sele)


def _residue_key(atom: dict) -> Tuple[str, str, str, str]:
    return (
        str(atom.get("model") or ""),
        str(atom.get("chain") or ""),
        str(atom.get("resn") or ""),
        str(atom.get("resi") or ""),
    )


def _residue_preview_label(atom: dict) -> str:
    resn = str(atom.get("resn") or "").strip().upper()
    resi = str(atom.get("resi") or "").strip()
    if len(resn) == 3:
        display = resn
    else:
        display = resn_one_letter(resn) or resn
    if display and resi:
        return "%s %s" % (display, resi)
    return display or resi or "?"


def atom_insertion_preview_label(atom: dict) -> str:
    """Human-readable atom label, e.g. ``CA · A · GLY 42``."""
    name = str(atom.get("name") or atom.get("elem") or "").strip()
    chain = str(atom.get("chain") or "").strip()
    residue = _residue_preview_label(atom)
    parts = [part for part in (name, chain, residue) if part]
    if parts:
        return " · ".join(parts)
    model = abbreviate_object_name(str(atom.get("model") or ""))
    return model or "Atom"


def _preview_with_coords(label: str, atom: dict) -> str:
    return "%s  (%.2f, %.2f, %.2f)" % (
        label,
        float(atom["x"]),
        float(atom["y"]),
        float(atom["z"]),
    )


def insertion_selection_summary(
    cmd_,
    *,
    interactive_only: bool = True,
) -> Tuple[int, str]:
    """Return ``(count, preview_text)`` for the current PyMOL selection."""
    atoms = _selection_atom_dicts(cmd_, interactive_only=interactive_only)
    count = len(atoms)
    if count == 0:
        return 0, INSERTION_NOTHING_SELECTED
    if count == 1:
        return 1, _preview_with_coords(atom_insertion_preview_label(atoms[0]), atoms[0])
    residue_keys = {_residue_key(atom) for atom in atoms}
    if len(residue_keys) == 1:
        label = _residue_preview_label(atoms[0])
        return count, "%s (%d atoms)" % (label, count)
    hint = atom_insertion_preview_label(atoms[0])
    return count, "%d atoms selected  (%s, …)" % (count, hint)


def _residue_caption_label(atom: dict) -> str:
    """``RESI RESN`` label, e.g. ``42 GLY``."""
    resn = str(atom.get("resn") or "").strip().upper()
    resi = str(atom.get("resi") or "").strip()
    if len(resn) != 3:
        resn = resn_one_letter(resn) or resn
    if resi and resn:
        return "%s %s" % (resi, resn)
    return resi or resn or "?"


def insertion_selection_caption(
    cmd_,
    *,
    interactive_only: bool = True,
) -> str:
    """Short name of the live selection, e.g. ``42 GLY (3 atoms)``.

    Large selections only read the first atom so idle polls stay cheap.
    """
    if cmd_ is None:
        return INSERTION_SELECTION_EMPTY
    count = insertion_selection_count(cmd_, interactive_only=interactive_only)
    if count <= 0:
        return INSERTION_SELECTION_EMPTY
    noun = "atom" if count == 1 else "atoms"
    if count > INSERTION_CAPTION_ITERATE_LIMIT:
        sele = _active_selection(cmd_, interactive_only=interactive_only)
        if sele is None:
            return INSERTION_SELECTION_EMPTY
        rows = []
        if not _iterate_atoms(cmd_, "first %s" % sele, rows, _current_state(cmd_)) or not rows:
            return "%d %s" % (count, noun)
        atom = _atom_row_as_dict(rows[0])
        return "%s (%d %s)" % (_residue_caption_label(atom), count, noun)
    atoms = _selection_atom_dicts(cmd_, interactive_only=interactive_only)
    if not atoms:
        return INSERTION_SELECTION_EMPTY
    residue_keys = []
    labels = []
    for atom in atoms:
        key = _residue_key(atom)
        if key in residue_keys:
            continue
        residue_keys.append(key)
        labels.append(_residue_caption_label(atom))
        if len(labels) == 3:
            break
    if not labels:
        return "%d %s" % (count, noun)
    if len(residue_keys) == 1:
        return "%s (%d %s)" % (labels[0], count, noun)
    shown = labels[:2]
    if len({_residue_key(atom) for atom in atoms}) > 2:
        shown_text = "%s, …" % ", ".join(shown)
    else:
        shown_text = ", ".join(labels)
    return "%s (%d %s)" % (shown_text, count, noun)


def insertion_selection_identifier(cmd_, *, interactive_only: bool = True):
    """Unwrapped PyMOL selection name for the live add target, or None."""
    expr = _active_selection(cmd_, interactive_only=interactive_only)
    if expr is None:
        return None
    name = _unwrap_selection_name(expr).strip()
    return name or None


def insertion_add_label(count: int, name: Optional[str] = None) -> str:
    """Add-button copy. The live selection name is shown under the button."""
    return ADD_POINT_LABEL


def _first_atom_identity(cmd_, sele: str):
    """``(model, atom_id)`` of the first atom in ``sele``, or None.

    Uses ``first`` only. Never iterates the rest of a large selection.
    """
    rows = []
    if not _iterate_atoms(cmd_, "first %s" % sele, rows, _current_state(cmd_)) or not rows:
        return None
    atom = _atom_row_as_dict(rows[0])
    return (str(atom.get("model") or ""), int(atom["index"]))


def insertion_preview_fingerprint(
    cmd_,
    source: str,
    snap: bool = False,
    *,
    interactive_only: bool = True,
):
    """Cheap poll key: selection count + first atom id + whether sele is live.

    Camera source uses a rounded view-center instead of iterating atoms.
    """
    source = str(source)
    if source == INSERT_SOURCE_CAMERA:
        center = None
        try:
            view = tuple(cmd_.get_view())
            pos = screen_center(view)
            center = (
                round(float(pos[0]), 3),
                round(float(pos[1]), 3),
                round(float(pos[2]), 3),
            )
        except Exception:
            pass
        return (source, bool(snap), center, False)
    sele = _active_selection(cmd_, interactive_only=interactive_only)
    live = bool(sele) and _selection_is_enabled(cmd_, sele)
    if not live:
        # Count must be 0 here: refresh_preview uses fingerprint[1] as the
        # Add-button N, and count_atoms("sele") stays non-zero after disable.
        return (source, 0, None, False)
    count = _count_selection_atoms(cmd_, sele, _current_state(cmd_))
    first_id = _first_atom_identity(cmd_, sele) if count else None
    return (source, int(count), first_id, True)


def insertion_selection_count(cmd_, interactive_only: bool = True) -> int:
    sele = _active_selection(cmd_, interactive_only=interactive_only)
    if sele is None:
        return 0
    return _count_selection_atoms(cmd_, sele, _current_state(cmd_))


def insertion_can_add(
    cmd_,
    source: str,
    *,
    interactive_only: bool = True,
) -> bool:
    if str(source) in (INSERT_SOURCE_CAMERA, INSERT_SOURCE_FRESH):
        return True
    return insertion_selection_count(cmd_, interactive_only=interactive_only) > 0


def insertion_preview_text(
    cmd_,
    source: str,
    snap: bool = False,
    *,
    interactive_only: bool = True,
) -> str:
    """Live-preview copy. Callers must re-read via ``resolve_insertion_points`` on Add."""
    if str(source) == INSERT_SOURCE_FRESH:
        return INSERTION_FRESH_HINT
    if str(source) == INSERT_SOURCE_CAMERA:
        if bool(snap):
            atom = nearest_atom_at_view_center(cmd_)
            if atom is not None:
                return _preview_with_coords(atom_insertion_preview_label(atom), atom)
        pt = camera_center_point(
            cmd_, snap_to_atom=bool(snap), existing=(), hook_to_selection=False,
        )
        if pt.source == "selection" and pt.atom_ref is not None:
            ref = pt.atom_ref
            label = atom_insertion_preview_label({
                "model": ref.model,
                "chain": ref.chain,
                "resn": "",
                "resi": ref.resi,
                "name": ref.name,
                "elem": ref.elem,
                "x": pt.x,
                "y": pt.y,
                "z": pt.z,
            })
            if not label or label == abbreviate_object_name(ref.model):
                label = atom_anchor_label(ref) or pt.name
            return "%s  (%.2f, %.2f, %.2f)" % (label, pt.x, pt.y, pt.z)
        return "Camera center  (%.2f, %.2f, %.2f)" % (pt.x, pt.y, pt.z)
    _, text = insertion_selection_summary(
        cmd_, interactive_only=interactive_only,
    )
    return text


def resolve_insertion_points(
    cmd_,
    source: str,
    existing: Sequence[VisualPoint] = (),
    snap: bool = False,
    hook: bool = True,
) -> List[VisualPoint]:
    """Re-read PyMOL at click time; never insert from a stale preview."""
    if str(source) == INSERT_SOURCE_CAMERA:
        return [camera_center_point(
            cmd_,
            snap_to_atom=bool(snap),
            existing=existing,
            hook_to_selection=bool(hook),
        )]
    return selection_points(
        cmd_,
        existing=existing,
        interactive_only=True,
        hook_to_selection=bool(hook),
    )
