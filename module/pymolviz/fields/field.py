"""First-class reusable spatial Field (recipe + domain + cached brick)."""

from __future__ import annotations

from typing import Iterable, List, Optional

from ..Displayable import Displayable
from .domain import Domain
from .generators import generate_brick, imported_generator_from_grid
from .identity import (
    GEN_IMPORTED,
    GEN_PYMOL_MAP,
    KIND_SCALAR,
    canonical_field_spec,
    field_identity_hash,
    find_equivalent_field,
    normalize_kind,
    specs_equal,
)

_WRAPS = {}


class Field(Displayable):
    """Sampleable spatial data: generator recipe, domain, and cached GridData."""

    renders_cgo = False
    is_visual = False

    def __init__(
        self,
        name=None,
        kind=KIND_SCALAR,
        units=None,
        generator=None,
        domain=None,
        provenance=None,
        grid_data=None,
        categories=None,
        obj_id=None,
        default_color_field_id=None,
    ):
        self.kind = normalize_kind(kind)
        self.units = str(units) if units else None
        self.generator = dict(generator or {"type": GEN_IMPORTED})
        self.domain = domain if isinstance(domain, Domain) else Domain.from_dict(domain)
        self.provenance = dict(provenance or {})
        self.grid_data = grid_data
        self.categories = list(categories) if categories else None
        fid = default_color_field_id or self.provenance.get("default_color_field_id")
        self.default_color_field_id = str(fid) if fid else None
        super().__init__(name=name, obj_id=obj_id)
        if grid_data is not None:
            self.dependencies = [grid_data]

    def spec(self) -> dict:
        return canonical_field_spec(self)

    def identity_hash(self) -> str:
        return field_identity_hash(self.spec())

    @property
    def values(self):
        brick = ensure_brick(self)
        return None if brick is None else brick.values

    def equivalent_to(self, other) -> bool:
        if other is None:
            return False
        other_spec = canonical_field_spec(other) if not isinstance(other, dict) else other
        return specs_equal(self.spec(), other_spec)

    def _script_string(self):
        brick = ensure_brick(self)
        if brick is not None and hasattr(brick, "_script_string"):
            return brick._script_string()
        return ""

    def load(self, cmd=None):
        brick = ensure_brick(self, cmd=cmd)
        from ..Displayable import call_load
        call_load(brick, cmd)


def ensure_brick(field, cmd=None):
    """Return cached GridData, generating or wrapping it if needed."""
    if field is None:
        return None
    if type(field).__name__ == "GridData":
        return field
    cached = getattr(field, "grid_data", None)
    if cached is not None and type(cached).__name__ == "GridData":
        return cached
    generator = getattr(field, "generator", None) or {}
    domain = getattr(field, "domain", None)
    name = getattr(field, "_name", None) or getattr(field, "name", None) or "field"
    grid, kind, categories = generate_brick(generator, domain, name=name, cmd=cmd)
    if grid is None:
        return None
    field.grid_data = grid
    if kind:
        field.kind = kind
    if categories is not None:
        field.categories = list(categories)
    stops = getattr(grid, "color_stops", None)
    if stops:
        field.color_stops = list(stops)
    field.dependencies = [grid]
    try:
        grid._name = name
    except Exception:
        pass
    return grid


def as_field(obj, *, intern=False) -> Optional[Field]:
    """Wrap GridData / native maps as Field. Reuse an existing wrap when possible."""
    if obj is None:
        return None
    if type(obj).__name__ == "Field":
        if intern:
            return intern_field(obj)
        return obj
    from ..util.field_sample import PYMOL_MAP_ID_PREFIX, resolve_grid

    grid = resolve_grid(obj)
    if grid is None:
        return None
    oid = str(getattr(obj, "id", "") or getattr(grid, "id", "") or "")
    if oid:
        cached = _WRAPS.get(oid)
        if cached is not None:
            if type(obj).__name__ == "GridData":
                cached.grid_data = obj
            elif getattr(cached, "grid_data", None) is None:
                cached.grid_data = grid
            return cached
    from ..util.field_sample import field_label

    label = field_label(obj) or field_label(grid) or "field"
    if oid.startswith(PYMOL_MAP_ID_PREFIX):
        map_name = oid[len(PYMOL_MAP_ID_PREFIX):]
        field = Field(
            name=label,
            kind=KIND_SCALAR,
            generator={"type": GEN_PYMOL_MAP, "map_name": map_name},
            domain=Domain.from_grid(grid),
            provenance={"source": "pymol_map", "map_name": map_name},
            grid_data=grid,
            obj_id=oid,
        )
        try:
            field._name = label
        except Exception:
            pass
    else:
        field = Field(
            name=label,
            kind=KIND_SCALAR,
            generator=imported_generator_from_grid(grid),
            domain=Domain.from_grid(grid),
            provenance={"source": "imported"},
            grid_data=grid,
            obj_id=oid or None,
        )
        try:
            field._name = label
        except Exception:
            pass
    wrap_key = str(field.id)
    _WRAPS[wrap_key] = field
    if intern:
        return intern_field(field)
    return field


def forget_field_wrap(field_id) -> None:
    if not field_id:
        return
    _WRAPS.pop(str(field_id), None)


def remember_wrap(field) -> None:
    if field is None:
        return
    oid = str(getattr(field, "id", "") or "")
    if oid:
        _WRAPS[oid] = field


def clear_wraps() -> None:
    _WRAPS.clear()


def intern_field(field, objects=None) -> Field:
    """Reuse an existing session Field with the same canonical spec."""
    if field is None:
        raise ValueError("intern_field requires a Field")
    spec = canonical_field_spec(field)
    existing = find_equivalent_field(spec, iter_fields(objects))
    if existing is not None:
        if getattr(existing, "grid_data", None) is None and getattr(field, "grid_data", None) is not None:
            existing.grid_data = field.grid_data
        return existing
    try:
        from ..runtime.session import add

        add(field)
    except Exception:
        pass
    _WRAPS[str(field.id)] = field
    return field


def iter_fields(objects: Optional[Iterable] = None) -> List[Field]:
    if objects is None:
        try:
            from ..runtime.session import all_objects

            objects = all_objects()
        except Exception:
            objects = []
    out = []
    seen = set()
    for obj in objects:
        field = obj if type(obj).__name__ == "Field" else None
        if field is None:
            continue
        oid = str(getattr(field, "id", "") or "")
        if oid and oid in seen:
            continue
        if oid:
            seen.add(oid)
        out.append(field)
    return out


def register_generated_field(field, cmd=None) -> Field:
    """Bake the brick, auto-reuse identical fields, and remember in session."""
    ensure_brick(field, cmd=cmd)
    return intern_field(field)
