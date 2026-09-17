"""First-class reusable spatial Field (recipe + domain + voxel lattice)."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

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
from .lattice import (
    copy_lattice_onto,
    has_lattice,
    init_lattice,
    load_lattice,
    script_string as lattice_script_string,
    to_points as lattice_to_points,
    get_positions as lattice_get_positions,
    cut as lattice_cut,
)

_WRAPS = {}


class Field(Displayable):
    """Sampleable spatial data: generator recipe, domain, and baked voxels."""

    renders_cgo = False
    is_visual = False

    def __init__(
        self,
        values=None,
        positions=None,
        step_sizes=None,
        step_counts=None,
        origin=None,
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
        self.provenance = dict(provenance or {})
        self.categories = list(categories) if categories else None
        fid = default_color_field_id or self.provenance.get("default_color_field_id")
        self.default_color_field_id = str(fid) if fid else None
        self._values = None
        self.origin = None
        self.step_sizes = None
        self.step_counts = None
        self.sorted_indices = None
        self.A_to = None
        self.is_loaded = False
        super().__init__(name=name, obj_id=obj_id)
        if values is not None:
            init_lattice(self, values, positions, step_sizes, step_counts, origin)
        elif grid_data is not None and grid_data is not self:
            copy_lattice_onto(self, grid_data)
        if generator is not None:
            self.generator = dict(generator)
        elif has_lattice(self):
            self.generator = imported_generator_from_grid(self)
        else:
            self.generator = {"type": GEN_IMPORTED}
        if domain is not None:
            self.domain = domain if isinstance(domain, Domain) else Domain.from_dict(domain)
        elif has_lattice(self):
            self.domain = Domain.from_grid(self)
        else:
            self.domain = Domain.from_dict(None)

    def spec(self) -> dict:
        return canonical_field_spec(self)

    def identity_hash(self) -> str:
        return field_identity_hash(self.spec())

    @property
    def grid_data(self):
        return self if has_lattice(self) else None

    @grid_data.setter
    def grid_data(self, brick):
        if brick is None or brick is self:
            return
        copy_lattice_onto(self, brick)

    @property
    def values(self):
        if self._values is None:
            ensure_brick(self)
        return self._values

    @values.setter
    def values(self, values):
        self._values = None if values is None else np.asarray(values).reshape(-1)
        self.is_loaded = False

    def equivalent_to(self, other) -> bool:
        if other is None:
            return False
        other_spec = canonical_field_spec(other) if not isinstance(other, dict) else other
        return specs_equal(self.spec(), other_spec)

    def get_positions(self):
        ensure_brick(self)
        return lattice_get_positions(self)

    def to_points(self, filter=None, *args, **kwargs):
        ensure_brick(self)
        return lattice_to_points(self, filter=filter, *args, **kwargs)

    def cut(self, point, normal, interpolation="NN"):
        ensure_brick(self)
        return lattice_cut(self, point, normal, interpolation)

    def _script_string(self):
        brick = ensure_brick(self)
        if brick is None:
            return ""
        return lattice_script_string(brick)

    def load(self, cmd=None):
        brick = ensure_brick(self, cmd=cmd)
        if brick is None:
            return
        load_lattice(brick, cmd)

    @staticmethod
    def from_xyz(path, in_bohr=True, *args, **kwargs):
        from ..util.io import grid_from_xyz

        return grid_from_xyz(path, in_bohr, *args, **kwargs)

    @staticmethod
    def from_mtz(path, factor_column="FWT", phase_column="PHWT", sample_rate=2.6, min_pos=None, max_pos=None, step_sizes=None, *args, **kwargs):
        from ..util.io import grid_from_mtz

        return grid_from_mtz(
            path,
            factor_column,
            phase_column,
            sample_rate,
            min_pos,
            max_pos,
            step_sizes,
            *args,
            **kwargs
        )

    @staticmethod
    def from_orca3d(path, *args, **kwargs):
        from ..util.io import grid_from_orca3d

        return grid_from_orca3d(path, *args, **kwargs)

    @staticmethod
    def from_ccp4(path):
        import gemmi  # noqa: F401


def ensure_brick(field, cmd=None):
    """Return the Field with a baked lattice, generating it if needed."""
    if field is None:
        return None
    if has_lattice(field):
        return field
    generator = getattr(field, "generator", None) or {}
    domain = getattr(field, "domain", None)
    name = getattr(field, "_name", None) or getattr(field, "name", None) or "field"
    grid, kind, categories = generate_brick(generator, domain, name=name, cmd=cmd)
    if grid is None:
        return None
    if grid is not field:
        copy_lattice_onto(field, grid)
    if kind:
        field.kind = kind
    if categories is not None:
        field.categories = list(categories)
    stops = getattr(grid, "color_stops", None) or getattr(field, "color_stops", None)
    if stops:
        field.color_stops = list(stops)
    field.dependencies = []
    try:
        field._name = name
    except Exception:
        pass
    return field


def as_field(obj, *, intern=False) -> Optional[Field]:
    """Treat a lattice or native map as a Field. Identity when ``obj`` already is one."""
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
    if type(grid).__name__ == "Field":
        if intern:
            return intern_field(grid)
        return grid
    oid = str(getattr(obj, "id", "") or getattr(grid, "id", "") or "")
    if oid:
        cached = _WRAPS.get(oid)
        if cached is not None:
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
    _WRAPS[str(field.id)] = field
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
        if not has_lattice(existing) and has_lattice(field):
            copy_lattice_onto(existing, field)
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
    """Bake the lattice, auto-reuse identical fields, and remember in session."""
    ensure_brick(field, cmd=cmd)
    return intern_field(field)
