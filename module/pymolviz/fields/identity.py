"""Canonical Field spec and content identity (hash + deep compare)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from .domain import Domain

KIND_SCALAR = "scalar"
KIND_VECTOR = "vector"
KIND_CATEGORICAL = "categorical"
FIELD_KINDS = (KIND_SCALAR, KIND_VECTOR, KIND_CATEGORICAL)

GEN_IMPORTED = "imported"
GEN_PYMOL_MAP = "pymol_map"
GEN_GAUSSIAN = "gaussian_atoms"
GEN_DISTANCE = "distance_to_atoms"
GEN_SIGNED_VDW = "signed_vdw_distance"
GEN_NEAREST_PROP = "nearest_atom_property"
GEN_NEAREST_COLOR = "nearest_atom_color"
GEN_DERIVED = "derived"
GEN_GRADIENT = "gradient"

IMPLEMENTED_GENERATORS = (
    GEN_IMPORTED,
    GEN_PYMOL_MAP,
    GEN_GAUSSIAN,
    GEN_DISTANCE,
    GEN_SIGNED_VDW,
    GEN_NEAREST_PROP,
    GEN_NEAREST_COLOR,
)
SCHEMA_ONLY_GENERATORS = (GEN_DERIVED, GEN_GRADIENT)

_FLOAT_NDIGITS = 6


def normalize_kind(kind) -> str:
    text = str(kind or KIND_SCALAR).strip().lower()
    if text in FIELD_KINDS:
        return text
    if text in ("category", "discrete"):
        return KIND_CATEGORICAL
    if text in ("vec", "vector3"):
        return KIND_VECTOR
    return KIND_SCALAR


def _round_float(value) -> float:
    return round(float(value), _FLOAT_NDIGITS)


def canonical_value(value: Any) -> Any:
    """JSON-stable form: sorted dicts, rounded floats, lists not tuples."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    try:
        import numpy as np

        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return _round_float(value)
        if isinstance(value, np.ndarray):
            return canonical_value(value.tolist())
    except Exception:
        if isinstance(value, float):
            return _round_float(value)
    if isinstance(value, float):
        return _round_float(value)
    if isinstance(value, dict):
        return {str(k): canonical_value(value[k]) for k in sorted(value.keys(), key=str)}
    if isinstance(value, (list, tuple)):
        return [canonical_value(v) for v in value]
    return str(value)


def canonical_atom_records(records) -> list:
    rows = []
    for item in records or ():
        if not isinstance(item, dict):
            continue
        xyz = item.get("xyz")
        if xyz is None:
            continue
        try:
            x, y, z = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        except (TypeError, ValueError, IndexError):
            continue
        row = {
            "xyz": [_round_float(x), _round_float(y), _round_float(z)],
            "elem": str(item.get("elem") or "C"),
        }
        for key in (
            "object",
            "atom_id",
            "chain",
            "resi",
            "name",
            "vdw",
            "b_factor",
            "occupancy",
            "property",
            "color",
        ):
            if key not in item or item[key] is None:
                continue
            row[key] = canonical_value(item[key])
        rows.append(row)
    rows.sort(
        key=lambda r: (
            str(r.get("object") or ""),
            int(r.get("atom_id") or 0),
            r["xyz"][0],
            r["xyz"][1],
            r["xyz"][2],
            str(r.get("elem") or ""),
            tuple(r["color"]) if isinstance(r.get("color"), list) else (),
        )
    )
    return rows


def canonical_generator(generator) -> dict:
    gen = dict(generator or {})
    kind = str(gen.get("type") or GEN_IMPORTED)
    out = {"type": kind}
    if kind == GEN_PYMOL_MAP:
        out["map_name"] = str(gen.get("map_name") or "")
        return canonical_value(out)
    if kind == GEN_IMPORTED:
        for key in ("path", "kind", "values_digest", "origin", "step_sizes", "step_counts", "A_to"):
            if gen.get(key) is not None:
                out[key] = gen[key]
        return canonical_value(out)
    if kind in (GEN_GAUSSIAN, GEN_DISTANCE, GEN_SIGNED_VDW, GEN_NEAREST_PROP, GEN_NEAREST_COLOR):
        out["atoms"] = canonical_atom_records(gen.get("atoms") or ())
        if kind == GEN_GAUSSIAN:
            out["quality"] = int(gen.get("quality") or 3)
            out["resolution"] = float(gen.get("resolution") or 2.0)
            out["b_floor"] = float(gen.get("b_floor") or 20.0)
        if kind == GEN_NEAREST_PROP:
            out["property"] = str(gen.get("property") or "b_factor")
            out["color_snapshot"] = bool(gen.get("color_snapshot", True))
        if kind == GEN_NEAREST_COLOR:
            from .color_blend import normalize_color_blend, normalize_color_sigma

            out["color_snapshot"] = bool(gen.get("color_snapshot", True))
            out["blend"] = normalize_color_blend(gen.get("blend"))
            out["sigma"] = float(normalize_color_sigma(gen.get("sigma")))
        if gen.get("selection_expr"):
            out["selection_expr"] = str(gen["selection_expr"])
        return canonical_value(out)
    if kind == GEN_DERIVED:
        out["op"] = str(gen.get("op") or "add")
        out["source_a"] = str(gen.get("source_a") or "")
        out["source_b"] = str(gen.get("source_b") or "")
        return canonical_value(out)
    if kind == GEN_GRADIENT:
        out["source"] = str(gen.get("source") or "")
        return canonical_value(out)
    return canonical_value(gen)


def canonical_domain(domain) -> dict:
    if isinstance(domain, Domain):
        data = domain.to_dict()
    elif isinstance(domain, dict):
        data = Domain.from_dict(domain).to_dict()
    else:
        data = Domain().to_dict()
    return canonical_value(data)


def canonical_field_spec(field=None, *, generator=None, domain=None, kind=None, units=None) -> dict:
    """Identity payload: generator + domain + kind + units. No display name, no UUID."""
    if field is not None:
        generator = generator if generator is not None else getattr(field, "generator", None)
        domain = domain if domain is not None else getattr(field, "domain", None)
        kind = kind if kind is not None else getattr(field, "kind", KIND_SCALAR)
        units = units if units is not None else getattr(field, "units", None)
        upstream = getattr(field, "upstream_identities", None)
    else:
        upstream = None
    spec = {
        "kind": normalize_kind(kind),
        "units": None if not units else str(units),
        "generator": canonical_generator(generator),
        "domain": canonical_domain(domain),
    }
    if upstream:
        spec["upstream"] = canonical_value(list(upstream))
    return canonical_value(spec)


def field_identity_hash(spec) -> str:
    if spec is None:
        spec = {}
    if not isinstance(spec, dict) or "generator" not in spec:
        spec = canonical_field_spec(spec) if not isinstance(spec, dict) else canonical_field_spec(
            generator=spec.get("generator"),
            domain=spec.get("domain"),
            kind=spec.get("kind"),
            units=spec.get("units"),
        )
    blob = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def specs_equal(a, b) -> bool:
    """Deep compare of canonical dicts. Never use hash equality alone."""
    return canonical_value(a) == canonical_value(b)


def find_equivalent_field(spec, fields) -> Optional[object]:
    """Return the first field whose canonical spec matches ``spec``."""
    want = canonical_field_spec(
        generator=spec.get("generator") if isinstance(spec, dict) else None,
        domain=spec.get("domain") if isinstance(spec, dict) else None,
        kind=spec.get("kind") if isinstance(spec, dict) else None,
        units=spec.get("units") if isinstance(spec, dict) else None,
    ) if isinstance(spec, dict) and "generator" in spec else canonical_field_spec(spec)
    digest = field_identity_hash(want)
    for field in fields or ():
        other = canonical_field_spec(field)
        if field_identity_hash(other) != digest:
            continue
        if specs_equal(want, other):
            return field
    return None
