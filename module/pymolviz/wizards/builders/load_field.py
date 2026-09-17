"""Load a sampleable field from a file into PyMOL / GridData (no Qt)."""

from __future__ import annotations

from pathlib import Path

MAP_EXTENSIONS = (".ccp4", ".mrc", ".map", ".dx", ".xplor", ".grd")
XYZ_EXTENSIONS = (".xyz",)
MTZ_EXTENSIONS = (".mtz",)
ORCA_EXTENSIONS = (".txt", ".cube")
PMV_EXTENSIONS = (".pmv",)

KIND_MAP = "map"
KIND_XYZ = "xyz"
KIND_MTZ = "mtz"
KIND_ORCA = "orca"
KIND_PMV = "pmv"
KIND_IMPLICIT = "implicit"
KIND_FROM_SELECTION = "from_selection"
KIND_DERIVED = "derived"


def field_kind_from_path(path) -> str:
    ext = Path(str(path)).suffix.lower()
    if ext in MTZ_EXTENSIONS:
        return KIND_MTZ
    if ext in XYZ_EXTENSIONS:
        return KIND_XYZ
    if ext in ORCA_EXTENSIONS:
        return KIND_ORCA
    if ext in PMV_EXTENSIONS:
        return KIND_PMV
    return KIND_MAP


def default_field_name(path) -> str:
    stem = Path(str(path)).stem.strip() or "field"
    return stem.replace(" ", "_")


def load_field_file(cmd, path, kind=None, name=None):
    """Create or wrap a ``GridData`` from ``path``. Returns the field or None."""
    path = str(path)
    kind = kind or field_kind_from_path(path)
    label = name or default_field_name(path)
    from .object_names import unused_object_name

    label = unused_object_name(label, cmd)
    if kind == KIND_PMV:
        from ...io import intern_loaded, load

        objects = load(path)
        interned = intern_loaded(cmd, objects)
        for obj in interned:
            if type(obj).__name__ == "Field":
                return obj
        return interned[0] if interned else None
    if kind == KIND_XYZ:
        from ...util.io import grid_from_xyz

        grid = grid_from_xyz(path, name=label)
        return _register_grid(cmd, grid, path=path, kind=KIND_XYZ)
    if kind == KIND_ORCA:
        from ...util.io import grid_from_orca3d

        grid = grid_from_orca3d(path, name=label)
        return _register_grid(cmd, grid, path=path, kind=KIND_ORCA)
    if kind == KIND_MTZ:
        from ...util.io import grid_from_mtz

        grid = grid_from_mtz(path, name=label)
        return _register_grid(cmd, grid, path=path, kind=KIND_MTZ)
    return _load_pymol_map(cmd, path, label)


def load_mtz_fields(cmd, path, maps, *, stem=None):
    """Load each selected MTZ amplitude/phase pair as its own Field."""
    from ...util.io import grid_from_mtz, mtz_field_basename
    from .object_names import unused_object_name

    path = str(path)
    stem = stem or default_field_name(path)
    fields = []
    for product in maps or ():
        factor = str(product.get("factor") or "").strip()
        phase = str(product.get("phase") or "").strip()
        if not factor or not phase:
            continue
        label = unused_object_name(mtz_field_basename(stem, product), cmd)
        grid = grid_from_mtz(
            path,
            factor_column=factor,
            phase_column=phase,
            name=label,
        )
        field = _register_grid(
            cmd, grid, path=path, kind=KIND_MTZ,
            extra={
                "factor_column": factor,
                "phase_column": phase,
                "map_title": str(product.get("title") or ""),
            },
        )
        if field is not None:
            fields.append(field)
    return fields


def _register_grid(cmd, grid, *, path=None, kind=None, extra=None):
    from ...fields.field import as_field, intern_field
    from ...util.field_sample import remember_field

    field = as_field(grid)
    if field is not None and (path or kind or extra):
        gen = dict(field.generator or {})
        if path:
            gen["path"] = str(path)
        if kind:
            gen["kind"] = str(kind)
        if extra:
            gen.update(extra)
        field.generator = gen
    remember_field(field if field is not None else grid)
    try:
        from .field_visual import ensure_grid_ready

        ensure_grid_ready(cmd, getattr(field, "grid_data", None) or grid)
    except Exception:
        pass
    return field if field is not None else grid


def _load_pymol_map(cmd, path, name):
    from ...util.field_sample import forget_native_grid, grid_from_pymol_map, PYMOL_MAP_ID_PREFIX

    if cmd is None:
        return None
    cmd.load(path, name)
    try:
        cmd.set("map_auto_expand_sym", 0)
        cmd.set("map_auto_expand_sym", 0, name)
    except Exception:
        pass
    try:
        if hasattr(cmd, "object_types"):
            cmd.object_types[str(name)] = "object:map"
    except Exception:
        pass
    forget_native_grid(PYMOL_MAP_ID_PREFIX + str(name))
    grid = grid_from_pymol_map(name, cmd=cmd)
    if grid is None:
        return None
    from ...fields.field import as_field, intern_field

    field = as_field(grid)
    if field is not None:
        intern_field(field)
    return field if field is not None else grid


def _point_element(pt) -> str:
    ref = getattr(pt, "atom_ref", None)
    elem = str(getattr(ref, "elem", "") or "").strip()
    if elem:
        return elem
    src = getattr(pt, "point_source", None)
    elem = str(getattr(src, "elem", "") or "").strip()
    return elem or "C"


def centers_and_elements_from_points(points):
    """``(centers, elements)`` for enabled visual points."""
    from .points import enabled_points

    active = enabled_points(points)
    centers = []
    elements = []
    for pt in active:
        try:
            xyz = pt.resolve()
        except Exception:
            xyz = pt.xyz()
        centers.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
        elements.append(_point_element(pt))
    return centers, elements


def atom_records_from_points(points):
    """Stable atom snapshots for a From Selection Field (no live ``sele``)."""
    from ...util.solvent_surface import lookup_source_vdw
    from .points import enabled_points

    records = []
    for pt in enabled_points(points):
        try:
            xyz = pt.resolve()
        except Exception:
            xyz = pt.xyz()
        src = getattr(pt, "point_source", None)
        ref = getattr(pt, "atom_ref", None)
        rec = {
            "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
            "elem": _point_element(pt),
        }
        if ref is not None:
            rec["object"] = str(getattr(ref, "model", "") or "")
            rec["atom_id"] = int(getattr(ref, "atom_id", 0) or 0)
            rec["chain"] = str(getattr(ref, "chain", "") or "")
            rec["resi"] = str(getattr(ref, "resi", "") or "")
            rec["name"] = str(getattr(ref, "name", "") or "")
        elif src is not None:
            rec["object"] = str(getattr(src, "object", "") or "")
            rec["atom_id"] = int(getattr(src, "atom_id", 0) or 0)
            rec["chain"] = str(getattr(src, "chain", "") or "")
            rec["resi"] = str(getattr(src, "resi", "") or "")
            rec["name"] = str(getattr(src, "name", "") or "")
        vdw = lookup_source_vdw(src, None) if src is not None else None
        if vdw is not None:
            rec["vdw"] = float(vdw)
        b_factor = getattr(src, "last_b_factor", None) or getattr(pt, "b_factor", None)
        if b_factor is not None:
            rec["b_factor"] = float(b_factor)
        occupancy = getattr(src, "last_occupancy", None) or getattr(pt, "occupancy", None)
        if occupancy is not None:
            rec["occupancy"] = float(occupancy)
        else:
            rec["occupancy"] = 1.0
        records.append(rec)
    return records


def grid_from_implicit_atoms(
    cmd,
    centers,
    elements,
    name,
    *,
    quality=None,
    resolution=None,
    b_floor=None,
):
    """Gaussian density field from atom centers (PyMOL ``map_new gaussian``)."""
    from ...fields.domain import Domain
    from ...fields.field import Field, register_generated_field
    from ...fields.identity import GEN_GAUSSIAN
    from ...util.gaussian_map import (
        DEFAULT_GAUSSIAN_B_FLOOR,
        DEFAULT_GAUSSIAN_RESOLUTION,
    )
    from ...util.solvent_surface import DEFAULT_QUALITY, gauss_spacing
    from .object_names import unused_object_name

    label = unused_object_name(name or "implicit", cmd)
    q = DEFAULT_QUALITY if quality is None else int(quality)
    resol = DEFAULT_GAUSSIAN_RESOLUTION if resolution is None else float(resolution)
    floor = DEFAULT_GAUSSIAN_B_FLOOR if b_floor is None else float(b_floor)
    atoms = []
    for xyz, elem in zip(centers or (), elements or ()):
        atoms.append({
            "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
            "elem": str(elem or "C"),
        })
    field = Field(
        name=label,
        generator={
            "type": GEN_GAUSSIAN,
            "atoms": atoms,
            "quality": q,
            "resolution": resol,
            "b_floor": floor,
        },
        domain=Domain(spacing=gauss_spacing(q)),
        provenance={"source": "from_selection", "algorithm": GEN_GAUSSIAN},
    )
    try:
        field._name = label
    except Exception:
        pass
    return register_generated_field(field, cmd=cmd)


def field_from_selection(
    cmd,
    points,
    name,
    *,
    algorithm="gaussian_atoms",
    domain=None,
    quality=None,
    resolution=None,
    b_floor=None,
    property_key="b_factor",
    color_snapshot=True,
    units=None,
    register=True,
):
    """Build a Field from enabled table points.

    When ``register`` is False the brick is baked for preview only and the
    field is not interned into the session.
    """
    from ...fields.domain import Domain
    from ...fields.field import Field, register_generated_field
    from ...fields.identity import (
        GEN_DISTANCE,
        GEN_GAUSSIAN,
        GEN_NEAREST_PROP,
        GEN_SIGNED_VDW,
        KIND_CATEGORICAL,
        KIND_SCALAR,
    )
    from ...util.gaussian_map import (
        DEFAULT_GAUSSIAN_B_FLOOR,
        DEFAULT_GAUSSIAN_RESOLUTION,
    )
    from ...util.solvent_surface import DEFAULT_QUALITY, gauss_spacing
    from .object_names import unused_object_name

    atoms = atom_records_from_points(points)
    if not atoms:
        return None
    label = unused_object_name(name or "field", cmd)
    algo = str(algorithm or GEN_GAUSSIAN)
    q = DEFAULT_QUALITY if quality is None else int(quality)
    resol = DEFAULT_GAUSSIAN_RESOLUTION if resolution is None else float(resolution)
    floor = DEFAULT_GAUSSIAN_B_FLOOR if b_floor is None else float(b_floor)
    if domain is None:
        domain = Domain(spacing=gauss_spacing(q))
    generator = {"type": algo, "atoms": atoms}
    kind = KIND_SCALAR
    if algo == GEN_GAUSSIAN:
        generator.update({"quality": q, "resolution": resol, "b_floor": floor})
    elif algo == GEN_NEAREST_PROP:
        generator["property"] = str(property_key or "b_factor")
        generator["color_snapshot"] = bool(color_snapshot)
        if str(property_key or "") in ("elem", "element", "chain", "name", "resn", "resi"):
            kind = KIND_CATEGORICAL
    field = Field(
        name=label,
        kind=kind,
        units=units,
        generator=generator,
        domain=domain,
        provenance={"source": "from_selection", "algorithm": algo},
    )
    try:
        field._name = label
    except Exception:
        pass
    if not register:
        from ...fields.field import ensure_brick

        ensure_brick(field, cmd=cmd)
        return field
    return register_generated_field(field, cmd=cmd)


def commit_from_selection_preset(
    cmd,
    points,
    name,
    *,
    algorithm="gaussian_atoms",
    domain=None,
    quality=None,
    resolution=None,
    property_key="b_factor",
    color_mode=None,
    color_field_id=None,
    colormap=None,
    iso_level=None,
    visual_kind="IsoSurface",
    persist=True,
):
    """Intern From Selection Field(s) and ensure a loaded IsoSurface visual.

    Always interned (``register=True``). Preview must keep using
    ``field_from_selection(..., register=False)`` so draft bricks stay
    ephemeral. Loads the visual (``.load``), not the Field as CGO.
    """
    from .field_params import default_preset_iso_level
    from .field_visual import ensure_field_visual
    from .points import enabled_points
    from .surface_params import COLOR_MODE_FIELD, COLOR_MODE_UNIFORM

    geom = field_from_selection(
        cmd,
        points,
        name,
        algorithm=algorithm,
        domain=domain,
        quality=quality,
        resolution=resolution,
        property_key=property_key,
        register=True,
    )
    if geom is None:
        return None, None, None

    mode = str(color_mode or COLOR_MODE_UNIFORM)
    color_field = None
    uniform_color = None
    cmap = colormap
    if mode == COLOR_MODE_FIELD:
        remember_default_color_field(
            geom, color_field_id=color_field_id, colormap=cmap,
        )
    else:
        remember_default_color_field(geom, color_field_id=None)
        active = enabled_points(points)
        if active:
            rgba = active[0].rgba() if hasattr(active[0], "rgba") else active[0].color
            uniform_color = (float(rgba[0]), float(rgba[1]), float(rgba[2]))

    bound_color_id = default_color_field_id(geom)
    if iso_level is None:
        iso_level = default_preset_iso_level(algorithm, geom)
    visual = ensure_field_visual(
        cmd,
        geom,
        visual_kind,
        level=float(iso_level),
        color_field_id=bound_color_id,
        colormap=cmap or "RdYlBu_r",
        color=None if bound_color_id else uniform_color,
        persist=persist,
    )
    remember_default_iso_level(geom, iso_level)
    return geom, color_field, visual


def default_color_field_id(field):
    """Companion color Field id remembered on a geometry Field, if any."""
    if field is None:
        return None
    fid = getattr(field, "default_color_field_id", None)
    if fid:
        return str(fid)
    prov = getattr(field, "provenance", None) or {}
    fid = prov.get("default_color_field_id")
    return str(fid) if fid else None


def default_color_colormap(field):
    if field is None:
        return None
    prov = getattr(field, "provenance", None) or {}
    cmap = prov.get("default_colormap")
    return str(cmap) if cmap else None


def remember_default_color_field(geometry_field, color_field=None, *, color_field_id=None, colormap=None):
    """Store an optional default color Field on the geometry Field (not identity)."""
    if geometry_field is None:
        return geometry_field
    fid = color_field_id
    if not fid and color_field is not None:
        fid = getattr(color_field, "id", None)
    fid = str(fid) if fid else None
    geometry_field.default_color_field_id = fid
    prov = dict(getattr(geometry_field, "provenance", None) or {})
    if fid:
        prov["default_color_field_id"] = fid
    else:
        prov.pop("default_color_field_id", None)
    if colormap:
        prov["default_colormap"] = str(colormap)
    elif not fid:
        prov.pop("default_colormap", None)
    geometry_field.provenance = prov
    return geometry_field


def remember_default_iso_level(field, iso_level):
    """Preview / IsoSurface default; not part of Field identity."""
    if field is None or iso_level is None:
        return field
    prov = dict(getattr(field, "provenance", None) or {})
    prov["default_iso_level"] = float(iso_level)
    field.provenance = prov
    return field


def default_iso_level_from_field(field):
    if field is None:
        return None
    prov = getattr(field, "provenance", None) or {}
    value = prov.get("default_iso_level")
    if value is None:
        return None
    return float(value)


def field_is_from_selection(field) -> bool:
    from ...fields.identity import (
        GEN_DISTANCE,
        GEN_GAUSSIAN,
        GEN_NEAREST_COLOR,
        GEN_NEAREST_PROP,
        GEN_SIGNED_VDW,
    )

    gen = str((getattr(field, "generator", None) or {}).get("type") or "")
    return gen in (
        GEN_GAUSSIAN,
        GEN_DISTANCE,
        GEN_SIGNED_VDW,
        GEN_NEAREST_PROP,
        GEN_NEAREST_COLOR,
    )


def field_options(field) -> dict:
    """Generator / Domain / provenance knobs for the From Selection editor."""
    from ...fields.identity import GEN_GAUSSIAN
    from ...util.gaussian_map import DEFAULT_GAUSSIAN_RESOLUTION
    from ...util.solvent_surface import DEFAULT_QUALITY

    gen = dict(getattr(field, "generator", None) or {})
    domain = getattr(field, "domain", None)
    domain_dict = domain.to_dict() if domain is not None and hasattr(domain, "to_dict") else {}
    from .preview_mode import read_preview_mode

    return {
        "algorithm": str(gen.get("type") or GEN_GAUSSIAN),
        "quality": int(gen.get("quality") or DEFAULT_QUALITY),
        "resolution": float(gen.get("resolution") or DEFAULT_GAUSSIAN_RESOLUTION),
        "property": str(gen.get("property") or "b_factor"),
        "iso_level": default_iso_level_from_field(field),
        "domain": domain_dict,
        "color_field_id": default_color_field_id(field),
        "colormap": default_color_colormap(field),
        "preview_mode": read_preview_mode(field),
    }


def points_from_field(field):
    """Rebuild table points from a From Selection generator recipe."""
    from ...points import AtomPoint, FixedPoint
    from .points import AtomRef, VisualPoint, SOURCE_SELECTION

    gen = getattr(field, "generator", None) or {}
    points = []
    for i, rec in enumerate(gen.get("atoms") or ()):
        if not isinstance(rec, dict):
            continue
        xyz = rec.get("xyz") or (0.0, 0.0, 0.0)
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        elem = str(rec.get("elem") or "C")
        object_name = str(rec.get("object") or "")
        atom_id = rec.get("atom_id")
        chain = str(rec.get("chain") or "")
        resi = str(rec.get("resi") or "")
        atom_name = str(rec.get("name") or "")
        label = atom_name or ("p%d" % (i + 1))
        ref = None
        if object_name and atom_id:
            src = AtomPoint(
                object_name,
                int(atom_id),
                chain=chain,
                resi=resi,
                name=atom_name,
                elem=elem,
                last_xyz=(x, y, z),
                last_vdw=rec.get("vdw"),
            )
            if rec.get("b_factor") is not None:
                src.last_b_factor = float(rec["b_factor"])
            if rec.get("occupancy") is not None:
                src.last_occupancy = float(rec["occupancy"])
            ref = AtomRef(
                object_name,
                int(atom_id),
                chain,
                resi,
                atom_name,
                elem,
            )
        else:
            src = FixedPoint((x, y, z))
        points.append(
            VisualPoint(
                name=label,
                source=SOURCE_SELECTION,
                x=x,
                y=y,
                z=z,
                point_source=src,
                atom_ref=ref,
            )
        )
    return points
