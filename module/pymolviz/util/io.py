import numpy as np


def grid_from_xyz(path, in_bohr = True, *args, **kwargs):
    """
    Reads in a grid from an xyz file as output by turbomole.
    Assumes lines starting with # are comments and that the value is in the last column.
    """

    from ..fields.field import Field
    with open(path, "r") as f:
        coords = []
        values = []
        for line in f.readlines():
            if line.startswith("#"):
                continue
            line = line.split()
            if len(line) == 4:
                coords.append([float(line[0]), float(line[1]), float(line[2])])
                values.append(float(line[-1]))

        coords = np.array(coords)
        values = np.array(values)
        if in_bohr:
            coords /= 1.89

        return Field(values, coords, *args, **kwargs)


_MTZ_SKIP_LABELS = frozenset({
    "h", "k", "l", "free", "freer", "freerflag", "freer_flag", "free_r_flag",
    "rfree", "r-free-flags", "i_obs", "isym", "m/isym", "batch",
})
_MTZ_AMPLITUDE_TYPES = frozenset("FGD")
_MTZ_INTENSITY_TYPES = frozenset("IJKM")
_MTZ_PHASE_TYPES = frozenset("P")
_MTZ_PREFERRED_PAIRS = (
    ("FWT", "PHWT"),
    ("2FOFCWT", "PH2FOFCWT"),
    ("2FOFC_WT", "PH2FOFC_WT"),
    ("FOSC", "PHOSC"),
    ("FOFCWT", "PHFOFCWT"),
    ("DELFWT", "PHDELWT"),
    ("FC", "PHIC"),
    ("F", "PHI"),
    ("FP", "PHIB"),
    ("FOBS", "PHIB"),
    ("F-obs", "PHIB"),
    ("F-model", "PHIF"),
)
_MTZ_MAP_TITLES = {
    ("FWT", "PHWT"): "2Fo-Fc",
    ("2FOFCWT", "PH2FOFCWT"): "2Fo-Fc",
    ("2FOFC_WT", "PH2FOFC_WT"): "2Fo-Fc",
    ("FOSC", "PHOSC"): "Fo",
    ("FOFCWT", "PHFOFCWT"): "Fo-Fc",
    ("DELFWT", "PHDELWT"): "Fo-Fc",
    ("FC", "PHIC"): "Fc",
    ("F", "PHI"): "F",
    ("FP", "PHIB"): "FP",
    ("FOBS", "PHIB"): "Fobs",
    ("F-OBS", "PHIB"): "Fobs",
    ("F-MODEL", "PHIF"): "Fmodel",
}
MTZ_TYPE_LABELS = {
    "H": "index",
    "B": "batch",
    "Y": "M/ISYM",
    "I": "integer / intensity",
    "F": "amplitude",
    "D": "anomalous difference",
    "Q": "sigma",
    "G": "F(+)",
    "L": "F(-)",
    "K": "I(+)",
    "M": "I(-)",
    "P": "phase",
    "A": "Hendrickson-Lattman",
    "W": "weight",
    "J": "intensity",
}


def mtz_column_entries(mtz):
    """``[(label, type_char), ...]`` for columns on a gemmi MTZ (or test fake)."""
    out = []
    for col in getattr(mtz, "columns", ()) or ():
        label = str(getattr(col, "label", "") or "").strip()
        if not label:
            continue
        ctype = str(getattr(col, "type", "") or "").strip()
        out.append((label, ctype[:1].upper() if ctype else ""))
    return out


def _mtz_label_map(entries):
    by_lower = {}
    for label, ctype in entries:
        by_lower.setdefault(label.lower(), (label, ctype))
    return by_lower


def _mtz_resolve_label(entries, name):
    if not name:
        return None
    text = str(name).strip()
    if not text:
        return None
    mapped = _mtz_label_map(entries)
    hit = mapped.get(text.lower())
    return None if hit is None else hit[0]


def _mtz_skip_label(label) -> bool:
    compact = str(label or "").strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    if compact in _MTZ_SKIP_LABELS:
        return True
    return compact in ("h", "k", "l")


def _mtz_labels_of_type(entries, types):
    return [
        label for label, ctype in entries
        if ctype in types and not _mtz_skip_label(label)
    ]


def _mtz_companion_phase(factor_label, entries):
    mapped = _mtz_label_map(entries)
    factor = str(factor_label or "")
    rest = factor[1:] if factor[:1].upper() == "F" else factor
    candidates = []
    if rest:
        candidates.extend(("PH" + rest, "PHI" + rest, "P" + rest))
    candidates.extend(("PH" + factor, "PHI" + factor, "PHWT", "PHI", "PHIB", "PHIC", "PHIF"))
    seen = set()
    for raw in candidates:
        key = raw.lower()
        if key in seen:
            continue
        seen.add(key)
        hit = mapped.get(key)
        if hit is not None:
            return hit[0]
    phases = _mtz_labels_of_type(entries, _MTZ_PHASE_TYPES)
    return phases[0] if phases else None


def iter_mtz_map_column_pairs(mtz, factor_column=None, phase_column=None):
    """Yield ``(F, PHI)`` label pairs, preferred names first then any typed pair."""
    entries = mtz_column_entries(mtz)
    seen = set()

    def _emit(factor, phase):
        if not factor or not phase:
            return
        key = (str(factor), str(phase))
        if key in seen or factor == phase:
            return
        seen.add(key)
        yield key

    f_req = _mtz_resolve_label(entries, factor_column)
    p_req = _mtz_resolve_label(entries, phase_column)
    if f_req and p_req:
        yield from _emit(f_req, p_req)
    if f_req:
        yield from _emit(f_req, p_req or _mtz_companion_phase(f_req, entries))
    for fav, pav in _MTZ_PREFERRED_PAIRS:
        yield from _emit(_mtz_resolve_label(entries, fav), _mtz_resolve_label(entries, pav))
    amplitudes = _mtz_labels_of_type(entries, _MTZ_AMPLITUDE_TYPES)
    intensities = _mtz_labels_of_type(entries, _MTZ_INTENSITY_TYPES)
    phases = _mtz_labels_of_type(entries, _MTZ_PHASE_TYPES)
    for factor in amplitudes:
        yield from _emit(factor, _mtz_companion_phase(factor, entries))
    for factor in amplitudes:
        for phase in phases:
            yield from _emit(factor, phase)
    for factor in intensities:
        yield from _emit(factor, _mtz_companion_phase(factor, entries))
        for phase in phases:
            yield from _emit(factor, phase)


def choose_mtz_map_columns(mtz, factor_column=None, phase_column=None):
    """Pick structure-factor and phase column labels that exist on ``mtz``."""
    pairs = list(iter_mtz_map_column_pairs(mtz, factor_column, phase_column))
    if pairs:
        return pairs[0]
    entries = mtz_column_entries(mtz)
    listing = ", ".join("%s (%s)" % (lab, typ or "?") for lab, typ in entries) or "(none)"
    raise ValueError(
        "No amplitude/phase column pair found in the MTZ file. Columns: %s" % listing
    )


def mtz_map_id(factor, phase) -> str:
    return "%s|%s" % (str(factor or "").strip(), str(phase or "").strip())


def mtz_map_title(factor, phase) -> str:
    factor = str(factor or "").strip()
    phase = str(phase or "").strip()
    title = _MTZ_MAP_TITLES.get((factor.upper(), phase.upper()))
    if title:
        return title
    if factor:
        return "%s / %s" % (factor, phase) if phase else factor
    return phase or "map"


def mtz_field_basename(stem, product) -> str:
    stem = str(stem or "mtz").strip().replace(" ", "_") or "mtz"
    title = ""
    if isinstance(product, dict):
        title = str(product.get("title") or product.get("factor") or "map")
    else:
        title = str(product or "map")
    slug = []
    for ch in title.replace(" ", "_"):
        slug.append(ch if ch.isalnum() or ch in "-_" else "_")
    slug = "".join(slug).strip("_") or "map"
    return "%s_%s" % (stem, slug)


def mtz_column_records(mtz):
    """Column dicts: label, type, type_label, dataset."""
    datasets = {}
    for index, dataset in enumerate(getattr(mtz, "datasets", ()) or ()):
        name = (
            getattr(dataset, "dataset_name", None)
            or getattr(dataset, "crystal_name", None)
            or getattr(dataset, "project_name", None)
            or ""
        )
        dsid = getattr(dataset, "id", None)
        if dsid is None:
            dsid = index
        datasets[dsid] = str(name or "")
    records = []
    for col in getattr(mtz, "columns", ()) or ():
        label = str(getattr(col, "label", "") or "").strip()
        if not label:
            continue
        ctype = str(getattr(col, "type", "") or "").strip()
        type_char = ctype[:1].upper() if ctype else ""
        dsid = getattr(col, "dataset_id", None)
        records.append({
            "label": label,
            "type": type_char,
            "type_label": MTZ_TYPE_LABELS.get(type_char, type_char or "unknown"),
            "dataset": str(datasets.get(dsid, "") or ""),
        })
    return records


def _mtz_map_product(factor, phase, types, *, default=False) -> dict:
    return {
        "id": mtz_map_id(factor, phase),
        "title": mtz_map_title(factor, phase),
        "factor": factor,
        "phase": phase,
        "factor_type": str(types.get(factor, "") or ""),
        "phase_type": str(types.get(phase, "") or ""),
        "default": bool(default),
    }


def mtz_map_products(mtz) -> list:
    """Named amplitude/phase maps that can become fields."""
    entries = mtz_column_entries(mtz)
    types = {label: ctype for label, ctype in entries}
    products = []
    seen = set()
    preferred = False
    for fav, pav in _MTZ_PREFERRED_PAIRS:
        factor = _mtz_resolve_label(entries, fav)
        phase = _mtz_resolve_label(entries, pav)
        if not factor or not phase:
            continue
        key = (factor, phase)
        if key in seen:
            continue
        seen.add(key)
        preferred = True
        products.append(_mtz_map_product(factor, phase, types, default=True))
    extras = list(_mtz_labels_of_type(entries, _MTZ_AMPLITUDE_TYPES))
    if not extras:
        extras = list(_mtz_labels_of_type(entries, _MTZ_INTENSITY_TYPES))
    for factor in extras:
        phase = _mtz_companion_phase(factor, entries)
        if not factor or not phase:
            continue
        key = (factor, phase)
        if key in seen:
            continue
        seen.add(key)
        products.append(_mtz_map_product(
            factor, phase, types, default=(not preferred and not products),
        ))
    return products


def mtz_inventory(mtz) -> dict:
    """Exploreable summary of an MTZ (columns, suggested maps, cell)."""
    cell = getattr(mtz, "cell", None)
    cell_abc = None
    if cell is not None:
        try:
            cell_abc = (
                float(getattr(cell, "a", 0.0) or 0.0),
                float(getattr(cell, "b", 0.0) or 0.0),
                float(getattr(cell, "c", 0.0) or 0.0),
                float(getattr(cell, "alpha", 0.0) or 0.0),
                float(getattr(cell, "beta", 0.0) or 0.0),
                float(getattr(cell, "gamma", 0.0) or 0.0),
            )
        except (TypeError, ValueError):
            cell_abc = None
    spacegroup = getattr(mtz, "spacegroup", None)
    space_name = ""
    if spacegroup is not None:
        space_name = str(getattr(spacegroup, "hm", None) or "")
        if not space_name:
            xhm = getattr(spacegroup, "xhm", None)
            space_name = str(xhm() if callable(xhm) else (xhm or spacegroup))
    nref = getattr(mtz, "nreflections", None)
    try:
        nref = int(nref) if nref is not None else None
    except (TypeError, ValueError):
        nref = None
    return {
        "title": str(getattr(mtz, "title", None) or "").strip(),
        "space_group": space_name.strip(),
        "cell": cell_abc,
        "n_reflections": nref,
        "columns": mtz_column_records(mtz),
        "maps": mtz_map_products(mtz),
    }


def inspect_mtz_file(path) -> dict:
    import gemmi

    inventory = mtz_inventory(gemmi.read_mtz_file(str(path)))
    inventory["path"] = str(path)
    return inventory


def selected_mtz_maps(inventory, ids) -> list:
    """Return map products for ``ids`` (``factor|phase``), in requested order."""
    by_id = {}
    for product in (inventory or {}).get("maps") or ():
        mid = str(product.get("id") or "")
        if mid:
            by_id[mid] = dict(product)
    maps = []
    seen = set()
    for raw in ids or ():
        mid = str(raw)
        if not mid or mid in seen:
            continue
        seen.add(mid)
        if mid in by_id:
            maps.append(by_id[mid])
            continue
        if "|" in mid:
            factor, phase = mid.split("|", 1)
            maps.append(_mtz_map_product(factor.strip(), phase.strip(), {}, default=False))
    return maps


def mtz_crop_shape(min_pos, max_pos, step_size):
    """Interpolate voxel counts, or ``None`` to keep the full FFT unit-cell map."""
    if min_pos is None or max_pos is None:
        return None
    span = np.asarray(max_pos, dtype=float).reshape(3) - np.asarray(min_pos, dtype=float).reshape(3)
    step = np.asarray(step_size if step_size is not None else (1.0, 1.0, 1.0), dtype=float).reshape(3)
    step = np.where(np.abs(step) < 1e-15, 1.0, np.abs(step))
    counts = np.ceil(np.abs(span) / step).astype(int)
    if np.any(counts < 2):
        return None
    return tuple(int(n) for n in counts)


def _gemmi_vec3(vec):
    if vec is None:
        return np.zeros(3, dtype=float)
    to_list = getattr(vec, "tolist", None)
    if callable(to_list):
        return np.asarray(to_list(), dtype=float).reshape(3)
    return np.array([
        float(getattr(vec, "x", 0.0)),
        float(getattr(vec, "y", 0.0)),
        float(getattr(vec, "z", 0.0)),
    ], dtype=float)


def _gemmi_spacegroup_name(obj) -> str:
    sg = getattr(obj, "spacegroup", None)
    if sg is None:
        return "P1"
    hm = getattr(sg, "hm", None)
    if hm:
        text = str(hm).strip()
        if text:
            return text
    xhm = getattr(sg, "xhm", None)
    if callable(xhm):
        try:
            text = str(xhm() or "").strip()
            if text:
                return text
        except Exception:
            pass
    text = str(sg).strip()
    return text or "P1"


def _attach_gemmi_crystal(grid, density):
    """Keep the FFT unit cell so lattice wrap/resample matches the protein."""
    cell = getattr(density, "unit_cell", None)
    if cell is None:
        return grid
    try:
        a = float(getattr(cell, "a", 0.0) or 0.0)
        b = float(getattr(cell, "b", 0.0) or 0.0)
        c = float(getattr(cell, "c", 0.0) or 0.0)
        alpha = float(getattr(cell, "alpha", 90.0) or 90.0)
        beta = float(getattr(cell, "beta", 90.0) or 90.0)
        gamma = float(getattr(cell, "gamma", 90.0) or 90.0)
    except (TypeError, ValueError):
        return grid
    if min(a, b, c) < 1e-6:
        return grid
    from ..fields.crystal_cell import (
        _cell_is_orthogonal,
        attach_crystal_axis_frame,
        attach_crystal_cell,
        orthogonalization_matrix,
    )

    O = orthogonalization_matrix(a, b, c, alpha, beta, gamma)
    sg = _gemmi_spacegroup_name(density)
    attach_crystal_cell(grid, O, sg)
    if not _cell_is_orthogonal(O):
        attach_crystal_axis_frame(grid, O)
        attach_crystal_cell(grid, O, sg)
    return grid


def griddata_from_gemmi_map(density, *args, **kwargs):
    """Field covering a gemmi FFT map (full unit cell)."""
    from ..fields.field import Field

    arr = np.asarray(density, dtype=float)
    if arr.ndim != 3 or arr.size == 0:
        raise ValueError("MTZ map is empty")
    nx, ny, nz = (int(n) for n in arr.shape)
    if min(nx, ny, nz) < 2:
        raise ValueError("MTZ map has fewer than 2 samples on an axis")
    getter = getattr(density, "get_position", None)
    if callable(getter):
        origin = _gemmi_vec3(getter(0, 0, 0))
        dx = _gemmi_vec3(getter(1, 0, 0)) - origin
        dy = _gemmi_vec3(getter(0, 1, 0)) - origin
        dz = _gemmi_vec3(getter(0, 0, 1)) - origin
        step_sizes = [
            float(np.linalg.norm(dx)) or 1.0,
            float(np.linalg.norm(dy)) or 1.0,
            float(np.linalg.norm(dz)) or 1.0,
        ]
    else:
        cell = getattr(density, "unit_cell", None)
        a = float(getattr(cell, "a", nx - 1) or (nx - 1))
        b = float(getattr(cell, "b", ny - 1) or (ny - 1))
        c = float(getattr(cell, "c", nz - 1) or (nz - 1))
        origin = np.zeros(3, dtype=float)
        step_sizes = [a / float(nx - 1), b / float(ny - 1), c / float(nz - 1)]
    grid = Field(
        arr.ravel(order="C"),
        step_sizes=step_sizes,
        step_counts=(nx - 1, ny - 1, nz - 1),
        origin=origin,
        *args,
        **kwargs,
    )
    return _attach_gemmi_crystal(grid, density)


def grid_from_mtz(path, factor_column="FWT", phase_column="PHWT", sample_rate=2.6, min_pos=None, max_pos=None, step_size=None, *args, **kwargs):
    import gemmi
    from ..fields.field import Field

    mtz = gemmi.read_mtz_file(path)
    pairs = list(iter_mtz_map_column_pairs(mtz, factor_column, phase_column))
    if not pairs:
        choose_mtz_map_columns(mtz, factor_column, phase_column)
    last_error = None
    density = None
    for factor, phase in pairs:
        try:
            density = mtz.transform_f_phi_to_map(factor, phase, sample_rate=sample_rate)
            break
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            last_error = exc
            density = None
    if density is None:
        listing = ", ".join("%s (%s)" % item for item in mtz_column_entries(mtz)) or "(none)"
        detail = str(last_error).strip() if last_error is not None else "unknown error"
        raise ValueError(
            "Could not build a map from MTZ columns (%s). Columns: %s" % (detail, listing)
        ) from last_error
    crop = mtz_crop_shape(min_pos, max_pos, step_size)
    if crop is None:
        return griddata_from_gemmi_map(density, *args, **kwargs)
    min_pos = np.asarray(min_pos, dtype=float).reshape(3)
    step_size = np.asarray(step_size if step_size is not None else (1.0, 1.0, 1.0), dtype=float).reshape(3)
    m = gemmi.Mat33()
    m.fromlist([[float(step_size[0]), 0., 0.], [0., float(step_size[1]), 0.], [0., 0., float(step_size[2])]])
    transform = gemmi.Transform(m, gemmi.Vec3(*[float(v) for v in min_pos]))
    values = np.zeros(crop, dtype=np.float32)
    density.interpolate_values(values, transform)
    grid = Field(
        values.ravel(order="C"),
        step_sizes=step_size,
        step_counts=(crop[0] - 1, crop[1] - 1, crop[2] - 1),
        origin=min_pos,
        *args,
        **kwargs,
    )
    return _attach_gemmi_crystal(grid, density)
            

def grid_from_orca3d(path, *args, **kwargs):
    from ..fields.field import Field
    step_counts = None
    origin = None
    step_sizes = None
    values = []
    start = 0
    with open(path, "r") as f:
        for i,line in enumerate(f.readlines()):
            if i == 0:
                try:
                    line = line.split(":")[1].strip()
                    step_counts = np.array(line.split()).astype(int) - 1
                except:
                    start = 1
                    step_counts = None
                continue
            elif i == start:
                step_counts = np.array(line.split()).astype(int) - 1
            elif i == start + 1:
                origin = np.array(line.split()).astype(float)
            elif i == start + 2:
                step_sizes = np.array(line.split()).astype(float)
            elif i > start + 2:
                if len(line.strip()) > 0:
                    values.append(float(line))
    values = np.array(values)
    return Field(values, step_counts=step_counts, origin=origin, step_sizes=step_sizes, *args, **kwargs)