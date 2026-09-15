"""Load and save PyMolViz objects in multiple file formats."""

from __future__ import annotations

from pathlib import Path

from .native import as_save_items, is_native_pack, load_native, save_native

FORMAT_NATIVE = "native"
FORMAT_SCRIPT = "script"

EXTENSIONS = {
    ".pmv": FORMAT_NATIVE,
    ".py": FORMAT_SCRIPT,
}

SAVE_FILTER = "PyMolViz (*.pmv);;Python script (*.py)"
OPEN_FILTER = "PyMolViz (*.pmv);;Python script (*.py);;All files (*)"
OPEN_NATIVE_FILTER = "PyMolViz (*.pmv);;All files (*)"


def format_from_path(path, format=None, *, default=FORMAT_SCRIPT) -> str:
    if format:
        text = str(format).strip().lower()
        if text in ("pmv", "pack", FORMAT_NATIVE):
            return FORMAT_NATIVE
        if text in ("py", "python", FORMAT_SCRIPT):
            return FORMAT_SCRIPT
        return text
    suffix = Path(str(path)).suffix.lower()
    return EXTENSIONS.get(suffix, default)


def complete_save_path(path, name_filter="") -> tuple:
    """Add a suffix from the chosen file-dialog filter when the path has none."""
    path = str(path or "")
    if not path:
        return path, FORMAT_NATIVE
    suffix = Path(path).suffix.lower()
    if suffix in EXTENSIONS:
        return path, EXTENSIONS[suffix]
    filt = str(name_filter or "").lower()
    if "*.py" in filt and "*.pmv" not in filt:
        return path + ".py", FORMAT_SCRIPT
    return path + ".pmv", FORMAT_NATIVE


def detect_format(path, format=None) -> str:
    if format:
        return format_from_path(path, format)
    path = str(path)
    suffix = Path(path).suffix.lower()
    if suffix in EXTENSIONS:
        return EXTENSIONS[suffix]
    if is_native_pack(path):
        return FORMAT_NATIVE
    return FORMAT_SCRIPT


def save(obj, path, format=None) -> None:
    """Write ``obj`` (one displayable or a sequence) to ``path``."""
    fmt = format_from_path(path, format, default=FORMAT_SCRIPT)
    items = as_save_items(obj)
    if fmt == FORMAT_SCRIPT:
        from ..Script import Script

        Script(items).write(path)
        return
    if fmt == FORMAT_NATIVE:
        save_native(items, path)
        return
    raise ValueError("Unknown export format %r" % fmt)


def load(path, format=None) -> list:
    """Load displayables from ``path``. Python scripts are export-only."""
    fmt = detect_format(path, format)
    if fmt == FORMAT_NATIVE:
        return load_native(path)
    if fmt == FORMAT_SCRIPT:
        from ..serialization import SerializationError

        raise SerializationError(
            "Python scripts are export-only. Run the script in PyMOL, "
            "or save a .pmv pack to load data back."
        )
    from ..serialization import SerializationError

    raise SerializationError("Unknown file format %r" % fmt)


def intern_loaded(cmd, objects) -> list:
    """Put loaded displayables into the live session and materialize them."""
    from ..Displayable import call_load
    from ..fields.field import intern_field
    from ..meshes.CGOCollection import CGOCollection
    from ..runtime.session import add as session_add
    from ..wizards.builders.field_visual import is_field_visual, persist_field_visual
    from ..wizards.builders.preview import persist_collection

    interned = []
    for obj in list(objects or []):
        kind = type(obj).__name__
        if kind == "Field":
            interned.append(intern_field(obj))
            continue
        if is_field_visual(obj):
            persist_field_visual(cmd, obj)
            interned.append(obj)
            continue
        if isinstance(obj, CGOCollection) or kind == "CGOCollection":
            persist_collection(cmd, obj, obj_id=getattr(obj, "id", None))
            interned.append(obj)
            continue
        call_load(obj, cmd)
        session_add(obj)
        interned.append(obj)
    return interned
