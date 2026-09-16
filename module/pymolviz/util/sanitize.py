"""PyMOL object names and selection-safe quoting."""

_OBJECT_NAME_UNSAFE = " .:|&?!+-()[],"


def sanitize_pymol_string(string):
    if string is None:
        return None
    s = str(string)
    for ch in _OBJECT_NAME_UNSAFE:
        s = s.replace(ch, "_")
    if s[:1].isdigit():
        s = "_" + s
    return s


def quote_pymol_name(name):
    """Wrap a PyMOL object name for use in selection algebra (hyphens preserved)."""
    s = str(name).strip()
    if s.startswith("(") and s.endswith(")"):
        return s
    s = s.replace('"', "")
    return '"%s"' % s
