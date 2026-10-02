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


def object_sele(name):
    """``object "name"`` so maps are not parsed as atom selections."""
    text = str(name or "").strip()
    if not text:
        return ""
    if text.startswith("(") and text.endswith(")"):
        return text
    if text.lower().startswith("object "):
        return text
    return "object %s" % quote_pymol_name(text)


def parse_object_sele(expr):
    """Bare object name from ``object "name"``, or the original string."""
    text = str(expr or "").strip()
    if text.lower().startswith("object "):
        text = text[7:].strip()
        if len(text) >= 2 and text[0] == text[-1] == '"':
            return text[1:-1]
    return text
