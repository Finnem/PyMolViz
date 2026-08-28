"""Script and live backends for Displayable.render."""

from __future__ import annotations


class ScriptBackend:
    """Today's ``_script_string()`` path; AtomPoint xyz is baked."""

    def visit(self, obj):
        if hasattr(obj, "_try_rebuild"):
            obj._try_rebuild()
        elif hasattr(obj, "rebuild"):
            from .context import try_context
            obj.rebuild(try_context())
        return obj._script_string()


class LiveBackend:
    def __init__(self, runtime=None):
        self.runtime = runtime

    def visit(self, obj):
        from .runtime import get_runtime
        rt = self.runtime or get_runtime()
        rt.sync(obj)
        return obj
