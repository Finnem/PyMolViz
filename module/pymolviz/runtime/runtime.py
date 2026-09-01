"""Materialize / sync / remove / reconcile displayables in PyMOL."""

from __future__ import annotations

from ..points import PointUnresolvedError, has_dynamic_sources
from ..serialization import style_hash
from ..util.pymol_helpers import load_cgo_no_zoom, replace_cgo_no_zoom, set_cgo_transparency
from ..util.sanitize import sanitize_pymol_string
from .bindings import BindingRegistry, PyMOLBinding
from .context import ResolveContext
from .renderer import resolved_cgo_tokens

_DEFAULT = None


def get_runtime(cmd=None):
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = PyMOLRuntime(cmd)
    elif cmd is not None:
        _DEFAULT.cmd = cmd
    return _DEFAULT


def reset_runtime():
    global _DEFAULT
    _DEFAULT = None


def binding_name(obj) -> str:
    stored = getattr(obj, "_name", None)
    if stored:
        return sanitize_pymol_string(obj.name)
    kind = type(obj).__name__.lower()
    return "_pmv_%s_%s" % (kind, obj.id[:8])


def _scalar_transparency(obj) -> float:
    t = getattr(obj, "transparency", 0)
    try:
        t[0]
        return float(min(t))
    except (TypeError, IndexError):
        try:
            return float(t)
        except (TypeError, ValueError):
            return 0.0


class PyMOLRuntime:
    def __init__(self, cmd=None):
        if cmd is None:
            from pymol import cmd as _cmd
            cmd = _cmd
        self.cmd = cmd
        self.bindings = BindingRegistry()

    def _context(self):
        try:
            state = int(self.cmd.get_state())
        except Exception:
            state = 1
        return ResolveContext(self.cmd, state)

    def _apply_transparency(self, obj, name):
        alpha = 1.0 - max(0.0, min(1.0, _scalar_transparency(obj)))
        set_cgo_transparency(self.cmd, name, alpha)

    def _load(self, obj, name, replace=False):
        context = self._context()
        try:
            tokens = resolved_cgo_tokens(obj, context)
        except PointUnresolvedError:
            tokens = resolved_cgo_tokens(obj, None)
        state = int(getattr(obj, "state", 1) or 1)
        if replace:
            replace_cgo_no_zoom(self.cmd, tokens, name, state)
        else:
            load_cgo_no_zoom(self.cmd, tokens, name, state)
        self._apply_transparency(obj, name)

    def materialize(self, obj):
        name = binding_name(obj)
        self._load(obj, name, replace=False)
        binding = PyMOLBinding(obj.id, name, "cgo", style_hash=style_hash(obj))
        self.bindings.put(binding)
        return name

    def sync(self, obj):
        binding = self.bindings.get(obj.id)
        if binding is None:
            return self.materialize(obj)
        self._load(obj, binding.pymol_name, replace=True)
        binding.style_hash = style_hash(obj)
        return binding.pymol_name

    def replace_cgo(self, obj):
        """Reload CGO tokens without resolving or remeshing."""
        binding = self.bindings.get(obj.id)
        if binding is None:
            return self.materialize(obj)
        tokens = list(resolved_cgo_tokens(obj, None))
        state = int(getattr(obj, "state", 1) or 1)
        replace_cgo_no_zoom(self.cmd, tokens, binding.pymol_name, state)
        self._apply_transparency(obj, binding.pymol_name)
        return binding.pymol_name

    def remove(self, obj):
        binding = self.bindings.pop(obj.id)
        name = binding.pymol_name if binding is not None else binding_name(obj)
        try:
            self.cmd.delete(name)
        except Exception:
            pass

    def reconcile(self, objects):
        try:
            existing = set(self.cmd.get_names("objects"))
        except Exception:
            existing = set()
        for obj in objects:
            name = binding_name(obj)
            if name in existing:
                self.bindings.put(PyMOLBinding(obj.id, name, "cgo", style_hash=style_hash(obj)))
                if has_dynamic_sources(obj):
                    self.sync(obj)
            else:
                self.materialize(obj)
