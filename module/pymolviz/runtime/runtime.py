"""Materialize / sync / remove / reconcile displayables in PyMOL."""

from __future__ import annotations

from ..points import PointUnresolvedError, has_dynamic_sources
from ..serialization import style_hash
from ..util.pymol_helpers import load_cgo_no_zoom, replace_cgo_no_zoom, set_cgo_specular, set_cgo_transparency
from ..util.sanitize import sanitize_pymol_string
from .bindings import BindingRegistry, PyMOLBinding
from .context import ResolveContext
from .renderer import FIELD_VISUAL_TYPES, renders_cgo, resolved_cgo_tokens

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

    def _apply_object_look(self, obj, name):
        alpha = 1.0 - max(0.0, min(1.0, _scalar_transparency(obj)))
        set_cgo_transparency(self.cmd, name, alpha)
        set_cgo_specular(self.cmd, name, bool(getattr(obj, "specular", True)))

    def _load(self, obj, name, replace=False, rebuild=True):
        if not renders_cgo(obj):
            return
        context = self._context() if rebuild else None
        try:
            tokens = resolved_cgo_tokens(obj, context)
        except PointUnresolvedError:
            tokens = resolved_cgo_tokens(obj, None)
        state = int(getattr(obj, "state", 1) or 1)
        if replace:
            replace_cgo_no_zoom(self.cmd, tokens, name, state)
        else:
            load_cgo_no_zoom(self.cmd, tokens, name, state)
        self._apply_object_look(obj, name)

    def _load_field_visual(self, obj):
        """Load Volume / IsoSurface / IsoMesh via native PyMOL cmds, not CGO."""
        from ..Displayable import call_load

        call_load(obj, self.cmd)

    def _bind_field_visual(self, obj):
        name = getattr(obj, "_name", None) or binding_name(obj)
        kind = type(obj).__name__
        representation = "volume" if kind in ("Volume", "IsoVolume") else kind.lower()
        self.bindings.put(
            PyMOLBinding(obj.id, str(name), representation, style_hash=style_hash(obj))
        )
        return str(name)

    def materialize(self, obj, rebuild=True):
        name = binding_name(obj)
        if not renders_cgo(obj):
            if type(obj).__name__ in FIELD_VISUAL_TYPES:
                from .presence import pause_presence_sync

                with pause_presence_sync():
                    self._load_field_visual(obj)
                return self._bind_field_visual(obj)
            return name
        self._load(obj, name, replace=False, rebuild=rebuild)
        binding = PyMOLBinding(obj.id, name, "cgo", style_hash=style_hash(obj))
        self.bindings.put(binding)
        return name

    def sync(self, obj):
        if not renders_cgo(obj):
            return self.materialize(obj)
        binding = self.bindings.get(obj.id)
        if binding is None:
            return self.materialize(obj)
        self._load(obj, binding.pymol_name, replace=True)
        binding.style_hash = style_hash(obj)
        return binding.pymol_name

    def replace_cgo(self, obj):
        """Reload CGO tokens without resolving or remeshing."""
        if not renders_cgo(obj):
            return binding_name(obj)
        binding = self.bindings.get(obj.id)
        if binding is None:
            return self.materialize(obj)
        tokens = list(resolved_cgo_tokens(obj, None))
        state = int(getattr(obj, "state", 1) or 1)
        replace_cgo_no_zoom(self.cmd, tokens, binding.pymol_name, state)
        self._apply_object_look(obj, binding.pymol_name)
        return binding.pymol_name

    def remove(self, obj):
        from .presence import pause_presence_sync

        binding = self.bindings.pop(obj.id)
        names = set()
        if binding is not None:
            names.add(binding.pymol_name)
        names.add(binding_name(obj))
        stored = getattr(obj, "_name", None)
        if stored:
            names.add(str(stored))
        with pause_presence_sync():
            for name in names:
                if not name:
                    continue
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
            if not renders_cgo(obj):
                if type(obj).__name__ in FIELD_VISUAL_TYPES:
                    name = getattr(obj, "_name", None) or binding_name(obj)
                    if name not in existing:
                        self._load_field_visual(obj)
                    self._bind_field_visual(obj)
                continue
            name = binding_name(obj)
            if name in existing:
                self.bindings.put(PyMOLBinding(obj.id, name, "cgo", style_hash=style_hash(obj)))
                if has_dynamic_sources(obj):
                    self.sync(obj)
            else:
                self.materialize(obj)
