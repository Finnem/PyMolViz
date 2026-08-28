"""In-memory PyMOL object bindings (never persisted)."""

from __future__ import annotations


class PyMOLBinding:
    def __init__(self, model_id, pymol_name, representation="cgo", style_hash=None):
        self.model_id = str(model_id)
        self.pymol_name = str(pymol_name)
        self.representation = representation
        self.style_hash = style_hash


class BindingRegistry:
    def __init__(self):
        self._by_id = {}
        self._by_name = {}

    def get(self, model_id):
        return self._by_id.get(str(model_id))

    def get_by_name(self, pymol_name):
        return self._by_name.get(str(pymol_name))

    def put(self, binding: PyMOLBinding):
        self._by_id[binding.model_id] = binding
        self._by_name[binding.pymol_name] = binding

    def pop(self, model_id):
        binding = self._by_id.pop(str(model_id), None)
        if binding is not None:
            self._by_name.pop(binding.pymol_name, None)
        return binding

    def clear(self):
        self._by_id.clear()
        self._by_name.clear()

    def __contains__(self, model_id):
        return str(model_id) in self._by_id

    def __iter__(self):
        return iter(self._by_id.values())
