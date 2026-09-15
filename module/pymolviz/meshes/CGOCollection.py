import numpy as np
import logging
from .Points import Points
from ..Displayable import Displayable

class CGOCollection(Displayable, list):
    """A Collection is a container for different meshes rendered as a single CGO object."""

    type_name = "CGOCollection"

    def __init__(self, CGOs: list = None, name: str = None, state: int = 1, transparency: float = 0, obj_id=None, specular: bool = True) -> None:
        self.state = state
        self.transparency = transparency
        self.specular = bool(specular)
        super().__init__(name, obj_id=obj_id)
        self.extend(CGOs if CGOs else [])

    def __setitem__(self, index, item):
        if not issubclass(type(item), Points):
            raise TypeError(
                f"Tried to add {type(item)} to a CGOCollection. "
                "CGOCollection only accepts classes deriving from Points."
            )
        super().__setitem__(index, item)

    def insert(self, index, item):
        if not issubclass(type(item), Points):
            raise TypeError(
                f"Tried to add {type(item)} to a CGOCollection. "
                "CGOCollection only accepts classes deriving from Points."
            )
        super().insert(index, item)

    def append(self, item):
        if not issubclass(type(item), Points):
            raise TypeError(
                f"Tried to add {type(item)} to a CGOCollection. "
                "CGOCollection only accepts classes deriving from Points."
            )
        super().append(item)

    def extend(self, other):
        if isinstance(other, type(self)):
            super().extend(other)
        else:
            for item in other:
                if not issubclass(type(item), Points):
                    raise TypeError(
                        f"Tried to add {type(item)} to a CGOCollection. "
                        "CGOCollection only accepts classes deriving from Points."
                    )
            super().extend(item for item in other)

    def invalidate_merged_cache(self) -> None:
        """Drop concatenated CGO tokens. Child meshes keep their own caches."""
        self._cached_merged_resolved = None
        self._child_spans = None
        self._child_serials = None

    def rebuild(self, context=None) -> None:
        self.invalidate_merged_cache()
        for child in self:
            if hasattr(child, "rebuild"):
                child.rebuild(context)

    def prepare_child_look(self) -> None:
        """Copy collection specular onto children and drop stale CGO caches."""
        spec = bool(getattr(self, "specular", True))
        for child in self:
            if bool(getattr(child, "specular", True)) != spec:
                if hasattr(child, "invalidate_cgo_cache"):
                    child.invalidate_cgo_cache()
            child.specular = spec

    def _merged_cgo_list(self) -> list:
        self.prepare_child_look()
        merged = []
        for child in self:
            merged.extend(child._create_CGO_list())
        return merged

    def _create_CGO_list(self) -> list:
        return self._merged_cgo_list()

    def _script_string(self) -> str:
        self._try_rebuild()
        self.prepare_child_look()
        look = ""
        if not bool(getattr(self, "specular", True)):
            look = '\ncmd.set("cgo_lighting", 0, "%s")' % self.name
        cgo_string_builder = []
        cgo_string_builder.append(f"""
{self.name} = [
        """)
        content = ",\n".join([",".join([str(e) for e in child._create_CGO_list()]) for child in self])
        cgo_string_builder.append(content)
        cgo_string_builder.append(f"""
            ]
cmd.load_cgo({self.name}, "{self.name}", state={self.state})
cmd.set("cgo_transparency", {self.transparency}, "{self.name}"){look}
        """)
        return "\n".join(cgo_string_builder)

    def load(self, context=None):
        if context is not None:
            self.rebuild(context)
        else:
            self._try_rebuild()
        from pymol import cmd
        from ..util.cgo import resolve_cgo_tokens
        from ..util.pymol_helpers import set_cgo_specular
        from ..util.sanitize import sanitize_pymol_string

        cgo_name = sanitize_pymol_string(self.name)
        content = resolve_cgo_tokens(self._merged_cgo_list())
        cmd.load_cgo(content, cgo_name, self.state)
        cmd.set("cgo_transparency", self.transparency, cgo_name)
        set_cgo_specular(cmd, cgo_name, bool(getattr(self, "specular", True)))

    def to_dict(self) -> dict:
        from ..serialization import displayable_to_dict
        return displayable_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict):
        from ..serialization import displayable_from_dict
        return displayable_from_dict(data)
