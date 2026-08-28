import numpy as np
import logging
from .Points import Points
from ..Displayable import Displayable

class CGOCollection(Displayable, list):
    """A Collection is a container for different meshes rendered as a single CGO object."""

    type_name = "CGOCollection"

    def __init__(self, CGOs: list = None, name: str = None, state: int = 1, transparency: float = 0, obj_id=None) -> None:
        self.state = state
        self.transparency = transparency
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

    def rebuild(self, context=None) -> None:
        for child in self:
            if hasattr(child, "rebuild"):
                child.rebuild(context)

    def _merged_cgo_list(self) -> list:
        merged = []
        for cgo in self:
            merged.extend(cgo._create_CGO_list())
        return merged

    def _create_CGO_list(self) -> list:
        return self._merged_cgo_list()

    def _script_string(self) -> str:
        self._try_rebuild()
        cgo_string_builder = []
        cgo_string_builder.append(f"""
{self.name} = [
        """)
        content = ",\n".join([",".join([str(e) for e in CGO._create_CGO_list()]) for CGO in self])
        cgo_string_builder.append(content)
        cgo_string_builder.append(f"""
            ]
cmd.load_cgo({self.name}, "{self.name}", state={self.state})
cmd.set("cgo_transparency", {self.transparency}, "{self.name}")
        """)
        return "\n".join(cgo_string_builder)

    def load(self, context=None):
        if context is not None:
            self.rebuild(context)
        else:
            self._try_rebuild()
        from pymol import cmd
        from ..util.cgo import resolve_cgo_tokens
        from ..util.sanitize import sanitize_pymol_string

        cgo_name = sanitize_pymol_string(self.name)
        content = resolve_cgo_tokens(self._merged_cgo_list())
        cmd.load_cgo(content, cgo_name, self.state)
        cmd.set("cgo_transparency", self.transparency, cgo_name)

    def to_dict(self) -> dict:
        from ..serialization import displayable_to_dict
        return displayable_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict):
        from ..serialization import displayable_from_dict
        return displayable_from_dict(data)
