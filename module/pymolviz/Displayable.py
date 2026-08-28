import logging
import uuid
from collections import defaultdict

_pmv_default_name_counter = defaultdict(int)


class Displayable():
    """
    Base class for all objects that can be displayed in PyMol.

    Attributes:
        id (str): Stable identifier (uuid4 hex), independent of name.
        name (str): Optional. User-facing / PyMOL object name.
        dependencies (list): Optional. Objects this displayable depends on.
    """

    def __init__(self, name=None, dependencies=None, obj_id=None, *args, **kwargs):
        obj_id = obj_id or kwargs.pop("id", None) or kwargs.pop("obj_id", None)
        self._id = obj_id or uuid.uuid4().hex
        self.name = name
        self.dependencies = dependencies if dependencies else []
        super().__init__(*args, **kwargs)

    @property
    def id(self) -> str:
        return self._id

    @id.setter
    def id(self, value: str) -> None:
        self._id = str(value)

    def _script_string(self):
        raise NotImplementedError

    def rebuild(self, context=None) -> None:
        """Resolve semantic inputs and refresh baked geometry."""

    def _try_rebuild(self):
        try:
            from .runtime.context import try_context
            self.rebuild(try_context())
        except Exception:
            try:
                self.rebuild(None)
            except Exception:
                pass

    def render(self, backend) -> None:
        backend.visit(self)

    @property
    def name(self):
        global _pmv_default_name_counter

        if self._name is None:
            class_name = type(self).__name__
            new_name = f"{class_name}_{_pmv_default_name_counter[class_name]}"
            _pmv_default_name_counter[class_name] += 1
            logging.warning(
                "No name provided for %s. Using default name: %s. "
                "It is highly recommended to provide meaningful names.",
                class_name,
                new_name,
            )
            self._name = new_name
        return self._name

    @name.setter
    def name(self, value):
        if value is None:
            self._name = None
            return
        new_name = str(value).replace(" ", "_")
        new_name = str(new_name).replace(".", "_")
        new_name = str(new_name).replace(":", "_")
        new_name = str(new_name).replace("|", "_")
        new_name = str(new_name).replace("&", "_")
        new_name = str(new_name).replace("?", "_")
        new_name = str(new_name).replace("!", "_")
        new_name = str(new_name).replace("+", "_")
        new_name = str(new_name).replace("-", "_")
        if new_name[0].isdigit():
            new_name = "_" + new_name
        if any(s in new_name for s in ["(", ")", "[", "]"]):
            logging.warning(
                "Name %s contains parentheses. This may cause issues with PyMol. "
                "Consider changing the name.",
                new_name,
            )
        self._name = new_name.replace("(", "_").replace(")", "_").replace("[", "_").replace("]", "_")

    def to_script(self):
        from .Script import Script
        return Script([self])

    def write(self, filename):
        self.to_script().write(filename)
