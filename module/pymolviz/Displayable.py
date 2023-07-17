import logging
from collections import defaultdict

_pmv_default_name_counter = defaultdict(int)
class Displayable():
    def __init__(self, name = None, dependencies = None, *args, **kwargs):
        self.name = name
        self.dependencies = dependencies if dependencies else []
        super().__init__(*args, **kwargs) # multiple inheritance support

    def _script_string(self):
        raise NotImplementedError

    @property
    def name(self):
        global _pmv_default_name_counter

        if self._name is None:
            class_name = type(self).__name__
            new_name = f"{class_name}_{_pmv_default_name_counter[class_name]}"
            _pmv_default_name_counter[class_name] += 1
            logging.warning(f"No name provided for {class_name}. Using default name: {new_name}. It is highly recommended to provide meaningful names.")
            self._name = new_name
        return self._name

    @name.setter
    def name(self, value):
        if value is None:
            self._name = None
            return
        new_name = str(value).replace(" ", "_")
        if any(s in new_name for s in ["(", ")", "[", "]"]):
            logging.warning(f"Name {new_name} contains parentheses. This may cause issues with PyMol. Consider changing the name.")
        self._name = new_name.replace("(", "_").replace(")", "_").replace("[", "_").replace("]", "_")
        self._name = str(value).replace(" ", "_")

    def to_script(self):
        from .Script import Script
        return Script([self])

    def write(self, filename):
        self.to_script().write(filename)