import logging
from collections import defaultdict

_pmv_default_name_counter = defaultdict(int)
class Displayable():
    def __init__(self, name = None, dependencies = None):
        self._name = name
        self.dependencies = dependencies if dependencies else []

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
        self._name = value

    def to_script(self):
        from .Script import Script
        return Script([self])

    def write(self, filename):
        self.to_script().write(filename)