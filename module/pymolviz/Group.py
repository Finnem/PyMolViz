from .Displayable import Displayable

class Group(Displayable, list):
    """ A group can be used to aggregate multiple displayables into a PyMOL group.
    
    
    Attributes:
        displayables (list): A list of displayables.
        name (str): Optional. Defaults to None. The name of the object.
        state (int): Optional. Defaults to 1. The state of the object.

    """



    def __init__(self, displayables : list = None, name : str = None, state : int = 1) -> None:
        self.state = state
        
        super().__init__(name, displayables if displayables else [])
        self.extend(displayables if displayables else [])

    def __setitem__(self, index, item):
        if not item in self.dependencies:
            self.dependencies.append(item)
        super().__setitem__(index, item)
    
    def insert(self, index, item):
        if not item in self.dependencies:
            self.dependencies.append(item)
        super().insert(index, item)

    def append(self, item):
        if not item in self.dependencies:
            self.dependencies.append(item)
        super().append(item)

    def extend(self, other):
        for item in other:
            if not item in self.dependencies:
                self.dependencies.append(item)
        super().extend(item for item in other)

    
    def _script_string(self) -> str:
        """ Creates a CGO string from the meshes informations.
        
        Returns:
            None
        """

        cgo_string_builder = []
        
        cgo_string_builder.append(f"""
{self.name} = cmd.group("{self.name}")
cmd.group("{self.name}", "open")
""")
        content = ",\n".join([f"""cmd.group("{self.name}", "{item.name}", "add")""" for item in self])
        cgo_string_builder.append(content)
        # ending

        return "\n".join(cgo_string_builder)
