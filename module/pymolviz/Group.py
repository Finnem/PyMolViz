from .Displayable import Displayable

class Group(Displayable, list):
    """ A group can be used to aggregate multiple displayables into a PyMOL group.
    
    
    Attributes:
        name (str): The name of the CGO object.
        meshes (list): A list of meshes.
        transformation (np.array): A 4x4 transformation matrix.
    """



    def __init__(self, displayables : list = None, name : str = None, state : int = 1, transparency : float = 0) -> None:
        self.state = state
        self.transparency = transparency
        
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
