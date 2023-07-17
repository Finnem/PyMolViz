import numpy as np
import logging
from .Points import Points
from ..Displayable import Displayable

class CGOCollection(Displayable, list):
    """ A Collection is a container for different meshes. All meshes in a collection are rendered as a single CGO object.
    
    
    Attributes:
        name (str): The name of the CGO object.
        meshes (list): A list of meshes.
        transformation (np.array): A 4x4 transformation matrix.
    """



    def __init__(self, CGOs : list = None, name : str = None, state : int = 1, transparency : float = 0) -> None:
        self.state = state
        self.transparency = transparency
        
        super().__init__(name)
        self.extend(CGOs if CGOs else [])

    def __setitem__(self, index, item):
        if not issubclass(type(item), Points):
            raise TypeError(f"Tried to add {type(item)} to a CGOCollection. CGOCollection only accepts classes deriving from Points. You might consider using a Group instead.")
        super().__setitem__(index, item)
    
    def insert(self, index, item):
        if not issubclass(type(item), Points):
            raise TypeError(f"Tried to add {type(item)} to a CGOCollection. CGOCollection only accepts classes deriving from Points. You might consider using a Group instead.")
        super().insert(index, item)

    def append(self, item):
        if not issubclass(type(item), Points):
            raise TypeError(f"Tried to add {type(item)} to a CGOCollection. CGOCollection only accepts classes deriving from Points. You might consider using a Group instead.")
        super().append(item)

    def extend(self, other):
        if isinstance(other, type(self)):
            super().extend(other)
        else:
            for item in other:
                if not issubclass(type(item), Points):
                    raise TypeError(f"Tried to add {type(item)} to a CGOCollection. CGOCollection only accepts classes deriving from Points. You might consider using a Group instead.")
            super().extend(item for item in other)



    
    def _script_string(self) -> str:
        """ Creates a CGO string from the meshes informations.
        
        Returns:
            None
        """

        cgo_string_builder = []
        
        cgo_string_builder.append(f"""
{self.name} = [
        """)
        content = ",\n".join([",".join([str(e) for e in CGO._create_CGO_list()]) for CGO in self])
        cgo_string_builder.append(content)

        # ending
        cgo_string_builder.append(f"""
            ]
cmd.load_cgo({self.name}, "{self.name}", state={self.state})
cmd.set("cgo_transparency", {self.transparency}, "{self.name}")
        """)

        return "\n".join(cgo_string_builder)



