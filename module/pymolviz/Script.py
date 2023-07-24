from .Displayable import Displayable

class Script(object):
    """A class wrapping one or multiple displayables to be added with a single script. Use write function to write the script to a file.

    Attributes:
        displayables (list): A list of displayables.
        

    """


    def __init__(self, displayables : list = None, *args, **kwargs) -> None:
        self.displayables = []
        if displayables:
            self.add(displayables, *args, **kwargs)
    
    def add(self, displayable, **kwargs):
        """ Adds a displayable to this script. Specifically check for dependencies and add them as well.
        
        Args:
            object: A single or multiple displayable instance.
            
        Returns:
            None

        """
        if isinstance(displayable, Displayable):
            for dependency in displayable.dependencies:
                if dependency not in self.displayables:
                    self.add(dependency, **kwargs)
            self.displayables.append(displayable)
        else:
            try:    
                for obj in displayable:
                    self.add(obj, **kwargs)
            except TypeError:
               raise TypeError(f"Object {displayable.name} is not a Displayable instance.")
            

    def load(self):
        raise NotImplementedError


    def write(self, out) -> str:
        """
        Creates a PyMol script to load all given displayables.
        
        Args:
            out (str): The output file name.
            
        Returns:
            None
        """

        with open(out, "w") as f:
            f.write(str(self))

    def __repr__(self) -> str:
        cgo_string_builder = ['''
from pymol.cgo import *
from pymol import cmd
import numpy as np
from chempy.brick import Brick

        '''
        ]

        for displayable in self.displayables:
            cgo_string_builder.append(displayable._script_string())

        final_string = "\n".join(cgo_string_builder)
        return final_string