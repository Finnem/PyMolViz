import numpy as np
import logging
from .meshes.Mesh import Mesh

_pmv_collection_counter = 0
class Collection:
    """ A Collection is a container for different meshes. All meshes in a collection are rendered as a single CGO object.
    
    
    Attributes:
        name (str): The name of the CGO object.
        meshes (list): A list of meshes.
        transformation (np.array): A 4x4 transformation matrix.
    """



    def __init__(self, name : str = None, meshes : list = None, transformation : np.array = None, opacity : float = 0) -> None:
        if name:
            self.name = name
        else:
            global _pmv_collection_counter
            logging.warning("No name provided for Collection. Using default name: Collection_{}. It is highly recommended to provide meaningful names.".format(_pmv_collection_counter))
            self.name = "Collection_{}".format(_pmv_collection_counter)
            _pmv_collection_counter += 1

        self.meshes = meshes if meshes else []
        self.transformation = transformation if transformation else []
        self.opacity = opacity



    def add(self, mesh : Mesh):
        self.meshes.append(mesh)


    def to_script(self):
        """ Creates a script from the collection.
        
        Returns:
            Script: A script object.
        """
        from .Script import Script
        return Script([self])


    def load(self):
        """ Loads the collection into PyMOL.
        
        Returns:
            None
        """

        from pymol import cmd
        cmd.load_cgo(self._create_CGO(), self.name)
        cmd.set("cgo_transparency", self.opacity, self.name)

    
    def _create_CGO_script(self) -> str:
        """ Creates a CGO string from the meshes informations.
        
        Returns:
            None
        """

        cgo_string_builder = []
        
        cgo_name = self.name.replace(" ", "_")

        cgo_string_builder.append(f"""
{cgo_name} = [
        """)

        content = ",\n".join([",".join([str(e) for e in mesh._create_CGO()]) for mesh in self.meshes])
        cgo_string_builder.append(content)

        # ending
        cgo_string_builder.append(f"""
            ]
cmd.load_cgo({cgo_name}, "{self.name}")
cmd.set("cgo_transparency", {self.opacity}, "{self.name}")
        """)

        return "\n".join(cgo_string_builder)


    def _create_CGO(self):
        """ Creates a CGO object from the CGOProxy object.
        
        Returns:
            CGO: A CGO object.
        """

        # combine all meshes' cgo_lists into a single cgo_list
        combined_list = []
        for mesh in self.meshes:
            combined_list.extend(mesh.create_CGO())

        # convert cgo constant strings to actual constants
        from pymol.cgo import BEGIN, END, TRIANGLES, COLOR, VERTEX, NORMAL, SPHERE, POINTS, LINES, LINEWIDTH
        string_mapping = {
            "BEGIN" : BEGIN,
            "END" : END,
            "TRIANGLES" : TRIANGLES,
            "COLOR" : COLOR,
            "VERTEX" : VERTEX,
            "NORMAL" : NORMAL,
            "SPHERE" : SPHERE,
            "POINTS" : POINTS,
            "LINES" : LINES,
            "LINEWIDTH" : LINEWIDTH,
        }
        combined_list = [string_mapping.get(item, item) for item in combined_list]

        # create the CGO object
        return combined_list

    

