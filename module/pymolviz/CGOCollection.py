from .CGOProxy import CGOProxy
from .meshes import Mesh

class CGOCollection(object):
    """A collection of soon-to-be named CGO objects. The CGO objects are stored
    as dictionaries mapping vertex "positions" to a list of 3D coordinates,
    "faces" to a list of vertex indices, and "normals" to a list of 3D
    coordinates. The "faces" and "normals" are optional. The "positions" are
    required. The "faces" are required if "normals" are provided.

    Attributes:
        cgo_proxies (list): A list of CGO proxies.

    """

    cgo_proxies = []

    def __init__(self, cgo_proxies : list = None) -> None:
        self.cgo_proxies = cgo_proxies if cgo_proxies else []
    
    def add_proxy(self, cgo_proxy : CGOProxy):
        """ Adds a CGO proxy to the collection.
        
        Args:
            cgo_proxy (CGOProxy): A CGO proxy object.
            
        Returns:
            None

        """

        self.cgo_proxies.append(cgo_proxy)

    def add_mesh(self, name : str, mesh : Mesh, **kwargs):
        """ Adds a mesh to the collection.

        Args:
            name (str): The name of the CGO object.
            mesh (Mesh): A Mesh object.

        Returns:
            None
        """

        self.cgo_proxies.append(CGOProxy(name, [mesh], **kwargs))

    def create_CGO_script(self, out) -> str:
        """
        Creates a CGO script from the CGO proxies.
        
        Args:
            out (str): The output file name.
            
        Returns:
            None
        """

        cgo_string_builder = ['''
from pymol.cgo import *
from pymol import cmd

        '''
        ]

        for cgo_proxy in self.cgo_proxies:
            cgo_string_builder.append(cgo_proxy.create_CGO_script())

        final_string = "\n".join(cgo_string_builder)

        with open(out, "w") as f:
            f.write(final_string)