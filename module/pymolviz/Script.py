from .Collection import Collection
from .meshes import Mesh
import logging
class Script(object):
    """A collection of soon-to-be named CGO objects. The CGO objects are stored
    as dictionaries mapping vertex "positions" to a list of 3D coordinates,
    "faces" to a list of vertex indices, and "normals" to a list of 3D
    coordinates. The "faces" and "normals" are optional. The "positions" are
    required. The "faces" are required if "normals" are provided.

    Attributes:
        cgo_proxies (list): A list of CGO proxies.

    """

    collections = []

    def __init__(self, cgo_proxies : list = None) -> None:
        self.cgo_proxies = cgo_proxies if cgo_proxies else []
    
    def add(self, object, name = None, **kwargs):
        """ Adds a collection to this script.
        
        Args:
            object: A Mesh, Collection of Meshes, list of meshes and collections or a dict mapping names to meshes and collections.
                When passing a dictionary, the names provided will overwrite names previously assigned to the objects.
            name (str): The name of the object. Should only be provided if object is a Mesh or a list of meshes.
            
        Returns:
            None

        """
        if issubclass(type(object), Mesh):
            if type(name) == str:
                collection = Collection(name, [object], **kwargs)
                self.collections.append(collection)
                return
        elif issubclass(type(object), Collection):
            self.collections.append(object)
            return
        elif type(object) == list:
            mesh_count = 0
            for o in object:
                if issubclass(type(o), Mesh):
                    mesh_count += 1
                    if name is None:
                        collection = Collection(meshes = [o], **kwargs)
                        self.collections.append(collection)
                        logging.warning("No name provided for mesh when creating script. Try passing a name(s) to the add function or a dictionary mapping names to the objects.")
                    if type(name) == str:
                        collection = Collection(name, [o], **kwargs)
                        self.collections.append(collection)
                    elif type(name) == list:
                        collection = Collection(name[mesh_count], [o], **kwargs)
                        self.collections.append(collection)
                elif issubclass(type(o), list):
                    mesh_count += 1
                    if name is None:
                        collection = Collection(meshes = o, **kwargs)
                        self.collections.append(collection)
                        logging.warning("No name provided for mesh when creating script. Try passing a name(s) to the add function or a dictionary mapping names to the objects.")
                    if type(name) == str:
                        collection = Collection(name, o, **kwargs)
                        self.collections.append(collection)
                    elif type(name) == list:
                        if not all([issubclass(type(mesh), Mesh) for mesh in o]):
                            raise TypeError("Passed sublist contains object of type {}. Only Meshes are allowed as part of sublists.".format(type(o)))
                        collection = Collection(name[mesh_count], o, **kwargs)
                        self.collections.append(collection)
                elif issubclass(type(o), Collection):
                    self.collections.append(o)
                else:
                    raise TypeError("Passed list contains object of type {}. Only Meshes, Collections and lists of Meshes are allowed.".format(type(o)))
            return
        elif type(object) == dict:
            for name, o in object.items():
                if issubclass(type(o), Mesh):
                    collection = Collection(name, [o], **kwargs)
                    self.collections.append(collection)
                elif issubclass(type(o), list):
                    if not all([issubclass(type(mesh), Mesh) for mesh in o]):
                        raise TypeError("Passed sublist contains object of type {}. Only Meshes are allowed as part of sublists.".format(type(o)))
                    collection = Collection(name, o, **kwargs)
                    self.collections.append(collection)
                elif issubclass(type(o), Collection):
                    o.name = name
                    self.collections.append(o)
                else:
                    raise TypeError("Passed dictionary contains object of type {}. Only Meshes, Collections, and lists of Meshes are allowed.".format(type(o)))
            return
        else:
            raise TypeError("Tried to add an object of type {} to a script. Only Meshes, Collections, lists of Meshes and lists of Collections are allowed.".format(type(object)))

        self.collections.append(collection)


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

        for collection in self.collections:
            cgo_string_builder.append(collection._create_CGO_script())

        final_string = "\n".join(cgo_string_builder)

        with open(out, "w") as f:
            f.write(final_string)