from .Collection import Collection
from .meshes import Mesh
import logging
class Script(object):
    """A class wrapping one or multiple collections to be added with a single script. Use write function to write the script to a file.

    Attributes:
        collections (list): A list of collections.

    """


    def __init__(self, collections : list = None, *args, **kwargs) -> None:
        self.collections = []
        if collections:
            self.add(collections, *args, **kwargs)
    
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
                collection = Collection([object], name, **kwargs)
                self.collections.append(collection)
                return
        elif issubclass(type(object), Collection):
            self.collections.append(object)
            return
        elif type(object) == list:
            mesh_count = 0
            for o in object:
                if issubclass(type(o), Mesh):
                    if name is None:
                        collection = Collection(meshes = [o], **kwargs)
                        self.collections.append(collection)
                        logging.warning("No name provided for mesh when creating script. Try passing a name(s) to the add function or a dictionary mapping names to the objects.")
                    if type(name) == str:
                        collection = Collection([o], name, **kwargs)
                        self.collections.append(collection)
                    elif type(name) == list:
                        collection = Collection([o], name[mesh_count], **kwargs)
                        self.collections.append(collection)
                    mesh_count += 1
                elif issubclass(type(o), list):
                    if name is None:
                        collection = Collection(meshes = o, **kwargs)
                        self.collections.append(collection)
                        logging.warning("No name provided for mesh when creating script. Try passing a name(s) to the add function or a dictionary mapping names to the objects.")
                    if type(name) == str:
                        collection = Collection(o, name, **kwargs)
                        self.collections.append(collection)
                    elif type(name) == list:
                        if not all([issubclass(type(mesh), Mesh) for mesh in o]):
                            raise TypeError("Passed sublist contains object of type {}. Only Meshes are allowed as part of sublists.".format(type(o)))
                        collection = Collection(o, name[mesh_count], **kwargs)
                        self.collections.append(collection)
                    mesh_count += 1
                elif issubclass(type(o), Collection):
                    self.collections.append(o)
                else:
                    raise TypeError("Passed list contains object of type {}. Only Meshes, Collections and lists of Meshes are allowed.".format(type(o)))
            return
        elif type(object) == dict:
            for name, o in object.items():
                if issubclass(type(o), Mesh):
                    collection = Collection([o], name, **kwargs)
                    self.collections.append(collection)
                elif issubclass(type(o), list):
                    if not all([issubclass(type(mesh), Mesh) for mesh in o]):
                        raise TypeError("Passed sublist contains object of type {}. Only Meshes are allowed as part of sublists.".format(type(o)))
                    collection = Collection(o, name, **kwargs)
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


    def load(self):
        """
        
        Loads all collections in this script into the current PyMOL session.
        
        """
        name_counter = {}
        for collection in self.collections:
            if collection.name in name_counter:
                name_counter[collection.name] += 1
            else:
                name_counter[collection.name] = 1
        for name, counter in name_counter.items():
            if counter > 1:
                logging.warning("Multiple collections with the name {} found. Adding to multiple states...".format(name))
            cur_counter = 0
            for collection in self.collections:
                if collection.name == name:
                    if collection.state is None:
                        collection.state = cur_counter
                        cur_counter += 1
                    else:
                        ValueError("State only set for some collections with the name {}. State must be set for all collections with the same name, or for none.".format(name))
        
        for collection in self.collections:
            collection.load()

    def write(self, out) -> str:
        """
        Creates a CGO script from the CGO proxies.
        
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

        '''
        ]

        for collection in self.collections:
            cgo_string_builder.append(collection._create_CGO_script())

        final_string = "\n".join(cgo_string_builder)
        return final_string