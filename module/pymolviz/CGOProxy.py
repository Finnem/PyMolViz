import numpy as np

class CGOProxy:
    """ CGOProxy is a proxy object for CGO objects. It is used to store the
    CGO objects in the CGOCollection object. It holds the geometric information required
    to create a CGO object. A single CGO may contain multiple geometric objects.
    
    
    Attributes:
        name (str): The name of the CGO object.
        meshes (list): A list of meshes.
        transformation (np.array): A 4x4 transformation matrix.
    """



    def __init__(self, name : str, meshes : list = None, transformation : np.array = None, opacity : float = 0) -> None:
        self.name = name
        self.meshes = meshes if meshes else []
        self.transformation = transformation if transformation else []
        self.opacity = opacity

    def add_mesh(self, mesh):
        self.meshes.append(mesh)

    def create_CGO_script(self) -> str:
        """ Creates a CGO string from the mesh information. The base class assumes a triangle mesh.
        
        Returns:
            None
        """

        cgo_string_builder = []
        
        cgo_name = self.name.replace(" ", "_")

        cgo_string_builder.append(f"""
{cgo_name} = [
        """)

        content = ",\n".join([",".join([str(e) for e in mesh.create_CGO()]) for mesh in self.meshes])
        cgo_string_builder.append(content)

        # ending
        cgo_string_builder.append(f"""
            ]
cmd.load_cgo({cgo_name}, "{self.name}")
cmd.set("cgo_transparency", {self.opacity}, "{self.name}")
        """)

        return "\n".join(cgo_string_builder)

    def as_CGO(self):
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

