from .ColorRamp import ColorRamp
from .IsoSurface import IsoSurface

class IsoMesh(IsoSurface):

    def _script_string(self):
        """ Creates a pymol script to create an isomesh representation of the given regular data.
        
        Returns:
            str: The script.
        """

        optional_arguments = []
        if not(self.selection is None):
            optional_arguments.append(f"selection = \"{self.selection}\"")
        if self.carve is not None:
            optional_arguments.append(f"carve = {self.carve}")

        if issubclass(type(self.color), ColorRamp):
            color_string = f'cmd.color("{self.color.name}", "{self.name}")'
        else:
            color_string = f'''cmd.set_color("{self.name}_color", {self.color})
cmd.color("{self.name}_color", "{self.name}")
'''

        
        result = f"""
cmd.isomesh("{self.name}", "{self.grid_data.name}", {self.level}, {" , ".join(optional_arguments)}{"," if len(optional_arguments) > 0 else ""})
{color_string}
cmd.set("transparency", {self.transparency}, "{self.name}")
        """
        
        return result