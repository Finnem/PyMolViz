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
    
    def load(self, cmd=None):
        if cmd is None:
            from pymol import cmd
        from ..Displayable import call_load
        from ..wizards.builders.field_visual import bind_iso_color_ramp

        bind_iso_color_ramp(self)
        from ..wizards.builders.field_visual import load_geometry_map, sync_visual_grid_from_field

        sync_visual_grid_from_field(self, cmd)
        map_name, _rebuilt = load_geometry_map(cmd, self.grid_data, getattr(self, "clip_aabb", None))
        if not map_name:
            return
        cmd.isomesh(self.name, map_name, level=self.level, selection=self.selection, carve=self.carve)
        if issubclass(type(self.color), ColorRamp):
            call_load(self.color, cmd)
            cmd.color(self.color.name, self.name)
        else:
            cmd.set_color(self.name + "_color", self.color)
            cmd.color(self.name + "_color", self.name)
        cmd.set("transparency", self.transparency, self.name)
            
        