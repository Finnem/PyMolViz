from chempy import cpv
from pymol import cmd
from ..Displayable import Displayable

viewports = {}

def get_viewport_callback(position):
    global viewports
    if position[0] in viewports and position[1] in viewports[position[0]]:
        return viewports[position[0]][position[1]]
    return None


class ViewportCallback(Displayable):
    prev_v = None

    def __init__(self, names, x, y):
        self.names = names
        self.cb_name = cmd.get_unused_name('_cb')
        cmd.create(self.cb_name, 'none') # necessary if multiple instances are initialized but not loaded yet to reserve the name
        self.x = x
        self.y = y
        self.dependencies = [viewport_callback]

    @property
    def names(self):
        return self._names
    
    @names.setter
    def names(self, names):
        self._names = names

    def add_object(self, name):
        names = self.names
        names.append(name)
        self.__init__(names, self.x, self.y) 
        
    def load(self):
        cmd.load_callback(self, self.cb_name)
        if self.x not in viewports:
            viewports[self.x] = {}
        if self.y not in viewports[self.x]:
            viewports[self.x][self.y] = self
    
    def _script_string(self):
        result = []
        for name in self.names:
            result.append(f"""
if {self.x} in positions_viewport_callbacks and {self.y} in positions_viewport_callbacks[{self.x}]:
    viewport = positions_viewport_callbacks[{self.x}][{self.y}]
    viewport.add_object("{name}")
else:
    viewport = ViewportCallback(["{name}"], {self.x}, {self.y})
    if {self.x} not in positions_viewport_callbacks:
        positions_viewport_callbacks[{self.x}] = {{}}
    positions_viewport_callbacks[{self.x}][{self.y}] = viewport""")
        return "\n".join(result)
    
    def __call__(self):
        
        change = False
        for name in self.names:
            if name not in cmd.get_names('objects'):
                import threading
                threading.Thread(None, cmd.delete, args=(self.cb_name,)).start()
            else:
                change = True
        if not change:
            viewports[self.x].pop(self.y)
            if len(viewports[self.x]) == 0:
                viewports.pop(self.x)
            return
        
        v = cmd.get_view()
        if v == self.prev_v:
            return
        self.prev_v = v
        
        # inverted world to camera space => camera to world space matrix
        R_mc = [v[0:3], v[3:6], v[6:9]]
        
        # target position in camera space
        z = v[11]/50 # zoom
        width, height = [v for v in cmd.get_viewport()]
        x, y = self.x * 8.5, self.y * 8.5 # 8.5 is a magic number
        position = [ x * z * width / height, y * z, 0] # for some reason this is still not quite accurate when resizing
        
        
        # target orientation in world space
        model_space_position = cpv.transform(R_mc, position)
        
        t = v[12:15] # offset from model origin
        
        # target position in world space
        t = cpv.add(t, model_space_position)
        
        t = [n/-z for n in t]

        m = [-z, 0, 0, 0,
            0, -z, 0, 0, 
            0, 0, -z, 0, 
            *t, 1]
        
        for name in self.names:
            if name in cmd.get_names('objects'):
                cmd.set_object_ttt(name, m)
        
        cmd.set('auto_zoom', 0)

class ViewportCallbackFunction(Displayable):
    def __init__(self):
        super().__init__(name = "dummy")

    def _script_string(self):
        return """
from chempy import cpv

class ViewportCallback(object):
    prev_v = None

    def __init__(self, names, x, y):
        self.names = names
        self.cb_name = cmd.get_unused_name('_cb')
        cmd.create(self.cb_name, 'none')
        self.x = x
        self.y = y

    def load(self):
        cmd.load_callback(self, self.cb_name)
        
    def add_object(self, name):
        names = self.names
        names.append(name)
        self.__init__(names, self.x, self.y) # this is a way the __call__ function knows about the update of the attribute names, just appending name to names did not work for me

    def __call__(self):
        
        change = False
        for name in self.names:
            if name not in cmd.get_names('objects'):
                import threading
                threading.Thread(None, cmd.delete, args=(self.cb_name,)).start()
            else:
                change = True
        if not change:
            viewports[self.x].pop(self.y)
            if len(viewports[self.x]) == 0:
                viewports.pop(self.x)
            return
        
        v = cmd.get_view()
        if v == self.prev_v:
            return
        self.prev_v = v
        
        # inverted world to camera space => camera to world space matrix
        R_mc = [v[0:3], v[3:6], v[6:9]]
        
        # target position in camera space
        z = v[11]/50 # zoom
        width, height = [v for v in cmd.get_viewport()]
        x, y = self.x * 8.5, self.y * 8.5 # 8.5 is a magic number
        position = [ x * z * width / height, y * z, 0] # for some reason this is still not quite accurate when resizing
        
        
        # target orientation in world space
        model_space_position = cpv.transform(R_mc, position)
        
        t = v[12:15] # offset from model origin
        
        # target position in world space
        t = cpv.add(t, model_space_position)
        
        t = [n/-z for n in t]

        m = [-z, 0, 0, 0,
            0, -z, 0, 0, 
            0, 0, -z, 0, 
            *t, 1]
        
        for name in self.names:
            if name in cmd.get_names('objects'):
                cmd.set_object_ttt(name, m)
        
        cmd.set('auto_zoom', 0)
    """
viewport_callback = ViewportCallbackFunction()