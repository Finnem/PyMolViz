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
        self.names.append(name)
        
    def load(self):
        from pymol import cmd
        self.cb_name = cmd.get_unused_name('_cb')
        cmd.load_callback(self, self.cb_name)
        if self.x not in viewports:
            viewports[self.x] = {}
        if self.y not in viewports[self.x]:
            viewports[self.x][self.y] = self
    
    def _script_string(self):
        result = []
        for name in self.names:
            result.append(f"""
viewport = positions_viewport_callbacks[{self.x}][{self.y}]
viewport.x,viewport.y = {self.x},{self.y}
viewport.add_object("{name}")""")
        return "\n".join(result)
    
    def __call__(self):
        from pymol import cmd
        from chempy import cpv
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
        self.x = x
        self.y = y

    def load(self):
        self.cb_name = cmd.get_unused_name('_cb')
        cmd.load_callback(self, self.cb_name)
        
    def add_object(self, name):
        self.names.append(name)
        
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