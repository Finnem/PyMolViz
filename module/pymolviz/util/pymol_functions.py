from ..Displayable import Displayable

class ViewportCallback(Displayable):
    def __init__(self):
        super().__init__(name = "dummy")

    def _script_string(self):
        return """
from chempy import cpv

class ViewportCallback(object):
    prev_v = None

    def __init__(self, name, x, y):
        self.name = name
        self.cb_name = cmd.get_unused_name('_cb')
        self.x = x
        self.y = y
        self()

    def load(self):
        cmd.load_callback(self, self.cb_name)

    def __call__(self):
        if self.name not in cmd.get_names('objects'):
            import threading
            threading.Thread(None, cmd.delete, args=(self.cb_name,)).start()
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
        
        cmd.set_object_ttt(self.name, m)
    """
viewport_callback = ViewportCallback()
