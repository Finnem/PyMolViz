
from pymol.cgo import *
from pymol import cmd

        

Collection_0 = [
        
COLOR,1.0,1.0,1.0,SPHERE,4.5025,23.3395,28.4671,0.1

            ]
cmd.load_cgo(Collection_0, "Collection_0")
cmd.set("cgo_transparency", 0, "Collection_0")
        