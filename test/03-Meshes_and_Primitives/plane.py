
from pymol.cgo import *
from pymol import cmd
import numpy as np
from chempy.brick import Brick

        

Plane_0 = [
        
BEGIN,TRIANGLES,COLOR,1.0,0.0,0.0,VERTEX,2.5,-2.5,0.0,NORMAL,0.0,0.0,1.0,COLOR,1.0,0.0,0.0,VERTEX,2.5,2.5,0.0,NORMAL,0.0,0.0,1.0,COLOR,1.0,0.0,0.0,VERTEX,-2.5,-2.5,0.0,NORMAL,0.0,0.0,1.0,COLOR,1.0,0.0,0.0,VERTEX,2.5,2.5,0.0,NORMAL,0.0,0.0,1.0,COLOR,1.0,0.0,0.0,VERTEX,-2.5,2.5,0.0,NORMAL,0.0,0.0,1.0,COLOR,1.0,0.0,0.0,VERTEX,-2.5,-2.5,0.0,NORMAL,0.0,0.0,1.0,END

            ]
cmd.load_cgo(Plane_0, "Plane_0", state=1)
cmd.set("cgo_transparency", 0, "Plane_0")
        