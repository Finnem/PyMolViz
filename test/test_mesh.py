
from pymol.cgo import *
from pymol import cmd

        

mesh_points = [
        
COLOR,1.0,0.0,0.0,SPHERE,0.0,0.0,0.0,0.1,COLOR,0.0,0.5019607843137255,0.0,SPHERE,0.0,0.0,1.0,0.1,COLOR,0.0,0.0,1.0,SPHERE,0.0,1.0,0.0,0.1

            ]
cmd.load_cgo(mesh_points, "mesh_points")
cmd.set("cgo_transparency", 0, "mesh_points")
        

mesh = [
        
BEGIN,TRIANGLES,COLOR,1.0,0.0,0.0,VERTEX,0.0,0.0,0.0,NORMAL,0,0,0,COLOR,0.0,0.5019607843137255,0.0,VERTEX,0.0,0.0,1.0,NORMAL,0,0,0,COLOR,0.0,0.0,1.0,VERTEX,0.0,1.0,0.0,NORMAL,0,0,0,END

            ]
cmd.load_cgo(mesh, "mesh")
cmd.set("cgo_transparency", 0, "mesh")
        