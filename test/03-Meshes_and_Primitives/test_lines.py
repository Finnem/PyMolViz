
from pymol.cgo import *
from pymol import cmd
import numpy as np
from chempy.brick import Brick
from collections import defaultdict
positions_viewport_callbacks = defaultdict(lambda: defaultdict(lambda: ViewportCallback([],0,0)))


Lines_0 = [
        
LINEWIDTH,1,BEGIN,LINES,COLOR,1.0,0.0,0.0,VERTEX,2.749000186473004,-0.8750002617531515,-0.05722880237544803,COLOR,1.0,0.0,0.0,VERTEX,3.9760001864730037,-0.8750002617531515,-0.05622880237544803,COLOR,1.0,0.0,0.0,VERTEX,2.749,-0.975,-0.057,COLOR,1.0,0.0,0.0,VERTEX,3.976,-0.975,-0.056,COLOR,1.0,0.0,0.0,VERTEX,2.7489998135269964,-1.0749997382468486,-0.056771197624551975,COLOR,1.0,0.0,0.0,VERTEX,3.9759998135269963,-1.0749997382468486,-0.055771197624551974,END

            ]
cmd.load_cgo(Lines_0, "Lines_0", state=1)
cmd.set("cgo_transparency", 0, "Lines_0")
        

for x in positions_viewport_callbacks:
    for y in positions_viewport_callbacks[x]:
        positions_viewport_callbacks[x][y].load()
