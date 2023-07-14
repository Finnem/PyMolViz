
from pymol.cgo import *
from pymol import cmd
import numpy as np
from chempy.brick import Brick

        
cmd.pseudoatom("Labels_1", label="A", pos = [0, 0, 0], state = 1)
cmd.pseudoatom("Labels_1", label="B", pos = [0, 0, 1], state = 1)
cmd.pseudoatom("Labels_1", label="C", pos = [0, 1, 0], state = 1)
cmd.pseudoatom("Labels_1", label="D", pos = [0, 1, 1], state = 1)
cmd.pseudoatom("Labels_1", label="E", pos = [1, 0, 0], state = 1)
cmd.pseudoatom("Labels_1", label="F", pos = [1, 0, 1], state = 1)
cmd.pseudoatom("Labels_1", label="G", pos = [1, 1, 0], state = 1)
cmd.pseudoatom("Labels_1", label="H", pos = [1, 1, 1], state = 1)