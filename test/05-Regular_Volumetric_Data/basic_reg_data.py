
from pymol.cgo import *
from pymol import cmd
import numpy as np
from chempy.brick import Brick

        

RegularData_0_data = np.array([[[0,1],
  [1,2]],

 [[1,2],
  [2,3]]])
RegularData_0 = Brick.from_numpy(RegularData_0_data, [1.,1.,1.], origin=[0,0,0])
cmd.load_brick(RegularData_0, "RegularData_0")
