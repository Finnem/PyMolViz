
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

cmd.volume_ramp_new("Volume_0_volume_color_ramp", [\
    -0.2320508075688772, 0.19215686274509805,0.21176470588235294,0.5843137254901961, 0.03,\
    1.5, 0.9999231064975009,0.9976163014225298,0.7454056132256824, 0.005,\
    3.232050807568877, 0.6470588235294118,0.0,0.14901960784313725, 0.1,\
])

cmd.volume("Volume_0", "RegularData_0", "Volume_0_volume_color_ramp", )
        