from motion import lateral_motionLST_and_longitudinal_motionVK

# initial state

p0 = 2.0  # current lateral position [m]
dp0 = 0.0  # current lateral speed [m/s]
ddp0 = 0.0  # current lateral acceleration [m/s]
s0 = 0.0  # current course position
ds0 = 10.0 / 3.6  # current speed [m/s]

frenet_p = lateral_motionLST_and_longitudinal_motionVK(p0,dp0,ddp0,s0,ds0)

