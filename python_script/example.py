import numpy as np

import mag_damping as mag

# length of cell
Lx = 0.154
# breadth of cell
Ly = 0.094


# declearing fluid properties

gal = mag.metal(
    name="GaInSn",
    density=6345,
    kinematic_viscosity=3.48e-7,
    electric_conductivity=3.31e6,
)


# variable to store dimensions
dim = mag.geometry(length=Lx, width=Ly, metal_height=0.049)

# external magnetic field strength
B0 = mag.magField(Bz=0.05)  

Ha = mag.Ha_number(B0.Bz, dim.length, gal.electric_conductivity, 
                   gal.density, gal.kinematic_viscosity)

# electrical boundary condition
BC_cond = mag.elec_bcond(cond='bc1')


# calculated wave mode
wave = mag.wavemode(m=1, n=1)

mag.parameter_check(gal, dim, BC_cond, wave)


# wave-number dependent magnetic damping rates
damp = mag.calculate_magnetic_damping(
    metal=gal,
    geometry=dim,
    elec_bcond=BC_cond,
    magField=B0,
    wavemode=wave,
)

print(Ha)

print(
    f"Magnetic damping rate of wave mode {wave} is:",
    f"{damp:.4f}",
)

print(
    f"The corresponding magnetic damping time of wave mode {wave} is:",
    f"{1 / damp:.4f}",
)
