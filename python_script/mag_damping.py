import os
import numpy as np
from numpy import pi, tanh, sqrt
import cmath as cmth
from dataclasses import dataclass
from itertools import product

np.set_printoptions(legacy="1.25")


##### PHYSICAL PARAMETERS ##############
g = 9.81


##### MATH FUNCTION ##############
def coth(x):
    return 1 / np.tanh(x)


def csqrt(x):
    return cmth.sqrt(x)


###################################


## DEFINING CLASSES ############

@dataclass
class metal:
    name: str
    density: float  # kg/m^3
    kinematic_viscosity: float  # m2/s (Kinematic Viscosity)
    electric_conductivity: float  # S/m electrical conductivity

@dataclass
class magField:
    Bz: float #magnetic field in tesla

@dataclass
class geometry:
    # all values in m
    length: float  # length of geometry
    width: float  # width of geometry
    metal_height: float  # layer height of metal

@dataclass
class wavemode:
    # wave mode (m, n) as in the manuscript
    m: int
    n: int

@dataclass
class elec_bcond:
    cond : str

#########################################################
# %%
####### TRIGONOMETRIC FUNCTIONS ############
def sinh(x):
    
    if abs(x) < 600:
        return np.sinh(x)
    
    else:
        return x / abs(x) * np.sinh(600)

def cosh(x):
    
    if abs(x) < 600:
        return np.cosh(x)
    
    else:
        return np.cosh(600)
    
    
def csch(x): 
    
    return 1 / sinh(x)

def sech(x):
    
    x = np.array(x, dtype=np.float128)
    
    return 1 / cosh(x)

def coth(x):
    
    x = np.array(x, dtype=np.float128)
    
    return 1 / tanh(x)
###############################################################################
# USER INPUT
instruct = """Enter electrical boundary condition configuration. Check Table I. \n
           bc1: All walls are perfect conductors.
           bc2: X-normal walls are perfect conductors; the rest are perfect insulators.
           bc3: Bottom wall is a perfect conductor; the rest are perfect insulators. 
           bc4: All walls are perfect insulators
           """
print(instruct)

####### RAW FUNCTIONS ######################
def epsilon(r):
    return 2 if r == 0 else 1


# wave number function
def kmn(m, n, Lx, Ly):
    """
    Wave number function. Equation (15).
    """
    
    if m <= 0 and n <= 0:
        raise ValueError("Incorrect wave modes")
    else:
        return pi * sqrt(pow(m / Lx, 2) + pow(n / Ly, 2))


# wave frequency
def wmn(k, h):
    """
    Gravity-wave frequency. Equation (16).
    """
    return sqrt(g * k * tanh(k * h))

# Hartmann number
def Ha_number(B, L, sig, rho, nu):
    """
    Hartmann number. Equation (18).
    """
    
    return B * L * sqrt(sig / (rho * nu))
    
###############################################################################
# B.C 1 formulae
def Q_ind(sigma, rho, w, Bz, k, Lx, Ly, h, m,n):
    """
    Ohmic dissipation when all walls are perfect conductors. Equation (39).
    """
    res = (
            epsilon(m * n) *
            ( sigma * w**2 * Bz**2 *
              ( coth(k * h) + k * h * csch(k * h)**2 )
            ) / (8 * k) * Lx * Ly
        )
        
    return res
###############################################################################

###############################################################################
# B.C 2 formulae
def Bab_bc2(w, Bz, k, Lx, Ly, h, m, n, a, b):
    """
    Coefficient of the induced electric potential of the B.C 2 case. Equation (23).
    """
    xi_y = pi * sqrt(pow(a / Lx, 2) + pow(b / h, 2))

    oe_term_const = ((1 - pow(-1, n)) / (2 * csch(0.5 * xi_y * Ly)) 
                   + (1 + pow(-1, n)) / (2 * sech(0.5 * xi_y * Ly))
                   )
    frac1 = (( -2 * w * Bz * m * pi * ((-1)**(n + b)) ) 
             / ( epsilon(b) * xi_y * Lx * oe_term_const )
             )
    frac2 = h / (k**2 * h**2 + b**2 * pi**2)
    return frac1 * frac2

def Qcorr_bc2(sigma, rho, w, Bz, k, Lx, Ly, h, m, n, itr):
    sumB = 0 
    a = m
    """
    Correction term for B.C 2 case. Equation (42).
    """
    for b in range(itr):
        if a==0 and b==0:
            return 0
        else:
            xi_y = pi * sqrt(pow(a / Lx, 2) + pow(b / h, 2))

            oe_term = ((1 - pow(-1, n)) / (2 * cosh(0.5 * xi_y * Ly)) 
                       + (1 + pow(-1, n)) / (2 * sinh(0.5 * xi_y * Ly))
                       )

            B_mb = Bab_bc2(w, Bz, k, Lx, Ly, h, m, n, a, b)
            term = (   (
                        -B_mb * sigma * w * Bz *
                        h**2 * m * pi * ((-1) ** (n + b))
                    ) / (
                        (k**2 * h**2 + b**2 * pi**2) * oe_term
                    )
                )
            sumB += term
            
    return sumB
###############################################################################

###############################################################################
# B.C 3 formulae
def A0_bc3(w, Bz, k, Lx, Ly, h, m, n):
    """
    The constant term of the induced electric potential of the B.C 3 case. Equation (26).
    """
    frac1 = 4 * w * Bz * (pow(-1, m + 1) + pow(-1, m + n) ) / (k * Ly)
    
    frac2 = (2 * k * h + pi * csch(k * h) ) / (4 * pow(k * h, 2) + pi**2)
    
    return -frac1 * frac2

def B0_bc3(w, Bz, k, Lx, Ly, h, m, n):
    """
    The constant term of the induced electric potential of the B.C 3 case. Equation (27).
    """
    frac1 = 4 * w * Bz * (pow(-1, n + 1) + pow(-1, m + n) ) / (k * Lx)
    
    frac2 = (2 * k * h + pi * csch(k * h) ) / (4 * pow(k * h, 2) + pi**2)
    
    return frac1 * frac2

def Aab_bc3(w, Bz, k, Lx, Ly, h, m, n, a, b):
    """
    Coefficient of the induced electric potential of the B.C 3 case. Equation (28).
    """
    xi_x = pi * sqrt(pow(a / Ly, 2) + pow((1 + 2 * b) / (2 * h), 2))

    oe_term_constX = ((1 - pow(-1, m)) / (2 * csch(0.5 * xi_x * Lx)) 
                    + (1 + pow(-1, m)) / (2 * sech(0.5 * xi_x * Lx))
                    )

    num1 = 8 * w * Bz * n**2 * (pow(-1, m + 1) + pow(-1, a + m + n))
    den1 = epsilon(a) * (a**2 - n**2) * oe_term_constX * xi_x * Ly * k
    
    num2 = 2 * k * h * pow(-1, b) + (1 + 2 * b) * pi * csch(k * h)
    den2 = 4 * k**2 * h**2 + (1 + 2 * b)**2 * pi**2

    if a==0 and b==0:
        return 0 
    elif a==n:
        return 0 
    else:
        return (num1 / den1) * (num2 / den2)

def Bab_bc3(w, Bz, k, Lx, Ly, h, m, n, a, b):
    """
    Coefficient of the induced electric potential of the B.C 3 case. Equation (29).
    """
    xi_y = pi * sqrt(pow(a / Lx, 2) + pow((1 + 2 * b) / (2 * h), 2))

    oe_term_constY = ((1 - pow(-1, n)) / (2 * csch(0.5 * xi_y * Ly)) 
                    + (1 + pow(-1, n)) / (2 * sech(0.5 * xi_y * Ly))
                    )

    num1 = -8 * w * Bz * m**2 * (pow(-1, n + 1) + pow(-1, a + m + n))
    den1 = epsilon(a) * (a**2 - m**2) * oe_term_constY * xi_y * Lx * k
    
    num2 = 2 * k * h * pow(-1, b) + (1 + 2 * b) * pi * csch(k * h)
    den2 = 4 * k**2 * h**2 + (1 + 2 * b)**2 * pi**2

    if a==0 and b==0:
        return 0 
    elif a==m:
        return 0 
    else:
        return (num1 / den1) * (num2 / den2)

def Qcorr_bc3(sigma, rho, w, Bz, k, Lx, Ly, h, m, n, itr1, itr2):
    """
    Correction term for B.C 3 case. Equation (43).
    """
    def A0termFunc(sigma, rho, w, Bz, k, Lx, Ly, h, m, n):
        deltaMN = (1 + pow(-1, m)) * (1 - pow(-1, n))**2
        deltaOMN = (-1 + pow(-1, m)) * (1 - pow(-1, n))**2
        
        ph = pi / (2 * h)
        
        A0bc3 = A0_bc3(w, Bz, k, Lx, Ly, h, m, n)
        fac = A0bc3 * sigma * w * Bz
        
        num1 = deltaMN * tanh(0.5 * ph * Lx) + deltaOMN * coth(0.5 * ph * Lx)
        den1 = 2 * pi * (pow(m * h, 2) + 0.25 * Lx**2)
        frac1 = num1 / den1
        
        num2 = pow(h * Lx, 2) * (pi * csch(k * h) + 2 * k * h)
        den2 = k * (4 * pow(k * h, 2) + pi**2)
        frac2 = num2 / den2
        
        return fac * frac1 * frac2

    def B0termFunc(sigma, rho, w, Bz, k, Lx, Ly, h, m, n):
        deltaNM = (1 + pow(-1, n)) * (1 - pow(-1, m))**2
        deltaONM = (-1 + pow(-1, n)) * (1 - pow(-1, m))**2
        
        ph = pi / (2 * h)
        
        B0bc3 = B0_bc3(w, Bz, k, Lx, Ly, h, m, n)
        fac = B0bc3 * sigma * w * Bz
        
        num1 = deltaNM * tanh(0.5 * ph * Ly) + deltaONM * coth(0.5 * ph * Ly)
        den1 = 2 * pi * (pow(n * h, 2) + 0.25 * Ly**2)
        frac1 = num1 / den1
        
        num2 = pow(h * Ly, 2) * (pi * csch(k * h) + 2 * k * h)
        den2 = k * (4 * pow(k * h, 2) + pi**2)
        frac2 = num2 / den2
        
        return fac * frac1 * frac2
    
    A0term = A0termFunc(sigma, rho, w, Bz, k, Lx, Ly, h, m, n)
    
    B0term = B0termFunc(sigma, rho, w, Bz, k, Lx, Ly, h, m, n)

    sumA = 0
    for a in range(itr1):
        for b in range(itr2):
            xi_x = pi * sqrt(pow(a / Ly, 2) + pow((1 + 2 * b) / (2 * h), 2))
            Aab = Aab_bc3(w, Bz, k, Lx, Ly, h, m, n, a, b)
            oe_termM = ((1 - pow(-1, m)) / (2 * cosh(0.5 * xi_x * Lx)) 
                      + (1 + pow(-1, m)) / (2 * sinh(0.5 * xi_x * Lx))
                      )

            num1 = a**2 * m**2 * pi**2 + xi_x**2 * n**2 * Lx**2
            alt_sign = pow(-1, m + 1) + pow(-1, a + m + n)
            den1 = (a**2 - n**2) * (m**2 * pi**2 + xi_x**2 * Lx**2)

            num2 = h * ((1 + 2*b) * pi * csch(k * h) + 2 * k * h * (-1)**b)
            den2 = k * (4 * k**2 * h**2 + (1 + 2*b)**2 * pi**2)

            if a==0 and b==0:
                sumA += 0
            elif a==n:
                sumA += 0
            else:
                frac1 = ((4 * Aab * w * sigma * Bz * num1 * alt_sign) 
                         / (den1 * oe_termM)
                         )
                frac2 = num2 / den2
                sumA += frac1 * frac2

    sumB = 0
    for a in range(itr1):
        for b in range(itr2):
            xi_y = pi * sqrt(pow(a / Lx, 2) + pow((1 + 2 * b) / (2 * h), 2))
            Bab = Bab_bc3(w, Bz, k, Lx, Ly, h, m, n, a, b)
            oe_termN = ((1 - pow(-1, n)) / (2 * cosh(0.5 * xi_y * Ly)) 
                      + (1 + pow(-1, n)) / (2 * sinh(0.5 * xi_y * Ly))
                       )

            num1 = a**2 * n**2 * pi**2 + xi_y**2 * m**2 * Ly**2
            alt_sign = pow(-1, n + 1) + pow(-1, a + m + n)
            den1 = (a**2 - m**2) * (n**2 * pi**2 + xi_y**2 * Ly**2)

            num2 = h * ((1 + 2*b) * pi * csch(k * h) + 2 * k * h * (-1)**b)
            den2 = k * (4 * k**2 * h**2 + (1 + 2*b)**2 * pi**2)

            if a==0 and b==0:
                sumB += 0
            elif a==m:
                sumB += 0
            else:
                frac1 = ((4 * Bab * w * sigma * Bz * num1 * alt_sign) 
                         / (den1 * oe_termN)
                         )
                frac2 = num2 / den2
                sumB += frac1 * frac2

    return A0term - B0term + sumA - sumB
###############################################################################

###############################################################################
# B.C 4 formulae
def A0_const(w, Bz, k, Lx, Ly, h, m, n):
    """
    The constant term of the induced electric potential of the B.C 3 case. Equation (32).
    """
    term = -(
            w * Bz * (pow(-1, m + 1) + pow(-1, m + n)) / (k**2 * Ly * h)
        )
    return term

def B0_const(w, Bz, k, Lx, Ly, h, m, n):
    """
    The constant term of the induced electric potential of the B.C 3 case. Equation (32).
    """
    term = (
            w * Bz * (pow(-1, n + 1) + pow(-1, m + n)) / (k**2 * Lx * h)
        )
    return term

def Aab_bc4(w, Bz, k, Lx, Ly, h, m, n, a, b):
    """
    Coefficient of the induced electric potential of the B.C 4 case. Equation (33).
    """
    xi_x = pi * sqrt(pow(a / Ly, 2) + pow(b / h, 2))
    oe_term_constX = ((1 - pow(-1, m)) / (2 * csch(0.5 * xi_x * Lx)) 
                    + (1 + pow(-1, m)) / (2 * sech(0.5 * xi_x * Lx))
                    )
                        
    num1 = 4 * w * Bz * n**2 * (pow(-1, m + b + 1) + pow(-1, a + b + m + n))
    den1 = epsilon(a) * epsilon(b) * (a**2 - n**2) * oe_term_constX * xi_x * Ly
    
    frac2 = h / (k ** 2 * h ** 2 + b ** 2 * pi ** 2)

    if a==0 and b==0:
        return 0 
    elif a==n:
        return 0 
    else:
        return (num1 / den1) * frac2

def Bab_bc4(w, Bz, k, Lx, Ly, h, m, n, a, b):
    """
    Coefficient of the induced electric potential of the B.C 4 case. Equation (34).
    """
    xi_y = pi * sqrt(pow(a / Lx, 2) + pow(b / h, 2))
    oe_term_constY = ((1 - pow(-1, n)) / (2 * csch(0.5 * xi_y * Ly)) 
                      + 
                      (1 + pow(-1, n)) / (2 * sech(0.5 * xi_y * Ly))
                      )

    num1 = -4 * w * Bz * m**2 * (pow(-1, n + b + 1) + pow(-1, a + b + m + n))
    den1 = epsilon(a) * epsilon(b) * (a**2 - m**2) * oe_term_constY * xi_y * Lx 
    
    frac2 = h / (k ** 2 * h ** 2 + b ** 2 * pi ** 2)

    if a==0 and b==0:
        return 0 
    elif a==m:
        return 0 
    else:
        return (num1 / den1) * frac2


def Qcorr_bc4(sigma, rho, w, Bz, k, Lx, Ly, h, m, n, itr1, itr2):
    """
    Correction term for B.C 4 case. Equation (45).
    """
    def DeltaM(m,n): #delta function, Equation (46)
        if m==0:
            return 1
        else:
            term = (pow(-1, n) * (1 - pow(-1, m))**2  ) / (pow(m*pi, 2))
            return term
        
    def DeltaN(m,n): #delta function, Equation (46)
        if n==0:
            return 1
        else:
            term = (pow(-1, m) * (1 - pow(-1, n))**2  ) / (pow(n*pi, 2))
            return term
        
    A0 = A0_const(w, Bz, k, Lx, Ly, h, m, n)
    
    B0 = B0_const(w, Bz, k, Lx, Ly, h, m, n)

    A0term = A0 * sigma * w * Bz * Lx * (-1 + pow(-1, n)) / k**2 * DeltaM(m,n)
    
    B0term = B0 * sigma * w * Bz * Ly * (-1 + pow(-1, m)) / k**2 * DeltaN(m,n)

    sumA = 0
    for a in range(itr1):
        for b in range(itr2):
            xi_x = pi * sqrt(pow(a / Ly, 2) + pow(b / h, 2))

            if a==0 and b==0:
                sumA += 0
            elif a==n:
                sumA += 0
            else:
                Aab = Aab_bc4(w, Bz, k, Lx, Ly, h, m, n, a, b)

                num1 = a**2 * m**2 * pi**2 + xi_x**2 * n**2 * Lx**2
                alt_sign = pow(-1, m + b + 1) + pow(-1, a + b + m + n)
                # 
                den1 = (a**2 - n**2) * (m**2 * pi**2 + xi_x**2 * Lx**2)

                frac2 = h**2 / (k ** 2 * h ** 2 + b ** 2 * pi ** 2)
                
                oe_termM = ((1 - pow(-1, m)) / (2 * cosh(0.5 * xi_x * Lx)) 
                            + 
                            (1 + pow(-1, m)) / (2 * sinh(0.5 * xi_x * Lx))
                            )
                frac1 = (( 2 * Aab * w * sigma * Bz * num1 * alt_sign) 
                         / (den1 * oe_termM)
                        )
                sumA += frac1 * frac2

    sumB = 0
    for a in range(itr1):
        for b in range(itr2):
            xi_y = pi * sqrt(pow(a / Lx, 2) + pow(b / h, 2))

            if a==0 and b==0:
                sumB += 0
            elif a==m:
                sumB += 0
            else:
                Bab = Bab_bc4(w, Bz, k, Lx, Ly, h, m, n, a, b)

                num1 = a**2 * n**2 * pi**2 + xi_y**2 * m**2 * Ly**2
                alt_sign = pow(-1, n + b + 1) + pow(-1, a + b + m + n)
                # 
                den1 = (a**2 - m**2) * (n**2 * pi**2 + xi_y**2 * Ly**2)

                frac2 = h**2 / (k ** 2 * h ** 2 + b ** 2 * pi ** 2)
                
                oe_termN = ((1 - pow(-1, n)) / (2 * cosh(0.5 * xi_y * Ly)) 
                            + 
                            (1 + pow(-1, n)) / (2 * sinh(0.5 * xi_y * Ly))
                            )
                frac1 = (( 2 * Bab * w * sigma * Bz * num1 * alt_sign) 
                         / (den1 * oe_termN)
                         )
                sumB += frac1 * frac2

    return -A0term + B0term + sumA - sumB
###############################################################################

def OhmicDamp(sigma, rho, w, Bz, k, Lx, Ly, h, m, n, itr1, itr2, BC):
    """
    Damping due to Ohmic dissipation. Equation (38).
    """
    
    qind = Q_ind(sigma, rho, w, Bz, k, Lx, Ly, h, m, n)

    if BC == 'bc1':
        qcorr = 0
        
    elif BC == 'bc2':
        qcorr = Qcorr_bc2(sigma, rho, w, Bz, k, Lx, Ly, h, m, n, itr1+itr2)
        
    elif BC == 'bc3':
        qcorr = Qcorr_bc3(sigma, rho, w, Bz, k, Lx, Ly, h, m, n, itr1, itr2)
        
    elif BC == 'bc4':
        qcorr = Qcorr_bc4(sigma, rho, w, Bz, k, Lx, Ly, h, m, n, itr1, itr2)
        
    K = 0.25 * epsilon(m * n) * rho * g * Lx * Ly 

    return (qind - qcorr) / (2 * K)

def Hartmann_damp(nu, sigma, rho, w, Bz, k, Lx, Ly, h, m, n):
    """
    Damping due to Hartmann boundary layer. Equation (56).
    """
    tau = rho / (sigma * Bz**2)

    fac = ((nu * k) / (2 * sqrt(2))) * (1 / sqrt(tau * nu))
    
    num1 = sqrt(1 + sqrt(1 + (tau ** 2) * (w ** 2)))
    
    den1 = sinh(k * h) * cosh(k * h)
    
    return fac * num1 / den1

def Shercliff_damp(nu, sigma, rho, w, Bz, k, Lx, Ly, h, m, n):
    """
    Damping due to Shercliff boundary layer. Equation (57).
    """
    tau = rho / (sigma * Bz**2)

    fac1 = nu / (sqrt(2) * k)
    
    inner_sqrt = 1 + (w**2 * h**2 * tau) / nu
    
    outer_sqrt = np.sqrt(1 + sqrt(inner_sqrt))
    
    fac2 = outer_sqrt / (Lx * Ly * sqrt(h * sqrt(tau * nu)))

    term1 = ((h * (n**2 * np.pi**2 - k**2 * Ly**2)) 
             / (epsilon(m) * Ly * sinh(k * h) * cosh(k * h))
             )
    
    term2 = (n**2 * np.pi**2 + k**2 * Ly**2) / (epsilon(m) * Ly * k)
    
    term3 = ((h * (m**2 * np.pi**2 - k**2 * Lx**2)) 
             / (epsilon(n) * Lx * sinh(k * h) * cosh(k * h))
             )
    
    term4 = (m**2 * np.pi**2 + k**2 * Lx**2) / (epsilon(n) * Lx * k)

    bracket = term1 + term2 + term3 + term4

    return fac1 * fac2 * bracket


# %%

# parameter check
def parameter_check(metal, geometry, elec_bcond, wavemode=None):
    """
    Validate physical and geometrical parameters of the free-surface system.
    """
    
    # initiate variables
    rho = metal.density
    nu = metal.kinematic_viscosity
    sig = metal.electric_conductivity

    Lx, Ly = geometry.length, geometry.width
    h = geometry.metal_height
    
    bc_cond = elec_bcond.cond
        
    if rho <= 0:
        raise ValueError("Incorrect density value")
    elif nu <= 0:
        raise ValueError("Incorrect kinematic viscosity value")
    elif sig <= 0:
        raise ValueError("Non physical electric conductivity")
    elif Lx <= 0 or Ly <= 0:
        raise ValueError("Incorrect geometry")
    elif h <= 0:
        raise ValueError("Incorrect free-surface height")
    
    valid_bcs = {'bc1', 'bc2', 'bc3', 'bc4'}
    
    if bc_cond not in valid_bcs:
        raise ValueError(
            f"Invalid electrical boundary condition configuration '{bc_cond}'. "
            "Please enter either bc1, bc2, bc3, or bc4."
        )
    
    if (wavemode is not None):
        m, n = wavemode.m, wavemode.n

        k = kmn(m, n, Lx, Ly)
        
        w = wmn(k, h)
        
        wavelength = 2 * pi / k
        
        
        if wavelength / h >= 20:
            print("Shallow water approximation can be applied to this cell configuration")
            
        else:
            print("Shallow water approximation cannot be applied in this case")
    
    print("parameter check: OK \n")


# %%
####### USER FUNCTIONS ######################

def calculate_magnetic_damping(metal, geometry, elec_bcond, wavemode, magField):
    """
    Compute the magnetic damping rate as the sum of the Ohmic and boundary layer damping.
    """
    # initiate variables
    rho = metal.density
    nu = metal.kinematic_viscosity
    sigma = metal.electric_conductivity
    
    Bz = magField.Bz
    
    m, n = wavemode.m, wavemode.n
    
    Lx, Ly = geometry.length, geometry.width
    
    h = geometry.metal_height
    
    BC = elec_bcond.cond
    
    k = kmn(m, n, Lx, Ly); w = wmn(k, h)
    
    itr1 = 50; itr2 = 50
    
    mag_damp = (
        OhmicDamp(sigma, rho, w, Bz, k, Lx, Ly, h, m, n, itr1, itr2, BC)
        +
        Hartmann_damp(nu, sigma, rho, w, Bz, k, Lx, Ly, h, m, n)
        +
        Shercliff_damp(nu, sigma, rho, w, Bz, k, Lx, Ly, h, m, n)
        )
    
    return mag_damp

def calculate_magnetic_damping_time(metal, geometry, elec_bcond, wavemode, magField):
    """
    Compute the magnetic damping time: inverse of magnetic damping rate.
    """
    # initiate variables
    rho = metal.density
    nu = metal.kinematic_viscosity
    sigma = metal.electric_conductivity
    
    Bz = magField.Bz
    
    m, n = wavemode.m, wavemode.n
    
    Lx, Ly = geometry.length, geometry.width
    
    h = geometry.metal_height
    
    BC = elec_bcond.cond
    
    k = kmn(m, n, Lx, Ly); w = wmn(k, h)
    
    itr1 = 20; itr2 = 20
    
    mag_damp = (
        OhmicDamp(sigma, rho, w, Bz, k, Lx, Ly, h, m, n, itr1, itr2, BC)
        +
        Hartmann_damp(nu, sigma, rho, w, Bz, k, Lx, Ly, h, m, n)
        +
        Shercliff_damp(nu, sigma, rho, w, Bz, k, Lx, Ly, h, m, n)
        )
    
    return 1 / mag_damp
