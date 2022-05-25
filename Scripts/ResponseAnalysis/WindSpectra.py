import numpy as np
import math


def SinglePointSpectra(comp, V, sigma, z, w, x_unit, Au=6.8, Aw=9.4):
    """Generate single-point turbulence (Kaimal) spectrum

    Args:
        comp (str): Turbulence component, 'u', 'v' or 'w'
        V (float): Mean wind velocity
        sigma (float): standard deviation of the wind velocity
        z (float): Height at bridge site, meter above sea level
        w (array): Frequency axis, 1D-array
        x_unit (str): Unit of frequency axis. 'F' for [cycle/s], 'W' for [rad/s]
        Au (float, optional): Factor. Defaults to 6.8.
        Aw (float, optional): Factor. Defaults to 9.4.

    Returns:
        array: Turbulence spectrum
    """
    z_1 = 10
    L_1 = 100
    xL_u = L_1*(z/z_1)**0.3    
    A_u = Au
    A_v = 9.4
    A_w = Aw
    if comp == 'u':
        A = A_u
        xL_u = xL_u
    elif comp == 'v':
        A = A_v
    elif comp == 'w':
        A = A_w
        xL_u = xL_u * 1/12
    else:
        print('Specify turbulence component.')


    I_u = sigma/V
    if x_unit == 'F':
        w_hat = w*xL_u/V                                             #for spectra w.r.t f [cycle/s]
        S_ = sigma**2*A*w_hat/(w*(1+1.5*A*w_hat)**(5/3))             #for spectra w.r.t f [cycle/s]
    elif x_unit == 'W':
        w_hat = w*xL_u/(2*np.pi*V)                                   #for spectra w.r.t w [rad/s]
        S_ = sigma**2*xL_u*A/(2*np.pi*V*(1+1.5*A*w_hat)**(5/3))      #for spectra w.r.t w [rad/s]
    else:
        print('x_unit is not defined')

    return S_










