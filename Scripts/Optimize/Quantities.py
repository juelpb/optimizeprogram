""" 
Functions to calculate material quanities for arbitrary configuration
"""
import os
import sys
path = os.getcwd()
sys.path.append(path)

from Scripts.DeflectionPrediction.DefPred import Deflection
from Scripts.CrossSection.GenProp import Properties


import numpy as np
from matplotlib import pyplot as plt

def CableVolume(t_height, g_height, p_type, max_stress=500e6, z_cable_midspan=88.8, L_bridgedeck=1235):
    """Calculating volume of cable for current configuration

    Args:
        t_height (float): Tower height
        g_height (float): Girder height
        p_type (int): Parametrization type. 2020, 2021, 2022
        max_stress (float, optional): Target stress in main cable [Pa]. Defaults to 500e6.
        z_cable_midspan (float, optional): Elevation of cable at midspan. Defaults to 88.8.
        L_bridgedeck (float, optional): Span length. Defaults to 1235.

    Returns:
        float: Cable volume [m^3]
    """
    # Creating cable vector
    x = np.array([-L_bridgedeck/2, 0, L_bridgedeck/2])
    z = np.array([t_height - z_cable_midspan, 0, t_height - z_cable_midspan])
    p = np.polyfit(x, z, 2)
    
    x = np.linspace(-L_bridgedeck/2, L_bridgedeck/2, L_bridgedeck)
    z_cable = np.zeros_like(x)
    for i in range(len(z_cable)):
        z_cable[i] = p[0]*x[i]**2 + p[1]*x[i] + p[2]
    
    # Summing segment lengths
    cl = 0
    for i in range(len(z_cable)-1):
        dx = x[i+1] - x[i]
        dz = z_cable[i+1] - z_cable[i]
        dl = np.sqrt(dx**2 + dz**2)
        cl += dl
       
    # Calculating demanded cable area for selected geometry
    cable_sf_max = Deflection(t_height, g_height, p_type, 'cable_sf_max')
    cable_area = cable_sf_max/max_stress
    
    cv = 2*cl*cable_area
    return cv


def HangerVolume(t_height, g_height, p_type, max_stress=500e6, z_cable_midspan=88.8, z_cog_south=69, z_cog_midspan=76.6, z_cog_north=69, dx_hanger=12, L_bridgedeck=1235):
    """Calculate total hanger volume for current configuration

    Args:
        t_height (float): Tower height
        g_height (float): Girder height
        p_type (int): Parametrization type. 2020, 2021, 2022
        max_stress (float, optional): Target stress in hangers [Pa]. Defaults to 500e6.
        z_cable_midspan (float, optional): Elevation of cable at midspan. Defaults to 88.8.
        z_cog_south (int, optional): Elevation of girder center at south end. Defaults to 69.
        z_cog_midspan (float, optional): Elevation of girder center at midspan. Defaults to 76.6.
        z_cog_north (int, optional): Elevation of girder center at north end. Defaults to 69.
        dx_hanger (float, optional): Hanger spacing [m]. Defaults to 12.
        L_bridgedeck (float, optional): Span length. Defaults to 1235.

    Returns:
        float: Total hanger volume [m^3]
    """
    # Creating cable vector
    x = np.array([-L_bridgedeck/2, 0, L_bridgedeck/2])
    z = np.array([t_height - z_cable_midspan, 0, t_height - z_cable_midspan])
    p = np.polyfit(x, z, 2)
    
    x = np.linspace(-L_bridgedeck/2, L_bridgedeck/2, L_bridgedeck)
    z_cable = np.zeros_like(x)
    for i in range(len(z_cable)):
        z_cable[i] = p[0]*x[i]**2 + p[1]*x[i] + p[2]

    # Creating birdgedeck vector
    x = np.array([-L_bridgedeck/2, 0, L_bridgedeck/2])
    z = np.array([(z_cog_south + g_height/2) - z_cable_midspan, (z_cog_midspan + g_height/2) - z_cable_midspan, (z_cog_north + g_height/2) - z_cable_midspan])
    p = np.polyfit(x, z, 2)
    
    x = np.linspace(-L_bridgedeck/2, L_bridgedeck/2, L_bridgedeck)
    z_bridgedeck = np.zeros_like(x)
    for i in range(len(z_bridgedeck)):
        z_bridgedeck[i] = p[0]*x[i]**2 + p[1]*x[i] + p[2]
        
    # Defnining hanger positions
    rest = L_bridgedeck % dx_hanger
    x_hanger = [int(rest/2)]
    
    for i in range(1,L_bridgedeck//dx_hanger+1):
        x_hanger.append(x_hanger[i-1] + dx_hanger)
    
    hl = 0
    for i in range(len(x_hanger)):
        dl = z_cable[x_hanger[i]] - z_bridgedeck[x_hanger[i]]
        hl += dl
    
    # Calculating demanded hanger area for selected geometry
    hanger_sf_max = Deflection(t_height, g_height, p_type, 'hanger_sf_max')
    hanger_area = hanger_sf_max/max_stress
    
    hv = 2*hl*hanger_area
    return hv



def GirderVolume(g_height, p_type, L_bridgedeck=1235):
    """Calculating solid volume of girder

    Args:
        g_height (float): Girder height
        p_type (int): Parametrization type. 2020, 2021, 2022
        L_bridgedeck (float, optional): Span length. Defaults to 1235.

    Returns:
        float: Girder solid volume
    """
    girder_area = Properties(g_height, p_type, 'girder_area')
    return girder_area*L_bridgedeck


        

def TowerVolume(t_height, elev = [0, 40, 180, 206], h_elev = [7.5, 5, 5, 5], b_elev = [7.5, 5, 4, 4], t_elev = [1.0, 1.0, 0.6, 0.6], dy_legs = [40, 3], crossB_elev = [60, 205]):
    """Calculate tower volume (cross-beam included)

    Args:
        t_height (float): Tower height
        elev (list, optional): elevations of which the tower dimension are defined. Defaults to [0, 40, 180, 206].
        h_elev (list, optional): tower leg cross section height at the elevations. Defaults to [7.5, 5, 5, 5].
        b_elev (list, optional): tower leg cross section width at the elevations. Defaults to [7.5, 5, 4, 4].
        t_elev (list, optional): tower leg cross section wall thickness at the elevations. Defaults to [1.0, 1.0, 0.6, 0.6].
        dy_legs (list, optional): horizontal spacing between the legs at bottom and top. Defaults to [40, 3].
        crossB_elev (list, optional): elevations of which the cross beams are located. Defaults to [60, 205].

    Returns:
        float: Total tower volume [m^3]
    """
    
    if t_height != 206.0:
        crossB_elev = [60, int(round(t_height)) - 2]
    
    z = np.linspace(0,t_height,int(t_height))
    # Tower legs
    h = np.zeros_like(z)
    b = np.zeros_like(z)
    t = np.zeros_like(z)
    dz = 1
    p_h = np.polyfit(elev, h_elev, 2)
    p_b = np.polyfit(elev, b_elev, 2)
    p_t = np.polyfit(elev, t_elev, 2)
    V = 0
    for i in range(len(z)):
        h[i] = p_h[0]*z[i]**2 + p_h[1]*z[i] + p_h[2]
        b[i] = p_b[0]*z[i]**2 + p_b[1]*z[i] + p_b[2]
        t[i] = p_t[0]*z[i]**2 + p_t[1]*z[i] + p_t[2]
        V += (h[i]*b[i] - (h[i]-t[i])*(b[i]-t[i]))*dz
    V = V*4
    
    # Crossbeams
    dy = np.zeros_like(z)   
    for i in range(len(z)):
        dy[i] = dy_legs[0] - z[i]*((dy_legs[0]-dy_legs[1])/t_height) - b[i]
    h = 6
    b = 4
    t = 0.6
    A_crossB = h*b - (h-t)*(b-t)
    for i in range(len(crossB_elev)):
        V += dy[crossB_elev[i]]*A_crossB
    
    return V




