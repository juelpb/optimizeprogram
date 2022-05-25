""" 
Funtions to write to input file, run global model input file, read output file, write static deflections to csv-file, interpolate static deflections for arbitrary configuration
"""

import sys
import os
import time
import shutil
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp2d

path = os.getcwd()
sys.path.append(path)


import suspensionbridge
from Scripts.CrossSection.GenProp import Properties
from Scripts.HelperFunctions import sleep


def find_nearest(array, value):
    """Function to find the index of the nearest array-element

    Args:
        array (array): Array to search thorugh
        value (float): Value to return the neares index for

    Returns:
        int: Index of "array"-element nearest "value"
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx



def Run(t_height, g_height, p_type, initial):
    """Writes to input file, runs input file, reads output file and writes results to csv-file

    Args:
        t_height (float): Tower height [m]
        g_height (float): Girder height [m]
        p_type (int): p_type (int): Parametrization type. Girder sections from earlier Langenuen studies are implemented. 2020, 2021, 2022
        initial (list): Suggested initial offset of geometry
    """
    path = os.getcwd()    
    
    
    UserParameterFolder=path + '/Scripts/DeflectionPrediction'
    UserParameterFileName='/DefPredInput.py'
    
    # Searches through csv-file to find "initial" for the most similar configuration with tower height as priority
    with open(path + '/Scripts/DeflectionPrediction/DeflectionPrediction.csv') as file:
        data = np.atleast_2d(np.loadtxt(file, delimiter=',', skiprows=1))
    if data.shape[1] == 0:
        deflections = initial
    else:
        data_new = []
        for i in range(len(data)):
            if data[i,2] == p_type:
                data_new.append(data[i,:])
        if len(data_new) == 0:
            deflections = initial
        else:
            data_arr = np.array(data_new)
            t_heights = data_arr[:,0]
            deflections = data_arr[find_nearest(t_heights,t_height),:]

    # Finds section properties for current girder height
    girder_area = Properties(g_height, p_type, 'girder_area')
    girder_I11 = Properties(g_height, p_type, 'girder_I11')
    girder_I22 = Properties(g_height, p_type, 'girder_I22')
    girder_It = Properties(g_height, p_type, 'girder_It')
    mass = Properties(g_height, p_type, 'mass')
    i11 = Properties(g_height, p_type, 'i11')
    i22 = Properties(g_height, p_type, 'i22')

    # Writing to input script
    with open(UserParameterFolder + UserParameterFileName, 'r', encoding='utf-8') as file:
        data = file.readlines()    
    data[12] = f't_height={t_height}\n' 
    data[13] = f'girder_def_mid={deflections[4]}\n'
    data[14] = f'girder_def_south={deflections[3]}\n'
    data[15] = f'girder_def_north={deflections[5]}\n'
    data[16] = f'cable_def_mid={deflections[6]}\n'
    data[17] = f'tower_def={deflections[7]}\n'
    data[18] = f'cable_sf_max={deflections[8]}\n'
    data[19] = f'hanger_sf_max={deflections[9]}\n'
    data[20] = f'girder_area={girder_area}\n' 
    data[21] = f'girder_I11={girder_I11}\n'
    data[22] = f'girder_I22={girder_I22}\n'
    data[23] = f'girder_It={girder_It}\n'
    data[24] = f'mass={mass}\n'
    data[25] = f'i11={i11}\n'
    data[26] = f'i22={i22}\n'
    
    with open(UserParameterFolder + UserParameterFileName, 'w', encoding='utf-8') as file:
        file.writelines(data)        
    file.close()
    
    # Remove existing folders with content
    folder = path + '/FEM_Results/GlobalAnalysis'
    if os.path.isdir(folder):
        shutil.rmtree(folder)
        time.sleep(1)

    # Create folder
    os.makedirs(folder)
    
    # ----------------#
    # RUN ANALYSIS
    # ----------------#
    
    suspensionbridge.MainSuspensionBridge(UserParameterFileName,UserParameterFolder)
    
    sleep(3)
    
    # ---------------- #
    # READ ANALYSIS
    # ---------------- #

    script_name = UserParameterFolder + '/ReadOdb'

    # Run odb-reader
    print('--------------------------------------------')
    print('Reading analysis (1)')
    t0 = time.time()

    os.system('abaqus cae noGUI=' + script_name)

    t1 = time.time()
    print('Reading analysis (1) done - ' + str(round(t1 - t0, 3)) + ' sec.')
    print('--------------------------------------------')
    print()

    # Wait 3 sec for the Abaqus analysis to properly close
    time.sleep(3)
    
    
    with open(UserParameterFolder + '/step1_disp.csv', 'r') as file:
        disp = np.loadtxt(file, delimiter=",")
        
    # Collecting tower top deflection ((101001 - 1) returns tower top node)
    for i in range(len(disp)):
        if disp[i,0] == 101001:
            t_d=np.abs(disp[i-1,1]) 

    
    with open(UserParameterFolder + '/step4_disp.csv', 'r') as file:
        disp = np.loadtxt(file, delimiter=",")
    
    # Collecting deflections at particular node-numbers: 1001 - girder south end | 1156 - girder midspan | 1311 - girder north end | 10103 - cable midspan |
    for i in range(len(disp)):
        if disp[i,0] == 1001: 
            g_d_s=np.abs(disp[i,3])
        if disp[i,0] == 1156:
            g_d_m=np.abs(disp[i,3])
        if disp[i,0] == 1311:
            g_d_n=np.abs(disp[i,3])
        if disp[i,0] == 10103:
            c_d_m=np.abs(disp[i,3])
            
    # Max cable SF for elements in range 20051-20154 and max hanger SF for elements in range 90001-90103
    with open(UserParameterFolder + '/step4_sf.csv', 'r') as file:
        sf = np.loadtxt(file, delimiter=",")
    
    sf_cable = []
    sf_hanger = []
    
    for i in range(len(sf)):
        if 20050 < sf[i,0] < 20155:
            sf_cable.append(sf[i,:])
    
    sf_cable = np.array(sf_cable)
    sf_max_cable = np.amax(sf_cable[:,1])
    
    for i in range(len(sf)):
        if 90000 < sf[i,0] < 90104:
            sf_hanger.append(sf[i,:])
    
    sf_hanger = np.array(sf_hanger)
    sf_max_hanger = np.amax(sf_hanger[:,1])
    

    with open(path + '/Scripts/DeflectionPrediction/DeflectionPrediction.csv', 'a') as file:
        file.write(f'{t_height}, {g_height}, {p_type}, {g_d_s}, {g_d_m}, {g_d_n}, {c_d_m}, {t_d}, {sf_max_cable}, {sf_max_hanger}\n')
            



def Deflection(t_height, g_height, p_type, var):
    """Function to interpolate deflections for arbitrary configuration of tower height and girder height

    Args:
        t_height (float): Tower height [m]
        g_height (float): Girder height [m]
        p_type (int): Parametrization type. 2020, 2021, 2022
        var (string): Variable to return interpolated value for. Valid inputs are:
                'girder_def_south','girder_def_mid','girder_def_north','cable_def_mid','tower_def', 'cable_sf_max', 'hanger_sf_max'

    Returns:
        float: Interpolated value of variable for input girder height
    """
    path = os.getcwd() + '/Scripts/DeflectionPrediction'
    
    # extracts data from csv-file
    data = np.atleast_2d(np.loadtxt(path + '\DeflectionPrediction.csv', delimiter=',', skiprows=1))
    data_new = []
    
    # stores data for selected p_type
    for i in range(len(data)):
        if data[i,2] == p_type:
            data_new.append(data[i,:])
    data_arr = np.atleast_2d(np.array(data_new))
    x_data = data_arr[:,0]
    y_data = data_arr[:,1]
    
    # list of valid variable inputs
    vars = ['girder_def_south','girder_def_mid','girder_def_north','cable_def_mid','tower_def', 'cable_sf_max', 'hanger_sf_max']
    
    for i in range(len(vars)):
        if var == vars[i]:
            z_data = data_arr[:,i+3]
    
    f = interp2d(x_data, y_data, z_data, kind='quintic')
    
    return f(t_height, g_height)[0]



