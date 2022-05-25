""" 
Functions to write input file, run input file, read output file, write results and interpolate section parameters
"""

import os
import sys
import time
import numpy as np
import shutil

path = os.getcwd()
sys.path.append(path)
from Scripts.CrossSection import Geometry

def shoelace_area(x_list,y_list):
    """Calculate area of polygon from corner coordinates

    Args:
        x_list (list): x-coordinates of vertices
        y_list (list): y-coordinates of vertices

    Returns:
        float: area within polygon
    """
    a1,a2=0,0
    x_list.append(x_list[0])
    y_list.append(y_list[0])
    for j in range(len(x_list)-1):
        a1 += x_list[j]*y_list[j+1]
        a2 += y_list[j]*x_list[j+1]
    l=abs(a1-a2)/2
    return l


def Run(g_height, p_type, tef):
    """Writes to input file, runs input file, reads output file and writes results to csv-file

    Args:
        g_height (float): Girder height [m]
        p_type (int): Parametrization type. Girder sections from earlier Langenuen studies are implemented. 2020, 2021, 2022
        tef (float): Effective thickness of thin-walled cross-section [m]
    """
    path = os.getcwd()
    
    # Writing to input file (section analysis)
    H=g_height
    tef=tef
    p_type=p_type
    seed_size=0.01
    
    with open(path + '/Scripts/CrossSection/GenPropInput.py', 'r') as file:
        data = file.readlines()    
    data[28] = f'H={H}\n' 
    data[29] = f'tef={tef}\n'
    data[30] = f'p_type={p_type}\n'
    data[31] = f'seed_size={seed_size}\n'

    with open(path + '/Scripts/CrossSection/GenPropInput.py', 'w', encoding='utf-8') as file:
        file.writelines(data)        
    file.close()

    # Remove existing folders with content
    folder = path + '/FEM_Results/CrossSectionAnalysis'
    if os.path.isdir(folder):
        shutil.rmtree(folder)
        time.sleep(1)

    # Create folder
    os.makedirs(folder)

    # ---------------- #
    # RUN SECTION MODEL
    # ---------------- #

    # Variables
    script_name = path + '/Scripts/CrossSection/GenPropInput'
    #script_name = path + '/GenPropInput'

    # Run script
    print('--------------------------------------------')
    print('Running section analysis (1)')
    t0 = time.time()

    os.chdir(path + '/FEM_Results/CrossSectionAnalysis') # Change dir to store results files

    os.system('abaqus cae noGUI=' + script_name) # Initiate 

    os.chdir(path) #

    t1 = time.time()
    print('Section analysis (1) done - ' + str(round(t1 - t0, 3)) + ' sec.')
    print('--------------------------------------------')
    print()

    # Wait 3 sec for the Abaqus analysis to properly close
    time.sleep(5)


    props = []
    # Extracting results
    with open(path + '/FEM_Results/CrossSectionAnalysis/section_properties.bsp', 'r') as file:
        props=file.readlines()
        
    E = 70e9
    G = 26e9      
        
    props_lst = []
    for i in range(1, len(props), 2):
        props_lst.append(props[i].replace(' ','').split(","))
    
    girder_area=float(props_lst[0][0])/E
    girder_I11=float(props_lst[0][1])/E
    girder_I12=float(props_lst[0][2])/E
    girder_I22=float(props_lst[0][3])/E
    girder_It=float(props_lst[0][4])/G        
    girder_mass = float(props_lst[1][0])
    girder_i11 = float(props_lst[1][1])
    girder_i12 = float(props_lst[1][2])
    girder_i22 = float(props_lst[1][3])
    
    xs = Geometry.GenerateCorners(H, p_type)[0]
    ys = Geometry.GenerateCorners(H, p_type)[1]
    girder_per = 0
    for i in range(len(xs)-1):
        dx = xs[i+1] - xs[i]
        dy = ys[i+1] - ys[i]
        girder_per += np.sqrt(dx**2 + dy**2)
    
    tot_area = shoelace_area(xs, ys)
    
    # Additional mass (80 mm asphalt, 16 eff. mm bulkhead pr 3.9 m, hanger heads, other equipment)
    m_asphalt = 80e-3*(xs[3]-xs[1])*2500
    m_bulkhead = tot_area*16e-3*2700/3.9
    
    mass = girder_mass + m_asphalt + m_bulkhead + 30 + 400
    
    # Additional moment inertia from asphalt and bulkhead (neglected for hanger head and other equipment)
    i11 = girder_i11 + m_asphalt*1**2/12 + m_bulkhead*g_height**2/12
    i22 = girder_i22 + m_asphalt*(xs[3]-xs[1])**2/12 + m_bulkhead*(0.9*(xs[4]-xs[0]))**2/12

    
    with open(path + '/Scripts/CrossSection/SectionProperties.csv', 'a') as file:
        file.write(f'{g_height}, {p_type}, {tef}, {girder_area}, {girder_I11}, {girder_I12}, {girder_I22}, {girder_It}, {girder_mass}, {mass}, {girder_i11}, {girder_i12}, {girder_i22}, {i11}, {i22}, {girder_per}, {tot_area}\n')


def Properties(g_height, p_type, var): 
    """Function to interpolate section properties for an arbitrary girder height

    Args:
        g_height (float): Girder height [m]
        p_type (int): Parametrization type. 2020, 2021, 2022
        var (str): Variable to return interpolated value for. Valid inputs are:
                'girder_area', 'girder_I11', 'girder_I12', 'girder_I22', 'girder_It', 'girder_mass', 'mass', 'girder_i11', 'girder_i12', 'girder_i22', 'i11', 'i22', 'girder_per', 'tot_area'

    Returns:
        float: Interpolated value of variable for input girder height
    """
    
    path = os.getcwd()
    
    # extracts data from csv-file
    data = np.loadtxt(path + '/Scripts/CrossSection/SectionProperties.csv', delimiter=',', skiprows=1)
    props_lst = []
    
    # stores data for selected p_type
    for i in range(len(data)):
        if data[i,1] == p_type:
            props_lst.append(data[i,:])
    
    props = np.atleast_2d(np.array(props_lst))
            
    x_data = props[:,0]
    
    # list of valid variable inputs
    vars = ['girder_area', 'girder_I11', 'girder_I12', 'girder_I22', 'girder_It', 'girder_mass', 'mass', 'girder_i11', 'girder_i12', 'girder_i22', 'i11', 'i22', 'girder_per', 'tot_area']
    
    
    for i in range(len(vars)):
        if var == vars[i]:
            y_data = props[:,i+3]
    
    return np.interp(g_height, x_data, y_data)


