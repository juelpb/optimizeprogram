import sys
import os
import time
import shutil


path = os.getcwd()
sys.path.append(path)


import suspensionbridge
from Scripts.CrossSection.GenProp import Properties
from Scripts.DeflectionPrediction.DefPred import Deflection



def Run(UserParameterFileName, UserParameterFolder, t_height, g_height, p_type):
    """Function to initiate the modal analysis

    Args:
        UserParameterFileName (str): Input file name
        UserParameterFolder (_type_): Path to input file name
        t_height (float): Tower height
        g_height (float): Girder height
        p_type (int): Parametrization type. 2020, 2021, 2022
    """
    path = os.getcwd()    
    # Interpolating to find deflections and section properties for current configurations. Writes this to input file.
    girder_def_mid = Deflection(t_height, g_height, p_type, 'girder_def_mid')
    girder_def_south = Deflection(t_height, g_height, p_type, 'girder_def_south')
    girder_def_north = Deflection(t_height, g_height, p_type, 'girder_def_north')
    cable_def_mid = Deflection(t_height, g_height, p_type, 'cable_def_mid')
    tower_def = Deflection(t_height, g_height, p_type, 'tower_def')
    cable_sf_max = Deflection(t_height, g_height, p_type, 'cable_sf_max')
    hanger_sf_max = Deflection(t_height, g_height, p_type, 'hanger_sf_max')
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
    data[13] = f'girder_def_mid={girder_def_mid}\n'
    data[14] = f'girder_def_south={girder_def_south}\n'
    data[15] = f'girder_def_north={girder_def_north}\n'
    data[16] = f'cable_def_mid={cable_def_mid}\n'
    data[17] = f'tower_def={tower_def}\n'
    data[18] = f'cable_sf_max={cable_sf_max}\n'
    data[19] = f'hanger_sf_max={hanger_sf_max}\n'
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
    
    # Run analysis
    suspensionbridge.MainSuspensionBridge(UserParameterFileName,UserParameterFolder)
    
    