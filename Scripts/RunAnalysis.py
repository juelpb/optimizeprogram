import sys
import os
import time
import numpy as np
from matplotlib import pyplot as plt
from HelperFunctions import sleep

path = os.getcwd()

sys.path.append(path)

print(path)


import suspensionbridge



UserParameterFolder= path 
UserParameterFileName='LangenuenGeneralSection.py'

def Run(section_height, tower_height, p_type=2021):
    
    # Writing to input file (section analysis)
    H = section_height
    tef = 0.0445
    p_type = p_type
    seed_size = 0.010
    
    filename_sectionInput = path + '/Scripts/CrossSection/GenProp.py'

   

    with open(filename_sectionInput, 'r', encoding='utf-8') as file:
        data = file.readlines()    
    data[28] = f'H={H}\n' 
    data[29] = f'tef={tef}\n'
    data[30] = f'p_type={p_type}\n'
    data[31] = f'seed_size={seed_size}\n'

    with open(filename_sectionInput, 'w', encoding='utf-8') as file:
        file.writelines(data)        
    file.close()



    # ------------------- #
    #* RUN SECTION MODEL
    # ------------------- #

    # Variables
    script_name = path+'/Scripts/CrossSection/generate_properties'
    SectionAnalysisFolderName = '/FEM_Results/CrossSectionAnalysis'
    # Run script
    print('-----------------------------------------------------------------------')
    print(f'Running section analysis with section height: H = {section_height}')
    print('-----------------------------------------------------------------------')
    
    t0 = time.time() # Variable for t_0 (time started)
    
    os.chdir(path+SectionAnalysisFolderName) # Change dir to store results files
    
    os.system('abaqus cae noGUI=' + script_name) # Initiate 
    
    os.chdir(path) # Change back to root-folder
    
    t1 = time.time() # Variable for t_1 (time ended)
    
    print('-----------------------------------------------------------------------')
    print('Section analysis is Done! - ' + str(round(t1 - t0, 3)) + ' sec.')
    print('-----------------------------------------------------------------------')
    print()

    # Wait for the Abaqus analysis to properly close
    sleep(5)

    # Extracting results
    props = []
    with open(path + SectionAnalysisFolderName + '/section_properties.bsp', 'r') as file:
        props=file.readlines()

    E = 70e9
    G = 26e9
    girder_area=float(props[1][1:12])/E
    girder_I11=float(props[1][14:25])/E
    girder_I12=float(props[1][27:38])/E
    girder_I22=float(props[1][40:51])/E
    girder_It=float(props[1][53:64])/G

        

    # ------------------- #
    #* RUN GLOBAL MODEL
    # ------------------- #
    
    # Writing to input script
    filename_mainInput= path +'/Scripts/LangenuenGeneralSection.py'

    
    with open(filename_mainInput, 'r', encoding='utf-8') as file:
        data = file.readlines()
    data[12] = f'tower_height={tower_height}\n'    
    data[20] = f'girder_area={girder_area}\n' 
    data[21] = f'girder_I11={girder_I11}\n'
    data[22] = f'girder_I22={girder_I22}\n'
    data[23] = f'girder_It={girder_It}\n'

    with open(filename_mainInput, 'w', encoding='utf-8') as file:
        file.writelines(data)        
    file.close()

    # Run global model
    suspensionbridge.MainSuspensionBridge(UserParameterFileName,UserParameterFolder)            




Run(5.5, 200)


from Scripts.HelperFunctions import Pull_ModalData
Pull_ModalData()
# %%
