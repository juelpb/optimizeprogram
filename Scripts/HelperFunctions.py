"""
Collection of common helper functions 
- Sleep
- Pull_modaldata
- Pull_modeshapes
"""

import time
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def sleep(sec):
    print(f'Sleep for {sec} seconds')
    for sec in range(sec):
        print(".", end="")
        time.sleep(1)
    print("")
    
    
def Pull_ModalData():
    """
    Initiate the OdbPullModalData.py script
    """
    t0 = time.time()
    print('-----------------------------------------------------------------------')
    print("Pulling Modal Data")
    
        
    path = os.getcwd()
    script_name = path+'/Scripts/OdbPullModalData'
    Folder = path + '/FEM_Results/ModalData'

    os.chdir(Folder)
    os.system('abaqus cae noGUI=' + script_name)
    os.chdir(path)
    
    t1 = time.time()
    
    print('Done! -- ' + str(round(t1 - t0, 3)) + ' sec.')
    print('-----------------------------------------------------------------------')
    

def Pull_modeshapes():
    """
    Initiate the OdbPullModshapes.py script
    """
    
    t0 = time.time()
    print('-----------------------------------------------------------------------')
    print("Pulling Modeshapes")
    
        
    path = os.getcwd()
    script_name = path+'/Scripts/OdbPullModeshapes'
    Folder = path + '/FEM_Results/ModalData'

    os.chdir(Folder)
    os.system('abaqus cae noGUI=' + script_name)
    os.chdir(path)
    
    t1 = time.time()
    
    print('Done! -- ' + str(round(t1 - t0, 3)) + ' sec.')
    print('-----------------------------------------------------------------------')
    
    