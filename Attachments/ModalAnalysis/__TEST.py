import numpy as np
import os
import sys
import csv
import shutil
# --------------------------------------------------
# SETTING UP PATH IN ORDER TO CALL SCRIPTS
# --------------------------------------------------
#?os.chdir('C:/Users/juelpb/Desktop/langenuenprogram')
path = os.getcwd()
sys.path.append(path)
ThisFolder = path + '/Attachments/ModalAnalysis'


t_H = 206
g_H = 4.0

FolderPath = ThisFolder + f'/Data/MD_{t_H}_{g_H}'

if os.path.exists(FolderPath):
    shutil.rmtree(FolderPath)
    
    
source_dir  = path + '/03_Results/ModalData/'
shutil.copytree(source_dir, FolderPath)


