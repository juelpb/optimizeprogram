"""
Script for running a singe analysis/iteration. 
Comes in handy when testing out i.e. different GPR-parameters for the AD surrogate models.
"""

import os
import sys
path = os.getcwd()
sys.path.append(path)

from Scripts.Optimize.ModalAnalysis import Run
from Scripts.FlutterAnalysis.Functioncode import flutter_speed
from Scripts.HelperFunctions import Pull_ModalData, Pull_modeshapes, sleep

UserParameterFolder = path + '/Scripts/Optimize/'
UserParameterFilename = 'LangenuenInput.py'


#------------------------------
# Set parameters for analysis
t_height    = 206
g_height    = 4.0
p_type      = 2022
#------------------------------


Run(UserParameterFilename, UserParameterFolder, t_height, g_height, p_type) # Global analysis
Pull_ModalData()
Pull_modeshapes()
#sleep(2)                                                                    # Give abaqus proper time to close
#flutter_speed(g_height, p_type,p=True,plot=True)                            # Run critical wind speed analysis