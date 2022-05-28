# -*- coding: utf-8 -*-

import os 
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath

os.chdir('C:/Users/juelpb/Desktop/langenuenprogram')
path = os.getcwd()
sys.path.append(path)

from Scripts.FlutterAnalysis.Functioncode import flutter_speed


U_flutter, U_modeshape, U_eigen = flutter_speed(4.0, 2022,plot=True)

#%% 

real = np.abs(U_modeshape)
