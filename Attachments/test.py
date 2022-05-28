import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import csv
path = os.getcwd()
sys.path.append(path)
ThisFolder = path + '/Attachments/ModalAnalysis'
from Scripts.FunctionsOld.Evaluate_modeshapes import Evaluate_modeshapes, get_modeshape


modeshape = get_modeshape(2, 2)

plt.figure()
plt.plot(modeshape)
plt.ylim(-1,1)
plt.show()