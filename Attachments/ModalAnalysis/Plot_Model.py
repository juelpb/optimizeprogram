import numpy as np
import os
import sys
import csv
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
# --------------------------------------------------
# SETTING UP PATH IN ORDER TO CALL SCRIPTS
# --------------------------------------------------
#?os.chdir('C:/Users/juelpb/Desktop/langenuenprogram')
path = os.getcwd()
sys.path.append(path)
ThisFolder = path + '/Attachments/ModalAnalysis'

Data = np.loadtxt(ThisFolder + '/Eigfreq_3D_data.csv',delimiter=',')

def plot_mode_data(n):
    mode = n+1
    eF  = Data[:,mode] 
    print(eF)
    eF = np.reshape(eF, (5,5)).T
    print(eF)
    t_H = np.linspace(180, 220,5)
    g_H = np.array([3.5 ,3.75 ,4.0 ,4.25 ,4.5])

    X, Y = np.meshgrid(t_H,g_H)
    return X, Y, eF
    
    
fig = plt.figure(figsize=(12,5))

ax = fig.add_subplot(121, projection='3d')
X, Y, eF = plot_mode_data(2)                
surf = ax.plot_surface(X, Y, eF, cmap='plasma', linewidth=0, alpha=0.5)
ax.scatter(X, Y, eF, color='black',alpha=0.5)
ax.set_xlabel('Tower height [m]')
ax.set_ylabel('Girder height [m]')
ax.set_zlabel('Frequency [$f$]')
ax.set_title('First vertical asymmetric')
ax.set_yticks([3.5 ,3.75 ,4.0 ,4.25 ,4.5])
ax.set_xticks(np.linspace(180, 220,5))

ax = fig.add_subplot(122, projection='3d')
X, Y, eF = plot_mode_data(30)
surf = ax.plot_surface(X, Y, eF, rstride=1, cstride=1, cmap='plasma', linewidth=0, alpha=0.5)
ax.scatter(X, Y, eF, color='black',alpha=0.5)
ax.set_xlabel('Tower height [m]')
ax.set_ylabel('Girder height [m]')
ax.set_zlabel('Frequency [$f$]')
ax.set_title('First torsional asymmetric')
ax.set_yticks([3.5 ,3.75 ,4.0 ,4.25 ,4.5])
ax.set_xticks(np.linspace(180, 220,5))


#plt.tight_layout()
#plt.savefig(ThisFolder+'/Eigenfreqs.png', dpi=300)
plt.show()


