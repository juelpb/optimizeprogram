"""
Script to plot the change in eigenfrequencies throughout the parameter change
Loop:
- Enter correct modal data folder
- Evaluate modeshape to identify TS1 and VS1
- Use the idx'es to extract eigenfrequencies
- 3 plots. VS1, TS1 freq and difference
"""
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
#from Scripts.FunctionsOld.Evaluate_modeshapes import Evaluate_modeshapes
#from Scripts.FunctionsOld.ReferenceModes.generate_ref_mode import GENERATE_REF_MODES
ThisFolder = path + '/Attachments/ModalAnalysis'

p_type_ = [2022]
G_HEIGHTS = [3.5 ,3.75 ,4.0 ,4.25 ,4.5]
T_HEIGHTS = np.arange(180,222,2)

V_f     = np.zeros((len(G_HEIGHTS),len(T_HEIGHTS))) # Vertrical frequencies
T_f     = np.zeros((len(G_HEIGHTS),len(T_HEIGHTS))) # Torsional frequencies
Ratio   = np.zeros((len(G_HEIGHTS),len(T_HEIGHTS))) # omega_theta / omega_vertical


idx = np.loadtxt(ThisFolder + '/INDEXES.csv',delimiter=',',unpack=True, encoding="utf-8-sig")
idx = idx.T



for i,g_H in enumerate(G_HEIGHTS):
    for j,t_H in enumerate(T_HEIGHTS):
        MD_path = path + f'/FEM_Results/MD_database/MD_{int(t_H)}_{g_H}'
        eF = np.loadtxt(MD_path + '/Eigenfrequency.csv',delimiter=',')
             
        x = int(idx[i,j])
        

        vf = eF[3,1]
        tf = eF[x,1]
        
        print(t_H,g_H,x)
        
        V_f[i,j]    = vf
        T_f[i,j]    = tf
        Ratio[i,j]  = tf / vf

# print(V_f)
# print(np.min(V_f),np.max(V_f))
# print()
# print(T_f)
# print(np.min(T_f),np.max(T_f))

X, Y = np.meshgrid(T_HEIGHTS,G_HEIGHTS)

fig = plt.figure(figsize=(12,5))
# FIRST FIGURE
ax = fig.add_subplot(131, projection='3d')
surf = ax.plot_surface(X, Y, V_f, cmap='plasma', linewidth=0, alpha=0.5)
ax.scatter(X, Y, V_f, color='black',alpha=0.5,s=1)
ax.set_xlabel('Tower height [m]')
ax.set_ylabel('Girder height [m]')
ax.set_zlabel('Frequency [$f$]')
ax.set_title('First vertical symmetric, VS1')
ax.set_yticks([3.5 ,3.75 ,4.0 ,4.25 ,4.5])
ax.set_xticks(np.linspace(180, 220,5))

# SECOND FIGURE
ax = fig.add_subplot(132, projection='3d')
surf = ax.plot_surface(X, Y, T_f, rstride=1, cstride=1, cmap='plasma', linewidth=0, alpha=0.5)
ax.scatter(X, Y, T_f, color='black',alpha=0.5,s=1)
ax.set_xlabel('Tower height [m]')
ax.set_ylabel('Girder height [m]')
ax.set_zlabel('Frequency [$f$]')
ax.set_title('First torsional symmetric, TS1')
ax.set_yticks([3.5 ,3.75 ,4.0 ,4.25 ,4.5])
ax.set_xticks(np.linspace(180, 220,5))

#ThIRD FIGURE
ax = fig.add_subplot(133, projection='3d')
surf = ax.plot_surface(X, Y, Ratio, rstride=1, cstride=1, cmap='plasma', linewidth=0, alpha=0.5)
ax.scatter(X, Y, Ratio, color='black',alpha=0.5,s=1)
ax.set_xlabel('Tower height [m]')
ax.set_ylabel('Girder height [m]')
ax.set_zlabel('$f_{\Theta}$/$f_{z}$')
ax.set_title('Frequency ratio')
ax.set_yticks([3.5 ,3.75 ,4.0 ,4.25 ,4.5])
ax.set_xticks(np.linspace(180, 220,5))

plt.show()
#plt.savefig(ThisFolder +'/PlotModal.png',dpi=300)