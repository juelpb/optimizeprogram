""" 
Functions to calculate the constraint function (stability limit) and objective function (cost) for configurations
"""
import os
import sys
import numpy as np
import time
import ModalAnalysis
import datetime
import matplotlib.pyplot as plt
import Quantities
from Scripts.FlutterAnalysis.FunctioncodePrelim import flutter_speed_DB
from Scripts.FlutterAnalysis.Functioncode import flutter_speed
from Scripts.HelperFunctions import Pull_ModalData, Pull_modeshapes, sleep
from scipy import optimize as Opt
import sklearn.gaussian_process as gpr
from mpl_toolkits import mplot3d

path = os.getcwd()

def Constraint(x0, p_type):
    """Calculates stability limit for configuration

    Args:
        x0 (list): Main parameters. [Tower height, Girder height]
        p_type (int): Parametrization type.

    Returns:
        float: Stability limit (Critical wind velocity)
    """
    t_height = x0[0]
    g_height = x0[1]
    p_type = p_type
    
    
    UserParameterFolder = path + '/Scripts/Optimize/'
    UserParameterFilename = 'LangenuenInput.py'
    #Running modal analysis
    ModalAnalysis.Run(UserParameterFilename, UserParameterFolder, t_height, g_height, p_type)
    
    sleep(2)
    #Pulling data from modal analysis
    Pull_ModalData()
    Pull_modeshapes()
    
    #Calculating stability limit and fluttermode
    
    V_cr = flutter_speed(g_height,p_type,plot=False,p=False)
    
    return V_cr

def ConstraintDB(x0, p_type):
    """Calculates stability limit for configuration

    Args:
        x0 (list): Main parameters. [Tower height, Girder height]
        p_type (int): Parametrization type.

    Returns:
        float: Stability limit (Critical wind velocity)
    """
    t_height = x0[0]
    g_height = x0[1]
    p_type = p_type
    
      
    # Calculating stability limit and fluttermode
    V_cr = flutter_speed_DB(g_height,t_height,p_type,plot=False,p=False)
    
    return V_cr
    
def Objective(x0, p_type):
    """Calculate material cost for configuration

    Args:
        x0 (list): Main parameters. [Tower height, Girder height]
        p_type (int): Parametrization type. 2020, 2021, 2022

    Returns:
        float: Material cost of configuration
    """
    t_height = x0[0]
    g_height = x0[1]
    p_type = p_type
    
    #! Prices
    Alu_girder_cost = 326948    # [NOK/m3]
    Conc_tower_cost = 86535     # [NOK/m3]
    Steel_cable_cost = 691358  # [NOK/m3]
    Steel_hanger_cost = 1543210 # [NOK/m3]
    Total_cost = Quantities.GirderVolume(g_height, p_type)*Alu_girder_cost + Quantities.TowerVolume(t_height)*Conc_tower_cost + Quantities.CableVolume(t_height, g_height, p_type)*Steel_cable_cost + Quantities.HangerVolume(t_height, g_height, p_type)*Steel_hanger_cost

    return Total_cost



def Optimize(p_type, threshold, plot=True):
    """Creates surrogate models for stability limit and material cost from sampling data, stores configuration which passes stability criteria and find optimal solution

    Args:
        p_type (int): Parametrization type. 2020, 2021, 2022
        threshold (float): Stability limit criteria (wind speed [m/s])
        plot (bool, optional): Wether to plot surrugate models. Defaults to True.

    Returns:
        list: Optimal main parameters. [Tower height, Girder height]
    """
    # Create GPR for V_cr
    vcr_data = np.loadtxt(path + f'/Scripts/Optimize/SamplingData/FlutterSpeeds_{p_type}.csv', delimiter=',')
    cost_data = np.loadtxt(path + f'/Scripts/Optimize/SamplingData/ConfigCost_{p_type}.csv', delimiter=',')
    t_heights = vcr_data[:,0]
    g_heights = vcr_data[:,1]
    X = [[i,j] for i in t_heights for j in g_heights]
    X = np.array(X)
    V_cr = vcr_data[:,2]
    Cost = cost_data[:,2]
    Xs = []
    for i in range(len(t_heights)):
        Xs.append([t_heights[i], g_heights[i]]) 
    
    Xs = np.array(Xs)

    t_height_low = np.amin(t_heights)
    t_height_high = np.amax(t_heights)
    t_height_step = (t_height_high - t_height_low)/len(t_heights)
    
    g_height_low = np.amin(g_heights)
    g_height_high = np.amax(g_heights)
    g_height_step = (g_height_high - g_height_low)/len(g_heights)
    
    vcr_kernel = gpr.kernels.Matern([t_height_step, g_height_step], ([1, 10],[1, 10]), nu=2.5)
    vcr_model = gpr.GaussianProcessRegressor(kernel=vcr_kernel,
                                        alpha=1e-10)
                                        #normalize_y=True)
    cost_kernel = gpr.kernels.Matern([t_height_step, g_height_step], ([1, 10],[1, 10]), nu=2.5)
    cost_model = gpr.GaussianProcessRegressor(kernel=cost_kernel,
                                        alpha=1e-10)
    
    vcr_model.fit(Xs, V_cr)
    cost_model.fit(Xs, Cost)

    mesh = 50
    x0 = np.linspace(t_height_low, t_height_high, mesh)
    x1 = np.linspace(g_height_low, g_height_high, mesh)
    x0x1 = [[i,j] for i in x0 for j in x1]
    x0x1 = np.array(x0x1)
    x0,x1 = np.meshgrid(x0,x1)


    vcr_pred, std = vcr_model.predict(x0x1, return_std=True)
    cost_pred, std = cost_model.predict(x0x1, return_std=True)
    
    # Stores configurations passing the criteria
    configs = []
    for i in range(len(vcr_pred)):
        if vcr_pred[i] > threshold:
            configs.append([x0x1[i,0], x0x1[i,1], cost_pred[i], vcr_pred[i]])
    
    # Chooses cheapest alternative 
    configs_arr = np.array(configs)
    Optimal = configs_arr[np.argmin(configs_arr[:,2])]
    
    Z_vcr = np.reshape(vcr_pred,(mesh,mesh), order='F')
    Z_cost = np.reshape(cost_pred,(mesh,mesh), order='F')

    if plot == True:
        # Plot GPR, scatter samples
        fig = plt.figure(figsize=(10,5))
        from matplotlib.gridspec import GridSpec
        # --------------------
        # FIGURE 1
        # --------------------
        gs = GridSpec(3,4)
        ax = plt.subplot(gs[:2,:2],projection='3d')            
        surf = ax.plot_surface(x0, x1, Z_vcr, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False, alpha=0.7)
        
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d
        ax.scatter(Optimal[0], Optimal[1], Optimal[3], color='red',s=150,marker='*',label="Optimal")
        ax.scatter(t_heights, g_heights, V_cr, color='black',s=3,label='Data Points', alpha=0.5)
        ax.set_xlabel('$h_t$')
        ax.set_ylabel('$h_g$')
        ax.set_zlabel('m/s',rotation=90)
        ax.set_yticks([3.5 ,3.75 ,4.0 ,4.25 ,4.5])
        ax.text2D(-0.1, 0.95, 'Critical wind speed', transform=ax.transAxes)
        props = dict(boxstyle='square', facecolor='#cbcefb', alpha=0.9)
        txtstr = f'Optimal configuration: $h_t$ = {Optimal[0]}m | $h_g$ = {round(Optimal[1],3)}m | Cost = {round(Optimal[2]/1e6,2)}MNOK ' '| $V_{cr}$'f' = {round(Optimal[3],2)} m/s'
        ax.text2D(0.15, 1.1, txtstr,transform=ax.transAxes,bbox=props)
        ax.legend(loc=1,prop={'size':7})
        # --------------------
        # SUB-FIGURE 1
        # --------------------
        ax = plt.subplot(gs[-1,0],projection='3d')            
        surf = ax.plot_surface(x0, x1, Z_vcr, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False, alpha=0.7)
        
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d
        ax.scatter(Optimal[0], Optimal[1], Optimal[3], color='red',s=100,marker='*',label="Optimal")
        ax.scatter(t_heights, g_heights, V_cr, color='black',s=1,label='Data Points', alpha=0.5)
        ax.view_init(20,152)
        # --------------------
        # SUB-FIGURE 2
        # --------------------
        ax = plt.subplot(gs[-1,1],projection='3d')            
        surf = ax.plot_surface(x0, x1, Z_vcr, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False, alpha=0.7)
        
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d
        ax.scatter(t_heights, g_heights, V_cr, color='black',s=1,label='Data Points', alpha=1)
        ax.scatter(Optimal[0], Optimal[1], Optimal[3], color='red',s=100,marker='*',label="Optimal")
        ax.set_zticks([])
        ax.view_init(90,-90)
               
        # --------------------
        # FIGURE 2
        # --------------------
        ax = plt.subplot(gs[:,2:],projection='3d')           
        surf = ax.plot_surface(x0, x1, Z_cost, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False, alpha=0.7)
        #surf = ax.plot_trisurf(configs_arr[:,0], configs_arr[:,1], configs_arr[:,2], color='k', alpha=0.5,label='Stable Area')

        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d
        ax.scatter(Optimal[0], Optimal[1], Optimal[2], color='red',s=150,marker='*',label='Optimal')
        ax.scatter(t_heights, g_heights, Cost, color='black',s=3, alpha=0.8,label='Data point')
        ax.scatter(configs_arr[:,0], configs_arr[:,1], configs_arr[:,2], color='green', s=9, alpha=0.6, label='$V_{cr} > 76 m/s$')
        ax.set_xlabel('$h_t$')
        ax.set_ylabel('$h_g$')
        ax.set_zlabel('NOK',rotation=90)
        ax.set_yticks([3.5 ,3.75 ,4.0 ,4.25 ,4.5])
        ax.legend(loc=1,prop={'size':7})
        ax.text2D(0.05, 0.95, 'Configuration cost', transform=ax.transAxes)
        plt.tight_layout()
        plt.savefig(path+f'/Scripts/Optimize/Optimize_Figure_{p_type}.pdf')#,dpi=600)
        #plt.show()

    
    print(f'Optimal design is:\nTowerHeight: {Optimal[0]}m\nGirderHeight: {round(Optimal[1],3)}m\nfor the cost of {round(Optimal[2]/1e6,2)}MNOK\nCritical wind speed is {round(Optimal[3],2)}')
    return Optimal

Optimize(2022, 76, plot=True)