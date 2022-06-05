"""
Script to generate data for the optimization algorithm.
All neccecary changes can be made in the "LOOP CONTROL"
Data is stored in: Scripts/Optimize/SampleData
Can be used to generate data for the Langenuen configurations decribed in 2020, 2021 and 2022 master's theses.

Some configurations yield time-increment error in Abaqus. 
Exception's is therefore included in the loop to make sure every configuration make it through the solver.
"""
import os
import sys
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import OptimizeFunctions as OptFunc
import win32com.client as win32

path = os.getcwd()

def GenListOfHeights(lb,ub,step):
    """
    Generate list of heights. 
    Dedicated function to make sure that the upper bound is included.
    
    Input:
    lb, ub, step : Lower bound, Upper bound and step, floats
    
    Output:
    List of heights
    """  
    heights = np.arange(lb, ub, step)
    if heights[-1] != ub:
        heights = np.append(heights, ub)
    op = []
    for h in heights:
        op.append(round(h,2))
    return op

#--------------------------------------------------------------------------------------
# LOOP CONTROL
#--------------------------------------------------------------------------------------

GirderHeights   = {
                    2020 : [5.5,7.0], 
                    2021 : [4.9,6.1], 
                    2022 : [3.5,4.5]
                    }

GirderHeightStep = {'2020' : 0.25, '2021' : 0.25, '2022' : 0.25}
t_heights = GenListOfHeights(180, 220, 2) # 
OverWrite       = {2020 : True, 2021 : True, 2022: True} # If false -> the data will append to the .csv files


p_type          = [2022] # Append year to include multiple years in the same run. 

#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------

for p_type in p_type:
    
    G_heights = GirderHeights[p_type]
    overwrite = OverWrite[p_type]
    step = GirderHeightStep[str(p_type)]    
    g_heights = GenListOfHeights(G_heights[0],G_heights[-1], step)

    V_crs = []
    Costs = []
    
    if overwrite == True:
        w_or_a = 'w'
    elif overwrite == False:
        w_or_a = 'a' 
        
    i = 1
    err_m = 0 # Variable to keep track of abaqus error messages

    for t_height in t_heights:
        for g_height in g_heights:
            ok = 0
            j = 0
            while ok != 1:
                try:
                    j += 1
                    print(f'Sampling data iteration {i} of {len(t_heights)*len(g_heights)}')
                    V_cr = OptFunc.Constraint([t_height, g_height], p_type)
                    V_crs.append(V_cr)
                    
                    if i > 1:
                        w_or_a = 'a'
                        
                    with open(path + f'/Scripts/Optimize/SamplingData/FlutterSpeeds_{p_type}.csv', f'{w_or_a}') as file:
                        file.write(f'{round(t_height,2)}, {round(g_height,2)}, {V_cr}\n')
                        
                    Cost = OptFunc.Objective([t_height, g_height], p_type)
                    Costs.append(Cost)
                    
                    with open(path + f'/Scripts/Optimize/SamplingData/ConfigCost_{p_type}.csv', f'{w_or_a}') as file:
                        file.write(f'{round(t_height,2)}, {round(g_height,2)}, {Cost}\n')
                    i += 1
                    
                
                except Exception: # To handle abaqus time increment error. Minimal adjustments to the heights until the analysis starts.
                    print("--- !! EXCEPTION !! ---")
                    err_m += 1
                    
                    if j == 10:
                        break
                    elif j%2 == 0:
                        if round(t_height,5) == round(t_heights[-1],5):
                            t_height -= 0.005
                        else:
                            t_height += 0.001
                            
                    else:
                        if round(g_height,5) == round(g_heights[-1],5):
                            g_height -= 0.005
                        else:
                            g_height += 0.001
                    continue
                
                
                ok += 1



print(f"--- Error messages: {err_m} ---")