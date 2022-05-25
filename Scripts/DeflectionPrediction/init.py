""" 
Script to iterate thorugh girder height and tower height and write deflections to csv-file
"""
import DefPred
import numpy as np

# Comment out to overwrite csv-file and write column labels
#with open(path + '/Scripts/DeflectionPrediction/DeflectionPrediction.csv', 'w') as file:
    #file.write('Tower height, Girder height, Param. type, Girder deformation south end, Girder deformation midspan, Girder deformation north end, Cable deformation midspan, Tower deformation, Max. tension in cables, Max. tension in hangers\n')


p_type = 2022
# Tower height interval
lb_t = 180 # lower bound 
ub_t = 225 # upper bound 
step_t = 5 # 

# Girder height interval
lb_g = 3.5 # lower bound 
ub_g = 4.6 # upper bound 
step_g = 25 


# Initial deflection predictions (Used only for first iteration)
c_d_m=14.76 # Cable deflection at midspan (U3)
g_d_m=9.87 # Girder deflection at midspan (U3)
g_d_s=0.42 # Girder deflection at south end (U3)
g_d_n=0.42 # Girder deflection at north end (U3)
t_d=0.69 # Tower pullback deflection (in x-dir) (U1)
c_sf_max=194e6 # Max. tension in main cables
h_sf_max=793e3 # Max. tension in hangers


initial = [lb_t, lb_g/100, p_type, g_d_s, g_d_m, g_d_n, c_d_m, t_d, c_sf_max, h_sf_max]
# Iterating through girder heights and tower heights
for i, g_height in enumerate(np.arange(lb_g, ub_g, step_g)):
    for j, t_height in enumerate(np.arange(lb_t, ub_t, step_t)):  
        DefPred.Run(t_height, g_height, p_type, initial)




