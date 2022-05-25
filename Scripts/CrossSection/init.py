""" 
Script to iterate through a range of cross-section height and write their properties to a csv-file. Calls on GenProp.py
"""

import GenProp
import numpy as np


# Comment out to overwrite csv-file and write column labels
#with open(path + '/Scripts/CrossSection/SectionProperties.csv', 'w') as file:
    #file.write('Girder height [m], Param. type, Eff. thickness [m], Solid area [m^2], I11 [m^4], I12 [m^4], I22 [m^4], It [m^4], Girder mass [kg/m], Total mass [kg/m], Girder i11 [kg m^2/m], Girder i12 [kg m^2/m], Girder i22 [kg m^2/m], i11 [kg m^2/m], i22 [kg m^2/m], Perimeter [m], Inner area [m^2]\n')

p_type = 2022
tef = 0.035 # effecktive thickness of thin-walled cross section
# Girder height interval
lb_g = 3.5 # Lower bound girder height [m]
ub_g = 4.5 # Upper bound girder height [m]
step_g = 0.125

# Iterating through girder heights
for i, g_height in enumerate(np.arange(lb_g, ub_g, step_g)):
    GenProp.Run(g_height, p_type, tef)