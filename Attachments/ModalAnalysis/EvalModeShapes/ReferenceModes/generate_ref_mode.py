# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:37:01 2022

@author: juelpb
"""

import os 
import csv
import matplotlib.pyplot as plt

#os.chdir('C:/Users/juelp/Desktop/langenuenprogram')
path = os.getcwd()

# #  ---------------- Change ref modes here! ------------------------
# current_ref_modes = {'VS1' : 4,
#                     'TS1' : 14,
#                   }
# # -----------------------------------------------------------------
# ---------------- Change ref modes here! ------------------------
current_ref_modes = {'HS1' : 1,
                     'VA1' : 2,
                     'HA1' : 3,
                     'VS1' : 4,
                     'VS2' : 5,
                     'VA2' : 8,
                     'HS2' : 12,
                     'VS3' : 13,
                     'VA3' : 16,
                     'TS1' : 19,
                     'VS4' : 31,
                     'TA1' : 30,
                     'HA2' : 35,
                     'VA4' : 37
                     }
# -----------------------------------------------------------------
def Normalize_lst(lst):
    #TODO: Function should be moved to HelperFunctions
    """
    Input:
    lst = 1-dimentional list
    
    Output:
    Normalizes list's between -1 (min val) and 1 (max val)
    """
    
    
    norm_lst = []
    max_val = max(lst)
    min_val = min(lst)
    for i in range(len(lst)):
        val = 2*(lst[i] - min_val) / (max_val - min_val) -1
        norm_lst.append(val)

    return norm_lst


def generate_ref_mode(mode,row_idx,filename,tH,gH):

    PathToModeShape = path + f'/FEM_Results/MD_database/MD_{tH}_{gH}/ModeShapes'
    
    file = open(PathToModeShape + f'/Mode_{mode}.csv','r')
    reader = csv.reader(file)
    data = []
    for row in reader:
        data.append(float(row[row_idx]))

    data_norm = Normalize_lst(data)
    #data_norm = data
    

    # plt.figure()
    # plt.plot(data)
    # plt.plot(data_norm)
    # plt.title(filename)
    # plt.show()


    with open(path + f'/Attachments/ModalAnalysis/EvalModeShapes/ReferenceModes/{filename}_ref.csv', 'w', newline='') as output:
        writer = csv.writer(output)
        writer.writerows([[num] for num in data_norm])
        
        
def GENERATE_REF_MODES(current_ref_modes,tH,gH):

    PathToModeShape = path + f'/FEM_Results/MD_database/MD_{tH}_{gH}/ModeShapes'

    for name, idx in current_ref_modes.items():
        if (name[:-1] == 'VS') or name[:-1] == 'VA':
            direc = 3
            generate_ref_mode(idx, direc, name,tH,gH)
            
        if (name[:-1] == 'TS') or (name[:-1] == 'TA'):
            direc = 4
            generate_ref_mode(idx, direc, name,tH,gH)
            
        if (name[:-1] == 'HS') or (name[:-1] == 'HA'):
            direc = 2
            generate_ref_mode(idx, direc, name,tH,gH)
        




GENERATE_REF_MODES(current_ref_modes, 206,4.0)
