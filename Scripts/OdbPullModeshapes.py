"""
Pull modeshapes from the latest abaqus global analysis. 
The data is stored under: FEM_Results/ModalData/Modeshapes (overwrites current data)

Written in abaqus-python and have to be initiated from Pull_modeshapes in Helperfunctions.py
"""

from odbAccess import *
from abaqusConstants import *
from odbMaterial import *
from odbSection import *
import codecs
import csv
import os

#* Access odb-file

OrigPath = os.getcwd() # Store this
os.chdir('..') # Move one folder up -> 03/FEM_Results
ResultsPath = os.getcwd() + '/GlobalAnalysis/LangenuenGlobal.odb'
odb = openOdb(path= ResultsPath)
os.chdir(OrigPath) # Back to Eigvals to write results

#* Process odb-file

CenterGirderNodes = odb.rootAssembly.instances['SUSPENSIONBRIDGE'].nodeSets['BRIDGEDECK_COG'] # Define the node set that we want to plot

Step = odb.steps['STEP_MODAL'] # Define step
n_steps = 50

# SUSPENSIONBRIDGE.BRIDGEDECK_COG

n_modes_to_plot = 40

for mode in range(n_modes_to_plot):

    frame = Step.frames[mode] 
    disp = frame.fieldOutputs['U'] 
    rotation = frame.fieldOutputs['UR']

    disp = disp.getSubset(region=CenterGirderNodes)
    rotation = rotation.getSubset(region=CenterGirderNodes)

    U = disp.values
    UR = rotation.values

    output_lst = [] # Empty list to collect outputs 

    for i in range(len(U)):
        lst = [U[i].nodeLabel,U[i].data[0],U[i].data[1],U[i].data[2],UR[i].data[0],UR[i].data[1],UR[i].data[2]]
        output_lst.append(lst)


    #* Writing output-file.csv

    with open('ModeShapes/Mode_' + str(mode) + '.csv','wb') as output: # 'wb' to avoid writing every other line
        writer = csv.writer(output)
        for i in range(len(output_lst)):
            writer.writerow(output_lst[i])
        





