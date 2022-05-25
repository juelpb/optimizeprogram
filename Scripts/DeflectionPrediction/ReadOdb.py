""" 
Abaqus input script to read odb-file
"""

from odbAccess import *
from abaqusConstants import *
from odbMaterial import *
from odbSection import *
import os

path = os.getcwd()

odb = openOdb(path= path + '/FEM_Results/GlobalAnalysis/TestLangenuen.odb')

lastFrame = odb.steps['STEP1'].frames[-1]

displacement=lastFrame.fieldOutputs['U']

fieldValuesU=displacement.values

disp = []

for v in fieldValuesU:
    disp.append('%d,%6.4f,%6.4f,%6.4f' % (v.nodeLabel,
    v.data[0], v.data[1], v.data[2]))


    
with open(path + '/Scripts/DeflectionPrediction/step1_disp.csv','w') as file:
    for i in range(len(disp)):
        file.writelines(disp[i] + '\n')



lastFrame = odb.steps['STEP4'].frames[-1]

displacement=lastFrame.fieldOutputs['U']
forces=lastFrame.fieldOutputs['SF']

fieldValuesU=displacement.values
fieldValuesSF=forces.values

disp = []

for v in fieldValuesU:
    disp.append('%d,%6.4f,%6.4f,%6.4f' % (v.nodeLabel,
    v.data[0], v.data[1], v.data[2]))

sf = []

for v in fieldValuesSF:
    sf.append('%d,%6.4f, %6.4f, %6.4f' % (v.elementLabel, v.data[0], v.data[1], v.data[2]))


odb.close()
    
with open(path + '/Scripts/DeflectionPrediction/step4_disp.csv','w') as file:
    for i in range(len(disp)):
        file.writelines(disp[i] + '\n')

with open(path +'/Scripts/DeflectionPrediction/step4_sf.csv', 'w') as file:
    for i in range(len(sf)):
        file.writelines(sf[i] + '\n')
        


