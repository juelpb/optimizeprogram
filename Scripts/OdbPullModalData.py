"""
Pull modal data from the latest abaqus global analysis. 
The data is stored under: FEM_Results/ModalData (overwrites current data)

Written in abaqus-python and have to be initiated from Pull_modaldata in Helperfunctions.py
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
print("----------------------------")
print(ResultsPath)
print("----------------------------")
odb = openOdb(path=ResultsPath)
os.chdir(OrigPath) # Back to Eigvals to write results

#* Pulling filenames and call-names from HistoryOutputs.csv

HistoryOutputs = []
with open('HistoryOutputs.csv','r') as f:
    r = csv.reader(codecs.EncodedFile(f, 'utf-8', 'utf-8-sig'), delimiter=",")
    for row in r:
        HistoryOutputs.append(row)

#* Process odb-file

Step = odb.steps['STEP_MODAL'] # Define step
region = Step.historyRegions['Assembly ASSEMBLYSUSPENSIONBRIDGE']  # Define the only availible assembly

for name, call in HistoryOutputs:

    hist_output_data = region.historyOutputs[call].data # call on history output

    lst = []        
    for i in range(len(hist_output_data)): # Loop for string-converting and formatting
        col1 = hist_output_data[i][0]
        col2 = hist_output_data[i][1]
        lst.append('%d,%6.4f' % (col1,col2))
    
    #* Writing output-file.csv
    
    with open(name + '.csv','w') as output:
        for i in range(len(lst)):
            output.writelines(lst[i] + '\n')
        





