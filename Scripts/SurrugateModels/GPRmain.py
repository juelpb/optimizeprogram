"""
Script for generation of surrogate models Aerodynamic derivatives with Gaussian process regression.

The RAW AD-data is found under Scripts/SurrugateModels/WindTunnelData_{year}.
- Structure/Layout on the .xlsx-files are important in order for labdata_to_csv to run.

- The GP-regressor and functions is decribed in gpr.Functions.py

- Thought workflow is decribed below:
"""

import os
import sys
path = os.getcwd()
sys.path.append(path)
from Scripts.SurrugateModels.GPRfunctions import plot_single_model, plot_all_models, write_GPRdata_to_xl, push_GPRdata_to_fluttercode, labdata_to_csv

#* --------------------------------- STEP 1 -----------------------------------
    #*Select the desired year to work with
year = 22

#* --------------------------------- STEP 2 -----------------------------------
#   !(Should only be done once)
#*   Convert lab-data to .csv-files for speed and convenience

#labdata_to_csv(year)

#* --------------------------------- STEP 3 -----------------------------------
    #* Plot the surrogate models for the AD's to make sure these are OK.

#plot_single_model(year, 'P4')
plot_all_models(year)

#* --------------------------------- STEP 4 -----------------------------------
    #* Write predicted data to excel-sheet, and .csv-files for the fluttercode. 
    #* Storing in an easy-to-read excel comes in handy if encountering errors or unexpected results

#write_GPRdata_to_xl(year)
#push_GPRdata_to_fluttercode(year)