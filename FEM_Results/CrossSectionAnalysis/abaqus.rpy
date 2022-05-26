# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2017 replay file
# Internal Version: 2016_09_27-23.54.59 126836
# Run by sverr on Sat May 21 10:16:02 2022
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(1.00586, 1.00116), width=148.063, 
    height=99.3148)
session.viewports['Viewport: 1'].makeCurrent()
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
execfile(
    'c:/Users/sverr/Desktop/langenuenprogram/Scripts/CrossSection/GenPropInput.py', 
    __main__.__dict__)
#: Abaqus Warning: The following input options are not supported by parallel execution of element operations: beamsectiongenerate. Only the solver will be executed in parallel for this analysis.
print 'RT script done'
#: RT script done
