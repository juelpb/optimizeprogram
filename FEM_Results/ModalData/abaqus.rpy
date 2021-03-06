# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2019 replay file
# Internal Version: 2018_09_24-20.41.51 157541
# Run by juelpb on Tue Jun  7 11:41:25 2022
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(1.01054, 1.0075), width=148.752, 
    height=99.9438)
session.viewports['Viewport: 1'].makeCurrent()
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
execfile(
    'C:/Users/juelpb/Desktop/optimizeprogram/Scripts/OdbPullModeshapes.py', 
    __main__.__dict__)
#: Model: C:/Users/juelpb/Desktop/optimizeprogram/FEM_Results/GlobalAnalysis/LangenuenGlobal.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       148
#: Number of Node Sets:          47
#: Number of Steps:              5
print 'RT script done'
#: RT script done
