# Input file for the section analysis in Abaqus. Same geometry definitions as Geometry.py
from abaqus import *
from abaqusConstants import *
import __main__
import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior
import numpy as np
import math
import os
#_____________________________________________________________________

#!!! KEEP FOLLOWING LINES AT THEIR CURRENT POSITION
# Initial parameters
H=3.5
tef=0.035
p_type=2022
seed_size=0.01
#LINE 33


if p_type == 2021:
    A = 9.538
    B = 30.64
    C = 1
    D = 4.451
    theta = (A*H-B)/(C*H-D) # In degrees
    theta = theta*2*np.pi/360

 
elif p_type == 2020:
    A = 2.1404
    B = 7.5532
    C = 0.2062
    D = 1
    theta = (A*H-B)/(C*H-D) # In degrees
    theta = theta*2*np.pi/360

if p_type == 2022:
    theta = 16 # In degrees
    theta = theta*2*np.pi/360


if p_type == 2021 or p_type == 2020:
    # Points along mid-line
    x1 = 0
    y1 = 0

    x2 = 15.5
    y2 = -0.03*31/2

    x3 = x2+0.4*H
    y3 = y2-np.tan(np.pi/6)*0.4*H

    x4 = x3-(H+y3)/np.tan(theta)
    y4 = -H
    
    x7 = -15.5
    y7 = -0.03*31/2

    x6 = x7-0.4*H
    y6 = y7-np.tan(np.pi/6)*0.4*H

    x5 = x6+(H+y6)/np.tan(theta)
    y5 = -H

    x8 = x1
    y8 = y1
    
elif p_type == 2022:
    h_incU2 = 1.310
    h_incU1 = 1.190
    h_fall2 = 0.330
    h_fall1 = 0.450
    h_incL = H - h_fall1 - h_incU1

    w_t = 31.0
    w_l2 = 11.0
    w_l1 = 15.0
    w_incU = (w_t-w_l1-w_l2)/2
    w_incL = h_incL/np.tan(theta)
    w_L = w_t - 2*w_incL + 2*w_incU

    # Coordinates
    x1 = 0
    y1 = 0

    x2 = w_l1
    y2 = -h_fall1

    x3 = x2 + w_incU
    y3 = y2 - h_incU1

    x4 = x3 - w_incL
    y4 = -H

    x7 = -w_l2
    y7 = -h_fall2

    x6 = x7 - w_incU
    y6 = y7 - h_incU2

    x5 = x6 + w_incL
    y5 = -H

    x8 = x1
    y8 = y1

# Offsetting functions
#____________________________________________________________________
def normalizeVec(x,y):
    normx = x/np.sqrt(x**2+y**2+1e-20)
    normy = y/np.sqrt(x**2+y**2+1e-20)
    
    return normx, normy


def makeOffsetPoly(oldX, oldY, offset, outer_ccw = 1):
    num_points = len(oldX)
    newX = []
    newY = []
    
    for curr in range(num_points):
        prev = (curr + num_points - 1) % num_points
        next = (curr + 1) % num_points

        vnX =  oldX[next] - oldX[curr]
        vnY =  oldY[next] - oldY[curr]
        vnnX, vnnY = normalizeVec(vnX,vnY)
        nnnX = vnnY
        nnnY = -vnnX

        vpX =  oldX[curr] - oldX[prev]
        vpY =  oldY[curr] - oldY[prev]
        vpnX, vpnY = normalizeVec(vpX,vpY)
        npnX = vpnY * outer_ccw
        npnY = -vpnX * outer_ccw

        bisX = (nnnX + npnX) * outer_ccw
        bisY = (nnnY + npnY) * outer_ccw

        bisnX, bisnY = normalizeVec(bisX,  bisY)
        bislen = offset /  np.sqrt(1 + nnnX*npnX + nnnY*npnY)

        newX.append(oldX[curr] + bislen * bisnX)
        newY.append(oldY[curr] + bislen * bisnY)

    return newX, newY
#___________________________________________________________________________

# Offsetting points
tef = 1.414*tef
off = tef/2

x = [x1, x2, x3, x4, x5, x6, x7, x8]
y = [y1, y2, y3, y4, y5, y6, y7, y8]
z = [0, 0, 0, 0, 0, 0, 0]

x_i, y_i = makeOffsetPoly(x, y, -off)
x_o, y_o = makeOffsetPoly(x, y, off)

# Correcting round-off errors
temp_x = (x_i[0]+x_i[-1])/2
x_i[0] = temp_x
x_i[-1] = temp_x

temp_x = (x_o[0]+x_o[-1])/2
x_o[0] = temp_x
x_o[-1] = temp_x

temp_y = (y_i[0]+y_i[-1])/2
y_i[0] = temp_y
y_i[-1] = temp_y

temp_y = (y_o[0]+y_o[-1])/2
y_o[0] = temp_y
y_o[-1] = temp_y


# Creating point-coordinate lists
pts_i = np.zeros([len(x_i),2])
pts_o = np.zeros([len(x_i),2])
for i in range(len(x_i)):
    pts_i[i,0] = x_i[i]
    pts_i[i,1] = y_i[i]
    
for i in range(len(x_o)):
    pts_o[i,0] = x_o[i]
    pts_o[i,1] = y_o[i]
    
#__________________________________________________________





# -------------------------------------- INITIALIZATION -----------------------------------------

MOD = mdb.models["Model-1"]
myAssembly = MOD.rootAssembly
myAssembly.DatumCsysByDefault(CARTESIAN)

# -------------------------------------- GIRDER SECTION -----------------------------------------

# PART CREATION
MOD.ConstrainedSketch(name="polygon", sheetSize=200.0)

for i in range(len(pts_o)-1):
    MOD.sketches["polygon"].Line(point1=(pts_o[i][0], pts_o[i][1]), point2=(pts_o[i+1][0],pts_o[i+1][1]))
for i in range(len(pts_i)-1):
    MOD.sketches["polygon"].Line(point1=(pts_i[i][0], pts_i[i][1]), point2=(pts_i[i+1][0],pts_i[i+1][1]))
MOD.Part(name="GIRD_SEC", dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
PART = MOD.parts["GIRD_SEC"]

PART.BaseShell(sketch=MOD.sketches["polygon"])


# PART SETS
set_girder = MOD.parts["GIRD_SEC"].Set(name = "whole", faces = PART.faces)

# MATERIAL
MOD.Material(name="ALU")
MOD.materials['ALU'].Density(table=((2700.0, ), ))
MOD.materials["ALU"].Elastic(table=((70e9, 0.3),))

# SECTION ASSIGNMENT
MOD.HomogeneousSolidSection(material="ALU", name="sec", thickness=None)
MOD.parts["GIRD_SEC"].SectionAssignment(region = set_girder, sectionName = "sec")
MOD.parts["GIRD_SEC"].assignBeamSectionOrientation(region = set_girder, method = N1_COSINES,
n1 = (0.0, 0.0, -1.0))


# ASSEMBLY
inst_girder = myAssembly.Instance(name="GIRD_INST", part=PART, dependent=ON)


# MESH
PART.seedPart(size=seed_size, deviationFactor=0.01, minSizeFactor=0.01)
elemType1=mesh.ElemType(elemCode=WARP2D4)
PART.setElementType(regions=set_girder, elemTypes=(elemType1,))
PART.generateMesh()





MOD.keywordBlock.synchVersions(storeNodesAndElements=False)

line_num = 0
for n, line in enumerate(MOD.keywordBlock.sieBlocks):
    line_num += 1

kwds = "*STEP \n*BEAM SECTION GENERATE \n*SECTION ORIGIN, ORIGIN=CENTROID \n**SECTION POINTS \n**113, 190, 1 \n*END STEP"
MOD.keywordBlock.insert(position=line_num-1, text=kwds)




# ---------------------------------------- CREATE JOB -------------------------------------------
#path = os.getcwd()
jobName="section_properties"
myJob=mdb.Job(name=jobName, model="Model-1", multiprocessingMode=DEFAULT, numCpus=4,numDomains=4, numGPUs=0)
myJob.submit(consistencyChecking=OFF)

# ---------------------------------------- END --------------------------------------------------
