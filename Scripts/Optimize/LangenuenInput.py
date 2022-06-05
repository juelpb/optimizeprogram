# Input file for modal analysis in Abaqus
import numpy as np
from ypstruct import *
import numpy as np
import numtools
import gen
import warnings
import os

##########################
##########################
# !!! KEEP FOLLOWING LINES AT THEIR CURRENT POSITION !!!
t_height=206
girder_def_mid=8.721851949649537
girder_def_south=0.5243842016166426
girder_def_north=0.5232823103811524
cable_def_mid=10.804935343024706
tower_def=1.1921376126852121
cable_sf_max=154767878.83242053
hanger_sf_max=827265.338218634
girder_area=2.4019285714285714
girder_I11=6.363957142857143
girder_I22=186.54142857142858
girder_It=21.892346153846155
mass=13159.881192150466
i11=19009.24158953395
i22=864359.1905651536
# LINE 28
#
max_stress_cable = 500e6
cable_area = cable_sf_max/max_stress_cable

max_stress_hanger = 500e6
hanger_area = hanger_sf_max/max_stress_hanger

abaqus=struct()


path = os.getcwd()
abaqus.FolderNameModel=path + '/FEM_Results/GlobalAnalysis'
abaqus.InputName='LangenuenGlobal'
abaqus.JobName='LangenuenGlobal'
abaqus.PartName='SuspensionBridge'
abaqus.AssemblyName='AssemblySuspensionBridge'
abaqus.RunJob=True
abaqus.cmd='abaqus'
abaqus.cpus=np.array(4)
abaqus.restart=False
abaqus.halt_error=True

##########################
##########################
##########################

step=struct()

step.time=[None]*4
step.time[0]=np.array([1e-1, 1, 1e-6, 1])
step.time[1]=np.array([1e-8, 1, 1e-12, 1])
step.time[2]=np.array([1e-8, 1, 1e-12, 1])
step.time[3]=np.array([1e-1, 1, 1e-6, 1])

##########################
##########################
##########################

modal=struct()

modal.N_modes=np.array(100)
modal.normalization='displacement'

##########################
##########################
##########################

cable=struct()
cable.cs=struct()

cable.cs.A=np.array(cable_area) #Initial 0.37
cable.cs.I11=0.8*cable.cs.A**2/(4*np.pi)*0.01
cable.cs.I22=0.8*cable.cs.A**2/(4*np.pi)*0.01
cable.cs.I12=np.array(0)
cable.cs.It=cable.cs.A**2/(2*np.pi)*0.01
cable.cs.rho=np.array(7850)
cable.cs.E=np.array(200e9)
cable.cs.G=np.array(80e9)
cable.cs.sigma_target=np.nan

cable.normaldir=np.array([0,1,0])
cable.eltype='B33'
cable.meshsize_approx=np.nan
cable.N_element=np.array(50)
cable.tempsupport=True
cable.N_tempsupport=np.array(11)
cable.polycoeff_hanger_adjust=np.nan

cable.NodeNumberBase=np.array([10e3,20e3])
cable.ElementNumberBase=np.array([10e3,20e3])

##########################
##########################
##########################

bridgedeck=struct()
bridgedeck.cs=struct()
bridgedeck.inertia=struct()
bridgedeck.gapbeam=struct()

bridgedeck.N_box=1

bridgedeck.cs.A=np.array([girder_area])
bridgedeck.cs.I11=np.array([girder_I11])
bridgedeck.cs.I22=np.array([girder_I22])
bridgedeck.cs.I12=np.array([0])
bridgedeck.cs.It=np.array([girder_It])
bridgedeck.cs.rho=np.array([0.0])
bridgedeck.cs.E=np.array([70e9])
bridgedeck.cs.G=np.array([26e9])
bridgedeck.cs.sc1=np.array([0.0])
bridgedeck.cs.sc2=np.array([-1.4])

bridgedeck.inertia.m=np.array([mass])
bridgedeck.inertia.x1=np.array([0.0])
bridgedeck.inertia.x2=np.array([0.0])
bridgedeck.inertia.alpha=np.array([0])
bridgedeck.inertia.I11=np.array([i11])
bridgedeck.inertia.I22=np.array([i22])
bridgedeck.inertia.I12=np.array([0])

bridgedeck.normaldir=np.array([0,1,0])
bridgedeck.eltype='B31'
bridgedeck.meshsize=np.array(4)
bridgedeck.shell=True

bridgedeck.gapbeam.type='box'
bridgedeck.gapbeam.h=np.array(2)
bridgedeck.gapbeam.b=np.array(2)
bridgedeck.gapbeam.t=np.array(0.02)

bridgedeck.NodeNumberBase=np.array([1e3,2e3,3e3,4e3,5e3])
bridgedeck.ElementNumberBase=np.array([1e3,2e3])
bridgedeck.NodeNumberBaseOuter=np.array([11e3,12e3])
bridgedeck.ElementNumberBaseConnLat=np.array([3e3,4e3,5e3,6e3,7e3,8e3])

##########################
##########################
##########################

hanger=struct()
hanger.cs=struct()

hanger.cs.A=np.array(hanger_area) # Initial 1.7e-3
hanger.cs.I11=np.array(1e-8)
hanger.cs.I22=np.array(1e-8)
hanger.cs.I12=np.array(0)
hanger.cs.It=np.array(1e-8)
hanger.cs.rho=np.array(7850.0)
hanger.cs.E=np.array(160e9)
hanger.cs.G=np.array(60e9)

hanger.normaldir=np.array([0,1,0])
hanger.eltype='B31'

hanger.ElementNumberBase=np.array([80e3,90e3])

##########################
##########################
##########################

tower=struct()
tower.cs=struct()

tower.cs.type='box'
tower.cs.h_vec=np.array([7.5,5,5,5])
tower.cs.b_vec=np.array([7.5,5,4,4])
tower.cs.t_vec=np.array([1.0,1.0,0.6,0.6])
tower.cs.z_vec=np.array([0.0,40.0,180.0,220.0])
tower.cs.rho=np.array(2500.0)
tower.cs.E=np.array(35e9)
tower.cs.v=np.array(0.2)

tower.normaldir=np.array([0,1,0])
tower.eltype='B31'

tower.F_pullback_south=np.array(-6e5) #np.nan
tower.F_pullback_north=np.array(6e5) #np.nan

tower.z_crossbeam_south=np.array([60.0,t_height-1])
tower.z_crossbeam_north=np.array([60.0,t_height-1])

tower.h_crossbeam=np.array([6.0,4.0])
tower.b_crossbeam=np.array([4.0,4.0])
tower.t_crossbeam=np.array([0.6,0.6])

tower.N_element=np.nan
tower.meshsize=np.array(5.0)

tower.NodeNumberBase=np.array([100e3,110e3,200e3,210e3])
tower.ElementNumberBase=np.array([100e3,110e3,200e3,210e3])

##########################
##########################
##########################

bearing=struct()

bearing.type='tri'
bearing.stiffness_south=np.array([3e7,1e12,1e6,1e0,1e0,1e0])
bearing.stiffness_north=np.array([3e7,1e12,1e6,1e0,1e0,1e0])

bearing.NodeNumberBase=np.array([400e3])
bearing.ElementNumberBase=np.array([400e3])

##########################
##########################
##########################

sadle=struct()

sadle.stiffness=np.array([1e12])

sadle.ElementNumberBase=np.array([300e3])

##########################
##########################
##########################

geo=struct()

geo.L_bridgedeck=np.array(1235.0)
geo.gap=np.nan

geo.dx_hanger=np.array(12.0)
geo.dx_endpiece_max=np.array(12.0)

geo.dx_pullback_south=np.array(-tower_def) #*np.nan, initial: -0.8
geo.dx_pullback_north=np.array(tower_def) #*np.nan, initial: 0.8

geo.z_cog_south=np.array(69.0)
geo.z_cog_north=np.array(69.0)
geo.z_cog_midspan=np.array(76.6)
geo.dz_cog_midspan_deflection=np.array(girder_def_mid) # initial: 7.58
geo.dz_cog_south_deflection=np.array(girder_def_south) # initial: 1.0
geo.dz_cog_north_deflection=np.array(girder_def_north) # initial: 1.0

geo.z_cable_top_south=np.array(t_height+0.1)
geo.z_cable_top_north=np.array(t_height+0.1)
geo.z_cable_midspan=np.array(88.8)
geo.dz_cable_deflection=np.array(cable_def_mid) # initial: 9.20

geo.dy_cable_anch_south=np.array(40.0)
geo.dy_cable_top_south=np.array(3.0)
geo.dy_cable_midspan=np.array(32.0)
geo.dy_cable_top_north=np.array(3.0)
geo.dy_cable_anch_north=np.array(40.0)

geo.dy_cog_hanger=np.array(16.0)
geo.dz_cog_hanger=np.array(1.0)
geo.dy_cog_inner=np.array(8.0)*np.nan
geo.dz_cog_inner=np.array(-0.5)*np.nan

geo.dy_pendulum=np.array(4.5)
geo.dz_slider=np.array(-3.0)
geo.dx_bearing_base=np.array(1.0)

geo.z_tower_base_south=np.array(0.0)
geo.z_tower_base_north=np.array(0.0)

geo.z_tower_top_south=np.array(t_height)
geo.z_tower_top_north=np.array(t_height)

geo.dy_tower_base_south=np.array(40.0)
geo.dy_tower_base_north=np.array(40.0)

geo.dy_tower_top_south=np.array(3.0)
geo.dy_tower_top_north=np.array(3.0)

geo.dx_tower_anch_south=np.array(285.0)
geo.dx_tower_anch_north=np.array(285.0)

geo.z_anch_south=np.array(206.0-158.0)
geo.z_anch_north=np.array(206.0-158.0)

geo.x_tower_south=np.array(-1235.0/2)
geo.x_tower_north=np.array(1235.0/2)

