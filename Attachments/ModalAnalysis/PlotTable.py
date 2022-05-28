import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import csv
path = os.getcwd()
sys.path.append(path)
ThisFolder = path + '/Attachments/ModalAnalysis'
from Scripts.FunctionsOld.Evaluate_modeshapes import Evaluate_modeshapes, get_modeshape

ResultPath = path + '/FEM_Results/ModalData'

eval = Evaluate_modeshapes(ResultPath)

header_ = ['Mode','Abaqus\nno.','Generalized mass\n$\widetilde{M}_{i}$[kg]','Frequency\n[Hz]','Horizontal','Vertical','Torsional']

with open(path + '/FEM_Results/ModalData/Eigenfrequency.csv','r') as f:
        Eigfreq = csv.reader(f)
        eF = []
        for i, row in enumerate(Eigfreq):
            eF.append(row[1])
            
with open(path + '/FEM_Results/ModalData/Generalized_mass.csv','r') as f:
        genMass = csv.reader(f)
        gm = []
        for i, row in enumerate(genMass):
            gm.append(round(float(row[1])))           

def gen_row_data(mode,name):
    D = {}
    
    D['Mode']               = name
    D['Abaqus no']          = mode
    D['Generalized Mass']   = gm[mode-1]
    D['Eigenfreq']          = eF[mode-1]
    
    D['H'] = list(get_modeshape(mode, 2,ResultPath))
    D['V'] = list(get_modeshape(mode, 3,ResultPath))
    D['T'] = list(get_modeshape(mode, 4,ResultPath))
     
    return D

# D = gen_row_data(3, 'VS1')
# plt.figure()
# plt.plot(D['H'])
# plt.show()



ncols = 7
nrows = len(eval.values())+1

header_height = 0.2
row_height = 0.1
height_ratios = []
for i in range(nrows):
    if i == 0:
        height_ratios.append(header_height)
    else:
        height_ratios.append(0.1)

fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12,6),
                         gridspec_kw={"width_ratios":[0.5,0.7,1.0,0.7,1,1,1],"height_ratios":height_ratios},
                                                 
                         )
                        
fig.subplots_adjust(0.01,0.01,0.99,0.99, wspace=0.0, hspace=0)

for ax in axes.flatten():
    ax.tick_params(labelbottom=0, labelleft=0, bottom=0, top=0, left=0, right=0)
    ax.ticklabel_format(useOffset=False, style="plain")
    for _,s in ax.spines.items():
        s.set_visible(True)

#------------
# HEADER
#------------
text_kw = dict(ha="center", va="center", size=13)
for i,ax in enumerate(axes[0,:]):
    ax.text(0.5, 0.5, header_[i], transform=ax.transAxes, **text_kw)
    ax.patch.set_color('#cbcefb')
#------------
# ROWS
#------------
mode_name = list(eval.keys())
mode_number = list(eval.values())
print(mode_number)

for row in range(1,nrows):
    row_data = gen_row_data(mode_number[row-1], mode_name[row-1])
    
    row_data_lst = list(row_data.values())
    for i,ax in enumerate(axes[row,:]):
        if i <= 3:
            ax.text(0.5, 0.5, row_data_lst[i], transform=ax.transAxes, **text_kw)
        elif i == 4:
            ax.plot(row_data['H'], color="#1f78b4", linewidth=1)
            ax.set_ylim(-1.4,1.4)
        elif i == 5:
            ax.plot(row_data['V'], color="#e31a1c", linewidth=1)
            ax.set_ylim(-1.4,1.4)
        elif i == 6:
            ax.plot(row_data['T'], color="#33a02c", linewidth=1)
            if row_data['Mode'][:-2] == 'T':
                ax.set_ylim(-0.1,0.1)
            else:
                ax.set_ylim(-1,1)
            
            

plt.savefig(ThisFolder + '/Table_modal.png',dpi=300)