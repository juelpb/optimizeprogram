import os
import numpy as np
import csv
import matplotlib.pyplot as plt
path = os.getcwd()
import sys
sys.path.append(path)

from Attachments.ModalAnalysis.EvalModeShapes.ReferenceModes.generate_ref_mode import GENERATE_REF_MODES

#! --------------------
ceil = 40
#!--------------------

def Normalize_lst(lst):
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


def get_refmodes():
        
        RefModePath = path + '/Attachments/ModalAnalysis/EvalModeShapes/ReferenceModes/'
        
        RefModes = {} 
        
        for file in os.listdir(RefModePath):
            if file.endswith('.csv'):
                with open(RefModePath + file ,'r') as f:
                    csvreader = csv.reader(f)
                    RefModes[str(file[:-4])] = [float(num[0]) for num in csvreader]
        return RefModes

def simularity(data, ref):

          
        
          data = np.array(data)
          ref = np.array(ref)
          n = len(data)
          rev_refdata = ref[::-1]
          
          RMS = np.sqrt(np.sum((ref-data)**2)/n)
          RMS_rev = np.sqrt(np.sum((rev_refdata-data)**2)/n)
          
          RMS_oposite_rev = np.sqrt(np.sum(((-1*rev_refdata)-data)**2)/n)
          RMS_oposite = np.sqrt(np.sum(((1*ref)-data)**2)/n)
          
          RMS = min(RMS,RMS_rev,RMS_oposite_rev,RMS_oposite)
          
          simularity = round(abs(100-(RMS*100)),1)
          
          return simularity
      
def get_modeshape(mode,direc,PathToModeShape,n=None):
		"""
		Get modeshape pulled from Global analysis.
	
		Input:
		Mode : # mode
		Direc : Direction (row index in modeshapes.csv files)
		n : None, returns full list. n (int) partition list
		"""
	
		if mode == 0 or direc > 6:
			raise Exception("-- Error in get_modeshape --")
		
		
		file = open(PathToModeShape + f'/ModeShapes/Mode_{mode}.csv','r')
		csvreader = csv.reader(file)
		
		output = []
		for row in csvreader:
			output.append(float(row[direc])) 
			
		if n != None:
			return np.array(output[::n])
		
		else:
			return np.array(output)


def Evaluate_modeshapes(tH,gH):
        """
        Evaluate modeshapes with respect to ReferenceModes. 
        Used to generate the modal info file for the flutter analysis
        
        Output:
        output : dictonary, Modeshape-name as key, mode# / index as value
        """

        ModalDataPath = path + f'/FEM_results/MD_database/MD_{tH}_{gH}'
        
        output = {}
        
        PathModeShapes = ModalDataPath + '/ModeShapes'
        n_modes = len([file for file in os.listdir(PathModeShapes)])
        
        Z_idx   = 3
        Xr_idx  = 4
        Y_idx   = 2
        
        RefModes = get_refmodes()
        
        for refname in RefModes.keys():
            output[refname[:-4]] = []
        for refname, ref in RefModes.items():
                if ('VS' in refname) or ('VA' in refname):
                    #print("ver")
                    for i in range(1,n_modes):
                        if i >= ceil:
                            continue
                        Z_data = get_modeshape(i, Z_idx,ModalDataPath)
                        Z_data = Normalize_lst(Z_data)
                        #Z_data = list(Z_data)
                                           
                        sim = simularity(Z_data, ref)
                        # print(sim,i)
                        # plt.figure()
                        # plt.plot(Z_data)
                        # plt.plot(ref)
                        # plt.show()
                        
                        if sim > 98:
                            #print(f"Mode {i} matches {refname} -- Sim = {sim}")
                            output[refname[:-4]].append(i)

                if ('TS' in refname) or ('TA' in refname):
                    #print("hei")
                    for i in range(1,n_modes):
                        if i >= ceil:
                            continue
                        Xr_data = get_modeshape(i, Xr_idx,ModalDataPath)
                        Xr_data = Normalize_lst(Xr_data)
                        #Xr_data = list(Xr_data)

                        
                        sim = simularity(Xr_data, ref)
                        #print(sim,i)
                        # plt.figure()
                        # plt.plot(Xr_data)
                        # plt.plot(ref)
                        # plt.show()

                        if sim > 98:
                            #print(f"Mode {i} matches {refname} -- Sim = {sim}")
                            output[refname[:-4]].append(i)
                            
                if ('HS' in refname) or ('HA' in refname):
                    #print("hor")
                    for i in range(1,n_modes):
                        Y_data = get_modeshape(i, Y_idx,ModalDataPath)
                        Y_data = Normalize_lst(Y_data)
                        
                        sim = simularity(Y_data, ref)

                        if sim > 90:
                            #print(f"Mode {i} matches {refname} -- Sim = {sim}")
                            output[refname[:-4]].append(i)

        #print(output)
                
        for modename, idx in output.items(): # If more than 1 mode passes the threshold
            if len(idx) > 1:
                if 'T' in modename:
                    #print("hei")
                    f_em_x = open(ModalDataPath + '/EM_x_component.csv','r')
                    r_em_x = csv.reader(f_em_x)
                    em_x   = []
                    
                    for i, row in enumerate(r_em_x):
                        if i+1 in idx:
                            em_x.append(abs(float(row[1])))
                    
                    output[modename] = [idx[np.argmin(em_x)]]
                          
                if 'V' in modename:
                    #print("skeiver")
                    f_em_z = open(ModalDataPath + '/EM_z_component.csv','r')
                    r_em_z = csv.reader(f_em_z)
                    em_z   = []
                    
                    for i, row in enumerate(r_em_z):
                        if i+1 in idx:
                            em_z.append(float(row[1]))
                    
                    output[modename] = [idx[np.argmax(em_z)]]
                
                # else:  
                #     mass_file = open(ModalDataPath + '/Generalized_mass.csv','r')
                #     genMass = csv.reader(mass_file)
                #     gm = []
                
                #     for i, row in enumerate(genMass):
                #         if i in idx:
                #             gm.append(row[1])
                            
                #     for i, mass in enumerate(gm):
                #         if mass == max(gm):
                #             output[modename] = [idx[i]]
        #print(output)
        for modename, val in output.items(): # Convert values from lists to single floats
            output[modename] = val[0]
        
        # Sort
        output = {k: v for k, v in sorted(output.items(), key=lambda item: item[1])}
        
        return output
   




#ModalDataPath = path + '/FEM_Results/ModalData'
#modes = Evaluate_modeshapes(180,3.5)
#print(modes)



G_HEIGHTS = [3.5 ,3.75 ,4.0 ,4.25 ,4.5]
T_HEIGHTS = np.arange(180,222,2)

shape = (len(G_HEIGHTS),len(T_HEIGHTS))
idxes = np.zeros(shape)

current_ref_modes = {'VS1' : 4,
                    'TS1' : 14,
                  }

GENERATE_REF_MODES(current_ref_modes, 180,3.5)
#modes = Evaluate_modeshapes(180, 3.5)
#print(modes)

for j,tH in enumerate(T_HEIGHTS):
    for i,gH in enumerate(G_HEIGHTS):
         try:
            print("------------------------")
            print(f"gH : {gH} -- tH : {tH}")
            modes = Evaluate_modeshapes(tH, gH)
            print(modes)
            idxes[i,j] = modes['TS1']
            GENERATE_REF_MODES(modes, tH, gH)
         except IndexError:
             idxes[i,j] = 0
             continue
         
         
np.savetxt(path + '/Attachments/ModalAnalysis/INDEXES.csv', idxes,delimiter=',',fmt='%d')