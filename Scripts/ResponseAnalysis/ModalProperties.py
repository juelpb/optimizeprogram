#-------------------------------------------------
def warn(*args, **kwargs): #* Ignore warnings
    pass
import warnings
warnings.warn = warn
#-------------------------------------------------
import time
import numpy as np
import os
from scipy import interpolate
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal as signal
import pandas as pd
from   datetime import datetime 
import scipy 
import csv
import matplotlib.dates as mdates

# ---------------------------------------------------------
path = os.getcwd()
# ---------------------------------------------------------



t0 = time.time()


def partition ( lst, n ):
        """
        Partition list
        Input: 
        lst : 1-D list
        n   : Desired list length
        
        Output:
        output : new list with len == n
        
        """

        if n > len(lst):
            raise Exception('Reduce n')
        
        mat = np.array(lst[1:-1])
        chunked_arrays = np.array_split(mat, n)
        chunked_list = [list(array) for array in chunked_arrays]
        output = [np.mean(lst) for lst in chunked_list]
        output[0] = lst[0]
        output[-1] = lst[-1]
        return output



def get_modeshape(mode,direc,n=None):
		"""
		Get modeshape pulled from Global analysis.
	
		Input:
		Mode : # mode
		Direc : Direction (row index in modeshapes.csv files)
		n : None, returns full list. n (int) partition list
		"""
	
		if mode == 0 or direc > 6:
			raise Exception("-- Error in get_modeshape --")
		
		
		file = open(path + f'/FEM_Results/ModalData/ModeShapes/Mode_{mode}.csv','r')
		csvreader = csv.reader(file)
		
		output = []
		for row in csvreader:
			output.append(float(row[direc])) 
			
		if n != None:
			return np.array(output[::n])
		
		else:
			return np.array(output)


def generate_modal_info_all_modes():

    modal_info = {}
    
    PathModeShapes = path + '/FEM_Results/ModalData/ModeShapes'
    n_modes = len([file for file in os.listdir(PathModeShapes)])
    indexes = range(1,n_modes,1)
    
           
    #* ----------------------
    #* Mass
    #* ---------------------- 
      
    f_gen_mass = open(path + '/FEM_Results/ModalData/Generalized_mass.csv','r')
    gen_mass = csv.reader(f_gen_mass)
    
    Mass = np.zeros(len(indexes))
    
    i = 0
    for ix, row in enumerate(gen_mass):
        if (ix+1) in indexes:
            Mass[i] = row[1]
            i += 1
         
    modal_info['Mass'] = Mass
    
    #* ----------------------
    #* Omega
    #* ---------------------- 
    
    f_eigfreq = open(path + '/FEM_Results/ModalData/Eigenfrequency.csv','r')
    eigfreq   = csv.reader(f_eigfreq)
    
    ef = np.zeros(len(indexes))
    
    i = 0
    for ix, row in enumerate(eigfreq):
        if (ix+1) in indexes:
            ef[i] = row[1]
            i += 1
            
    Omega = np.array([val*2*np.pi for val in ef])
    
    modal_info['Omega'] = Omega
    
    #* ----------------------
    #* K_stru
    #* ---------------------- 
    
    K_stru = np.array([Omega[i]**2*Mass[i] for i in range(len(indexes))])
    
    modal_info['K_stru'] = K_stru
    
    #* ----------------------
    #* C_stru
    #* ---------------------- 
    ksi = 0.02 
    C_stru = np.array([2*Mass[i]*Omega[i]*ksi for i in range(len(indexes))])
    
    modal_info['C_stru'] = C_stru
        
    #* ----------------------
    #* Modeshape and names
    #* ---------------------- 

    modeshape = []
    n = 2
    
    for mode in indexes:
        x_disp  = get_modeshape(mode, 1, n)
        y_disp   = get_modeshape(mode, 2, n)
        z_disp  = get_modeshape(mode, 3, n)
        x_rot   = get_modeshape(mode, 4, n)
        y_rot  = get_modeshape(mode, 5, n)
        z_rot   = get_modeshape(mode, 6, n)
        
        
        temp = []
        
        for i in range(len(x_disp)):
            temp.append(x_disp[i])
            temp.append(y_disp[i])
            temp.append(z_disp[i])
            temp.append(x_rot[i])
            temp.append(y_rot[i])
            temp.append(z_rot[i])
            
        
        modeshape.append(temp)
        
    modeshape = np.array(modeshape)
         
    modal_info['Modeshape'] = modeshape


    return modal_info

def Load_ADs(p_type, girder_height):
    """Loads ADs for p_type and girder height

    Args:
        p_type (int): Parametrization type. 2020, 2021, 2022
        girder_height (float): Girder height

    Returns:
        array: Array of AD values for current girder height
    """
    
    path = os.getcwd()
    try:
        with open(path + f'/Scripts/FlutterAnalysis/AD_Data_{p_type}/{girder_height}.csv') as f:
            AD_data = np.loadtxt(f,delimiter = ',')
        np.savetxt(path + '/Scripts/FlutterAnalysis/latest_AD_data.csv',AD_data, delimiter=',')
    
    except FileNotFoundError:
        n_lower, n_upper = file_round(girder_height, 0.05)

        with open(path + f'/Scripts/FlutterAnalysis/AD_Data_{p_type}/{n_lower}.csv') as f_lower:
            AD_data_lower = np.loadtxt(f_lower, delimiter = ',')
        
        with open(path + f'/Scripts/FlutterAnalysis/AD_Data_{p_type}/{n_upper}.csv') as f_upper:
            AD_data_upper = np.loadtxt(f_upper, delimiter = ',')
        
        nshape = np.shape(AD_data_lower)
        
        AD_data = np.zeros(nshape)
               
        H = [n_lower,n_upper]
            
        for col in range(1,nshape[1],2):
            for row in range(0,nshape[0],1):
                AD = [AD_data_lower[row,col],AD_data_upper[row,col]]
                K = [AD_data_lower[row,col-1], AD_data_upper[row, col-1]]
                AD_data[row,col] = np.interp(girder_height,H,AD)
                AD_data[row,col-1] = np.interp(girder_height,H,K)
        np.savetxt(path + '/Scripts/FlutterAnalysis/latest_AD_data.csv',AD_data, delimiter=',')
    return AD_data


def pull_AD(U_r,AD_name, AD_data):
    """Function to pull an AD value for a spesified reduced velocity

    Args:
        U_r (float): Reduced wind velocity
        AD_name (str): AD name. 'Pi', 'Hi' or 'Ai' for i in [1,6]
        AD_data (array): array to pull AD from

    Returns:
        float: AD value
    """

    direc = {'P1':1, 'P2':3, 'P3':5, 'P4':7,'P5':9, 'P6':11, 'H1':13, 'H2':15, 'H3':17, 'H4':19, 'H5':21, 'H6':23, 'A1':25, 'A2':27, 'A3':29, 'A4':31, 'A5':33, 'A6':35}
    
    idx = direc[AD_name]

    U_r_output = AD_data[:,idx-1]
    AD_output  = AD_data[:,idx]

      
    if U_r < min(U_r_output):
        return AD_output[0]

    if U_r > max(U_r_output):
        return AD_output[-1]

    else:
        AD_int = np.interp(U_r,U_r_output,AD_output)
        return AD_int

    

def aero_damp(omega,U,B,dL, AD_data):
    
    """
    U       :   Mean wind speed, flaot
    Omega   :   Frequency for mode, float
    B       :   Bridge width

    Aerodynamic damping matrix for element (6*6) 
          
    """


    
    const = 0.5*1.25*B**2*omega
    
    U_r   =  U/(B*omega) # Reduced velocity
    
     

    H1 = pull_AD(U_r,'H1', AD_data)
    H2 = pull_AD(U_r,'H2', AD_data)
    A1 = pull_AD(U_r,'A1', AD_data)
    A2 = pull_AD(U_r,'A2', AD_data)

    
    P1    =  0
    P5    =  0
    P2    =  0
    H5    =  0
    A5    =  0

    
    c_ae  =  np.zeros((6,6))
    
    c_ae[1,1] = P1
    c_ae[1,2] = P5
    c_ae[1,3] = P2*B
    c_ae[2,1] = H5
    c_ae[2,2] = H1
    c_ae[2,3] = H2*B
    c_ae[3,1] = A5*B
    c_ae[3,2] = A1*B
    c_ae[3,3] = A2*B**2

       
    return c_ae*const*dL

 
def aero_stiff(omega,U,B,dL, AD_data):
    
    """
    U       :   Mean wind speed, flaot
    Omega   :   Frequency for mode, float (frequency of motion)
    B       :   Bridge width

    Aerodynamic stiffness matrix for element (6*6)  
      
    
    """
     
    const = 0.5*1.25*B**2*omega**2
    
    U_r   =  U/(B*omega) # Reduced velocity


    H4 = pull_AD(U_r,'H4', AD_data)
    H3 = pull_AD(U_r,'H3', AD_data)
    A4 = pull_AD(U_r,'A4', AD_data)
    A3 = pull_AD(U_r,'A3', AD_data)
    

    
    P4    =  0
    P6    =  0
    P3    =  0
    H6    =  0
    A6    =  0
    
    k_ae  =  np.zeros((6,6))
    
    k_ae[1,1] = P4
    k_ae[1,2] = P6
    k_ae[1,3] = P3*B
    k_ae[2,1] = H6
    k_ae[2,2] = H4
    k_ae[2,3] = H3*B
    k_ae[3,1] = A6*B
    k_ae[3,2] = A4*B
    k_ae[3,3] = A3*B**2
    
    return k_ae*const*dL



def modal_aero(U,Omega,Modal, AD_data):
    
    """
    U       :   Mean wind speed, flaot
    Omega   :   Frequency for mode, float
    
    Construct the modal MCK matricces (including aero damping and stiffness matrices)
    
    """
    
    
    No_node = int(Modal['Modeshape'].shape[1]/6)
    No_mode = Modal['Omega'].size
    
    B = 31 
    dL     =  1264/(No_node-1)
  
    #create the global areo damping and stiffness in the nodal level
    C_ae_G  = np.zeros(( No_node * 6, No_node * 6))
    K_ae_G  = np.zeros(( No_node * 6, No_node * 6))
    
    for n in range(0,No_node):
        C_ae_G[n * 6:n * 6 + 6, n * 6:n * 6 + 6]  =  aero_damp (Omega,U,B,dL, AD_data)
        K_ae_G[n * 6:n * 6 + 6, n * 6:n * 6 + 6]  =  aero_stiff(Omega,U,B,dL, AD_data)
    
    M       = np.zeros(( No_mode, No_mode))
    C_stru  = np.zeros(( No_mode, No_mode))
    K_stru  = np.zeros(( No_mode, No_mode))
    
    # diagnal terms of the structural M,C,K 
    for n in range(0,No_mode):  
        M[n,n]       = Modal['Mass'][n]
        C_stru[n,n]  = Modal['C_stru'][n]
        K_stru[n,n]  = Modal['K_stru'][n]

    # Modal areo damping and stiffness 
    C_ae  =  Modal['Modeshape'] @ C_ae_G @ Modal['Modeshape'].T
    K_ae  =  Modal['Modeshape'] @ K_ae_G @ Modal['Modeshape'].T
    
    
    K   =  K_stru-K_ae
    C   =  C_stru-C_ae  
    
    return M,C,K







    
    
    

    



