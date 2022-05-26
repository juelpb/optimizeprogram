"""
Script to store the function for the critical wind speed analysis
"""
#-------------------------------------------------
def warn(*args, **kwargs): #* Ignore warnings
	pass
import warnings
warnings.warn = warn
#-------------------------------------------------
import time
import numpy as np
import os
import matplotlib
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal as signal
import pandas as pd
import math
from   datetime import datetime 
import scipy 
import csv
import matplotlib.dates as mdates


path = os.getcwd()


def flutter_speed(girder_height,tower_height,p_type,plot=None,p=None):
    
    """
    Calculate the critical wind speed for a specific configuration. 
    
    Input:
        girder_height   : Girder height, float
        p_type          : configuration year, int
        
    Return:
        U_flutter       : Critical wind speed [m/s] / flutter speed, float
    """
    
    t0 = time.time()
    
    MD_path = path + f'/FEM_Results/MD_database/MD_{tower_height}_{girder_height}'
    
    def file_round(x, base):
            n_lower = base * math.trunc(x/base)
            n_upper = base * math.ceil(x/base)
            return round(n_lower,2), round(n_upper,2)

    U_flutter = 150

    # ------------------------------------------------------------------------------------------
    # Laod AD-data for the specific girder height
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
        
    # -------------------------------------------------------------------------------------------

    def get_modeshape(mode,direc,n=None):
            """
            Get modeshape pulled from Global analysis.
        
            Input:
                Mode : # mode
                Direc : Direction (row index in modeshapes.csv files)
                n : Get every n'th object. Default: None, returns full modeshape
            Return:
                output  : Modeshape array
            """
        
            if mode == 0 or direc > 6:
                raise Exception("-- Error in get_modeshape --")
            
            
            file = open(MD_path + f'/ModeShapes/Mode_{mode}.csv','r')
            csvreader = csv.reader(file)
            
            output = []
            for row in csvreader:
                output.append(float(row[direc])) 
                
            if n != None:
                return np.array(output[::n])
            
            else:
                return np.array(output)
        

    
    def generate_modal_info_all_modes():
        """
        Return:
            Assemble the modal info to be used in the analysis.
        """

        modal_info = {}
        
        PathModeShapes = MD_path + '/ModeShapes'
        
        n_modes = len([file for file in os.listdir(PathModeShapes)])
        
        indexes = range(1,n_modes,1) # 40 first modes (multimodal)
        
               
        #* ----------------------
        #* Mass
        #* ---------------------- 
          
        f_gen_mass = open(MD_path + '/Generalized_mass.csv','r')
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
        
        f_eigfreq = open(MD_path + '/Eigenfrequency.csv','r')
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
        ksi = 0.005
        C_stru = np.array([2*Mass[i]*Omega[i]*ksi for i in range(len(indexes))])
        
        modal_info['C_stru'] = C_stru
            
        #* ----------------------
        #* Modeshape
        #* ---------------------- 

        modeshape = []
        n = 5 #! Partition, important for analysis speed
        
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
        modal_info['Name'] = indexes

        return modal_info

    def pull_AD(U_r,AD_name):
        """
        Pull AD from surrogate models.
        If U_r are outside domain, return nearest AD value
        
        Input:
            U_r       : Reduced velocity, float
            AD_name   : AD name, str
        Return
            AD        : AD, float  
        """

        direc = {'P1':1, 'P2':3, 'P3':5, 'P4':7,'P5':9, 'P6':11, 'H1':13, 'H2':15, 'H3':17, 'H4':19, 'H5':21, 'H6':23, 'A1':25, 'A2':27, 'A3':29, 'A4':31, 'A5':33, 'A6':35}
        
        idx = direc[AD_name]

        U_r_output = AD_data[:,idx-1]
        AD_output  = AD_data[:,idx]

        
        if U_r < min(U_r_output): # If outside domain on the lower side
            return AD_output[0]

        if U_r > max(U_r_output): # If outside domain on the upper side
            return AD_output[-1]

        else:
            AD_int = np.interp(U_r,U_r_output,AD_output) # Linear interpolation between the fine GPR-grid. 
            return AD_int

        
    def aero_damp(omega,U,B,dL):
        """
        Input:
            U       :   Mean wind speed, flaot
            omega   :   Frequency for mode, float
            B       :   Bridge width
            dL      :   Length increment

        Returns aerodynamic damping matrix for element (6*6).
        """
        
        const = 0.5*1.25*B**2*omega
        
        U_r   =  U/(B*omega) # Reduced velocity
        
         
        H1 = pull_AD(U_r,'H1') 
        H2 = pull_AD(U_r,'H2') 
        A1 = pull_AD(U_r,'A1') 
        A2 = pull_AD(U_r,'A2') 

        #print(f"H1: {round(float(H1*U_r),2)} -- H2: {round(float(H2*U_r),2)} -- A1: {round(float(A1*U_r),2)} -- A2: {round(float(A2*U_r),2)}") 
        
       
        P1    =  pull_AD(U_r,'P1') 
        P5    =  pull_AD(U_r,'P5') 
        P2    =  pull_AD(U_r,'P2') 
        H5    =  pull_AD(U_r,'H5') 
        A5    =  pull_AD(U_r,'A5') 

        
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

    def aero_stiff(omega,U,B,dL):
        """
        Input:
            U       :   Mean wind speed, flaot
            omega   :   Frequency for mode, float (frequency of motion)
            B       :   Bridge width
            dL      :   Length increment

        Returns aerodynamic stiffness matrix for element (6*6).  
        """
         
        const = 0.5*1.25*B**2*omega**2
        
        U_r   =  U/(B*omega) # Reduced velocity
        #print(U_r)

        H4 = pull_AD(U_r,'H4') 
        H3 = pull_AD(U_r,'H3') 
        A4 = pull_AD(U_r,'A4') 
        A3 = pull_AD(U_r,'A3') 
        
        #print(f"H4: {round(float(H4),2)} -- H3: {round(float(H3),2)} -- A4: {round(float(A4),2)} -- A3: {round(float(A3),2)}") 

        
        P4    =  pull_AD(U_r,'P4') 
        P6    =  pull_AD(U_r,'P6') 
        P3    =  pull_AD(U_r,'P3') 
        H6    =  pull_AD(U_r,'H6') 
        A6    =  pull_AD(U_r,'A6') 
        
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



    def modal_aero(U,Omega,Modal):
        """
        Input:     
            U       :   Mean wind speed, flaot
            Omega   :   Frequency for mode, float
            Modal   :   Modal info from global analysis       
        
        Return:
            Construct and returns M, C, K matrices.
            Including contributions from aero damping and stiffness matrices.
        """
        
        
        No_node = int(Modal['Modeshape'].shape[1]/6)
        No_mode = Modal['Omega'].size
        
        B      =  31
        dL     =  1264/(No_node-1)
      
        #create the global areo damping and stiffness in the nodal level
        C_ae_G  = np.zeros(( No_node * 6, No_node * 6))
        K_ae_G  = np.zeros(( No_node * 6, No_node * 6))
        
        for n in range(0,No_node):
            C_ae_G[n * 6:n * 6 + 6, n * 6:n * 6 + 6]  =  aero_damp (Omega,U,B,dL)
            K_ae_G[n * 6:n * 6 + 6, n * 6:n * 6 + 6]  =  aero_stiff(Omega,U,B,dL)
        
        M       = np.zeros(( No_mode, No_mode))
        C_stru  = np.zeros(( No_mode, No_mode))
        K_stru  = np.zeros(( No_mode, No_mode))
        
        # diagnal terms of the structural M,C,K 
        for n in range(0,No_mode):  
            M[n,n]       = Modal['Mass'][n]
            C_stru[n,n]  = Modal['C_stru'][n]
            K_stru[n,n]  = Modal['K_stru'][n]

        # Modal areo damping and stiffness (post and pre * with modeshapes)
        C_ae  =  Modal['Modeshape'] @ C_ae_G @ Modal['Modeshape'].T
        K_ae  =  Modal['Modeshape'] @ K_ae_G @ Modal['Modeshape'].T
        
        
               
        K   =  K_stru-K_ae
        C   =  C_stru-C_ae  
        return M,C,K



    def eigen_solve(U,Omega,Modal,flag,threshold,maxit):  
        """  
        Calculate the complex eigenvalues of the M C K system

        Input:
            U		    :   Mean wind speed, float
    	    Omega	    :   Eigenfrequency for current mode
    	    Modal	    :   Main info-file for modal analysis
    	    flag	    :   Variable to keep track of which mode we are refering too.
    	    AD_fit	    :   Flutter derivatives data and parameters
            threshold   :   Tolerance for while loop
            
        Return:
            Eigen_new   :   In-wind frequency 
            Miu         :   In wind damping 
            j           :   Loop variable
            Omega       :   Input Omega
            Omega_new   :   In-wind frequency for the flag-mode
        """

        M,C,K  = modal_aero(U,Omega,Modal)
        No_mode = Modal['Omega'].size    
        
        # Assemble the state space matrix, D
        D      = np.block([[-np.linalg.inv(M)@C,-np.linalg.inv(M)@K],[np.eye(No_mode),np.eye(No_mode)*0]])
        # Solve eigenvalue problem
        eigen, vector = scipy.linalg.eig(D)
       
        
        
        #index of the sorted eigen frequency from smallest to largest       
        idx = np.argsort(np.imag(eigen))[No_mode:None]
        Eigen_new = np.imag(eigen)[idx]
        Miu       = np.real(eigen)[idx]
                
        if Eigen_new[flag]==0:
            print("Reduce eigfreq")
            Omega_new=Omega*0.8
        else:
            Omega_new = Eigen_new[flag]    
        
        j=0 # Variable to count iterations
        
        while np.abs(Omega_new-Omega)>threshold:
                        
            if j >= maxit:
                break
            
            Omega  = Omega_new
            
            M,C,K  = modal_aero(U,Omega,Modal)
            
            # Assemble the state space matrix, D
            D      = np.block([[-np.linalg.inv(M)@C,-np.linalg.inv(M)@K],[np.eye(No_mode),np.eye(No_mode)*0]])
            # Solve eigenvalue problem
            eigen, vector = scipy.linalg.eig(D) 
            
       
            #index of the sorted eigen frequency from smallest to largest
            idx = np.argsort(np.imag(eigen))[No_mode:None]
            
            Eigen_new = np.imag(eigen)[idx]
            Miu       = np.real(eigen)[idx]
            
            
            if Eigen_new[flag]==0:
                print("Reduce eigfreq")
                Omega_new=Omega*0.8
            else:
                Omega_new = Eigen_new[flag]
            j=j+1
            
        return Eigen_new,Miu,j,Omega, Omega_new


    # --------------------------------------------------------------------------------------------------
    #! Starts here
    # --------------------------------------------------------------------------------------------------


    U_s     = 30
    U_e     = 150
    step    = 10
    num_U   = int((U_e-U_s)/step)+1
    U_range = np.linspace(U_s,U_e,num_U)
    threshold = 0.05
    
    maxit = 25
    Modal = generate_modal_info_all_modes()

    Omega_s  = np.zeros((U_range.size,Modal['Omega'].size))
    Damp_s   = np.zeros((U_range.size,Modal['Omega'].size))
    Zeta_s   = np.zeros((U_range.size,Modal['Omega'].size))

    if p == True:
        print("--------------------------------------------------")
        print(f"Loop #1 with step = {step} -- Interval [{U_s},{U_e}] ")
        print("--------------------------------------------------")

    U_lst = []
    Omega_lst = [list(Modal['Omega'])]
    Zeta_lst = []
    
    for i, U in enumerate(U_range):
        U_lst.append(U)
        Omega_lst_in = []
        Zeta_lst_in = []
        for flag, Omega in enumerate(Modal['Omega']):
            Eigen_new,Miu,j,Omega,Omega_new   =  eigen_solve(U,Omega_lst[-1][flag],Modal,flag,threshold,maxit)
            
            if j >= maxit:
                raise Exception('''
                    ----------------------------
                    ---EigenSolve Diverged! ---
                    ----------------------------''')
                

            # define the array to store the frequency and damping of each mode
            Omega_lst_in.append((Miu[flag]**2+Eigen_new[flag]**2)**0.5)
            Zeta_lst_in.append(-Miu[flag]/(Miu[flag]**2+Eigen_new[flag]**2)**0.5)
            Omega_s[i,flag]  = (Miu[flag]**2+Eigen_new[flag]**2)**0.5
            Damp_s[i,flag]   = Miu[flag]
            Zeta_s[i,flag]   = -Miu[flag]/(Miu[flag]**2+Eigen_new[flag]**2)**0.5

            
            # location where the eigen frequencies are not zero (not overdamped)        
            pos_omega = np.where(Omega_s[i,:]!=0)
            # locations of Miu when Miu is positive in the array when Omega_s is nonzero
            #print(f"Miu : {Miu[flag]}")
            pos = np.where(Miu[flag]>=0)[0]
            
            if p == True:
                print(f"U: {U} m/s -- Mode: {flag} -- Omega: {Omega}")
            if len(pos)>0:
                #Omega_lst.append(Omega_lst_in)
                #Zeta_lst.append(Zeta_lst_in)
                U_lst.pop(-1)
                if p == True:
                    print(f'Flutter speed @ {U} -- Mode = {flag} -- Omega= {Omega_s[i,flag]}')
                Speed = U
                break
        else:
            Omega_lst.append(Omega_lst_in)
            Zeta_lst.append(Zeta_lst_in)
            continue
        break


    if U < 150:
        U_s     = Speed-10
        U_e     = 150
        step    = 1
        num_U   = int((U_e-U_s)/step)+1
        U_range = np.linspace(U_s,U_e,num_U)
        threshold = 0.05
        
        Omega_s  = np.zeros((U_range.size,Modal['Omega'].size))
        Damp_s   = np.zeros((U_range.size,Modal['Omega'].size))
        Zeta_s   = np.zeros((U_range.size,Modal['Omega'].size))
        
        if p == True:
            print("--------------------------------------------------")
            print(f"Loop #2 with step = {step} -- Interval [{U_s},{U_e}] ")
            print("--------------------------------------------------")
        

        for i, U in enumerate(U_range):
            U_lst.append(U)
            Omega_lst_in = []
            Zeta_lst_in = []
            for flag,Omega in enumerate(Modal['Omega']):
                Eigen_new,Miu,j,Omega,Omega_new   =  eigen_solve(U,Omega_lst[-1][flag],Modal,flag,threshold,maxit)


                if j >= maxit:
                    raise Exception('''
                    ----------------------------
                    ---EigenSolve Diverged! ---
                    ----------------------------''')
                
                # define the array to store the frequency and damping of each mode
                Omega_lst_in.append(Eigen_new[flag])
                Zeta_lst_in.append(-Miu[flag]/(Miu[flag]**2+Eigen_new[flag]**2)**0.5)
                Omega_s[i,flag]  = Eigen_new[flag]
                Damp_s[i,flag]   = Miu[flag]
                Zeta_s[i,flag]   = -Miu[flag]/(Miu[flag]**2+Eigen_new[flag]**2)**0.5        
                
                # location where the eigen frequencies are not zero (not overdamped)        
                pos_omega = np.where(Omega_s[i,:]!=0)
                # locations of Miu when Miu is positive in the array when Omega_s is nonzero
                pos = np.where(Miu[flag]>=0)[0]
                
                if p == True:
                    print(f"U: {U} m/s -- Mode: {flag} -- Omega: {Omega}")
                if len(pos)>0:
                    #Omega_lst.append(Omega_lst_in)
                    #Zeta_lst.append(Zeta_lst_in)
                    U_lst.pop(-1)
                    if p == True:
                        print(f'Flutter speed @ {U} -- Mode = {flag} -- Omega= {Omega_s[i,flag]}')
                    Speed = U
                    break
            else:
                Omega_lst.append(Omega_lst_in)
                Zeta_lst.append(Zeta_lst_in)
                continue
            break
        U_flutter = Speed
        
        
        
    if U < 150:
        U_s     = Speed-1
        U_e     = 150
        step    = 0.1
        num_U   = int((U_e-U_s)/step)+1
        U_range = np.linspace(U_s,U_e,num_U)
        threshold = 0.05
        
        Omega_s  = np.zeros((U_range.size,Modal['Omega'].size))
        Damp_s   = np.zeros((U_range.size,Modal['Omega'].size))
        Zeta_s   = np.zeros((U_range.size,Modal['Omega'].size))
        
        if p == True:
            print("--------------------------------------------------")
            print(f"Loop #3 with step = {step} -- Interval [{U_s},{U_e}] ")
            print("--------------------------------------------------")
            

        for i, U in enumerate(U_range):
            U_lst.append(U)
            Omega_lst_in = []
            Zeta_lst_in = []
            for flag,Omega in enumerate(Modal['Omega']):
                Eigen_new,Miu,j,Omega,Omega_new   =  eigen_solve(U,Omega_lst[-1][flag],Modal,flag,threshold,maxit)
                
                if j >= maxit:
                    raise Exception('''
                    ----------------------------
                    ---EigenSolve Diverged! ---
                    ----------------------------''')


                # define the array to store the frequency and damping of each mode
                Omega_lst_in.append(Eigen_new[flag])
                Zeta_lst_in.append(-Miu[flag]/(Miu[flag]**2+Eigen_new[flag]**2)**0.5)
                Omega_s[i,flag]  = Eigen_new[flag]
                Damp_s[i,flag]   = Miu[flag]
                Zeta_s[i,flag]   = -Miu[flag]/(Miu[flag]**2+Eigen_new[flag]**2)**0.5 
                
                # location where the eigen frequencies are not zero (not overdamped)        
                pos_omega = np.where(Omega_s[i,:]!=0)
                # locations of Miu when Miu is positive in the array when Omega_s is nonzero
                pos = np.where(Miu[flag]>=0)[0]
                
                if p == True:
                    print(f"U: {U} m/s -- Mode: {flag} -- Omega: {Omega}")
                if len(pos)>0:
                    #Omega_lst.append(Omega_lst_in)
                    #Zeta_lst.append(Zeta_lst_in)
                    U_lst.pop(-1)
                    if p == True:
                        print(f'Flutter speed @ {U} -- Mode = {flag} -- Omega= {Omega_s[i,flag]}')
                    Speed = U
                    break
            else:
                Omega_lst.append(Omega_lst_in)
                Zeta_lst.append(Zeta_lst_in)
                continue
            break
        U_flutter = Speed
    
    t1 = time.time()
    print('Analysis is Done! - ' + str(round(t1 - t0, 3)) + ' sec.')
    
    if plot == True:
        
        plt.close('all')
        
        import seaborn as sns
        from matplotlib.pyplot import cm
        
        matplotlib.rc('xtick', labelsize=13) 
        matplotlib.rc('ytick', labelsize=13) 

        U_arr = np.array(U_lst)
        Omega_arr = np.array(Omega_lst[1:])
        Zeta_arr = np.array(Zeta_lst)
                
        fig = plt.gcf()
        plt.rcParams["figure.figsize"] = (13,7)
        ax1,ax2 = plt.subplots(nrows=2,ncols=1)
        
        
        U_new = np.arange(U_arr[0], np.amax(U_arr))
        
        
        colors = sns.color_palette('tab20',n_colors=len(Modal['Omega']))
        #colors = cm.rainbow(np.linspace(0, 1, len(Modal['Omega'])))
        
        ax1 = plt.subplot(211)
        for i, c in enumerate(colors):
        
            f_omega = (interpolate.interp1d(U_arr, Omega_arr[:,i]))
            Omega_new = f_omega(U_new)
            ax1.plot(U_new, Omega_new,color=c, label=f'{i}')
            #ax1.plot(U_arr, Omega_arr[:,i], 'o',markersize=3, color=c)

        #plt.title(f'Flutter analysis -- H = {girder_height} m, {p_type}')
        plt.xlabel('U [m/s]')
        plt.ylabel('$\omega$')
        
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
        ax1.legend(title='Mode',loc='upper left', bbox_to_anchor=(1.02, 0.5),
          fancybox=False, shadow=False, ncol=2)
                
        ax2 = plt.subplot(212)
        for i, c in enumerate(colors):
            f_zeta = (interpolate.interp1d(U_arr, Zeta_arr[:,i]))
            Zeta_new = f_zeta(U_new)
            ax2.plot(U_new, Zeta_new, label=f'{i}',color=c)
            #ax2.plot(U_arr, Zeta_arr[:,i],'o',markersize=3,color=c)

        #textstr =  r'$U_{flutter}=$' + str(U_flutter) + '$m/s$'
        #props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        #ax2.text(0.05, 0.90, textstr, transform=ax2.transAxes, fontsize=18,
                #verticalalignment='top', bbox=props)
        plt.xlabel('U [m/s]') 
        plt.ylabel('$\zeta$')
        box = ax2.get_position()
        #ax2.set_position([box.x0, box.y0 + box.height * 0.1,
                 #box.width, box.height * 0.9])
        #ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
          #fancybox=True, shadow=True, ncol=3)
        
        #plt.tight_layout()
        plt.subplots_adjust(right=0.85)
        plt.savefig(fname=path+f'/Scripts/FlutterAnalysis/Figures/{girder_height}_{p_type}.pdf',dpi=500,)
        #plt.show()
        print('Plot done!')

    return U_flutter


#flutter_speed(4.0, 2022,p=True,plot=True)