"""
Script to store functions used to generate surrugate models for the Aerodynamic derivatives from
the wind tunnel data. Surrogate models are generated using Gaussian process regression.
"""

def warn(*args, **kwargs): # Ignore warnings
    pass
import warnings
warnings.warn = warn

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import pandas as pd
import os
import math
import csv
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic, Exponentiation
from itertools import product
import glob
from xlwt import Workbook
import time

rootFolder = os.getcwd()
# Heights for each years confiurations
H_20 = [5500,5800,6100,6400,6700,7000]
H_21 = [4900,5200,5500,5800,6100]
H_22 = [3500,3750,4000,4250,4500]

sheetnames = ['Reduced Velocities', 'Aerodynamic Derivatives']

ad_colnames = ["P1", 
            "P2",
            "P3",
            "P4",
            "P5",
            "P6",
            "H1",
            "H2",
            "H3",
            "H4",
            "H5",
            "H6",
            "A1",
            "A2",
            "A3",
            "A4",
            "A5",
            "A6"]

Vred_colnames = ["Vred_P1",
            "Vred_P2",
            "Vred_P3",
            "Vred_P4",
            "Vred_P5",
            "Vred_P6",
            "Vred_H1",
            "Vred_H2",
            "Vred_H3",
            "Vred_H4",
            "Vred_H5",
            "Vred_H6",
            "Vred_A1",
            "Vred_A2",
            "Vred_A3",
            "Vred_A4",
            "Vred_A5",
            "Vred_A6"]



def load_data(year):
    """
    Load in the AD-data from wind tunnel testing.
    
    Return: 
        data, array
    """
    
    dataPath = rootFolder + '/Scripts/SurrugateModels/WindtunnelDataCSV/'
    fnames = glob.glob(dataPath + f'*{year}.csv')
    data = {}
    for np_name in fnames:
        data[np_name.split('\\')[-1]] = np.loadtxt(np_name)

    return data


def max_gap(lst):
    n = sorted(lst)
    ans = []
    for i in range(len(n)-1):
        gap = n[i+1]-n[i]
        ans.append(gap)
    return max(ans)


def gen_input_data(year, ad_idx):
    """
    Generate input data for the GPR.

    Input:
        year        : i.e. 22, int
        ad_idx      : index of decired ad, see ad_colnames, int
    Return:
        AD_data     : AD data for all heights for the respective year and AD.
        Vred_data   : Reduced velocity data for all heights for the respective year and AD.  
        heights     : list of heights for that years configurations
    """
    
    data = load_data(year)
    if year == 20:
        heights = H_20

    elif year == 21:
        heights = H_21
    
    else:
        heights = H_22
  
    
    ad_name = ad_colnames[ad_idx]
    data = load_data(year)
    AD_data = []
    Vred_data = []
    
    for h in (heights):
            d = data[f'AD_Vred_{h}_{year}.csv']
            ad_lst = d[:,ad_idx*2+1]
            Vred_lst = d[:,ad_idx*2]
           
            ad_lst = [ad_lst[i] for i in range(len(ad_lst))]

            AD_data.append(ad_lst)
            Vred_data.append(Vred_lst)
        
    return AD_data, Vred_data, heights


def GPR(Vred_data, AD_data,heights, n_h):
    """
    Generate data for the AD surrogate models using Gaussian process regression.
    GPR-parameters is controlled here.
    
    Input:
        Vred_data, AD_data, heights : generated with gen_input_data()
        n_h : # of data points on the girder height axis
    Return:
        X  : Reduced velocity input data
        Y  : AD input data
        Kp : Predicted reduced velocities                        
        Hp : Predicted heights           
        Zp : Predicted AD_data                         
        
    """
    
    x = Vred_data
    z = AD_data
    y = [h/1000 for h in heights]

    # ----------------------------
    # Zip-points
    X = []
    for i in range(len(y)):
        for j in range(len(z[i])):
            X.append([x[i][j],y[i]])
    X = np.asarray(X)

    Y = []
    for i in range(len(z)):
        for j in range(len(z[i])):
            Y.append(z[i][j])
    Y = np.asarray(Y)

    # Make gridpoints
    k_ax_n = 50
    h_ax_n = n_h
    k_ax = np.linspace(X[:,0].min()-1.1, X[:,0].max(),k_ax_n)
    h_ax = np.linspace(X[:,1].min(), X[:,1].max(),h_ax_n)
    Kp, Hp = np.meshgrid(k_ax, h_ax)

    # Gaussian process (GPR)
    #! Change GPR parameters here:
    #gap_Vred = max([max_gap(lst) for lst in Vred_data])
    #TODO: bounds = [(4,20),(0.1,10)] -- length_scale = [4, 0.25] -- 'alpha' : 0.13,
    bounds = [(4,20),(0.1,10)] 
    length_scale = [4, 0.25]

    gp_params = {
                'kernel' : Matern(length_scale, bounds, nu=2.5),
                'alpha' : 0.13,
                }

    gp = GaussianProcessRegressor(**gp_params)
    gp.fit(X, Y)

    Zp = [gp.predict([(Kp[i, j], Hp[i, j]) for i in range(Kp.shape[0])]) for j in range(Hp.shape[1])] # Regression part
    Zp = np.array(Zp).T
    
    return X, Y, Kp, Hp, Zp


def plot_all_models(year):
    """
    Input:
        year : int
    Retrun:
        Plotting all surrogate models at once
        Figures are saved in: Scripts/SurrogateModels/Figures and .png's
    """
    
    print("--------------------------------------------------------")
    data = load_data(year)
    
    for ad in range(len(ad_colnames)):
        AD_data, Vred_data, heights = gen_input_data(year, ad)
    
        spaceing = 0.05
        n_h = len(np.arange(min(heights)/1000, max(heights)/1000+spaceing,spaceing))

        X, Y, Kp, Hp, Zp = GPR(Vred_data, AD_data, heights,n_h)

        fig = plt.figure(figsize=(10,10),constrained_layout=True)
        ax = fig.add_subplot(111, projection = '3d')

        p_surf = ax.plot_surface(Kp, Hp, Zp,label="Predicted Surface", rstride=1, cstride=1,cmap='viridis', linewidth=0, alpha=0.7) # Plot surrugate model

        p_surf._facecolors2d = p_surf._facecolor3d
        p_surf._edgecolors2d = p_surf._edgecolor3d

        ax.legend()

        for i in range(len(Y)):
            if i == 0: # step for adding label.
                ax.scatter(X[i][0],X[i][1],Y[i],label="Data Points",color='black')
                ax.legend()
            else:
                ax.scatter(X[i][0],X[i][1],Y[i],color='black') # Plot obersvations


        plt.xlabel('$\hat{V}$')
        plt.ylabel('H')

        name = f"LN{year}_{ad_colnames[ad]}"
        fig.suptitle(name)

        filepath = rootFolder + '/Scripts/SurrugateModels/Figures/'
        file_name = str(f"{name}.png")
        plt.savefig(filepath+file_name,dpi=300)
        print(f"-- Saved: {file_name} {ad+1}/{len(ad_colnames)} --")
    
    print("--------------------------------------------------------")   
        
        

def plot_single_model(year, AD_name):
    """
    Input:
        year    : int
        AD_name : str
    Return:
        Plot singe Surrogate model for decired AD. 
        Figure open in matplotlib GUI which offer pan/zoom etc.
    """
    print("--------------------------------------------------------")
    ad_idx = ad_colnames.index(AD_name)
    AD_data, Vred_data, heights = gen_input_data(year, ad_idx)
    spaceing = 0.05
    n_h = len(np.arange(min(heights)/1000, max(heights)/1000+spaceing,spaceing))
    X, Y, Kp, Hp, Zp = GPR(Vred_data, AD_data, heights,n_h)

    fig = plt.figure(figsize=(10,10),constrained_layout=True)
    ax = fig.add_subplot(111, projection = '3d')

    p_surf = ax.plot_surface(Kp, Hp, Zp,label="Predicted Surface", rstride=1, cstride=1, cmap='viridis', linewidth=0, alpha=0.7) # Plot surrugate model

    p_surf._facecolors2d = p_surf._facecolor3d
    p_surf._edgecolors2d = p_surf._edgecolor3d

    ax.legend()

    for i in range(len(Y)):
        if i == 0: # Step for adding label.
            ax.scatter(X[i][0],X[i][1],Y[i],label="Data Points",color='black')
            ax.legend()
        else:
            ax.scatter(X[i][0],X[i][1],Y[i],color='black') # Plot obersvations


    plt.xlabel('K')
    plt.ylabel('H')

    name = f"LN{year}_{ad_colnames[ad_idx]}"
    fig.suptitle(name)

    filepath = rootFolder + '/Scripts/SurrugateModels/Figures/'
    file_name = str(f"{name}.png")
    plt.savefig(filepath+file_name,dpi=300)
    print(f"-- Saved: {file_name} {ad_idx+1}/{len(ad_colnames)} --")
    print("--------------------------------------------------------")
    plt.show()
    



def write_GPRdata_to_xl(year):
    """
    Input:
        year : int
    Return:
        Write an easy-to-read .xls-file with all the predicted GPR data
        Stored in Scripts/SurrugateModels/PredictedData_{year}.xls
    """
    print("--------------------------------------------------------")
    path = rootFolder + f'/Scripts/SurrugateModels/WindTunnelData_{year}/AerodynamicDerivatives_LN'
    sheetnames = ['Reduced Velocities', 'Aerodynamic Derivatives']
        
    wb = Workbook()
    
    for ad in range(len(ad_colnames)):
        
        print(f"Write to Excel -- {ad_colnames[ad]} -- {ad+1}/{len(ad_colnames)}")
        
        AD_data, Vred_data, heights = gen_input_data(year, ad)
        
        spaceing = 0.05 #! Change spaceing
        n_h = len(np.arange(min(heights)/1000, max(heights)/1000+spaceing,spaceing))   

        variance = np.var(AD_data)

        (X, Y, Kp, Hp, Zp) = GPR(Vred_data, AD_data,heights,n_h)

        if ad == 0:
                export_sheetnames = [wb.add_sheet(f'{str(round(Hp[i][0],2))}') for i in range(n_h)]

        rowSize = np.shape(Zp)[1]
        
        col_ind_k  = ad*2
        col_ind_ad = ad*2+1

        for count, sheet in enumerate(export_sheetnames):
            sheet.write(0,col_ind_k,Vred_colnames[ad])
            sheet.write(0,col_ind_ad,ad_colnames[ad])

            for j in range(1,rowSize,1):
                sheet.write(j,col_ind_k,Kp[count][j-1])
                sheet.write(j,col_ind_ad,Zp[count][j-1])


    wb.save(rootFolder + f'/Scripts/SurrugateModels/PredictedData_{year}.xls')
    print("Finish write GPR data to Excel!")
    print(f"Filename: PredictedData_{year}.xls")
    print("--------------------------------------------------------")



def push_GPRdata_to_fluttercode(year):
    """
    Input:
        year : int
    Return:
        Push the predicted GPR-data to the fluttercode
        Stored in: Scripts/FlutterAnalysis/AD_Data_{year}.csv
    """

    print("--------------------------------------------------------")
    print("Generating .csv files for FluttterAnalysis")
    y = year
    workbook_path = rootFolder + f'/Scripts/SurrugateModels/PredictedData_{y}.xls'

    output_folder = rootFolder + f'/Scripts/FlutterAnalysis/AD_Data_20{y}/' 
    print(output_folder)


    xl = pd.ExcelFile(workbook_path)

    heights = xl.sheet_names

    for h in heights:

        sh = xl.parse(h)

        n_rows = len(sh['P1'])

        def get_row(n):
            return [val for val in sh.loc[n]]

        with open(output_folder + f'{h}.csv','w+',newline='') as f:
            writer = csv.writer(f)
            for row_n in range(n_rows):
                writer.writerow(get_row(row_n))

    print("Finished!")
    print("--------------------------------------------------------")
    

def labdata_to_csv(year):
    """
    Input:
        year : int
    Return:
        Convert LAB-data to .csv-files for speed and convenience
        Stored in: Scripts/SurrogateModels/WindtunnelDataCSV
    """
    
    print("--------------------------------------------------------")
    if year == 20:
        H = H_20
    elif year == 21:
        H = H_21
    else:
        H = H_22
        
    for h in H:
        
        collected_data = np.zeros((16,36))
         
        workbook_path   = rootFolder + f'/Scripts/SurrugateModels/WindTunnelData_{year}/AerodynamicDerivatives_LN{year}_{h}.xlsx'
        output_path     = rootFolder + f'/Scripts/SurrugateModels/WindtunnelDataCSV/AD_Vred_{h}_{year}.csv'
        
        xl = pd.ExcelFile(workbook_path)
        
        SheetNames = xl.sheet_names
        
        n_rows = len(SheetNames[0])
        
        aD      = xl.parse('Aerodynamic Derivatives')
        Vred    = xl.parse('Reduced Velocities')
        
        def get_column(sheet,coln):
            df = pd.DataFrame(sheet,index=None)
            
            colname = df.columns[int(coln)]
                       
            return [val for val in df[colname]]
        
        
        n_cols = np.shape(collected_data)[1]    
        for i in range(0,n_cols,2):
            j = i/2
            collected_data[:,i]     = get_column(Vred, j)
            collected_data[:,i+1]   = get_column(aD, j)
        
        np.savetxt(output_path, collected_data,fmt='%f')
    
    print("LAB-data successfully converted to .csv\nReady for further processing")
    print("--------------------------------------------------------")
                





