import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt

path = os.getcwd()

def mooving_avg(arr, window_size):
    """Funtion to calculate the moving average of an array

    Args:
        arr (array): Array to average
        window_size (int): Number of points in window for each average value

    Returns:
        list: List of averaged values
    """
    numbers_series = pd.Series(arr)
    windows = numbers_series.rolling(window_size)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    final_list = moving_averages_list[window_size - 1:]
    #final_arr = np.array(final_list)
    return final_list

def pull_static_coeff(p_type, g_height, alpha=0):
    """Function to pull static coefficients

    Args:
        p_type (int): Parametrization type. 2020, 2021, 2022
        g_height (float): Girder height
        alpha (int, optional): Mean angle of attack. Defaults to 0.

    Returns:
        float: Static coefficients and their derivatives
    """
    if g_height == 3.5:
        g_height += 0.001
    elif g_height == 4.5:
        g_height -= 0.001
    if p_type == 2020:
        g_heights = [g_height, 5.5, 5.8, 6.1, 6.4, 6.7, 7.0]
        g_heights = sorted(g_heights)
        idx = g_heights.index(g_height)
        g_height1 = g_heights[idx-1]
        g_height2 = g_heights[idx+1]
        filename1 = path + f'/Scripts/ResponseAnalysis/StaticCoeff/StaticCoefficients_LN20_{int(g_height1*1000)}.xls'
        filename2 = path + f'/Scripts/ResponseAnalysis/StaticCoeff/StaticCoefficients_LN20_{int(g_height2*1000)}.xls'
        sheet_name = 0
        usecols = [0,1,2,3]
    elif p_type == 2021:
        g_heights = [g_height, 4.9, 5.2, 5.5, 5.8, 6.1]
        g_heights = sorted(g_heights)
        idx = g_heights.index(g_height)
        g_height1 = g_heights[idx-1]
        g_height2 = g_heights[idx+1]
        filename1 = path + f'/Scripts/ResponseAnalysis/StaticCoeff/StaticCoefficients_LN21_{int(g_height1*1000)}.xls'
        filename2 = path + f'/Scripts/ResponseAnalysis/StaticCoeff/StaticCoefficients_LN21_{int(g_height2*1000)}.xls'
        sheet_name = 0
        usecols = [0,1,2,3]
    elif p_type == 2022:
        g_heights = [g_height, 3.5, 3.75, 4.0, 4.25, 4.5]
        g_heights = sorted(g_heights)
        idx = g_heights.index(g_height)
        g_height1 = g_heights[idx-1]
        g_height2 = g_heights[idx+1]
        filename1 = path + f'/Scripts/ResponseAnalysis/StaticCoeff/StaticCoefficients_LN22_{int(g_height1*1000)}.xlsx'
        filename2 = path + f'/Scripts/ResponseAnalysis/StaticCoeff/StaticCoefficients_LN22_{int(g_height2*1000)}.xlsx'
        sheet_name = 2
        usecols = [1,2,3,4]
        
    df1 = pd.read_excel(filename1, sheet_name=sheet_name, usecols=usecols, skiprows=1)
    df2 = pd.read_excel(filename2, sheet_name=sheet_name, usecols=usecols, skiprows=1)
    arr1 = df1.to_numpy()
    arr2 = df2.to_numpy()
    
    arr1[:,0] = arr1[:,0]
    arr2[:,0] = arr2[:,0]
    
    arr1 = arr1[np.argsort(arr1[:,0])]
    arr2 = arr2[np.argsort(arr2[:,0])]
    
    lst1 = []
    lst2 = []
    for i in range(arr1.shape[1]):
        lst1.append(mooving_avg(arr1[:,i], 400))
        lst2.append(mooving_avg(arr2[:,i], 400))
    arr1 = np.array(lst1)
    arr1 = arr1.T
    arr2 = np.array(lst2)
    arr2 = arr2.T

    arr1 = arr1[np.argsort(arr1[:,0])]
    arr2 = arr2[np.argsort(arr2[:,0])]

    idx1 = np.argmin(np.abs(arr1[:,0]-alpha))
    idx2 = np.argmin(np.abs(arr2[:,0]-alpha))
    
    C1 = [g_height1]
    dC1 = [g_height1]
    C2 = [g_height2]
    dC2 = [g_height2]
    da1 = (np.amax(arr1[:,0])-np.amin(arr1[:,0]))/(len(arr1[:,0]))
    da2 = (np.amax(arr2[:,0])-np.amin(arr2[:,0]))/(len(arr2[:,0]))
    
    b_idx = 200 #indexes to average the derivatives over
    d_idx = 5 #indexes to average the derivatives over
    
    a1 = arr1[[idx1+k for k in range(-b_idx,b_idx,d_idx)], 0]
    a2 = arr2[[idx2+k for k in range(-b_idx,b_idx,d_idx)], 0]


    for i in range(3):
        C1.append(arr1[idx1,i+1])
        v = arr1[[idx1+k for k in range(-b_idx,b_idx,d_idx)], i+1]
        ds = np.diff(v)/np.diff(a1)
        dC1.append(np.mean(ds))
        C2.append(arr2[idx2,i+1])
        v = arr2[[idx2+k for k in range(-b_idx,b_idx,d_idx)], i+1]
        ds = np.diff(v)/np.diff(a2)
        dC2.append(np.mean(ds))
        
    C_data = np.array([np.array(C1), np.array(C2)])
    dC_data = np.array([np.array(dC1), np.array(dC2)])
    
    C_d = np.interp(g_height, C_data[:,0], C_data[:,1])
    C_l = np.interp(g_height, C_data[:,0], C_data[:,2])
    C_m = np.interp(g_height, C_data[:,0], C_data[:,3])
    dC_d = np.interp(g_height, dC_data[:,0], dC_data[:,1])
    dC_l = np.interp(g_height, dC_data[:,0], dC_data[:,2])
    dC_m = np.interp(g_height, dC_data[:,0], dC_data[:,3])
        
    return C_d, dC_d, C_l, dC_l, C_m, dC_m

