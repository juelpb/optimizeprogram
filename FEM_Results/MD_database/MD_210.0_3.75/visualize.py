import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import numpy as np

os.chdir('C:/Users/juelpb/Desktop/langenuenprogram')
path = os.getcwd()

fn = pd.read_csv(path + '/03_Results/ModalData/HistoryOutputs.csv',header=None)
filenames = [name[0] + '.csv' for name in fn.values]
callnames = [name[1] for name in fn.values]


def plot_csv_file(filename, plotType):
    data = pd.read_csv(path + '/03_Results/ModalData/' + filename, header=None)

    x_data = []
    y_data = []
    for val in data.values:
        x_data.append(val[0])
        y_data.append(val[1])

    plt.figure(constrained_layout=True)
    plt.title(filename)
    
    if plotType == "graph":
        plt.plot(x_data,y_data)

    if plotType == "bar":
        plt.bar(x_data,y_data)
        plt.xticks(np.arange(len(x_data)))
        
    plt.tight_layout()
    plt.show()
    

def plot_mode(mode):
    
    skip = [0,7,8,9,10,11,12,13,14,15]
    
    mode -= 1 
    mode_y_data = []
    mode_x_data = [(i+1) for i in range(len(filenames)-len(skip))]
    xticks = []
    
    
    
    for count, file in enumerate(filenames):
        if count in skip:
            continue
        else:
            data = pd.read_csv(path + '/03_Results/ModalData/' + file, header=None)
            data_vals = data.values
        
            xticks.append(file)
            mode_y_data.append(data_vals[mode][1])
    
    fig, ax = plt.subplots(1,1)
    ax.set_title(f"Properties for mode: {mode+1}")
    ax.bar(mode_x_data,mode_y_data)
    ax.set_xticks(mode_x_data)
    ax.set_xticklabels(xticks)
    
    plt.show()
    
    
    

        
    
    
        
plot_csv_file("EM_z_rotation.csv", 'bar')        
            
plot_mode(18)    
