import numpy as np
import os
import sys
import csv
import shutil
# --------------------------------------------------
# SETTING UP PATH IN ORDER TO CALL SCRIPTS
# --------------------------------------------------
#?os.chdir('C:/Users/juelpb/Desktop/langenuenprogram')
path = os.getcwd()
sys.path.append(path)
ThisFolder = path + '/Attachments/ModalAnalysis'
# --------------------------------------------------
# IMPORT SCRIPTS
# --------------------------------------------------
from Scripts.HelperFunctions import sleep, Pull_ModalData, Pull_modeshapes
from Scripts.Optimize.ModalAnalysis import Run
from Scripts.SendEmail import SendEmail


p_type = 2022
G_HEIGTS = [3.5 ,3.75 ,4.0 ,4.25 ,4.5]
T_HEIGTS = np.arange(180,220,2)

def write_csv_line(lst,p_type):
    with open(ThisFolder + f'/Eigfreq_3D_data_{p_type}.csv','a',newline='') as f:
        write = csv.writer(f)
        write.writerow(lst)
            
    

def Pull_Eigenfrequencies():
    eigfreqs = []
    with open(path + '/FEM_Results/ModalData/Eigenfrequency.csv','r') as f:
        ef   = csv.reader(f)
        for ix, row in enumerate(ef):
            eigfreqs.append(row[1])
    return eigfreqs
		

  
UserParameterFolder = path + '/Scripts/Optimize/'
UserParameterFileName = 'LangenuenInput.py'


err_m = 0
for t_H in T_HEIGTS:
    for g_H in G_HEIGTS:
        FolderPath = ThisFolder + f'/Data/MD_{t_H}_{g_H}'
        #if not os.path.exists(FolderPath):
            #os.makedirs(FolderPath)
        ok = 0
        print(t_H,g_H)

        j = 0
        while ok != 1:
            try:
                j += 1
                Run(UserParameterFileName, UserParameterFolder, t_H, g_H, p_type)
            except Exception:
                err_m += 1
                
                if j == 10:
                    break
                
                elif j%2 == 0:
                    if round(t_H,5) == round(T_HEIGTS[-1],5):
                        SendEmail("Program Notification", f"Abaqus Exception! @ {p_type}\nT = {t_H} -- Increment down!\nG = {g_H}", "juelpb@stud.ntnu.no")
                        t_H -= 0.005
                    else:
                        SendEmail("Program Notification", f"Abaqus Exception! @ {p_type}\nT = {t_H} -- Increment up!\nG = {g_H}", "juelpb@stud.ntnu.no")
                        t_H += 0.001
                        
                else:
                    if round(g_H,5) == round(G_HEIGTS[-1],5):
                        SendEmail("Program Notification", f"Abaqus Exception! @ {p_type}\nT = {t_H}\nG = {g_H} -- Increment down!", "juelpb@stud.ntnu.no")
                        g_H -= 0.005
                    else:
                        SendEmail("Program Notification", f"Abaqus Exception! @ {p_type}\nT = {t_H}\nG = {g_H} -- Increment up!", "juelpb@stud.ntnu.no")
                        g_H += 0.001
                continue
            
            ok += 1

        sleep(1)
        Pull_ModalData()
        sleep(1)
        Pull_modeshapes()
        
        # Copy modal data dictonary
        if os.path.exists(FolderPath):
            shutil.rmtree(FolderPath)

        source_dir = path + '/FEM_Results/ModalData/'
        destination_dir = FolderPath
        
        shutil.copytree(source_dir, destination_dir)
        shutil.copyfile(path+'/FEM_Results/GlobalAnalysis/LangenuenGlobal.odb', FolderPath+ f'/LangenuenGlobal_{t_H}_{g_H}.odb')

        
        
    