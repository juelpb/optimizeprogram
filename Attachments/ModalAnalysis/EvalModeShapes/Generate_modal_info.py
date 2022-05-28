def generate_modal_info():
          
        modal_info = {}
        
        Mode_eval = Evaluate_modeshapes()
        
        indexes = [idx for idx in Mode_eval.values()]
        
        
               
        #* ----------------------
        #* Mass
        #* ---------------------- 
          
        f_gen_mass = open(path + '/03_Results/ModalData/Generalized_mass.csv','r')
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
        
        f_eigfreq = open(path + '/03_Results/ModalData/Eigenfrequency.csv','r')
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
        names   = []
        modes = []
        n = 25
        
        for modename, mode in Mode_eval.items():
            
            modes.append(mode)
            
            x_disp  = get_modeshape(mode, 1, n)
            y_disp   = get_modeshape(mode, 2, n)
            z_disp  = get_modeshape(mode, 3, n)
            x_rot   = get_modeshape(mode, 4, n)
            y_rot  = get_modeshape(mode, 5, n)
            z_rot   = get_modeshape(mode, 6, n)
            
            
                   
            # qplot(x_disp,f'xdisp {modename}')
            # qplot(y_disp, f'ydisp {modename}')
            # qplot(z_disp,f'zdisp {modename}')
            # qplot(x_rot, f'xrot {modename}')
            # qplot(y_rot,f'yrot {modename}')
            # qplot(z_rot, f'zrot {modename}')
            
            temp = []
            
            for i in range(len(x_disp)):
                temp.append(x_disp[i])
                temp.append(y_disp[i])
                temp.append(z_disp[i])
                temp.append(x_rot[i])
                temp.append(y_rot[i])
                temp.append(z_rot[i])
                
            names.append(modename)
            modeshape.append(temp)
            
            
        modeshape = np.array(modeshape)
        
        #plt.figure()
        #plt.plot(modeshape[3][3::6])
        #plt.show()
        
        modal_info['Name'] = names
             
        modal_info['Modeshape'] = modeshape

        print(f"---- Modal info succecfully generated, used modes {modes} ----")

        return modal_info