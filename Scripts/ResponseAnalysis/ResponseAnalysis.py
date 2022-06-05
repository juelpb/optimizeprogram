import sys
import os
path = os.getcwd()
sys.path.append(path)

import ModalProperties as mp 
import numpy as np
from matplotlib import pyplot as plt

import csv
import WindSpectra as ws
import Static_coeff as sc
import scipy.integrate as integrate

plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=15)
plt.rcParams.update({'font.size': 15})


def ResponseAnalysis(p_type, g_height, U, I_u, alpha=0, plot=True):
    """Calculating buffeting response for latest modal- and AD data

    Args:
        p_type (int): Parametrization type. 2020, 2021, 2022
        g_height (float): Girder height
        U (float): Mean wind velocity
        I_u (float): Turbulence intensity. 
        alpha (float, optional): Mean angle of attack. Defaults to 0.
        plot (bool, optional): Plot figures. Defaults to True.

    Returns:
        float: Max. RMS standard deviation in the horizontal, vertical and pitching DOFs
    """
    Modal = mp.generate_modal_info_all_modes()

    D = g_height
    B = 31
    comp = ['u', 'w']                              # Turbulence components encounted
    U = U                                          # Mean wind [hor, vert]
    I_u = I_u                                      # Turbulence intensity
    sigma = [I_u*U, I_u*0.25*U]                    # Wind speed std.dev. [hor, vert]
    rho = 1.25                                     # Air density
    z = 76.6                                       # Girder elevation at midspan
    dw = 0.01                                      # Step size at frequency axis
    Omegas = np.arange(0, 3.0, dw)                 # Frequency axis
    n_nodes = int(Modal['Modeshape'].shape[1]/6)   # Number of nodes in modeshape
    n_modes = 39                                   # Number of modes included in analysis
    x = np.linspace(0,1235,n_nodes)                # x-axis
    dL = 1235/n_nodes                              # Element length

    C_d, dC_d, C_l, dC_l, C_m, dC_m = sc.pull_static_coeff(p_type, g_height, alpha)

    print(f'Force coefficients:\nC_d = {C_d}\nC_l = {C_l}\nC_m = {C_m}\ndC_d = {dC_d}\ndC_l = {dC_l}\ndC_m = {dC_m}')
    B_q1 = (rho*U*B/2)*np.array([[2*D*C_d/B, 0], [2*C_l, 0], [2*B*C_m, 0]])
    B_q2 = (rho*U*B/2)*np.array([[0, (D/B)*dC_d-C_l], [0, D*C_d/B+dC_l], [0, B*dC_m]])


    #___________N400 turbulence spectra_____
    # Single point spectra
    S_a = [ws.SinglePointSpectra('u', U, sigma[0], z, Omegas, 'W'), ws.SinglePointSpectra('w', U, sigma[1], z, Omegas, 'W')]
    dxdx = np.array([x]) - np.array([x]).T
    S_uu = np.zeros([len(Omegas), n_nodes, n_nodes])
    S_ww = np.zeros([len(Omegas), n_nodes, n_nodes])
    # Co-spectra
    for k in range(len(Omegas)):
        S_uu[k,:,:] = S_a[0][k]*np.exp(-10.0*np.abs(dxdx)*Omegas[k]/(U))
        S_ww[k,:,:] = S_a[1][k]*np.exp(-6.5*np.abs(dxdx)*Omegas[k]/(U))


    # Assembling modeshape vector
    phi_y = np.zeros([n_modes, n_nodes])
    phi_z = np.zeros([n_modes, n_nodes])
    phi_theta = np.zeros([n_modes, n_nodes])
    phi = np.zeros([n_modes, n_nodes, 3])
    for n in range(n_modes):
        for i in range(n_nodes):
            phi_y[n,i] = Modal['Modeshape'][n,i*6+1]
            phi_z[n,i] = Modal['Modeshape'][n,i*6+2]
            phi_theta[n,i] = Modal['Modeshape'][n,i*6+3]
            phi[n, i, 0] = phi_y[n,i]
            phi[n, i, 1] = phi_z[n,i]
            phi[n, i, 2] = phi_theta[n,i]




    #___________Modal load_______________
    S_Q = np.zeros([len(Omegas),n_modes])
    for n in range(n_modes):
        for k in range(len(Omegas)):
            integrand = phi[n,:,:]@B_q1@np.transpose(B_q1)@np.transpose(phi[n,:,:])*S_uu[k,:,:] + phi[n,:,:]@B_q2@np.transpose(B_q2)@np.transpose(phi[n,:,:])*S_ww[k,:,:]
            S_Q[k,n] = np.trapz(np.trapz(integrand,dx=dL),dx=dL)
        
    #__________Frequency response function________
    AD_data = mp.Load_ADs(p_type, g_height)
    H = []
    for i, Omega in enumerate(Omegas):
        M, C, K = mp.modal_aero(U, Omega, Modal, AD_data)
        H_temp = 2*(-Omega**2*M[:n_modes, :n_modes] + 1j*Omega*C[:n_modes, :n_modes] + K[:n_modes, :n_modes])**(-1)
        H.append(H_temp)
    H = np.array(H)

        
    S_eta = np.zeros([len(Omegas), n_modes, n_modes])

    for k in range(len(Omegas)):
        S_eta[k,:,:] = np.abs(np.conj(H[k,:,:]))*S_Q[k]*np.abs(np.transpose(H[k,:,:]))
    
    
    
    S_eta_tot = np.zeros([len(Omegas)])
    H_tot = np.zeros([len(Omegas)])
    for i in range(n_modes):
        S_eta_tot += S_eta[:,i,i]
        H_tot += np.abs(np.real(H[:,i,i]))
    

    #____Transferring to physical coordinates____

    sigma_ry = np.zeros([n_modes, n_nodes])
    sigma_ry_tot = np.zeros(n_nodes)
    sigma_rz = np.zeros([n_modes, n_nodes])
    sigma_rz_tot = np.zeros(n_nodes)
    sigma_rtheta = np.zeros([n_modes, n_nodes])
    sigma_rtheta_tot = np.zeros(n_nodes)
    
    stheta = np.zeros([len(Omegas), n_modes])
    
    for n in range(0,n_modes):
        for i in range(n_nodes):
            S_ry = phi[n,i,0]**2*S_eta[:,n,n]
            sigma_ry[n,i] = np.sqrt(np.trapz(S_ry, Omegas))
            sigma_ry_tot[i] += sigma_ry[n,i]**2
            S_rz = phi[n,i,1]**2*S_eta[:,n,n]
            sigma_rz[n,i] = np.sqrt(np.trapz(S_rz, Omegas))
            sigma_rz_tot[i] += sigma_rz[n,i]**2
            S_rtheta = (360/(2*np.pi)*phi[n,i,2])**2*S_eta[:,n,n]
            stheta[:,n] += phi[n,i,2]**2*S_eta[:,n,n]
            sigma_rtheta[n,i] = np.sqrt(np.trapz(S_rtheta, Omegas))
            sigma_rtheta_tot[i] += sigma_rtheta[n,i]**2

    sigma_ry_tot = np.sqrt(sigma_ry_tot)
    sigma_rz_tot = np.sqrt(sigma_rz_tot)
    sigma_rtheta_tot = np.sqrt(sigma_rtheta_tot)
    
    
    if plot==True:
        #________Plot_________
        modes = ['HS1', 'VA1', 'VS1', 'HA1', 'VS2', 'VA2', 'TS1']
        modes_numb = [0,1,2,3,4,6,22]
        idx1 = 0
        idx2 = 0
        idx3 = 0
        
        # Modal load spectrum
        fig = plt.figure(figsize=(12, 6))
        for i in range(23):
            if i in modes_numb:
                plt.plot(Omegas[:], S_Q[:,i], label=f'{modes[idx1]}')
                idx1 += 1
            else:
                plt.plot(Omegas[:], S_Q[:,i], color='#a6cee3')
        plt.title('Modal load spectrum')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('$S_Q$ [m^2/(rad/s)]')
        plt.yscale('log')
        plt.legend(loc='upper right')
        plt.grid()
        plt.savefig(path + '/Scripts/ResponseAnalysis/Figures/Modal_load.png',dpi=300)
        plt.show()
        
        
        # Frequency response function
        fig = plt.figure(figsize=(12, 6))
        for i in range(23):
            if i in modes_numb:
                plt.plot(Omegas[:], np.abs(H[:,i,i]), label=f'{modes[idx2]}')
                idx2 += 1
            else:
                plt.plot(Omegas[:], np.abs(H[:,i,i]), color='#a6cee3')
        plt.title('Frequency response function')
        plt.xlabel('Frequency [rad/s]')
        plt.ylabel('|H| [(m/$s^2$)/N]')
        plt.yscale('log')
        plt.legend(loc='upper right')
        plt.grid()
        plt.savefig(path + '/Scripts/ResponseAnalysis/Figures/FRF.png',dpi=300)
        plt.show()
        
        
        # Modal coordinate spectrum
        fig = plt.figure(figsize=(12, 6))
        #plt.plot(Omegas[:100], S_eta_tot[:100], label = 'Sum of modes', color='#1f78b4')
        for i in range(23):
            if i in modes_numb:
                plt.plot(Omegas[:], S_eta[:,i,i], label=f'{modes[idx3]}')
                idx3 += 1
            else:
                plt.plot(Omegas[:], S_eta[:,i,i], color='#a6cee3')
        plt.title('Modal coordinate spectrum')
        plt.xlabel('Frequency [rad/s]')
        plt.ylabel('$S_{\eta}$ [$m^2$/(rad/s)]')
        plt.yscale('log')
        plt.legend(loc='upper right')
        plt.grid()
        plt.savefig(path + '/Scripts/ResponseAnalysis/Figures/Modal_coord.png',dpi=300)
        plt.show()    
        
        
        # RMS of standard deviation
        fig = plt.figure(figsize=(14, 6))
        fig.tight_layout()
        #fig.suptitle('Standard deviation of buffeting response', fontsize=16)

        ax1 = fig.add_subplot(311)
        ax1.title.set_text('Horizontal response')
        ax1.set_xlabel('x [m]')
        ax1.set_ylabel('$\sigma_y$ [m]')
        ax1.plot(x, sigma_ry_tot, color='#1f78b4')
        ax1.grid()

        ax2 = fig.add_subplot(312)
        ax2.title.set_text('Vertical response')
        ax2.set_xlabel('x [m]')
        ax2.set_ylabel('$\sigma_z$ [m]')
        ax2.plot(x, sigma_rz_tot, color='#1f78b4')
        ax2.grid()

        ax3 = fig.add_subplot(313)
        ax3.title.set_text('Torsional response')
        ax3.set_xlabel('x[m]')
        ax3.set_ylabel('$\sigma_{\u03B8}$ [\N{DEGREE SIGN}]')
        ax3.plot(x, sigma_rtheta_tot, color='#1f78b4')
        ax3.grid()
        
        plt.subplots_adjust(hspace = 1.0)
        plt.savefig(path + '/Scripts/ResponseAnalysis/Figures/RMS_std.png',dpi=300)
        plt.show()
    return np.amax(sigma_ry_tot), np.amax(sigma_rz_tot), np.amax(sigma_rtheta_tot)




ResponseAnalysis(2022, 3.70, 30, 0.10, alpha=0, plot=True)


