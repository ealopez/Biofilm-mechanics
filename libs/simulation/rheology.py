# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:20:30 2018

@author: Enrique Alejandro

Description:  This library contains useful rheology based functions, for example for the interconversion
between generalized Maxwell and Voigt models, calculator of operator coefficients from Maxwell or Voigt 
parameters.
"""



import numpy as np
import os
import matplotlib.pyplot as plt
from numba import jit
from scipy import stats
import sys
#syspath = 'C:/Users/enrique/Desktop/QuiqueTemp/ModelFree_LossAngle'
syspath = 'd:/github/modelfree_lossangle'
sys.path.append(syspath)
from libs.simulation.afm_calculations import log_tw, log_scale, average_error, sparse, error_linscale
from libs.forcespec.fdanalysis import fdroi, moving_av, snr_fd, snr_roi, linear_fit_Nob
  
def j_storage(omega, Jg, J, tau):
    """this function gives an array of storage compliance on radian frequency
    
    Parameters:
    ---------- 
    omega : numpy.ndarray
        array of frequencies at which the loss compliance will be calculated
    Jg : float
        glassy compliance of the material
    J : numpy.ndarray or float
        array with compliances of the springs in the generalized voigt model, or single compliance in case of SLS model
    tau: numpy.ndarray or float
        array with retardation times in the generalized voigt model, or single tau in case if SLS model
    
    Returns:
    ---------- 
    J_prime : numpy.ndarray
        calculated storage moduli corresponding to the passed frequencies (omega)    
    """
    J_prime = np.zeros(len(omega))
    for i in range(len(omega)):
        if np.size(J) > 1:
            J_prime[i] = Jg + sum( J[:]/(1.0 + (  pow(omega[i],2)*pow(tau[:],2) ) ) )
        else: #the material is the standard linear solid (only one retardation time present)
            J_prime[i] = Jg + ( J/(1.0 + (  pow(omega[i],2)*pow(tau,2) ) ) )
            
    return J_prime

def j_loss(omega, Jg, J, tau, phi = 0.0):
    """this function returns an array of loss compliance on radian frequency
    
    Parameters:
    ---------- 
    omega : numpy.ndarray
        array of frequencies at which the loss compliance will be calculated
    Jg : float
        glassy compliance of the material
    J : numpy.ndarray or float
        array with compliances of the springs in the generalized voigt model, or single compliance in case of SLS model
    tau: numpy.ndarray or float
        array with retardation times in the generalized voigt model, or single tau in case if SLS model
    phi : float, optional
        steady state fluidity
    
    Returns:
    ---------- 
    J_biprime : numpy.ndarray
        calculated loss moduli corresponding to the passed frequencies (omega)    
    """
    J_biprime = np.zeros(len(omega))
    for i in range(len(omega)):
        if np.size(J)>1:
            J_biprime[i] = sum( J[:]*omega[i]*tau[:]/(1.0 + (pow(omega[i],2)*pow(tau[:],2)) ) ) + phi/omega[i]
        else:
            J_biprime[i] = ( J*omega[i]*tau/(1.0 + (pow(omega[i],2)*pow(tau,2)) ) ) + phi/omega[i]
    return J_biprime

def theta_v(omega, Jg, J, tau, phi = 0.0):    
    """this function returns an array of loss angle on radian frequency
    
    Parameters:
    ---------- 
    omega : numpy.ndarray
        array of frequencies at which the loss compliance will be calculated
    Jg : float
        glassy compliance of the material
    J : numpy.ndarray or float
        array with compliances of the springs in the generalized voigt model, or single compliance in case of SLS model
    tau: numpy.ndarray or float
        array with retardation times in the generalized voigt model, or single tau in case if SLS model
    phi : float, optional
        steady state fluidity
    
    Returns:
    ---------- 
    theta : numpy.ndarray
        calculated loss angle corresponding to the passed frequencies (omega)    
    """
    Jloss = j_loss(omega, Jg, J, tau, phi)
    Jstorage =  j_storage(omega, Jg, J, tau)
    theta = np.arctan(Jloss/Jstorage)*180/np.pi
    return theta

def g_loss(omega, G, tau, Ge = 0.0):
    """this function returns the value of G_loss for either a point value or an array of omega
    
    Parameters:
    ---------- 
    omega : numpy.ndarray
        array of frequencies at which the loss moduli will be calculated
    G : numpy.ndarray or float
        array with moduli of the springs in the generalized maxwell model, or single modulus in case of SLS model
    tau: numpy.ndarray or float
        array with relaxation times in the generalized maxwell model, or single tau in case if SLS model
    Ge : float, optional
        equilibrium modulus
    
    Returns:
    ---------- 
    G_biprime : numpy.ndarray
        calculated loss moduli corresponding to the passed frequencies (omega)    
    """
    if np.size(omega) == 1:  #calculation of point value of G_loss
        G_biprime = 0.0
        if np.size(G) >1: #the model has more than one arm
            G_biprime = sum(  G[:]*omega*tau[:]/( 1.0+pow(omega,2)*pow(tau[:],2) )  )
        else:  #the modela is the SLS
            G_biprime = (  G*omega*tau/( 1.0+pow(omega,2)*pow(tau,2) )  )
    else: #calculation of G_loss for an array of omega
        G_biprime = np.zeros(np.size(omega))
        if np.size(G) > 1: #the model has more than one arm
            for j in range(np.size(omega)):
                G_biprime[j] = sum(  G[:]*omega[j]*tau[:]/( 1.0+pow(omega[j],2)*pow(tau[:],2) )  )
        else: # the model is the SLS
            for j in range(np.size(omega)):
                G_biprime[j] = (  G*omega[j]*tau/( 1.0+pow(omega[j],2)*pow(tau,2) )  )              
    return G_biprime

def g_storage(omega, G, tau, Ge = 0.0):
    """this function returns the value of G_store for either a point value or an array of omega
    
    Parameters:
    ---------- 
    omega : numpy.ndarray
        array of frequencies at which the loss moduli will be calculated
    G : numpy.ndarray or float
        array with moduli of the springs in the generalized maxwell model, or single modulus in case of SLS model
    tau: numpy.ndarray or float
        array with relaxation times in the generalized maxwell model, or single tau in case if SLS model
    Ge : float, optional
        equilibrium modulus
    
    Returns:
    ---------- 
    G_prime : numpy.ndarray
        calculated storage moduli corresponding to the passed frequencies (omega)    
    """
    if np.size(omega) == 1:  #calculation of point value of G_loss
        G_prime = 0.0
        if np.size(G) >1: #the model has more than one arm
            Gg = Ge+sum(G[:])
            G_prime = Gg - sum(  G[:]/( 1.0+pow(omega,2)*pow(tau[:],2) )  )
        else:  #the model is the SLS
            Gg = Ge+G
            G_prime = Gg - (  G/( 1.0+pow(omega,2)*pow(tau,2) )  )
    else: #calculation of G_loss for an array of omega
        G_prime = np.zeros(np.size(omega))
        if np.size(G) > 1: #the model has more than one arm
            Gg = Ge+sum(G[:])
            for j in range(np.size(omega)):
                G_prime[j] = Gg - sum(  G[:]/( 1.0+pow(omega[j],2)*pow(tau[:],2) )  )
        else: # the model is the SLS
            for j in range(np.size(omega)):
                Gg = Ge+G
                G_prime[j] = Gg - (  G/( 1.0+pow(omega[j],2)*pow(tau,2) )  )             
    return G_prime    
    
def theta_g(omega, G, tau, Ge = 0):
    """this function returns the loss angle from Generalized Maxwell Prony Coefficients
    
    Parameters:
    ---------- 
    omega : numpy.ndarray
        array of frequencies at which the loss moduli will be calculated
    G : numpy.ndarray or float
        array with moduli of the springs in the generalized maxwell model, or single modulus in case of SLS model
    tau: numpy.ndarray or float
        array with relaxation times in the generalized maxwell model, or single tau in case if SLS model
    Ge : float, optional
        equilibrium modulus
        
    Returns:
    ---------- 
    theta : numpy.ndarray
        calculated loss angle corresponding to the passed frequencies (omega)    
    """
    Gloss = g_loss(omega, G, tau, Ge)
    Gstorage = g_storage(omega, G, tau, Ge)
    theta = np.arctan(Gloss/Gstorage)*180.0/np.pi
    return theta       


def chi_th(t, Jg, J, tau, phi = 0):
    """this function gives the strain response to a unit slope stress (the time varying fluidity)
    
    Parameters:
    ---------- 
    t : numpy.ndarray
        time trace
    Jg : float
        glassy compliance of the material
    J : numpy.ndarray or float
        array with compliances of the springs in the generalized voigt model, or single compliance in case of SLS model
    tau: numpy.ndarray or float
        array with retardation times in the generalized voigt model, or single tau in case if SLS model
    phi : float, optional
        steady state fluidity
    
    Returns:
    ---------- 
    chi : numpy.ndarray
        calculated time varying fluidity   
    """
    if np.size(J) > 1:
        Je = sum(J[:])+Jg
    else: #the model is the SLS
        Je = J+Jg
    chi = np.zeros(len(t))
    for i in range(len(t)):
        if np.size(J) >1 :
            chi[i] = Je*t[i] + sum(J[:]*tau[:]*(np.exp(-t[i]/tau[:])-1.0)) + 1/2.0*phi*pow(t[i],2)
        else:#the model is the SLS
            chi[i] = Je*t[i] + (J*tau*(np.exp(-t[i]/tau)-1.0)) + 1/2.0*phi*pow(t[i],2)
    return chi

def j_t(t, Jg, J, tau, phi=0):
    """this function returns the compliance in time t, for a model with given set of parameters
    
    Parameters:
    ---------- 
    t : numpy.ndarray
        time trace
    Jg : float
        glassy compliance of the material
    J : numpy.ndarray or float
        array with compliances of the springs in the generalized voigt model, or single compliance in case of SLS model
    tau: numpy.ndarray or float
        array with retardation times in the generalized voigt model, or single tau in case if SLS model
    phi : float, optional
        steady state fluidity
    
    Returns:
    ---------- 
    comp : numpy.ndarray
        calculated theoretical creep compliance   
    """
    if (np.size(J)) > 1:
        Je = sum(J[:])+Jg
    else: #the model is the SLS
        Je = J+Jg
    comp = np.zeros(len(t))
    for i in range (len(t)):
        if np.size(J) >1:
            comp[i] = Je - sum(J[:]*np.exp(-t[i]/tau[:])) + phi*t[i]
        else: #the model is the SLS
            comp[i] = Je - (J*np.exp(-t[i]/tau)) + phi*t[i]
    return comp

def g_t(t, G, tau, Ge = 0.0):
    """this function returns the relaxation modulus in time
    
    Parameters:
    ---------- 
    t : numpy.ndarray
        time trace
    G : numpy.ndarray or float
        array with moduli of the springs in the generalized maxwell model, or single modulus in case of SLS model
    tau: numpy.ndarray or float
        array with relaxation times in the generalized maxwell model, or single tau in case if SLS model
    Ge : float, optional
        equilibrium modulus
        
    Returns:
    ---------- 
    G_rel : numpy.ndarray
        calculated relaxation modulus    
    """
    G_rel = np.zeros(np.size(t))
    if np.size(G) == 1:  #the model is the SLS
        for i in range(np.size(t)):
            G_rel[i] = Ge + G*np.exp(-t[i]/tau)
    else: #the model has more than one arm
        for i in range(np.size(t)):
            G_rel[i] = Ge + sum(G[:]*np.exp(-t[i]/tau[:]))
    return G_rel

def u_t(t, J, tau, phi=0):
    """this function gives the response of a unit strain impulse
    
    It does not contain the term with the delta function: $J_g \delta (t)$, which has to be analitycally added
    
    Parameters:
    ---------- 
    t : numpy.ndarray
        time trace
    J : numpy.ndarray or float
        array with compliances of the springs in the generalized voigt model, or single compliance in case of SLS model
    tau: numpy.ndarray or float
        array with retardation times in the generalized voigt model, or single tau in case if SLS model
    phi : float, optional
        steady state fluidity
    
    Returns:
    ---------- 
    U : numpy.ndarray
        calculated theoretical unit strain impulse: $U(t) - J_g \delta (t)$
    """
    U = np.zeros(len(t))
    for i in range(len(t)):
        if np.size(J) > 1:
            U[i] = sum(J[:]/tau[:]*np.exp(-t[i]/tau[:])) + phi
        else: #the model is the SLS
            U[i] = J/tau*np.exp(-t[i]/tau) + phi
            
    return U

def conv_uf(t, F, Jg, J, tau, phi=0):
    """this function convolves force and the retardance U(t)
    
    Parameters:
    ---------- 
    t : numpy.ndarray
        time trace
    F : float
        modulus in the Maxwell arm that is in parallel with the spring
    Jg : float
        glassy compliance of the material
    J : numpy.ndarray or float
        array with compliances of the springs in the generalized voigt model, or single compliance in case of SLS model
    tau: numpy.ndarray or float
        array with retardation times in the generalized voigt model, or single tau in case if SLS model
    phi : float, optional
        steady state fluidity
    
    Returns:
    ---------- 
    conv : numpy.ndarray
        convolution of the viscoelastic retardation with the load history
    """
    dt = np.mean(np.diff(t))
    U = u_t(t, J, tau, phi)
    conv = np.convolve(U, F, mode='full')*dt
    conv = conv[range(len(F))] + Jg*F  #adding the contribution from the $J_g \delta(t)$ term
    return conv

def jt_sls(t, G, tau_m, Ge):
    """this function returns the compliance in time for a Maxwell SLS model (standard linear solid)
    
    Parameters:
    ---------- 
    t : numpy.ndarray
        time trace
    G : float
        modulus in the Maxwell arm that is in parallel with the spring
    tau: float
        relaxation time of the Maxwell arm
    Ge : float, optional
        equilibrium modulus
    
    Returns:
    ---------- 
    J_t : numpy.ndarray
        analytical creep compliance of the SLS model
    """
    Gg = G + Ge
    Jg = 1.0/Gg
    Je = 1.0/Ge
    J = Je - Jg
    tau_v = Gg/Ge*tau_m
    comp = np.zeros(len(t))
    for i in range (len(t)):
        comp[i] = Je - (J*np.exp(-t[i]/tau_v)) 
    return comp

def chi_sls(t, G, tau_m, Ge):
    """this function gives the strain response to a unit slope stress to the maxwell standard linear solid
    
    Parameters:
    ---------- 
    t : numpy.ndarray
        time trace
    G : float
        modulus in the Maxwell arm that is in parallel with the spring
    tau: float
        relaxation time of the Maxwell arm
    Ge : float, optional
        equilibrium modulus
    
    Returns:
    ---------- 
    chi : numpy.ndarray
        calculated time varying fluidity   
    """
    Gg = G + Ge
    Jg = 1.0/Gg
    Je = 1.0/Ge
    J = Je - Jg
    tau_v = Gg/Ge*tau_m
    chi = np.zeros(len(t))
    for i in range (len(t)):
        chi[i] = Je*t[i] + (J*tau_v*(np.exp(-t[i]/tau_v)-1.0)) 
    return chi
        
def jstar_obs(delta, Fts, t, R):
    """This function receives positive penetration history and loading history and calculates complex compliance, storage and loss compliances
    
    Parameters:
    ----------
    delta : numpy.ndarray
        indentation history during the force spectroscopy experiment, absolute value of tip position (e.g., z_sensor - deflection)
    fts : numpy.ndarray
        force history in the force spectroscopy approach curve
    t : numpy.ndarray
        time trace
    
    Returns:
    ----------
    nu : numpy.ndarray
        frequency axis of the complex compliance
    J_star_num : numpy.ndarray
        complex compliance
    J_store_num : numpy.ndarray
        storage compliance
    J_loss_num : numpy.ndarray
        loss compliance    
    """
    dt = np.mean(np.diff(t))
    N = np.size(t)
    sf = 1.0/dt
    nyquist = 1.0/2.0*sf
    
    Fdot, _,_,_,_ = stats.linregress(t, Fts)
    chi_t = 1.0/Fdot*16.0/3.0*np.sqrt(R)*(delta)**(3.0/2) # chi caculated from observables
    tmin = 0.95*t[len(t)-1]
    Je, _, _,_,_ = stats.linregress(t[t>tmin], chi_t[t>tmin]) #estimation of the equilibrium modulus
    kappa = -1.0*min(chi_t - Je*t)
    L_lambda_t =  chi_t - Je*t + kappa
    
    #GETTING FOURIER TRANSFORM TO GET COMPLEX COMPLIANCE IN TERMS OF fluidity (Chi_t, measured in experiments)
    Nu_total = np.fft.fftfreq(N,dt)
    nu = Nu_total[ (Nu_total>0) & (Nu_total<nyquist)]
    L_lambda_nu = np.fft.fft(L_lambda_t)*dt
    L_lambda_nu = L_lambda_nu[ (Nu_total>0) & (Nu_total<nyquist)]
    J_star_num = Je - (2.0*np.pi*nu)**2*L_lambda_nu  - 1.0j*(2.0*np.pi*nu)*kappa 
    J_store_num = np.real(J_star_num)
    J_loss_num = -1.0*np.imag(J_star_num)
    return nu, J_star_num, J_store_num, J_loss_num    


def jstar_chi(chi_t, Je, t):
    """This function receives theoretical chi (fluidity) and returns the complex compliance
    
    Parameters:
    ----------
    chi_t : numpy.ndarray
        time dependent fluidity, response of the viscoleastic material to a unit slope of stress
    t : numpy.ndarray
        time trace
    
    Returns:
    ----------
    nu : numpy.ndarray
        frequency axis of the complex compliance
    J_star_num : numpy.ndarray
        complex compliance
    J_store_num : numpy.ndarray
        storage compliance
    J_loss_num : numpy.ndarray
        loss compliance    
    """
    dt = np.mean(np.diff(t))
    N = np.size(t)
    sf = 1.0/dt
    nyquist = 1.0/2.0*sf
    
    kappa = -1.0*min(chi_t - Je*t)
    L_lambda_t =  chi_t - Je*t + kappa    
    #GETTING FOURIER TRANSFORM TO GET COMPLEX COMPLIANCE IN TERMS OF fluidity (Chi_t, measured in experiments)
    Nu_total = np.fft.fftfreq(N,dt)
    nu = Nu_total[ (Nu_total>0) & (Nu_total<nyquist)]
    L_lambda_nu = np.fft.fft(L_lambda_t)*dt
    L_lambda_nu = L_lambda_nu[ (Nu_total>0) & (Nu_total<nyquist)]
    J_star_num = Je - (2.0*np.pi*nu)**2*L_lambda_nu  - 1.0j*(2.0*np.pi*nu)*kappa 
    J_store_num = np.real(J_star_num)
    J_loss_num = -1.0*np.imag(J_star_num)
    return nu, J_star_num, J_store_num, J_loss_num 


def jcomplex_maxwell(omega, G, tau, Ge = 0.0):
    """caculation of complex compliance for the generalized maxwell model
    
    Parameters:
    ---------- 
    omega : numpy.ndarray
        array of frequencies at which the loss moduli will be calculated
    G : numpy.ndarray
        array with moduli of the springs in the generalized maxwell model, or single modulus in case of SLS model
    tau: numpy.ndarray
        array with relaxation times in the generalized maxwell model, or single tau in case if SLS model
    Ge : float, optional
        equilibrium modulus
    
    Returns:
    ----------
    J_star : numpy.ndarray
        complex compliance
    J_prime : numpy.ndarray
        storage compliance
    J_biprime : numpy.ndarray
        loss compliance    
    """
    Gg = Ge + sum(G[:])
    G_star = np.zeros(len(omega), dtype=complex)
    J_star = np.zeros(len(omega), dtype=complex)
    J_prime = np.zeros(len(omega))
    J_biprime = np.zeros(len(omega))
    for i in range(len(omega)):
        G_star[i] = Gg - sum( G[:] /(1.0+ 1.0j*omega[i]*tau[:]) )
        J_star[i] = 1.0/G_star[i]
        J_prime[i] = np.real(J_star[i])
        J_biprime[i] = -1.0*np.imag(J_star[i])
    return J_star, J_prime, J_biprime

def compliance_maxwell(N, G, tau , Ge = 0.0, dt = 0.0, simul_t = 0.0, printstep = 0.0, lw=0):
    """This function returns the numerical compliance of a Generalized Maxwell model.
    
    This numerical compliance is useful for interconversion from Gen Maxwell model to generalized Voigt model
    
    Parameters:
    ---------- 
    N: int
        number of relaxation times in the generalized maxwell model
    G :  numpy.ndarray
        moduli of the springs in the Maxwell arms of a generalized Maxwell model, if only one arm pass it as: np.array([G])
    tau: numpy.ndarray
        relaxation times of the Maxwell arms, if only one arm pass it as: np.array([tau])
    Ge : float, optional
        equilibrium modulus of the material, default value is zero 
    dt : float, optional
        simulation timestep
    simul_t : float, optional
        total simulation time
    printstep : float, optional
        timestep in the printed array of the returned compliance
    lw : int, optional
        flag to return calculated compliance with logarithmic weight, to activate flag int different than zero should be passed
    
    Returns:
    ---------- 
    np.array(t_r) : numpy.ndarray
        array containing the time trace
    np.array(J_r) : numpy.ndarray
        array containing the calculated creep compliance   
    """
    if dt == 0.0:  #if timestep is not user defined it will given as a fracion of the lowest characteristic time
        dt = tau[0]/100.0
    if simul_t == 0.0: #if simulation time is not defined it will be calculated with respect to largest retardation time
        simul_t = tau[N-1]*10.0e3
    if printstep == 0.0:
        printstep = dt
            
    G_a = []
    tau_a = []
    """
    this for loop is to make sure tau passed does not contain values lower than time step which would make numerical 
    integration unstable
    """
    for i in range(N):
        if tau[i] > dt*10.0:
            G_a.append(G[i])
            tau_a.append(tau[i])
    G = np.array(G_a)
    tau = np.array(tau_a)
    Gg = Ge
    for i in range(N): #this loop looks silly but if you replace it with Gg = Ge + sum(G[:]) it will conflict with numba making, simulation very slow
        Gg = Gg + G[i]
    eta = tau*G
    Jg =1.0/Gg  #glassy compliance
    N = len(tau)
    
    Epsilon_visco = np.zeros(N) #initial strain
    Epsilon_visco_dot = np.zeros(N) #initial strain velocity
        
    t_r = []  #creating list with unknown number of elements
    J_r = []  #creating list with unknown number of elements
    time = 0.0
    J_t = Jg #initial compliance
    print_counter = 1
        
    while time < simul_t: #CREEP COMPLIANCE SIMULATION, ADVANCING IN TIME
        time = time + dt
        sum_Gn_EpsVisco_n = 0.0   #this sum has to be resetted to zero every timestep
        for n in range(N):
            Epsilon_visco_dot[n] = G[n]*(J_t - Epsilon_visco[n])/eta[n]
            Epsilon_visco[n] = Epsilon_visco[n] + Epsilon_visco_dot[n]*dt
            sum_Gn_EpsVisco_n = sum_Gn_EpsVisco_n + G[n]*Epsilon_visco[n]
        J_t = (1 + sum_Gn_EpsVisco_n)/Gg 
        if time >= print_counter*printstep and time < simul_t:
            t_r.append(time)
            J_r.append(J_t)
            print_counter += 1
        if lw != 0:  #if logarithmic weight is activated, the data will be appended weighted logarithmically
            if print_counter == 10:
                printstep *= 10
                print_counter = 1
          
    return np.array(t_r), np.array(J_r)         

def relaxation_voigt(J, tau, Jg, phi_f = 0.0, dt = 1, simul_t = 1, lw = 0):
    """This function returns the numerical relaxation modulus of a Generalized Voigt model
        
    This numerical relaxation modulus is useful for interconversion from Gen Maxwell model to generalized Voigt model
    
    Parameters:
    ---------- 
    J :  numpy.ndarray
        compliances of the springs in the Voigt units of a generalized Voigt model
    tau: numpy.ndarray
        relaxation times of the Maxwell arms
    Jg : float
        glassy compliance of the material 
    dt : float, optional
        simulation timestep
    simul_t : float, optional
        total simulation time
    lw : int, optional
        flag to return calculated compliance with logarithmic weight
    
    Returns:
    ---------- 
    np.array(t_r) : numpy.ndarray
        array containing the time trace
    np.array(G_r) : numpy.ndarray
        array containing the calculated relaxation modulus   
    """    
    if dt == 1:  #if timestep is not user defined it will given as a fracion of the lowest characteristic time
        dt = tau[0]/100.0
    if simul_t ==1: #if simulation time is not defined it will be calculated with respect to largest retardation time
        simul_t = tau[len(tau)-1]*10.0e3
    
    J_a = []
    tau_a = []
    """
    this for loop is to make sure tau passed does not contain values lower than time step which would make numerical 
    integration unstable
    """
    for i in range(len(J)):
        if tau[i] > dt*10.0:
            J_a.append(J[i])
            tau_a.append(tau[i])
    J = np.array(J_a)
    tau = np.array(tau_a)
    
    Gg = 1.0/Jg
    N = len(tau)
    phi = J/tau
    #Defining initial conditions
    x = np.zeros(N)
    x_dot = np.zeros(N)
    t_r = []  #creating list with unknown number of elements
    G_r = []  #creating list with unknown number of elements
    time = 0.0
    G_t = Gg #initial relaxation modulus
    print_counter = 1
    tr = dt #printstep
    
    while time < simul_t: #RELAXATION MODULUS SIMULATION, ADVANCING IN TIME
        time = time + dt
        k = len(tau) - 1
        while k > -1:
            if k == len(tau) - 1:
                x_dot[k] = G_t*phi[k]
            else:
                x_dot[k] = G_t*phi[k] + x_dot[k+1]
            k -=1
        for i in range(len(tau)):
            x[i] = x[i] + x_dot[i]*dt
        G_t = Gg*(1.0-x[0])
        if time >= print_counter*tr and time <simul_t:
            t_r.append(time)
            G_r.append(G_t)
            print_counter += 1
        if lw != 0: #if logarithmic weight is activated, the data will be appended weighted logarithmically
            if print_counter == 10:
                tr = tr*10
                print_counter = 1
    
    return np.array(t_r), np.array(G_r) 


def chi_maxwell(t,J_t):
    """This function returns the fluidity from the compliance calculated by the compliance_maxwell function
    
    This function will perform numerical integration to obtain the fluidity from its derivative (i.e., the compliance)
    The fluidity at time zero is zero from linear viscoelastic theory
    
    Parameters:
    ----------
    t : numpy.ndarray
        time trace
    J_t : numpy.ndarray
        creep compliance
    
    Returns:
    ----------
    chi_t = numpy.ndarray
        fluidity of the viscoelastic material (response to unit slope of stress)    
    """
    chi_t = np.zeros(len(t))
    dt = np.mean(np.diff(t))
    for i in range(0,len(t)-1):
        chi_t[i+1] = chi_t[i] + J_t[i]*dt
    return chi_t

def U_maxwell(t, J_t):
    """This function gets the retardance from the creep compliance calculated numerically by compliance_maxwell function
    
    Parameters:
    ----------
    t : numpy.ndarray
        time trace
    J_t : numpy.ndarray
        creep compliance
    
    Returns:
    ----------
    U_t = numpy.ndarray
        retardance of the viscoelastic material (i.e., response of the material to unit stress impulse)
    """      
    U_t = np.zeros(np.size(J_t))
    for i in range(np.size(J_t)):  # calculation of derivative using central difference scheme
        if i == 0:
            U_t[i] = (J_t[1]-J_t[0])/(t[1] - t[0])
        elif i == np.size(J_t)-1:
            U_t[np.size(J_t)-1] = (J_t[np.size(J_t)-1]-J_t[np.size(J_t)-2]) / \
                                      (t[np.size(J_t) - 1] - t[np.size(J_t) - 2])
        else:
            U_t[i] = (J_t[i+1]-J_t[i-1])/(t[i + 1] - t[i - 1])
    return U_t

def jcomplex_modelfree(indent, Fts, t, R, model = False, model_params = [], t_res = 0.0, t_exp = 0.0):
    """This function performs the method of direct calculation of complex moduli and compare it in plots with the theoretical model
    
    Parameters
    ---------- 
    indent : numpy.ndarray
        array containing the indentation history (negative of the tip position history)
    Fts: numpy.ndarray
        tip-sample force array
    t: numpy.ndarray
        time trace   
    R : float, optional
        tip radius
    model :  boolean, optional
        it is assumed a theoretical model is not passed
    model_params : list, optional
        if theoretical model is available pass a list with this structure : [[G], [tau], Ge]
    t_res : float, optional
        time resolution if the experiment (i.e., inverse of sampling frequency)
    t_exp : float, optional
        total time of the experiment or simulation
        
    Returns:
    ----------
    nu_obs : numpy.ndarray
        frequency axis of the complex compliance
    Jprime_obs : numpy.ndarray
        storage compliance
    Jbiprime_obs : numpy.ndarray
        loss compliance       
    """
    nu_obs, _, Jprime_obs, Jbiprime_obs = jstar_obs(indent, Fts, t, R)
    if t_res == 0.0:
        t_res = np.mean(np.diff(t))
    if t_exp == 0.0:
        t_exp = t[len(t)-1]
    max_omega = 1.0/t_res*2.0*np.pi
    min_omega = 1.0/t_exp*2.0*np.pi
    omega = log_tw(min_omega, max_omega, 10)
    tmin = 0.95*t_exp #minimum time for calculation of equilibrium compliance
    if model: #model is available
        _, Jprime_an, Jbiprime_an = jcomplex_maxwell(omega, model_params[0], model_params[1], model_params[2])
        jit_compliance = jit()(compliance_maxwell)
        t_m, J_m, = jit_compliance(np.size(model_params[0]), model_params[0], model_params[1], model_params[2], t_res, t_exp)
        chi_m = chi_maxwell(t_m, J_m) #obtaining the theoretical fluidity chi of the generalized Maxwell model
        chim_Jet = chi_m - 1.0/model_params[2]*t_m    
    Fdot,_,_,_,_ = stats.linregress(t, Fts)
    chi_exp = 16.0/3.0*np.sqrt(R)*(indent)**1.5/Fdot
    Je_fit, b, _,_,_ = stats.linregress(t[t>tmin], chi_exp[t>tmin])  #calculation of Je  
    chi_Jet = chi_exp - Je_fit*t    
        

    plt.figure(1, figsize=(14,4))
    plt.subplot(1,2,1)
    if model:
        plt.plot(np.log10(omega/(2.0*np.pi)), np.log10(Jprime_an), 'y', lw=5,label = 'J_store analytical')
    plt.plot(np.log10(nu_obs), np.log10(Jprime_obs), 'b--', lw=3, label='J_store from fft experiment')
    plt.legend(loc='best')
    if model:
        plt.ylim(np.log10(min(Jprime_an)), np.log10(max(Jprime_an)*1.3))
    else:
        plt.ylim( np.log10(max(Jprime_obs)*1.0e-5), np.log10(max(Jprime_obs)) )
    plt.xlabel(r'$log(f), \, Hz$', fontsize='20',fontweight='bold')
    plt.ylabel(r'$log[J^{I}(2 \pi f)], \,Pa^{-1}$',fontsize='20',fontweight='bold')
    
    plt.figure(1, figsize=(14,4))
    plt.subplot(1,2,2)
    if model:
        plt.plot(np.log10(omega/(2.0*np.pi)), np.log10(Jbiprime_an), 'y', lw=5,label = 'J_loss analytical')
    plt.plot(np.log10(nu_obs), np.log10(Jbiprime_obs), 'b--', lw=3, label='J_loss from fft experiment')
    if model:
        plt.ylim(np.log10(min(Jbiprime_an)), np.log10(max(Jbiprime_an)*1.3))
    else:
        plt.ylim( np.log10(max(Jbiprime_obs)*1.0e-5), np.log10(max(Jbiprime_obs)) )
    plt.legend(loc='best')
    plt.xlabel(r'$log(f), \, Hz$', fontsize='20',fontweight='bold')
    plt.ylabel(r'$log[J^{II}(2 \pi f)], \,Pa^{-1}$',fontsize='20',fontweight='bold')

    plt.figure(2, figsize=(14,4))
    plt.subplot(1,2,1)
    if model:
        plt.plot(np.log10(omega/(2.0*np.pi)), np.arctan((Jbiprime_an)/(Jprime_an))*180.0/np.pi, 'y', lw=5,label = 'Loss angle analytical')
    plt.plot(np.log10(nu_obs), np.arctan((Jbiprime_obs)/(Jprime_obs))*180.0/np.pi, 'b--', lw=3, label='Loss angle from fft experiment')
    plt.ylim(0,90)
    plt.legend(loc='best')
    plt.xlabel(r'$log(f), \, Hz$', fontsize='20',fontweight='bold')
    plt.ylabel(r'$\theta(2 \pi f), \, deg$', fontsize='20',fontweight='bold')
    
    plt.subplot(1,2,2)
    if model:
        plt.plot(np.log10(t_m), (chim_Jet -  min(chim_Jet))*1.0e6, 'y', lw=5, label='theoretical')
    plt.plot(np.log10(t), (chi_Jet - min(chi_Jet))*1.0e6, 'b--', lw=3, label='Simulation')
    plt.xlabel('$log(time), s$', fontsize='20',fontweight='bold')
    plt.ylabel(r'$\chi - J_e t + \kappa, a.u.$', fontsize='20',fontweight='bold')
    
    return nu_obs, Jprime_obs, Jbiprime_obs 


def error_nls(indent, Fts, t, Jg_c, J_c, tau_c, R = 10.0e-9, linear_load = True, model=False, model_params = [], t_res=0.0, t_exp=0.0):
    """This function evaluates the error of the non-linear squares fit performed to obtain viscoelastic properties
    
    Parameters
    ---------- 
    indent : numpy.ndarray
        array containing the indentation history (negative of the tip position history)
    Fts: numpy.ndarray
        tip-sample force array
    t: numpy.ndarray
        time trace
    Jg_c : float
        glassy compliance retrieved in the nls fit
    J_c : numpy.ndarray or float
        array with compliances of the springs in the generalized voigt model retrieved in the nls fit, or single compliance in case of SLS model
    tau_c : numpy.ndarray or float
        array with retardation times in the generalized voigt model retrieved in the nls fit, or single tau in case if SLS model
    R : float, optional
        tip radius
    linear_load : boolean, optional
        method with linear load load assumption is assumed unless False is passed
    model :  boolean, optional
        it is assumed a theoretical model is not passed
    model params : list, optional
        if theoretical model is available pass a list with this structure : [Ge, [G], [tau]]
    t_res : float, optional
        time resolution if the experiment (i.e., inverse of sampling frequency)
    t_exp : float, optional
        total time of the experiment or simulation
        
    Returns
    ----------
    errors : list
        average percentage error of the non-linear square optimization, average % error compared to model of: compliance, storage compliance, loss compliance and loss angle   
    """
    if t_res == 0.0:
        t_res = np.mean(np.diff(t))
    if t_exp == 0.0:
        t_exp = t[len(t)-1]
    t_log = log_tw(t_res, t_exp)
    max_omega = 1.0/t_res*2.0*np.pi
    min_omega = 1.0/t_exp*2.0*np.pi
    omega = log_tw(min_omega, max_omega, 10)
    #calculation of harmonic responses of retrieved model
    J_prime_fit = j_storage(omega, Jg_c, J_c, tau_c)
    J_biprime_fit = j_loss(omega, Jg_c, J_c, tau_c)
    loss_angle_fit = J_biprime_fit/J_prime_fit
    Jt_fit = j_t(t_log, Jg_c, J_c, tau_c)
    if linear_load:
        Fdot,_,_,_,_ = stats.linregress(t, Fts) 
        indent_log, t_log = log_scale(indent, t, t_res, t_exp)
        chi_exp = 16.0/3*np.sqrt(R)*indent_log**1.5/Fdot
        chi_fit = chi_th(t_log, Jg_c, J_c, tau_c)
        err_nls, err_nls_point = average_error(chi_exp, chi_fit, t_log)
    else: #no assumption in loading history
        indent_log, _ = log_scale(indent, t, t_res, t_exp)
        convol_obs =  16.0/3*np.sqrt(R)*indent_log**1.5
        conv_fit = conv_uf(t, Fts, Jg_c, J_c, tau_c)  #convolution of retardation with load history
        convfit_log, t_log = log_scale(conv_fit, t, t_res, t_exp)
        indent_log, _ = log_scale(indent, t, t_res, t_exp)
        err_nls, err_nls_point = average_error(convol_obs, convfit_log, t_log)
        
        
    if model:
        Ge = float(model_params[0])
        G = np.array(model_params[1])
        tau = np.array(model_params[2])
        M = len(tau)
        jit_compliance = jit()(compliance_maxwell)
        t_m, J_m = jit_compliance(M, G, tau, Ge, t_res/10.0, t_exp)
        J_log, _ = log_scale(J_m, t_m, t_res, t_exp)
        chi_theoretical = chi_maxwell(t_m, J_m)
        _, J_primet, J_biprimet = jcomplex_maxwell(omega, G, tau, Ge)
        theta_model = J_biprimet/J_primet
        err_comp, err_comp_point = average_error(J_log, Jt_fit, t_log)
        err_Jprime, err_Jprime_point = average_error(J_primet, J_prime_fit, omega)
        err_Jbiprime, err_Jbiprime_point = average_error(J_biprimet, J_biprime_fit, omega)
        err_lossangle, err_lossangle_point = average_error(theta_model, loss_angle_fit, omega, True, False)   
    else: #no model available
        err_comp, err_Jprime, err_Jbiprime, err_lossangle = 0.0, 0.0, 0.0, 0.0
    
    #MAKING PLOTS
    i = 1
    plt.figure(i, figsize=(14,4))
    plt.subplot(1,2,1)
    if linear_load:     
        if model:
            chi_th_log, _ = log_scale(chi_theoretical, t_m, t_res, t_exp)
            plt.plot(np.log10(t_log), np.log10(chi_th_log), 'y', lw=5, label= 'Theoretical')
        plt.plot(np.log10(t_log), np.log10(chi_exp), 'r*', markersize=10, label=r'Exp observable, see Eq.(14)')
        plt.plot(np.log10(t_log), np.log10(chi_fit), 'b', lw = 2.0, label=r'Voigt Fit, Eq. (9)')
        plt.legend(loc='best', fontsize=13)
        plt.xlabel(r'$log(time), \,s$', fontsize='20',fontweight='bold')
        plt.ylabel(r'$log(\chi(t)), \,Pa^{-1}s$',fontsize='20',fontweight='bold')        
        
    else: #no assumption in loading history
        if model:
            u_maxw = U_maxwell(t_m, J_m)  #retardance of the generalized maxwell
            dt = np.mean(np.diff(t))    
            conv_th = np.convolve(u_maxw, Fts, mode='full')*dt
            conv_th = conv_th[range(len(Fts))] + (1.0/(Ge+sum(G[:])) )*Fts  #adding the contribution from the $J_g \delta(t)$ term
            convth_log, t_log = log_scale(conv_th, t, t_res, t_exp)            
            plt.plot(np.log10(t_log), np.log10(convth_log), 'y', lw=10.0, label= 'Theoretical')
        plt.plot(np.log10(t_log), np.log10(16.0/3.0*np.sqrt(R)*(indent_log)**(3.0/2)), 'r*', markersize=10, label = 'Observables')  #Lee Radok
        plt.plot(np.log10(t_log), np.log10(convfit_log), 'b', lw=2.0, label = 'Fit convolution')
        plt.legend(loc='best', fontsize=13)
        plt.xlabel(r'$time, \,s$', fontsize='20',fontweight='bold')
        plt.ylabel(r'$\int_0^t J(t-\zeta) \frac{dF(\zeta)}{d\zeta} d\zeta$',fontsize='20',fontweight='bold')  
        
    plt.subplot(1,2,2)
    plt.plot(np.log10(t_log), err_nls_point)
    plt.xlabel(r'$log(time), \,s$', fontsize='20',fontweight='bold')
    plt.ylabel(r'$Error$',fontsize='20',fontweight='bold')
    i+=1    
    
    plt.figure(i+1, figsize=(14,4))
    plt.subplot(1,2,1)
    if model:        
        plt.plot(np.log10(t_log), np.log10(J_log), 'r*', markersize=10, label=r'Theoretical, J(t)')
    plt.plot(np.log10(t_log), np.log10(Jt_fit), 'b', lw = 3.0, label=r'Voigt Fit, Eq. (9)')
    plt.legend(loc=4, fontsize=13)
    plt.xlabel(r'$time, \,s$', fontsize='20',fontweight='bold')
    plt.ylabel(r'$J(t), \,Pa^{-1}$',fontsize='20',fontweight='bold')
    
    if model:
        plt.subplot(1,2,2)
        plt.plot(np.log10(t_log), err_comp_point)
        plt.xlabel(r'$log(time), \,s$', fontsize='20',fontweight='bold')
        plt.ylabel(r'$Error$',fontsize='20',fontweight='bold')
    i += 1
    
    plt.figure(i+1, figsize=(14,4))
    plt.subplot(1,2,1)
    if model:        
        plt.plot(np.log10(omega), np.log10(J_primet), 'r*', markersize=10, label=r'Theoretical')
    plt.plot(np.log10(omega), np.log10(J_prime_fit), 'b', lw = 3.0, label=r'Voigt Fit, Eq. (9)')
    plt.legend(loc='best', fontsize=13)
    plt.xlabel(r'$log \omega, \,rad/s$', fontsize='20',fontweight='bold')
    plt.ylabel(r'$log J^{I}(\omega), \,Pa^{-1}$',fontsize='20',fontweight='bold')
    if model:
        plt.subplot(1,2,2)
        plt.plot(np.log10(omega), err_Jprime_point)
        plt.xlabel(r'$log(time), \,s$', fontsize='20',fontweight='bold')
        plt.ylabel(r'$Error$',fontsize='20',fontweight='bold')
    i += 1
    
    plt.figure(i+1, figsize=(14,4))
    plt.subplot(1,2,1)
    if model:
        plt.plot(np.log10(omega), np.log10(J_biprimet), 'r*', markersize=10, label=r'Theoretical')
    plt.plot(np.log10(omega), np.log10(J_biprime_fit), 'b', lw = 3.0, label=r'Voigt Fit, Eq. (9)')
    plt.legend(loc='best', fontsize=13)
    plt.xlabel(r'$\omega, \,rad/s$', fontsize='20',fontweight='bold')
    plt.ylabel(r'$J^{II}(\omega), \,Pa^{-1}$',fontsize='20',fontweight='bold')
    if model:
        plt.subplot(1,2,2)
        plt.plot(np.log10(t_log), err_Jbiprime_point)
        plt.xlabel(r'$log(time), \,s$', fontsize='20',fontweight='bold')
        plt.ylabel(r'$Error$',fontsize='20',fontweight='bold')
    i += 1
    
    plt.figure(i+1, figsize=(14,4))
    plt.subplot(1,2,1)
    if model:        
        plt.plot(np.log10(omega), np.arctan(J_biprimet/J_primet)*180.0/np.pi, 'r*', markersize=10, label=r'Theoretical')
    plt.plot(np.log10(omega), np.arctan(J_biprime_fit/J_prime_fit)*180.0/np.pi, 'b', lw = 3.0, label=r'Voigt Fit, Eq. (9)')
    plt.legend(loc='best', fontsize=13)
    plt.xlabel(r'$\omega, \,rad/s$', fontsize='20',fontweight='bold')
    plt.ylabel(r'$\theta(\omega),\,deg$',fontsize='20',fontweight='bold')
    if model:
        plt.subplot(1,2,2)
        plt.plot(np.log10(t_log), err_lossangle_point)
        plt.xlabel(r'$log(\omega), \,rad/s$', fontsize='20',fontweight='bold')
        plt.ylabel(r'$Error$',fontsize='20',fontweight='bold')
    
    return [err_nls, err_comp, err_Jprime, err_Jbiprime, err_lossangle] 


def error_conv(z_tot, d_tot, t_tot, Jg_nl, J_nl, tau_nl, k, R, gauss_filt = False, percent_roi=50.0, scale_indent = 5.0, tc=10.0, smooth=False, window=10, snr_min = 10, mparams = [], plotit =False, savefig=[], Hertz=False):
    """This function gives visualization of NLS fit for case on non-linear load, linear scale weight error minimization
    
    the inputs relate to the observables of and average fd curve (the outputs from avfd_time)
    or a single raw curve (the outputs from fd_align function)
    
    Parameters
    ---------- 
    z_tot : numpy.ndarray
        array containing the z-sensor history
    d_tot: numpy.ndarray
        deflection array
    t_tot: numpy.ndarray
        time trace
    Jg_nl : float
        glassy compliance retrieved in the nls fit
    J_nl : numpy.ndarray or float
        array with compliances of the springs in the generalized voigt model retrieved in the nls fit, or single compliance in case of SLS model
    tau_nl : numpy.ndarray or float
        array with retardation times in the generalized voigt model retrieved in the nls fit, or single tau in case if SLS model
    k : float
        cantilever stiffness
    R : float
        tip radius
    gauss_filt : boolean, optional
        flag to indicate if gauss filtering is needed to find snap-in position
    percent_roi : float, optional
        percentage of the approach size that is going to be included in the retract portion
    scale_indent: float, optional
        scaling of indentation for facilitating visualization, default is off but you may pass a value like 10.0 or 20.0
    tc : float, optional
        percentage bound allowed to the convolution kernel to be away from the observables
    mparams : list, optional
        if theoretical model is available pass a list with this structure : [Ge, [G], [tau]] 
    savefig : boolean, optional
        flag to indicate if one wants to save the figures generated by the function
    Hertz : boolean, optional
        boolean indicating if the Hertzian elastic fit will be drawn as well for reference, default is False    
    """
    
    if smooth:  #the input here are already generally smoothed, so this is not necessary
        z_tot = moving_av(z_tot, window)
        d_tot = moving_av(d_tot, window)
        
    t, zs, defl, indent = fdroi(t_tot, z_tot, d_tot, gauss_filt, percent_roi, False)
    Fts = defl*k    
    
    #cropping the inputs to a certain point where they fulfill imposed criteria of signal to noise
    tmini = snr_fd(t, Fts, indent, snr_min)
    t_nl, Fts_nl = snr_roi(t, Fts, tmini)
    _, indent_nl = snr_roi(t, indent, tmini)
    
    t_nl = t
    Fts_nl = Fts
    indent_nl = indent
    
    
    dt_default = np.mean(np.diff(t))
    convol_obs =  16.0/3*np.sqrt(R)*indent_nl**1.5
    conv_fit = conv_uf(t_nl, Fts_nl, Jg_nl, J_nl, tau_nl, 0.0)
    
    err_nls_point = error_linscale(convol_obs, conv_fit, t_nl)
    t_threshold = t_nl[err_nls_point < tc]
    
    d_ap = d_tot[:z_tot.argmax()]
    mini = d_ap[d_ap.argmin()]*k
    convfit = conv_uf(t, Fts, Jg_nl, J_nl, tau_nl)
    
    ind_fit = (3.0/16*(1.0/np.sqrt(R))*convfit)**(2.0/3) 
    
    if Hertz:
        nu = 0.5
        alfa = 4.0/3*np.sqrt(R)/(1.0-nu**2)
        #E, _,_,_,_ = stats.linregress(indent_nl**1.5, Fts/alfa)
        E = linear_fit_Nob(indent_nl**1.5, Fts/alfa)
    
    if plotit:
        plt.figure(figsize=(14,4))
        plt.subplot(1,2,1)
        if mparams != []:  #reference model is passed
            Ge = float(mparams[0])
            G = np.array(mparams[1])
            tau = np.array(mparams[2])
            M = len(tau)
            jit_compliance = jit()(compliance_maxwell)
            t_exp = t_nl[len(t_nl)-1]
            t_m, J_m = jit_compliance(M, G, tau, Ge, dt_default, t_exp)
            u_maxw = U_maxwell(t_m, J_m)  #retardance of the generalized maxwell
            u_maxw_nl, _ = sparse(u_maxw, t)        
            dt_nl = np.mean(np.diff(t_nl))    
            conv_th = np.convolve(u_maxw_nl, Fts_nl, mode='full')*dt_nl
            plt.plot(t_nl*1.0e3, conv_th*1.0e18, 'g', lw=2.0, label= 'Theoretical')    
        plt.plot(t_nl*1.0e3, convol_obs*1.0e18, 'y', lw=6, label = 'Experimental') 
        plt.plot(t_nl*1.0e3, conv_fit*1.0e18, 'b--', lw=3, label = 'Fit convolution')
        #plt.legend(loc='best', fontsize=15)
        plt.xlabel(r'$time, \, ms$', fontsize='20',fontweight='bold')
        plt.ylabel(r'$\int_0^t \; U(t-\zeta) p(\zeta) d\zeta, a.u.$',fontsize='20',fontweight='bold') 
        
    
        plt.subplot(1,2,2)    
        plt.plot(-(z_tot-d_tot)*1.0e9, (d_tot*k)*1.0e9, 'y', lw=6, label = 'Experimental')
        plt.plot(-ind_fit*1.0e9, (defl*k+mini)*1.0e9, 'b--', lw=4, label = 'Viscoelastic fit')
        if Hertz:
            plt.plot(-(z_tot-d_tot)*1.0e9, ((z_tot-d_tot)**1.5*alfa*E+mini)*1.0e9, color='r', ls=':', lw=4, label = 'Elastic fit')
        plt.legend(loc = 'best', fontsize=15, frameon=False)
        plt.xlabel(r'$Tip \, Position, \,nm$', fontsize='20',fontweight='bold')
        plt.ylabel(r'$Force, \,nN$', fontsize='20',fontweight='bold')
        plt.xlim(-1.2*max(indent)*1.0e9, 3.0*max(indent)*1.0e9)
        
        if savefig != []:
            os.chdir(savefig)
            plt.savefig('Convol_NLS_FDcurve.png', bbox_inches='tight')
                   

        plt.figure(figsize=(14,4))
        plt.subplot(1,2,1)
        plt.plot(t, zs*1.0e9, lw=3, label = 'z-sensor')
        if scale_indent == 0.0:
            plt.plot(t, indent*1.0e9, lw=3, label = 'indentation')
        else:
            plt.plot(t, indent*1.0e9*scale_indent, lw=3, label = 'scaled indentation')
            
        plt.plot(t, defl*k*1.0e9, lw=3, label='force')
        plt.legend(loc='best', fontsize=12)
        plt.xlabel(r'$time, \,s$', fontsize='20',fontweight='bold')
        plt.ylabel(r'$Position, \,nm$', fontsize='20',fontweight='bold')
        
        plt.subplot(1,2,2)    
        plt.plot(t_nl, err_nls_point)
        plt.xlabel(r'$log(time), \,s$', fontsize='20',fontweight='bold')
        plt.ylabel(r'$Error$',fontsize='20',fontweight='bold')   
        plt.ylim(0,100)
        if savefig != []:
            os.chdir(savefig)
            plt.savefig('Convol_NLS_error.png', bbox_inches='tight')
        
    maxi = z_tot.argmax()  
    ap_vel,_,_,_,_ = stats.linregress(t_tot[0:maxi],z_tot[0:maxi])  #estimated approach velocity
    return t_nl, convol_obs, conv_fit, (z_tot-d_tot), (d_tot*k), ind_fit, (defl*k+mini), ap_vel, t_threshold, err_nls_point


def visco_nlsfit(params_nl, t_res, t_exp, plotit = False, pathsave = [], params_ll=[], model_params = []):
    """This function evaluates viscoelastic functions
    
    It gives the calculated viscoelastic responses from the parameters calculated in the NLS optimization
    
    Parameters
    ---------- 
    params_nl : list
        [Jg_nl, tau_nl, J_nl]
    t_res : float
        time resolution if the experiment (i.e., inverse of sampling frequency)
    t_exp : float
        total time of the experiment or simulation
    plotit : boolean, optional
        if True it will plot the figures when called
    pathsave : list, optional
        if passed the figures will be saved to the string given
    params_ll : list, optional
        [Jg_l, tau_l, J_l]
    model :  boolean, optional
        it is assumed a theoretical model is not passed
    model params : list, optional
        if theoretical model is available pass a list with this structure : [Ge, [G], [tau]]       
    """
    t_log = log_tw(t_res, t_exp)
    max_omega = 1.0/t_res*2.0*np.pi
    min_omega = 1.0/t_exp*2.0*np.pi
    omega = log_tw(min_omega, max_omega, 10)
    
    if params_ll != []:
        Jg_l = params_ll[0]
        J_l = params_ll[2]
        tau_l = params_ll[1]
        #calculation of harmonic responses of retrieved model for linear load case
        J_prime_fit_l = j_storage(omega, Jg_l, J_l, tau_l)
        J_biprime_fit_l = j_loss(omega, Jg_l, J_l, tau_l)
        Jt_fit_l = j_t(t_log, Jg_l, J_l, tau_l)
    
    Jg_nl = params_nl[0]
    J_nl = params_nl[2]
    tau_nl = params_nl[1]
    #calculation of harmonic responses of retrieved model for non-linear load case
    J_prime_fit_nl = j_storage(omega, Jg_nl, J_nl, tau_nl)
    J_biprime_fit_nl = j_loss(omega, Jg_nl, J_nl, tau_nl)
    Jt_fit_nl = j_t(t_log, Jg_nl, J_nl, tau_nl)     
    Ut_fit_nl = u_t(t_log, J_nl, tau_nl)
        
    if model_params != []:
        Ge = float(model_params[0])
        G = np.array(model_params[1])
        tau = np.array(model_params[2])
        M = len(tau)
        jit_compliance = jit()(compliance_maxwell)
        t_m, J_m = jit_compliance(M, G, tau, Ge, t_res/10.0, t_exp)
        J_th, t_th = log_scale(J_m, t_m, t_res, t_exp)
        _,J_primet, J_biprimet = jcomplex_maxwell(omega, G, tau, Ge)
            
    #MAKING PLOTS
    if plotit:
        plt.figure(1, figsize=(7,4))
        if model_params != []:       
            plt.plot(np.log10(t_th), np.log10(J_th), 'r*', markersize=10, label=r'Theoretical, J(t)')
        if params_ll != []:
            plt.plot(np.log10(t_log), np.log10(Jt_fit_l), 'b', lw = 3.0, label=r'Fit, linear load assumption')
        plt.plot(np.log10(t_log), np.log10(Jt_fit_nl), 'orange', lw = 3.0, label=r'Fit, non-linear load assumption')
        #plt.legend(loc=4, fontsize=13)
        plt.xlabel(r'$log(time), \,s$', fontsize='20',fontweight='bold')
        plt.ylabel(r'$log[J(t)], \,Pa^{-1}$',fontsize='20',fontweight='bold')
        if pathsave != []:
            os.chdir(pathsave)
            plt.savefig('Compliance.png', bboxinches='tight')
    
        plt.figure(2, figsize=(7,4))
        if model_params != []:        
            plt.plot(np.log10(omega), np.log10(J_primet), 'r*', markersize=10, label=r'Theoretical')
        if params_ll != []:
            plt.plot(np.log10(omega), np.log10(J_prime_fit_l), 'b', lw = 3.0, label=r'Fit, linear load assumption')
        plt.plot(np.log10(omega), np.log10(J_prime_fit_nl), 'orange', lw = 3.0, label=r'Fit, non-linear load assumption')
        #plt.legend(loc='best', fontsize=13)
        plt.xlabel(r'$log (\omega), \,rad/s$', fontsize='20',fontweight='bold')
        plt.ylabel(r'$log [J^{I}(\omega)], \,Pa^{-1}$',fontsize='20',fontweight='bold')
        if pathsave != []:
            os.chdir(pathsave)
            plt.savefig('Jprime.png', bboxinches='tight')
           
        plt.figure(3, figsize=(7,4))
        if model_params != []: 
            plt.plot(np.log10(omega), np.log10(J_biprimet), 'r*', markersize=10, label=r'Theoretical')
        if params_ll != []:
            plt.plot(np.log10(omega), np.log10(J_biprime_fit_l), 'b', lw = 3.0, label=r'Fit, linear load assumption')
        plt.plot(np.log10(omega), np.log10(J_biprime_fit_nl), 'orange', lw = 3.0, label=r'Fit, non-linear load assumption')
        #plt.legend(loc='best', fontsize=13)
        plt.xlabel(r'$log (\omega), \,rad/s$', fontsize='20',fontweight='bold')
        plt.ylabel(r'$log[J^{II}(\omega)], \,Pa^{-1}$',fontsize='20',fontweight='bold')
        if pathsave != []:
            os.chdir(pathsave)
            plt.savefig('Jbiprime.png', bboxinches='tight')
           
        plt.figure(4, figsize=(7,4))
        if model_params != []:         
            plt.plot(np.log10(omega), np.arctan(J_biprimet/J_primet)*180.0/np.pi, 'r*', markersize=10, label=r'Theoretical')
        if params_ll != []:
            plt.plot(np.log10(omega), np.arctan(J_biprime_fit_l/J_prime_fit_l)*180.0/np.pi, 'b', lw = 3.0, label=r'Fit, linear load assumption')
        plt.plot(np.log10(omega), np.arctan(J_biprime_fit_nl/J_prime_fit_nl)*180.0/np.pi, 'orange', lw = 3.0, label=r'Fit, non-linear load assumption')
        #plt.legend(loc='best', fontsize=13)
        plt.xlabel(r'$log(\omega), \,rad/s$', fontsize='20',fontweight='bold')
        plt.ylabel(r'$\theta(\omega),\,deg$',fontsize='20',fontweight='bold')
        if pathsave != []:
            os.chdir(pathsave)
            plt.savefig('LossAngle.png', bboxinches='tight')
    return t_log, Ut_fit_nl, Jt_fit_nl, omega, J_prime_fit_nl, J_biprime_fit_nl, np.arctan(J_biprime_fit_nl/J_prime_fit_nl)*180.0/np.pi
    


def error_chi(z_tot, d_tot, t_tot, params_ll, t_res, t_exp, k, R, av = True, model=False, mparams = [], savefig =False):
    """This function gives visualization of NLS fit for case on non-linear load, linear scale weight error minimization
    
    the inputs relate to the observables of and average fd curve (the outputs from avfd_time)
    or a single raw curve (the outputs from fd_align function)
    
    Parameters
    ---------- 
    z_tot : numpy.ndarray
        array containing the z-sensor history
    d_tot: numpy.ndarray
        deflection array
    t_tot: numpy.ndarray
        time trace
    params_ll : list
        [Jg_l, tau_l, J_l]
    t_res : float, optional
        time resolution if the experiment (i.e., inverse of sampling frequency)
    t_exp : float, optional
        total time of the experiment or simulation
    k : float
        cantilever stiffness
    R : float
        tip radius
    av : boolean, optional
        flag to indicate that observables passed correspond to average force curve
    percent_roi : float, optional
        percentage of the approach size that is going to be included in the retract portion
    model :  boolean, optional
        it is assumed a theoretical model is not passed
    mparams : list, optional
        if theoretical model is available pass a list with this structure : [Ge, [G], [tau]] 
    savefig : boolean, optional
        flag to indicate if one wants to save the figures generated by the function
    """
    Jg_l = params_ll[0]
    J_l = params_ll[2]
    tau_l = params_ll[1]
    
    t, zs, defl, indent = fdroi(t_tot, z_tot, d_tot, av, 0.0)
    Fts = defl*k
    indent_l, t_l = sparse(indent, t, t_res, t_exp)  #adjusting to have the desired experimental resolution
    Fts_l, _ = sparse(Fts, t, t_res, t_exp) #adjusting to have the desired experimental resolution
    
    #MAKING PLOTS
    i = 1
    plt.figure(i, figsize=(14,4))
    plt.subplot(1,2,1)   
    if model:
        chi_th_log, _ = log_scale(chi_theoretical, t_m, t_res, t_exp)
        plt.plot(np.log10(t_log), np.log10(chi_th_log), 'y', lw=5, label= 'Theoretical')
    Fdot,_,_,_,_ = stats.linregress(t, Fts) 
    chi_exp = 16.0/3*np.sqrt(R)*indent_l**1.5/Fdot
    plt.plot(t_l, chi_exp, 'y', lw=6, label=r'Exp observables')
    chi_fit = chi_th(t_l, Jg_l, J_l, tau_l)
    plt.plot(t_l, chi_fit, 'b--', lw = 3.0, label=r'Fit linear load assumption')
    plt.legend(loc='best', fontsize=13)
    plt.xlabel(r'$time, \,s$', fontsize='20',fontweight='bold')
    plt.ylabel(r'$\chi(t), \,Pa^{-1}s$',fontsize='20',fontweight='bold') 
    
    plt.subplot(1,2,2)
    _, err_nls_point = average_error(chi_exp, chi_fit, t_l)
    plt.plot(t_l, err_nls_point)
    plt.xlabel(r'$time, \,s$', fontsize='20',fontweight='bold')
    plt.ylabel(r'$Error$',fontsize='20',fontweight='bold')
    if savefig:
        plt.savefig('FitLinearLoad_Chi.png', bbox_inches ='tight')
    i += 1
    
    plt.figure(i, figsize=(7,4))
    d_ap = d_tot[:z_tot.argmax()]
    mini = d_ap[d_ap.argmin()]*k
    chifit = chi_th(t, Jg_l, J_l, tau_l)
    ind_fit = (3.0/16*(1.0/np.sqrt(R))*Fdot*chifit)**(2.0/3)
    plt.plot((z_tot-d_tot)*1.0e9, (d_tot*k)*1.0e9, 'y', lw=6, label = 'Experimental')
    plt.plot(ind_fit*1.0e9, (defl*k+mini)*1.0e9, 'b--', lw=3, label = 'Fit linear load assumption')
    plt.xlim(-2.0*max(indent)*1.0e9, 1.5*max(indent)*1.0e9)
    plt.legend(loc = 'best', fontsize=15)
    plt.xlabel(r'$indentation, \,nm$', fontsize='20',fontweight='bold')
    plt.ylabel(r'$Force, \,nN$', fontsize='20',fontweight='bold')
    
    if savefig:
        plt.savefig('FitLinearLoad_ForceCurve.png', bbox_inches ='tight')
    
    
           
       