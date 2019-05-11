# -*- coding: utf-8 -*-
"""
Created on Mon Feb 05 16:38:01 2018

@author: Enrique Alejandro

Description: this library contains non linear least square fitting routines
             for extracting material properties from AFM results and
             other useful fitting functions.

You need to have installed:
    lmfit -- > this can be easily installed if you have the anaconda distribution via conda install method: conda install -c conda-forge lmfit
"""



import numpy as np
from lmfit import minimize, Parameters
from scipy import stats
import sys
#syspath = 'C:/Users/enrique/Desktop/QuiqueTemp/ModelFree_LossAngle'
syspath = 'e:/github/modelfree_lossangle'
sys.path.append(syspath)
from libs.simulation.afm_calculations import sparse, log_scale
from libs.forcespec.fdanalysis import moving_av, snr_roi, snr_fd, inicon

    
def comp_fit(params, t, compliance, Jg, arms=3):
    """This function contains the fitting model that will be passed to the non linear least square process to retrieve generalized voigt viscoelastic parameters
    
    This function will be called by the main wrap around funtion: comp_fit_main. This model will be compared with simulation or experimental data to retrieve generalized Voigt parameters from creep data
    
    Parameters:
    ---------- 
    params :  numpy.ndarray
        parameters defined by the non-linear optimization wrap around function: comp_fit_main
    t: numpy.ndarray
        time trace
    compliance: numpy.ndarray
        array contaning the creep compliance data
    Jg : float
        glassy compliance of the material
    arms : int, optional
        number of Voigt units used in the fitting model
        
    Returns:
    ---------- 
    (model - compliance)/compliance : float
        residual to be minimized by lmfit.minimize in comp_fit_main function    
    """
    p = params.valuesdict()
    J1 = p['J1']
    tau1 = p['tau1']
    if arms > 1:
        J2 = p['J2']
        tau2 = p['tau2']
        if arms > 2:
            J3 = p['J3']
            tau3 = p['tau3']
            if arms >3:
                J4 = p['J4']
                tau4 = p['tau4']
                if arms >4:
                    J5 = p['J5']
                    tau5 = p['tau5']
                    if arms >5:
                        J6 = p['J6']
                        tau6 = p['tau6']
                        if arms >6:
                            J7 = p['J7']
                            tau7 = p['tau7']
                            if arms >7:
                                J8 = p['J8']
                                tau8 = p['tau8']
                            else:
                                J8 = 0.0
                                tau8=0.0
                        else:
                            J7=0.0
                            tau7=0.0
                            J8 = 0.0
                            tau8=0.0
                    else:
                        J6 = 0.0
                        tau6 = 0.0
                        J7=0.0
                        tau7=0.0
                        J8 = 0.0
                        tau8=0.0
                else:
                    J5 = 0.0
                    tau5 = 0.0
                    J6 = 0.0
                    tau6 = 0.0
                    J7=0.0
                    tau7=0.0
                    J8 = 0.0
                    tau8=0.0
            else:
                J4 = 0.0
                tau4 = 0.0
                J5 = 0.0
                tau5 = 0.0
                J6 = 0.0
                tau6 = 0.0
                J7=0.0
                tau7=0.0
                J8 = 0.0
                tau8=0.0
            
        else:
            J3 = 0.0
            tau3 = 0.0
            J4 = 0.0
            tau4 = 0.0
            J5 = 0.0
            tau5 = 0.0
            J6 = 0.0
            tau6 = 0.0
            J7=0.0
            tau7=0.0
            J8 = 0.0
            tau8=0.0
    else:
        J2 = 0.0
        tau2 = 0.0
        J3 = 0.0
        tau3 = 0.0
        J4 = 0.0
        tau4 = 0.0    
        J5 = 0.0
        tau5 = 0.0  
        J6 = 0.0
        tau6 = 0.0
        J7=0.0
        tau7=0.0
        J8 = 0.0
        tau8=0.0         
    model =  Jg + ( J1*( 1.0 - np.exp(-t/tau1) ) + J2*(1.0 - np.exp(-t/tau2) )  + \
    J3*( 1.0 - np.exp(-t/tau3) ) + J4*( 1.0 - np.exp(-t/tau4)) + J5*(1.0- np.exp(-t/tau5) )  + \
    J6*( 1.0 - np.exp(-t/tau6) ) + J7*( 1.0 - np.exp(-t/tau7)) + J8*(1.0- np.exp(-t/tau8) )  )  # according to eq. 3.5-16 in Tschoegl book. #+J6*exp(-t/tau6)+J7*exp(-t/tau7) )
    return  (model - compliance)/compliance #calculating the residual     


def comp_fit_main(t, compliance, Jg, arms =3, J1_i = 1.0e-9, tau1_i = 1.0e-4, J2_i = 1.0e-8, tau2_i = 1.0e-3, J3_i = 1.0e-7, tau3_i = 1.0e-2, J4_i = 1.0e-6, tau4_i = 1.0e-1, J5_i = 1.0e-5, tau5_i = 1.0e0, J6_i = 1.0e-5, tau6_i = 1.0e0, J7_i = 1.0e-5, tau7_i = 1.0e0, J8_i = 1.0e-5, tau8_i = 1.0e0):
    """Wrap around function for the non-linear fitting to get generalized voigt parameters from generalized maxwell parameters
    
    This function uses the auxiliary comp_fit function defined above for the minimization of error
    
    Parameters:
    ---------- 
    t: numpy.ndarray
        time trace
    compliance: numpy.ndarray
        array contaning the creep compliance data
    Jg : float
        glassy compliance of the material, can be calculated as the inverse of the glassy modulus
    arms : int, optional
        number of Voigt units used in the fitting model
    J1_i : float, optional
        initial guess for compliance of spring in 1st voigt unit
    tau1_i : float, optional
        initial guess for retardation time of 1st voigt unit
    J2_i : float, optional
        initial guess for compliance of spring in 2nd voigt unit
    tau2_i : float, optional
        initial guess for retardation time of 2nd voigt unit
    J3_i : float, optional
        initial guess for compliance of spring in 3rd voigt unit
    tau3_i : float, optional
        initial guess for retardation time of 3rd voigt unit
    J4_i : float, optional
        initial guess for compliance of spring in 4th voigt unit
    tau4_i : float, optional
        initial guess for retardation time of 4th voigt unit
    J5_i : float, optional
        initial guess for compliance of spring in 5th voigt unit
    tau5_i : float, optional
        initial guess for retardation time of 5th voigt unit
    J6_i : float, optional
        initial guess for compliance of spring in 6th voigt unit
    tau6_i : float, optional
        initial guess for retardation time of 6th voigt unit
    J7_i : float, optional
        initial guess for compliance of spring in 7th voigt unit
    tau7_i : float, optional
        initial guess for retardation time of 7th voigt unit
    J8_i : float, optional
        initial guess for compliance of spring in 8th voigt unit
    tau8_i : float, optional
        initial guess for retardation time of 8th voigt unit        
        
    Returns:
    ---------- 
    tau_c : float
        retrieved retardation times corresponding to generalied Voigt model
    J_c : float
        retrieved values of compliances of springs in the Voigt units of the generalized Voigt model        
    """    
    params = Parameters() # creating a set of Parameters for the fitting model
    params.add('J1', value = J1_i, min=0)
    params.add('tau1', value = tau1_i, min=tau1_i/10.0, max=tau1_i*10.0)
    if arms > 1:
        params.add('J2', value = J2_i, min=0)
        params.add('tau2', value = tau2_i, min=tau2_i/10.0, max=tau2_i*10.0)
        if arms >2:
            params.add('J3', value = J3_i, min=0)
            params.add('tau3', value = tau3_i, min=tau3_i/10.0, max=tau3_i*10.0)
            if arms>3:
                params.add('J4', value = J4_i, min=0)
                params.add('tau4', value = tau4_i, min=tau4_i/10.0, max=tau4_i*10.0)
                if arms>4:
                    params.add('J5', value = J5_i, min=0)
                    params.add('tau5', value = tau5_i, min=tau5_i/10.0, max=tau5_i*10.0)
                    if arms>5:
                        params.add('J6', value = J5_i, min=0)
                        params.add('tau6', value = tau5_i, min=tau5_i/10.0, max=tau5_i*10.0)
                        if arms>6:
                            params.add('J7', value = J5_i, min=0)
                            params.add('tau7', value = tau5_i, min=tau5_i/10.0, max=tau5_i*10.0)
                            if arms>7:
                                params.add('J8', value = J5_i, min=0)
                                params.add('tau8', value = tau5_i, min=tau5_i/10.0, max=tau5_i*10.0)
                
    result = minimize(comp_fit, params, args=(t, compliance, Jg, arms), method='leastsq')
    J_c= np.zeros(arms)  #N is the number of voigt units retrieved
    tau_c = np.zeros(arms)
    J_c[0] = result.params['J1'].value
    tau_c[0]= result.params['tau1'].value
    if arms > 1:
        J_c[1] =result.params['J2'].value
        tau_c[1]= result.params['tau2'].value
        if arms >2:
            J_c[2] =result.params['J3'].value
            tau_c[2]= result.params['tau3'].value
            if arms>3:
                J_c[3] =result.params['J4'].value
                tau_c[3]= result.params['tau4'].value
                if arms>4:
                    J_c[4] =result.params['J5'].value
                    tau_c[4]= result.params['tau5'].value
                    if arms>5:
                        J_c[5] =result.params['J6'].value
                        tau_c[5]= result.params['tau6'].value
                        if arms>6:
                            J_c[6] =result.params['J7'].value
                            tau_c[6]= result.params['tau7'].value
                            if arms>7:
                                J_c[7] =result.params['J8'].value
                                tau_c[7]= result.params['tau8'].value
    return tau_c, J_c




def func_chi(params, t, chi_exp, arms=3, logscale = False):
    """This function contains the fitting model that will be passed to the non linear least square process to retrieve generalized voigt viscoelastic parameters from force spectroscopy data
    
    Auxiliary function to retrieve viscoelastic parameters from force spectroscopy data
    Assumes that load history is linear in time (a common physical assumption made in force spectroscopy experiments)
    This function will be called by the main wrap around funtion: nls_fit. This model will be compared with simulation or experimental data to retrieve generalized Voigt parameters from force spectroscopy data
    This function performs a non-linear fit based on Eq. 14 and 15 in: López‐Guerra, Enrique A., Babak Eslami, and Santiago D. Solares. Journal of Polymer Science Part B: Polymer Physics (2017).
    
    Parameters:
    ---------- 
    params :  numpy.ndarray
        parameters defined by the non-linear optimization wrap around function: nls_fit
    t: numpy.ndarray
        time trace weighted in logarithmic scale
    chi_exp : numpy.ndarray
        time varying fluidity measured in force spectroscopy (assuming force grows linearly in time), weighted in logarithmic scale    
    arms : int
        number of Voigt units used in the fitting model
    logscale : boolean, optional
        default is False, if True the minimization is done weighting in logarithmic scale
            
    Returns:
    ---------- 
    (model_log - tip_norm_log) /tip_norm_log : float
        residual to be minimized by lmfit.minimize in nls_fit wrap around function    
    """
    p = params.valuesdict()
    Jg = p['Jg']
    J1 = p['J1']
    tau1 = p['tau1']
    if arms > 1:
        J2 = p['J2']
        tau2 = p['tau2']
        if arms > 2:
            J3 = p['J3']
            tau3 = p['tau3']
            if arms >3:
                J4 = p['J4']
                tau4 = p['tau4']
                if arms >4:
                    J5 = p['J5']
                    tau5 = p['tau5']
                else:
                    J5 = 0.0
                    tau5 = 0.0
            else:
                J4 = 0.0
                tau4 = 0.0
                J5 = 0.0
                tau5 = 0.0
        else:
            J3 = 0.0
            tau3 = 0.0
            J4 = 0.0
            tau4 = 0.0
            J5 = 0.0
            tau5 = 0.0
    else:
        J2 = 0.0
        tau2 = 0.0
        J3 = 0.0
        tau3 = 0.0
        J4 = 0.0
        tau4 = 0.0    
        J5 = 0.0
        tau5 = 0.0           
    model = (Jg+J1+J2+J3+J4+J5)*t + J1*tau1*(np.exp(-t/tau1)-1.0) \
        + J2*tau2*(np.exp(-t/tau2)-1.0)  + J3*tau3*(np.exp(-t/tau3)-1.0) + J4*tau4*(np.exp(-t/tau4)-1.0) \
        + J5*tau5*(np.exp(-t/tau5)-1.0) #+ J6*tau_c[5]*(exp(-t/tau_c[5])-1.0)
    if logscale ==True:
        return np.abs((np.log10(model) - np.log10(chi_exp))/np.log10(chi_exp))
    else:
        return   np.abs((model-chi_exp)/chi_exp)

def func_conv(params, t, F, t_res_fit, t_exp, tip_norm, arms, dt, logscale=False):
    """This function contains the fitting model that will be passed to the non linear least square process to retrieve generalized voigt viscoelastic parameters from force spectroscopy data
    
    Auxiliary function to retrieve viscoelastic parameters from force spectroscopy data
    Doens't assume that load history is linear in time
    This function will be called by the main wrap around funtion: nls_fit. This model will be compared with simulation or experimental data to retrieve generalized Voigt parameters from force spectroscopy data
    This function performs a non-linear fit based on Eq. 13 and 16 in: López‐Guerra, Enrique A., Babak Eslami, and Santiago D. Solares. Journal of Polymer Science Part B: Polymer Physics (2017).
    
    Parameters:
    ---------- 
    params :  numpy.ndarray
        parameters defined by the non-linear optimization wrap around function: nls_fit
    t: numpy.ndarray
        time trace
    F : numpy.ndarray
        tip-sample force trace
    t_res_fit : float
        this is the time resolution of the fit, it is set as 10 times the experimental resolution (it is advised to keep this number larger than the experimental resolution)
    t_exp : float
        total experimental (or simulation) time
    tip_norm_log : float
        tip position weighted in logarithmic scale with resolution equal to t_res_fit resolution
    arms : int
        number of Voigt units used in the fitting model
    dt : float
        time resolution of the experiment or simulation (inverse of sampling frequency)
    logscale : boolean, optional
        default is False, if True the minimization is done weighting in logarithmic scale
        
    Returns:
    ---------- 
    (model_log - tip_norm_log) /tip_norm_log : float
        residual to be minimized by lmfit.minimize in nls_fit wrap around function    
    """    
    p = params.valuesdict()  #defining parameters
    Jg = p['Jg']
    J1 = p['J1']
    tau1 = p['tau1']
    U1 = J1/tau1*np.exp(-t/tau1)
    if arms > 1:
        J2 = p['J2']
        tau2 = p['tau2']
        U2 = J2/tau2*np.exp(-t/tau2)
        if arms > 2:
            J3 = p['J3']
            tau3 = p['tau3']
            U3 = J3/tau3*np.exp(-t/tau3)
            if arms >3:
                J4 = p['J4']
                tau4 = p['tau4']
                U4 = J4/tau4*np.exp(-t/tau4)
                if arms >4:
                    J5 = p['J5']
                    tau5 = p['tau5']
                    U5 = J5/tau5*np.exp(-t/tau5)
                else:
                    J5 = 0.0
                    tau5 = 0.0
                    U5 = 0.0
            else:
                J4 = 0.0
                tau4 = 0.0
                U4 = 0.0
                J5 = 0.0
                tau5 = 0.0
                U5 = 0.0
        else:
            J3 = 0.0
            tau3 = 0.0
            U3 = 0.0
            J4 = 0.0
            tau4 = 0.0
            U4 = 0.0
            J5 = 0.0
            tau5 = 0.0
            U5 = 0.0
    else:
        J2 = 0.0
        tau2 = 0.0
        U2 = 0.0
        J3 = 0.0
        tau3 = 0.0
        U3 = 0.0
        J4 = 0.0
        tau4 = 0.0    
        U4 = 0.0
        J5 = 0.0
        tau5 = 0.0    
        U5 = 0.0
    U_t = U1+U2+U3+U4+U5
    model = np.convolve(U_t, F,mode='full')*dt  #convolution is performed in linear scale with dt equal to experimental resolution
    model =  model[range(np.size(F))] + Jg*F #adding the term related with the dirac delta function to the convolution
    if logscale:
        model_log, _ = log_scale(model, t, t_res_fit, t_exp)
        tip_norm_log, _ = log_scale(tip_norm, t, t_res_fit, t_exp)
    else:  #linear scale
        model_sp, _ = sparse(model,t, t_res_fit, t_exp)
        tip_norm_sp,_ = sparse(tip_norm, t, t_res_fit, t_exp)
        
    return  np.abs((model_log - tip_norm_log) /tip_norm_log) if logscale==True else np.abs((model_sp - tip_norm_sp))# /tip_norm_sp)


def nls_fit(time_tot, indentation_tot, force_tot, R=10.0e-9, technique=0, arms = 0.0, ic = [], snr_min = 10, smooth=False, window=10, logscale = False, tlims=False):
    """Wrap around function for the non-linear fitting to get generalized voigt parameters from static force spectroscopy data.
    
    This function performs a non-linear fit based on either 'Eq. 14 and 15' or 'Eq. 13 and 16'  in: López‐Guerra, Enrique A., Babak Eslami, and Santiago D. Solares. Journal of Polymer Science Part B: Polymer Physics (2017).
    This function uses the auxiliary func_conv and func_chi functions defined above for the error minimization
    
    Parameters:
    ---------- 
    time : numpy.ndarray
        time trace
    indentation : numpy.ndarray
        indentation history (i.e., negative of the tip position, generally tip: z_sensor - deflection)
    force : numpy.ndarray
        tip-sample force trace from static force spectroscopy data         
    R : float, optional
        tip radius
    arms : int, optional
        number of voigt units used in the generalized voigt model to perform the fitting (recommended to make this number equal to the number of decades in logarithmic scale available in the time window)
    technique : int, optional
        default value is zero and the fit is performed based on Eq. 14 and 15, otherwise is performed based on Eq. 13 and 16 in López‐Guerra, et.al. J of Pol Sci B: Pol Phys (2017), 55(10), pp.804-813.
    ic = list, optional
        initial conditions, if passed they should have the structure: [Jg, [J], [tau]]        
        
    Returns:
    ---------- 
    Jg_c : float
        retrieved glassy compliance
    tau_c : np.array
        retrieved retardation times corresponding to generalied Voigt model
    J_c : np.array
        retrieved values of compliances of springs in the Voigt units of the generalized Voigt model        
    """   
     
    #cropping the inputs to a certain point where they fulfill imposed criteria of signal to noise
    tmini = snr_fd(time_tot, force_tot, indentation_tot, snr_min)
    print('tmin_snr:%2.5f'%tmini)
    time, force = snr_roi(time_tot, force_tot, tmini)
    _, indentation = snr_roi(time_tot, indentation_tot, tmini)
    
    
    time = time_tot
    force = force_tot
    indentation = indentation_tot
    
    if ic != []:  #assigning value of ic_0 if ic is not empty, this will be used in next 'if statement' to see if there is a nan value
        ic_0 = ic[0]
    else:
        ic_0 = 1.0
    
    if (np.size(indentation[indentation<0])*1.0/ np.size(indentation) < 0.1) and (np.isnan(ic_0)==False):  #if negative values in indentation do not exceed 10 percent do minimization process
                
        t_res = np.mean(np.diff(time))
        t_exp = time[len(time)-1]
        
        if smooth:  #Smoothening indentation and force traces with moving average if boolean is set to True
            indentation = moving_av(indentation,window)
            force = moving_av(force, window)    
        
        if ic == []: #calculating initial conditions if they are not passed
            Jg_i, J_i, tau_i, arms = inicon(time, indentation, force, R, arms)  #getting initial conditions to be passed to the minimization
              
        params = Parameters() # creating a set of Parameters
        if ic != []:
            Jg_i = ic[0]
        params.add('Jg', value = Jg_i, min=0)
        if ic == []:
            J1_i = J_i[0]
            tau1_i = tau_i[0]
        else:
            J1_i = ic[1][0]
            tau1_i = ic[2][0]
        params.add('J1', value = J1_i, min=0)
        params.add('tau1', value = tau1_i, min=tau1_i/10.0, max=tau1_i*10.0)
        if arms > 1:
            if ic == []:
                J2_i = J_i[1]
                tau2_i = tau_i[1]
            else:
                J2_i = ic[1][1]
                tau2_i = ic[2][1]            
            params.add('J2', value = J2_i, min=0)
            params.add('tau2', value = tau2_i, min=tau2_i/10.0, max=tau2_i*10.0)
            if arms >2:
                if ic == []:
                    J3_i = J_i[2]
                    tau3_i = tau_i[2]
                else:
                    J3_i = ic[1][2]
                    tau3_i = ic[2][2]
                params.add('J3', value = J3_i, min=0)
                params.add('tau3', value = tau3_i, min=tau3_i/10.0, max=tau3_i*10.0)
                if arms>3:
                    if ic == []:
                        J4_i = J_i[3]
                        tau4_i = tau_i[3]
                    else:
                        J4_i = ic[1][3]
                        tau4_i = ic[2][3]
                    params.add('J4', value = J4_i, min=0)
                    params.add('tau4', value = tau4_i, min=tau4_i/10.0, max=tau4_i*10.0)
                    if arms>4:
                        if ic == []:
                            J5_i = J_i[4]
                            tau5_i = tau_i[4]
                        else:
                            J4_i = ic[1][4]
                            tau4_i = ic[2][4]
                        params.add('J5', value = J5_i, min=0)
                        params.add('tau5', value = tau5_i, min=tau5_i/10.0, max=tau5_i*10.0)
        if technique == 0:  #it is done on the linear load in time assumption
            Fdot,_,_,_,_ = stats.linregress(time, force)
            if logscale: #the minimization will be performed doing logarithmic scale weighting                   
                chi_exp = 16.0/3*np.sqrt(R)*indentation**1.5/Fdot
                chi_exp_log, t_log = log_scale(chi_exp, time, t_res, t_exp)
                F_log, _ = log_scale(force, time, t_res, t_exp)        
                chi_log=chi_exp_log[np.invert(np.isnan(chi_exp_log))]
                time_log =t_log[np.invert(np.isnan(chi_exp_log))]
                result = minimize(func_chi, params, args=(time_log, chi_log, arms), method='leastsq')
            else: #the minimization is performed using linear scale (not log)
                chi_simul = 16.0/3*np.sqrt(R)*indentation**1.5/Fdot                
                chi_sim = chi_simul[np.invert(np.isnan(chi_simul))]
                t_sim = time[np.invert(np.isnan(chi_simul))]    
                result = minimize(func_chi, params, args=(t_sim[1:], chi_sim[1:], arms), method ='leastsq')            
        else:  #there is no assumption on loading history
            t_res_fit = tmini #this is advised to be larger than time_res, here given by time when snr reaches certain criterion  
            indent_norm = 16.0/3*np.sqrt(R)*indentation**1.5
            dt = np.mean(np.diff(time)) 
            result = minimize(func_conv, params, args=(time, force, t_res_fit, t_exp, indent_norm, arms, dt, logscale), method='leastsq')
    
        
        Jg_c = result.params['Jg'].value
        J_c= np.zeros(arms)  #N is the number of voigt units retrieved
        tau_c = np.zeros(arms)
        J_c[0] = result.params['J1'].value
        tau_c[0]= result.params['tau1'].value
        if arms > 1:
            J_c[1] =result.params['J2'].value
            tau_c[1]= result.params['tau2'].value
            if arms >2:
                J_c[2] =result.params['J3'].value
                tau_c[2]= result.params['tau3'].value
                if arms>3:
                    J_c[3] =result.params['J4'].value
                    tau_c[3]= result.params['tau4'].value
                    if arms>4:
                        J_c[4] =result.params['J5'].value
                        tau_c[4]= result.params['tau5'].value
        
        return Jg_c, tau_c, J_c, tmini, time[-1]

    else:  #negative values of indentation exceeded the 10%
        return np.nan, np.nan, np.nan, np.nan, np.nan
        

def linear_force(params, t, force):
    """
    Calculation of the residual to perform linear fit on a function forcing its intercept to be at the origin
    
    This auxiliary function returns the residual which is passed to the main wrap around fitting function: linear_fit_nob
    
    Parameters:
    ---------- 
    params :  numpy.ndarray
        parameters defined by the non-linear optimization wrap around function: nls_fit
    t : numpy.ndarray
        independent variable (generally time trace)
    force : numpy.ndarray
        dependent variable (generally force trace)
    
    Returns:
    ----------
    (model - force) : numpy.ndarray
        residual calculated to be passed to the optimization function: linear_fit_nob    
    """
    p = params.valuesdict()
    A = p['A']
    model = (A)*t
    return  (model - force) #calculating the residual

def linear_fit_nob(x,y):
    """This function performs linear fit in the special case where intercept is zero
    
    Wrap around fitting function to perform linear fit based on non linear least square optimization
    
    Parameters:
    ---------- 
    x : numpy.ndarray
        independent variable (generally time trace)
    y : numpy.ndarray
        dependent variable (generally force trace)
    
    Returns:
    ----------
    Fdot : float
        slope of the fitted line (recall intercept is at the origin)       
    """
    m,b,_,_,_ = stats.linregress(x,y)  #initial guess based on standard linear fitting
    params = Parameters()
    params.add('A', value = m, min=0)
    result = minimize(linear_force, params, args=(x,y), method='leastsq')
    Fdot = result.params['A'].value  
    return Fdot