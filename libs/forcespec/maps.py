# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 21:29:35 2018

@author: Enrique Alejandro
"""




import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
from glob import glob
import re
from scipy.stats import linregress
ruta = os.getcwd()

#importing customized libraries
import sys
#syspath = 'C:/Users/enrique/Desktop/QuiqueTemp/ModelFree_LossAngle'
syspath = 'e:/github/modelfree_lossangle'
sys.path.append(syspath)
from libs.forcespec.fdanalysis import fdroi, fd_align, fdcurve_ibw, loadibw, offsety, offsetx, moving_av, linear_fit_Nob
from libs.simulation.nls_fit import nls_fit
from libs.simulation.rheology import j_storage, j_loss, j_t



def map_Jinv_theta(t, viscomap, rows, columns):
    """This function receives a 2D map of viscoelastic paramters and returns 6 2D maps: 3 maps of 1/J(t) at three distinct times, 3 maps of theta at three distinct omegas
    
    Parameters:
    ----------
    t : np.array
        time trace, number of elements in t array will be equally to the number of 2D maps generated in the calculations
    viscomap : list
        list containing a 2D map of viscoelastic parameters, the returned matrix of map_viscoparams function    
    
    Returns:
    ----------
    Jinv : 3D np.array
        a list of 2D lists corresponding to the calculated inverse creep compliance for the specific elements of t
    theta : 3D np.array
        a list of 2D lists corresponding to the calculated loss angle for the specific elements of t  
    """
    tsize= len(t)  #number of Jinv maps to be returned
    omega = 2.0*np.pi/t #the size is number of theta maps to be returned
   
    Jinv = np.zeros((tsize, rows, columns))
    theta = np.zeros((tsize, rows, columns))
    
        
    for i in range(rows):
        for j in range(columns):
            Jg = viscomap[i][j][0]
            tau = viscomap[i][j][1]
            J = viscomap[i][j][2]
            Jt = j_t(t, Jg, J, tau)
            J_prime = j_storage(omega, Jg, J, tau)
            J_biprime = j_loss(omega, Jg, J, tau)
            loss_angle = np.arctan(J_biprime/J_prime)*180.0/np.pi
            for n in range(tsize):
                Jinv[n,i,j] = 1.0/Jt[n]
                theta[n,i,j] = loss_angle[n]                       
    return Jinv, theta

def map_Jinv_theta2(t, Jg, J, tau):
    """This function receives a 2D map of viscoelastic paramters and returns 6 2D maps: 3 maps of 1/J(t) at three distinct times, 3 maps of theta at three distinct omegas
    
    Parameters:
    ----------
    t : np.array
        time trace, number of elements in t array will be equally to the number of 2D maps generated in the calculations
    Jg : np.2darray
        2d array with glassy compliance per pixel in force map  
    J : np.2darray
        2d array with compliance of sls spring per pixel in force map
    tau : np.2darray
        2d array with retardation time per pixel in force map
    
    Returns:
    ----------
    Jinv : 3D np.array
        a list of 2D lists corresponding to the calculated inverse creep compliance for the specific elements of t
    theta : 3D np.array
        a list of 2D lists corresponding to the calculated loss angle for the specific elements of t  
    """
    tsize= len(t)  #number of Jinv maps to be returned
    omega = 2.0*np.pi/t #the size is number of theta maps to be returned
    omega = omega[::-1]   #reversing to have from lower to higher value of omega
   
    Jinv = np.zeros((tsize, int(np.shape(J)[0]), int(np.shape(J)[1])))
    theta = np.zeros((tsize, int(np.shape(J)[0]), int(np.shape(J)[1])))
            
    for i in range(int(np.shape(J)[0])):
        for j in range(int(np.shape(J)[1])):
            Jt = j_t(t, Jg[i,j], J[i,j], tau[i,j])
            J_prime = j_storage(omega, Jg[i,j], J[i,j], tau[i,j])
            J_biprime = j_loss(omega, Jg[i,j], J[i,j], tau[i,j])
            loss_angle = np.arctan(J_biprime/J_prime)*180.0/np.pi
            for n in range(tsize):
                Jinv[n,i,j] = 1.0/Jt[n]
                theta[n,i,j] = loss_angle[n]                       
    return Jinv, theta


    
def map_viscoparams(ruta, lines, points, invols=0.0, k=0.0, R=10.0e-9, nu=0.5, arms = 1, smooth=True, window=10, length=300, snr_min=100, percentroi=30.0, gauss_filt=False):
    """This function extracts the viscoelastic paramters of a 2D force map array
    
    Parameters:
    ----------
    ruta : str
        path containing the .ibw files of the force map
    size : float
        size in meters of the area where the force spectroscopy is taken   
    lines : int
        number of scanned lines in the force map
    points : int
        number of points per line scanned in the force map
    invols : float, optional
        inverse optical lever sensitivity value in m/V, if not passed it will acquire the value stored in the ibw file
    k : float, optional
        cantilever stifness in N/m, if not passed it will acquire the value stored in the ibw file
    R : float, optional
        tip radius of curvature
    nu = float, optional
        material's Poisson ratio, by default it is assumed to be an incompressible material: nu = 0.5  
    arms : int, optional
        number of voigt units used in the generalized voigt model to perform the fitting (recommended to make this number equal to the number of decades in logarithmic scale available in the time window)
    smooth : boolean, optional
        if True rolling average smoothing will be performed according to a user defined window
    window : int
        number of points to be used in the rolling average (i.e.,width of the rectangle window used in the convolution)
    length : int,optional
        approximate length of indentation and deflection arrays in the fitting region regardless the velocity (important to define if downsampling is needed)
    snr_min : int
        minimum signal to noise ratio permissible
    percentroi : float, optional
        percentage of the retract portion used in the NLS fitting
    gauss_filt : boolean, optional
        boolean flag indicating if gaussian filtering is needed to find the snap-in position
    
    Returns:
    ----------
    viscoparams : np.ndarray (2 dimensional)
        2D map of viscoelastic parameters obtained from each .ibw force spectroscopy file
    meanvel : float
        mean approach velocity in the force map
    """
    os.chdir(ruta)
    f = glob('*.ibw') 
    viscoparams = [[0] * points for i in range(lines)]  #initializing the 2D list
    size = int(lines*points)
    print('shape of visco params: %d, %d'%((np.shape(viscoparams)[0]), (np.shape(viscoparams)[1])))
    apvel = []
    tmini = []
    tmaxi = []
    i = 0
    for i in range(size):
        print('runing the %s file'%f[i])
        T,D,Z,params = fd_align(f[i],invols,k, gauss_filt, smooth, window, length)
        t_roi, z_roi, d_roi, ind_roi = fdroi(T, Z, D, gauss_filt, percentroi) #choosing the portion of experiment that will be passed to the minimization process
        Jg_l, tau_l, J_l = nls_fit(t_roi,ind_roi, d_roi*k, R, 0, arms, [],snr_min)
        Jg_nl, tau_nl, J_nl, tmin, tmax = nls_fit(t_roi,ind_roi, d_roi*k, R, 1, arms, [Jg_l, J_l, tau_l], snr_min, False, 1, False, True)
        print('Glassy Modulus: %5.5f MPa'%(1.0/Jg_nl/1.0e6))
        apvel.append(params[0]) #approach velocity
        tmini.append(tmin)
        tmaxi.append(tmax)
        fname = f[i].lower()
        row = re.search('line(.*)point',fname)
        row = row.group(1)
        row = int(row)
        column = re.search('point(.*).ibw', fname)
        column = column.group(1)
        column= int(column)
        print('row: %d, column: %d'%(row, column))
        viscoparams[row][column] = [Jg_nl, tau_nl, J_nl]   #populating the matrix of viscoelastic paramters
        i+=1
    apvel = np.array(apvel)  
    meanvel = np.nanmean(apvel) #average approach velocity of the different force spectroscopy curves
    meantmin = np.nanmean(np.array(tmini))
    meantmax = np.nanmean(np.array(tmaxi))
    return viscoparams, meanvel, meantmin, meantmax

def map_viscoparams2(ruta, lines, points, invols=0.0, k=0.0, R=10.0e-9, nu=0.5, arms = 1, smooth=True, window=10, length=300, snr_min=10, percentroi=0.0, gauss_filt=True, nls_method=0):
    """This function extracts the viscoelastic paramters of a 2D force map array
    
    Parameters:
    ----------
    ruta : str
        path containing the .ibw files of the force map
    size : float
        size in meters of the area where the force spectroscopy is taken   
    lines : int
        number of scanned lines in the force map
    points : int
        number of points per line scanned in the force map
    invols : float, optional
        inverse optical lever sensitivity value in m/V, if not passed it will acquire the value stored in the ibw file
    k : float, optional
        cantilever stifness in N/m, if not passed it will acquire the value stored in the ibw file
    R : float, optional
        tip radius of curvature
    nu = float, optional
        material's Poisson ratio, by default it is assumed to be an incompressible material: nu = 0.5  
    arms : int, optional
        number of voigt units used in the generalized voigt model to perform the fitting (recommended to make this number equal to the number of decades in logarithmic scale available in the time window)
    smooth : boolean, optional
        if True rolling average smoothing will be performed according to a user defined window
    window : int
        number of points to be used in the rolling average (i.e.,width of the rectangle window used in the convolution)
    length : int,optional
        approximate length of indentation and deflection arrays in the fitting region regardless the velocity (important to define if downsampling is needed)
    snr_min : int
        minimum signal to noise ratio permissible
    percentroi : float, optional
        percentage of the retract portion used in the NLS fitting
    gauss_filt : boolean, optional
        boolean flag indicating if gaussian filtering is needed to find the snap-in position
    nls_method : int, optional
        if 0 as default, linear load assumption is assumed, if 1 no assumption in load history
    
    Returns:
    ----------
    viscoparams : np.ndarray (2 dimensional)
        2D map of viscoelastic parameters obtained from each .ibw force spectroscopy file
    meanvel : float
        mean approach velocity in the force map
    """
    os.chdir(ruta)
    files = glob('*.ibw') 
    viscoparams = [[0] * points for i in range(lines)]  #initializing the 2D list
    #size = int(lines*points)
    print('shape of visco params: %d, %d'%((np.shape(viscoparams)[0]), (np.shape(viscoparams)[1])))
    apvel = []
    tmini = []
    tmaxi = []
    for f in files: #i in range(size):
        print('runing the %s file'%f)
        T,D,Z,params = fd_align(f,invols,k, gauss_filt, smooth, window, length)
        if (np.isnan(D).sum()>0) or np.isnan(Z).sum()>0 :
            Jg = np.nan
            tau = np.nan
            J = np.nan
        else:           
            t_roi, z_roi, d_roi, ind_roi = fdroi(T, Z, D, gauss_filt, percentroi) #choosing the portion of experiment that will be passed to the minimization process
            Jg_l, tau_l, J_l, tmin, tmax = nls_fit(t_roi,ind_roi, d_roi*k, R, 0, arms, [],snr_min)
            Jg = Jg_l
            tau = tau_l
            J = J_l
            if nls_method == 1:
                Jg_nl, tau_nl, J_nl, tmin, tmax = nls_fit(t_roi,ind_roi, d_roi*k, R, 1, arms, [Jg_l, J_l, tau_l], snr_min, False, 1, False, True)
                Jg = Jg_nl
                tau = tau_nl
                J = J_nl
        print('Glassy Modulus: %5.5f MPa'%(1.0/Jg/1.0e6))
        apvel.append(params[0]) #approach velocity
        tmini.append(tmin)
        tmaxi.append(tmax)
        fname = f.lower()
        row = re.search('line(.*)point',fname)
        row = row.group(1)
        row = int(row)
        column = re.search('point(.*).ibw', fname)
        column = column.group(1)
        column= int(column)
        print('row: %d, column: %d'%(row, column))
        #viscoparams[(lines-1)-row][column] = [Jg, tau, J]   #populating the matrix of viscoelastic paramters
        viscoparams[row][column] = [Jg, tau, J]   #populating the matrix of viscoelastic paramters
    apvel = np.array(apvel)  
    meanvel = np.nanmean(apvel) #average approach velocity of the different force spectroscopy curves
    meantmin = np.nanmean(np.array(tmini))
    meantmax = np.nanmean(np.array(tmaxi))
    return viscoparams, meanvel, meantmin, meantmax
        

def map_moduli2(ruta, lines, points, invols=0.0, k=0.0, R=10.0e-9, nu=0.5, max_ind=0.0, fig=False):
    """This function draws a 2D map of moduli from a raw .ibw force map
    
    Parameters:
    ----------
    ruta : str
        path containing the .ibw files of the force map
    size : float
        size in meters of the area where the force spectroscopy is taken   
    lines : int
        number of scanned lines in the force map
    points : int
        number of points per line scanned in the force map
    fig : boolean, optional
        if True the map moduli will be plotted
    
    Returns:
    ----------
    moduli : np.ndarray (2 dimensional)
        2D map of moduli from .ibw force spectroscopy file
    meanvel : float
        mean approach velocity in the force map
    """
    os.chdir(ruta)
    files = glob('*.ibw') 
    moduli = np.zeros((lines, points)) #initializing the 2D np.array
    #moduli = [[0] * lines for i in range(points)]  #initializing the 2D list
    apvel = []
    i = 0
    for f in files:
        i+=1
        print('runing the %d point'%i)
        young, vel = modulus(f,invols,k, R, nu, max_ind)
        print('Modulus: %5.5f MPa'%(young/1.0e6))
        apvel.append(vel)
        fname = f.lower()
        row = re.search('line(.*)point',fname)
        row = row.group(1)
        row = int(row)
        column = re.search('point(.*).ibw', fname)
        column = column.group(1)
        column= int(column)
        moduli[row,column] = young/1.0e6   #populating the moduli matrix in MPa
    apvel = np.array(apvel)  
    meanvel = np.mean(apvel) #average approach velocity of the different force spectroscopy curves
    if fig:
        plt.imshow(moduli, vmin = 0.0, vmax= 1.0e3, origin = ';lower')
        plt.colorbar()
    return np.array(moduli), meanvel

def map_moduli(ruta, size, lines, points, invols=0.0, k=0.0, R=10.0e-9, nu=0.5, max_ind=0.0, fig=False):
    """This function draws a 2D map of moduli from a raw .ibw force map
    
    Parameters:
    ----------
    ruta : str
        path containing the .ibw files of the force map
    size : float
        size in meters of the area where the force spectroscopy is taken   
    lines : int
        number of scanned lines in the force map
    points : int
        number of points per line scanned in the force map
    fig : boolean, optional
        if True the map moduli will be plotted
    
    Returns:
    ----------
    moduli : np.ndarray (2 dimensional)
        2D map of moduli from .ibw force spectroscopy file
    meanvel : float
        mean approach velocity in the force map
    """
    os.chdir(ruta)
    files = glob('*.ibw') 
    moduli = [[0] * lines for i in range(points)]  #initializing the 2D list
    apvel = []
    i = 0
    for f in files:
        i+=1
        print('runing the %d point'%i)
        young, vel = modulus(f,invols,k, R, nu, max_ind)
        print('Modulus: %5.5f MPa'%(young/1.0e6))
        apvel.append(vel)
        fname = f.lower()
        row = re.search('line(.*)point',fname)
        row = row.group(1)
        row = int(row)
        column = re.search('point(.*).ibw', fname)
        column = column.group(1)
        column= int(column)
        moduli[row][column] = young/1.0e6   #populating the moduli matrix in MPa
    apvel = np.array(apvel)  
    meanvel = np.mean(apvel) #average approach velocity of the different force spectroscopy curves
    if fig:
        plt.imshow(moduli, vmin = 0.0, vmax= 1.0e3, origin = ';lower')
        plt.colorbar()
    return np.array(moduli), meanvel
    

def modulus(f, invols=0.0, k=0.0, R=10.0e-9,nu=0.5, max_ind=0.0):
    """This function returns the Young's modulus of a force spectroscopy curve from the raw data (.ibw file)
    
    Parameters:
    ----------
    f : str
        string containing the name of the ibw file
    invols : float, optional
        inverse optical lever sensitivity value in m/V, if not passed it will acquire the value stored in the ibw file
    k : float, optional
        cantilever stifness in N/m, if not passed it will acquire the value stored in the ibw file
    R : float, optional
        tip radius of curvature
    nu : float, optional
        samples's Poison ratio
    max_ind : float, optional
        value of maximum indentation to be used in the calculation of modulus, it is advised to be lower than tip radius
    
     Returns:
    ----------
    E : float
        effective Young's modulus
    params[0] : float
        approach velocity during the force spectroscopy experiment
    """    
    alfa = 4.0/3*np.sqrt(R)/(1.0-nu**2)
    t, defl, zs, params = fd_align(f, invols, k)
                       
    #attaching values to lists containing stiffnesses and invols values
    if k == 0.0:
        k = params[3]
    if invols == 0.0:
        invols = params[2]
            
    #GETTING REPULSIVE PORTION TO CALCULATE STIFFNESS
    t_r, zs_r, defl_r, tip_r = fdroi(t, zs, defl, False, 0.0)
            
    if max_ind != 0.0: #Criterion was given of maximum indentation allowed
        tip_r = tip_r[tip_r<max_ind]
        defl_r = defl_r[:len(tip_r)]
        zs_r = zs_r[:len(tip_r)]
       
    #Making sure only positive values of indentation are considered:
    tip_rep = tip_r[tip_r>0]
    defl_rep = defl_r[tip_r>0]       
    
    #linear regression to get modulus (if info arrays are not empty)
    if len(tip_rep) > 0:
        E, _,_,_,_ = stats.linregress(tip_rep**1.5, defl_rep*k/alfa)
        #E = linear_fit_Nob(tip_rep**1.5, defl_rep*k/alfa)
        
    else:
        E = -1.0e6  #this negative value of -1MPa is a flag inndicating that the force curve is unphysical
    if np.isnan(E):
        E = -1.0e6  #this negative value of -1MPa is a flag inndicating that the force curve is unphysical
    return E, params[0]



def map2d_multiphys(path_origin, lines, points, path_save='None', invols=0.0, k=0.0, R=10.0e-9, nu=0.5, max_ind=0.0, smooth=False, window=10, gauss_filt = False):
    """This function returns 8 2D maps with physical relevant data retrieved from fd spectroscopy experiments
    
    Parameters:
    ----------
    path_origin : str
        path containing the .ibw files of the force map  
    lines : int
        number of scanned lines in the force map
    points : int
        number of points per line scanned in the force map
    path_save : str
        path where the 2D maps will be saved 
    
    Returns:
    ----------
    zs[defl.argmax()] : np.2d array
        2D map of height at trigger setpoint (contact mode topography)
    E : np.2d array
        2D map of apparent Young's modulus extracted from the FD curve assuming DMT contact mechanics
    adh_app: np.2d array
        2D map of maximum adhesion force in the approach portion (calculated at the jump to contact point)
    adh_ret: np.2d array
        2D map of maximum adhesion force in the retraction (calculated in the snap out point)
    z0: np.2d array
        2D map of z sensor position at the jump to contact point
    z1: np.2d array
        2D map of z sensor position at the zero force point in the approach portion beyond the jumpt to contact point
    max(indentation): np.2d array
        2D map of maximum indentation either in approach or retract during the force spectroscopy
    indentation[defl.argmax()]: np.2d array   
        2D map of indentation at the trigger point (point of maximum deflection)  
    """
    os.chdir(path_origin)
    files = glob('*.ibw') 
    z1_a, z0_a, topo_a, adh1_a, adh2_a, maxind_a, indtrig_a, E_a = np.zeros((lines, points)), np.zeros((lines, points)), np.zeros((lines, points)), np.zeros((lines, points)), np.zeros((lines, points)), np.zeros((lines, points)), np.zeros((lines, points)), np.zeros((lines, points)) #initializing the 2D np.array    
    i = 0
    apvel_a = []
    for f in files:
        i+=1
        print('evaluating point: %d'%i)
        topo, E, adh1, adh2, z0, z1, maxind, indtrig, apvel = multiparams_fd(f,k, invols, max_ind, R, nu, smooth, window, gauss_filt)
        apvel_a.append(apvel)
        fname = f.lower()
        row = re.search('line(.*)point',fname)
        row = row.group(1)
        row = int(row)
        column = re.search('point(.*).ibw', fname)
        column = column.group(1)
        column= int(column)
        z1_a[row,column] = z1*1.0e9   #populating the zero force height matrix in nm
        z0_a[row,column] = z0*1.0e9   #populating the jump-to-contact height matrix in nm
        topo_a[row,column] = topo*1.0e9   #contact mode topography in nN
        adh1_a[row,column] = adh1*1.0e9
        adh2_a[row,column] = adh2*1.0e9
        maxind_a[row,column] = maxind*1.0e9
        indtrig_a[row,column] = indtrig*1.0e9    
        E_a[row,column] = E/1.0e6   #Young's modulus in MPa        
    if path_save != 'None':
        os.chdir(path_save)
        np.savetxt('topo.txt', topo_a)
        np.savetxt('E.txt', E_a)
        np.savetxt('adh1.txt', adh1_a)
        np.savetxt('adh2.txt', adh2_a)
        np.savetxt('maxind.txt', maxind_a)
        np.savetxt('indtrig.txt', indtrig_a)
        np.savetxt('z1.txt', z1_a)
        np.savetxt('z0.txt', z0_a)    
    meanvel = np.array(apvel_a)
    meanvel = np.mean(meanvel)
    return topo_a, E_a, adh1_a, adh2_a, z0_a, z1_a, maxind_a, indtrig_a, meanvel
    #return topo_a[::-1,:], E_a[::-1,:], adh1_a[::-1,:], adh2_a[::-1,:], z0_a[::-1,:], z1_a[::-1,:], maxind_a[::-1,:], indtrig_a[::-1,:], meanvel


def igor2D(path):
    """this function gets the topography and adhesion map from a .txt file created by manual extraction of the FD data by exporting the waves to a text file
         
    Parameters:
    ----------
    path: string
        path where the .txt file has been saved
    
    Returns:
    ---------
    topoigor : np.2darray
        2D array containing the topography
    adhigor : np.2d array
        2D array containing the adhesion map
    """    
    os.chdir(path)  #changing directory to extract the igor topography
    igorf = glob('*.txt')
    igor = np.loadtxt(igorf[0])        
    topoigor = igor[:32,:]*1.0e9
    adhigor = igor[32:,:]*1.0e9   
    return topoigor[::-1,:], adhigor[::-1,:]

def multiparams_fd(f, k=0.0, invols=0.0, maxind = 0.0, R = 10.0e-9, nu=0.5, smooth=False, window=10, gauss_filt = False):
    """Multiparametric calculation of physical relevant data that can be extracted from force-distance curves
    
    this function is used by map2d_multiphys
    
    the file passed is an .ibw one that contains the force spectroscopy data for one curve
    
    Parameters:
    ----------
    f : str
        string containing the name of the ibw file
    k : float, optional
        spring constant, if not passed the value will be assigned to the stored value in the igor file
    invols : float, optional
        inverse optical lever sensitivity, if not passed it will be read from the ibw file channels    
    R : float, optional
        tip radius of curvature
    nu = float, optional
        material's Poisson ratio, by default it is assumed to be an incompressible material: nu = 0.5 
    smooth : boolean, optional
        if True a moving average smoothing will be performed
    window : int, optional
        width of the moving average smoothing to be performed on the data if smooth parameter is passed as True
    gauss_filt : boolean, optional
        boolean flag indicating if gaussian filtering is needed to find the snap-in position
        
    Returns:
    ----------
    zs[defl.argmax()] : float
        height at trigger setpoint (contact mode topography)
    E : float
        apparent Young's modulus extracted from the FD curve assuming DMT contact mechanics
    adh_app: float
        maximum adhesion force in the approach portion (calculated at the jump to contact point)
    adh_ret: float
        maximum adhesion force in the retraction (calculated in the snap out point)
    z0: float
        z sensor position at the jump to contact point
    z1: float
        z sensor position at the zero force point in the approach portion beyond the jumpt to contact point
    max(indentation): float
        maximum indentation either in approach or retract during the force spectroscopy
    indentation[defl.argmax()]: float    
        indentation at the trigger point (point of maximum deflection)
    approach_vel: float
        z-sensor approach velocity
    """
    x,y,z,ch,note = loadibw(f) 
    note =  {k.lower(): v for k, v in note.items()}  #making the dictionary to have all keys in lowercase
    ch = [u.lower() for u in ch]  #making channel to have all elements in lowercase
    if k == 0.0:
        k = note['springconstant']  
    inv = note['invols']
    if invols ==0.0:   #no correction needed
        correction = 1.0
    else: #correction needed due to late calibration
        correction = invols/inv
    defl = z[ch.index('defl')]
    defl *= correction
    zs = z[ch.index('zsnsr')]
    fs = note['numptspersec']    #sampling frequency
    t = np.arange(len(defl))/fs   #retrieving time array with the aid of sampling frequency  
    
    if smooth: #smoothing by means of moving average
        defl = moving_av(defl,window)
        zs = moving_av(zs, window) 
        
    if (zs[2] - zs[1]) < 0.0: #this is problematic
        flag = 0
        i=1
        while flag == 0:
            if zs[i] > zs[i-1]:
                flag =1
                r = i+1
            i += 1
        zs = zs[r:]
        defl = defl[r:]
        t = t[r:]           
               
    #point of zero force (z1,d1)
    max_z = zs.argmax()
    approach_vel,_,_,_,_ = linregress(t[0:max_z], zs[0:max_z]) #getting approach velocity
    defl_ap = defl[:max_z] #getting only approach portion which is useful to get the offsets
    zs_ap = zs[:max_z]   #approach portion of the z_sensor, useful to calculate z1 afterwards
    _, d1 = offsety(defl_ap)  
        
    #finding array value of minimum deflection (d0)
    if not gauss_filt: #no gaussian filtering
        min_defl = defl_ap.argmin()
    else: 
        min_defl = offsetx(defl_ap)  
    
    #finding z1
    dap_contact = defl_ap[min_defl:]
    zap_contact = zs_ap[min_defl:]
    zh_pos = np.argmin(np.abs(dap_contact-d1))
    z1 = zap_contact[zh_pos]
        
    #point of zero indentation (contact point), minimum deflection (z0, d0)
    d0 = defl[min_defl]
    z0 = zs[min_defl]
    
    #calculating force and indentation
    force = k*(defl-d1)
    indentation = (zs-z0) - (defl-d0)             
    adh_app = k*np.abs(d1-d0)   #maximum adhesion during approach
    
    #maximum adhesion during retract
    defl_ret = defl[zs.argmax():]
    adh_ret = k*np.abs(d1-min(defl_ret))
    
    #calculating the repulsive force for calculating modulus
    f_ap = force[:zs.argmax()]
    ind_ap = indentation[:zs.argmax()]
    f_rep = f_ap[min_defl:]   - force[min_defl] #DMT assumption means that adhesive force is constant during the whole contact portion
    ind_rep = ind_ap[min_defl:]
    
    #linear regression to get modulus (if info arrays are not empty)
    alfa = 4.0/3*np.sqrt(R)/(1.0-nu**2)
    ind_repulsive = ind_rep[ind_rep>0]
    f_rep = f_rep[ind_rep>0]
    if maxind != 0.0: #Criterion was given of maximum indentation allowed
            ind_repulsive = ind_repulsive[ind_repulsive<maxind]
            f_rep = f_rep[:len(ind_repulsive)]
    if len(ind_repulsive) > 0:
        E =linear_fit_Nob(ind_repulsive**1.5, f_rep/alfa)
    else:
        E = np.nan  #flag to indicate unphysical value of Young's modulus
    
    return  -zs[defl.argmax()], E, adh_app, adh_ret, -z0, -z1, max(indentation), indentation[defl.argmax()], approach_vel



def DMT_fit_plot(f, k=0.0, invols=0.0, R = 10.0e-9, nu=0.5, smooth=False, window=10, gauss_filt = False):
    """visual representation of the DMT fit of a force curve sent via .ibw file
    
    Parameters:
    ----------
    f : str
        string containing the name of the ibw file
    k : float, optional
        spring constant, if not passed the value will be assigned to the stored value in the igor file
    invols : float, optional
        inverse optical lever sensitivity, if not passed it will be read from the ibw file channels    
    R : float, optional
        tip radius of curvature
    nu = float, optional
        material's Poisson ratio, by default it is assumed to be an incompressible material: nu = 0.5 
    smooth : boolean, optional
        if True a moving average smoothing will be performed
    window : int, optional
        width of the moving average smoothing to be performed on the data if smooth parameter is passed as True
    gauss_filt : boolean, optional
        boolean flag indicating if gaussian filtering is needed to find the snap-in position
    """
    x,y,z,ch,note = loadibw(f) 
    note =  {k.lower(): v for k, v in note.items()}  #making the dictionary to have all keys in lowercase
    ch = [u.lower() for u in ch]  #making channel to have all elements in lowercase
    if k == 0.0:
        k = note['springconstant']  
    inv = note['invols']
    if invols ==0.0:   #no correction needed
        correction = 1.0
    else: #correction needed due to late calibration
        correction = invols/inv
    defl = z[ch.index('defl')]
    defl *= correction
    zs = z[ch.index('zsnsr')]
    fs = note['numptspersec']    #sampling frequency
    t = np.arange(len(defl))/fs   #retrieving time array with the aid of sampling frequency   
    if smooth: #smoothing by means of moving average
        defl = moving_av(defl,window)
        zs = moving_av(zs, window) 
        
    if (zs[2] - zs[1]) < 0.0: #this is problematic
        flag = 0
        i=1
        while flag == 0:
            if zs[i] > zs[i-1]:
                flag =1
                r = i+1
            i += 1
        zs = zs[r:]
        defl = defl[r:]
        t = t[r:]           
            
    #point of zero force (z1,d1)
    defl_ap = defl[:zs.argmax()] #getting only approach portion which is useful to get the offsets
    _, d1 = offsety(defl_ap)  
        
    #finding array value of minimum deflection (d0)
    if not gauss_filt: #no gaussian filtering
        min_defl = defl_ap.argmin()
    else: 
        min_defl = offsetx(defl_ap)  
        
    #point of zero indentation (contact point), minimum deflection (z0, d0)
    d0 = defl[min_defl]
    z0 = zs[min_defl]
    
    #calculating force and indentation
    force = k*(defl-d1)
    indentation = (zs-z0) - (defl-d0)             
    
    #calculating the repulsive force for calculating modulus
    f_ap = force[:zs.argmax()]
    ind_ap = indentation[:zs.argmax()]
    f_rep = f_ap[min_defl:]   - force[min_defl] #DMT assumption means that adhesive force is constant during the whole contact portion
    ind_rep = ind_ap[min_defl:]
    
    #linear regression to get modulus (if info arrays are not empty)
    alfa = 4.0/3*np.sqrt(R)/(1.0-nu**2)
    ind_repulsive = ind_rep[ind_rep>0]
    f_rep = f_rep[ind_rep>0]
    if len(ind_rep) > 0:
        E = linear_fit_Nob(ind_repulsive**1.5, f_rep/alfa)
                
    plt.plot(indentation*1.0e9, force*1.0e9, 'y', lw=5, label='total')
    plt.plot(ind_repulsive*1.0e9, f_rep*1.0e9 + force[min_defl]*1.0e9, 'b', lw=3, label='repulsive')
    plt.plot(ind_repulsive*1.0e9, (ind_repulsive**1.5)*E*alfa*1.0e9 + force[min_defl]*1.0e9, 'r--', lw='2', label='DMT fit')
    plt.legend(loc='best')
    plt.xlim(max(ind_rep)*1.0e9*-3.0, max(ind_rep)*1.0e9*1.5)
    plt.xlabel('indentation, nm')
    plt.ylabel('force, nN')


    
    
    