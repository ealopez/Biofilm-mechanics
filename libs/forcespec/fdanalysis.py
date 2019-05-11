# -*- coding: utf-8 -*-
"""

@author: Enrique Alejandro
Description: This library is built to be used for averaging FD curves
obtained in static force spectroscopy.
The files to be processed are individual .ibw files containing individual FD
curves. The original force map has to be extracted into single files through 
the Asylum software.

Updated on Tuesday May 22nd 2018
This library contains functions to process raw AFM spectrosocpy data.

Notes: this is the most updated version of the library that process and averages
FD curves from Asylum research software as of May 22nd 2018
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import splrep, splev
from scipy.stats import linregress
import os
import pandas as pd
from scipy import stats
from glob import glob
from PyAstronomy import pyasl
from lmfit import minimize, Parameters


def linear_fit(x, y):
    """This function receives a function and performs linear fit"""
    """Input: array of dependent variable, x. And array of dependent variable, y."""
    """Output: slope, intersect, and coefficient of determination (r^2 value) """
    """An alternative is to use: slope, intercept, r_value, p_value, std_err = stats.linregress(t, defl)"""
    m,b = np.polyfit(x, y, 1)
    mean = sum(y)/np.size(y)
    SS_tot = sum((y-mean)**2)
    SS_res = sum(   (y - (m*x+b))**2     )
    r_2 = 1.0 - SS_res/SS_tot
    return m, b, r_2

def linear_force(params, t, force):
    "this function returns the residual to perform linear fit"
    p = params.valuesdict()
    A = p['A']
    model = (A)*t
    return  (model - force) #calculating the residual

def linear_fit_Nob(x,y):
    """This function performs linear fit in the special case where intercept is zero"""
    m, b, _ = linear_fit(x,y)   #initial guess
    params = Parameters()
    params.add('A', value = m, min=0)
    result = minimize(linear_force, params, args=(x,y), method='leastsq')
    Fdot = result.params['A'].value  
    return Fdot


def fdcurve_ibw(f,invols=0.0, k=0.0, gauss_filt=False, smooth=True, window=10, length = 0):
    """This function receives an igor file containing a force distance curve and returns the curve 
    
    It applies an offset on z-sensor so it becomes zero when the tip first contacts the sample
    It also applies an offset to deflection so it becomes zero in the approach portion when the tip is far from the sample.
    
    Parameters:
    ----------
    f : str
        string containing the name of the ibw file
    invols : float, optional
        inverse optical lever sensitivity, if not passed it will be read from the ibw file channels
    k : float, optional
        spring constant, if not passed the value will be assigned to the stored value in the igor file
    gauss_filt : boolean, optional
        boolean flag indicating if gaussian filtering is needed to find the snap-in position
    
    Returns:
    ----------
    t : numpy.ndarray
        time trace of the experiment centered at time of contact
    force : numpy.ndarray
        array with force experimental force trace according to convention force = k*(defl -d1)
    indentation : numpy.ndarray
        indentation trace according to convention that indentation = (z-z0) - (d-d0)
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
    t -= t[min_defl]   #making time zero when tip first snaps into the sample      
    
    #Getting experimental Parameters
    maxi = zs.argmax()
    approach_vel,_,_,_,_ = linregress(t[0:maxi], zs[0:maxi])
    retract_vel,_,_,_,_ = linregress(t[maxi:maxi+int(len(t)*0.1)], zs[maxi:maxi+int(len(t)*0.1)])
    
    return t, force, indentation, z0, d0, d1, z1, zs, defl, [approach_vel, retract_vel, inv, k, fs]


def t_thvisco(t_th, dt=0.0, dp = 5):
    """finding the largest continuous portion in time where the integration kernel remains bounded within certain error
    
    first you have to run the nls_fit function and then the error_conv function, 
    then this function will return the initial and end time in the largest portion 
    of good behavior of the kernel
    
    Parameters
    -----------
    t_th : np.array
        time trace containing possibly discontinuous portions where the integration kernel remains valid (the last output returned from error_conv function)
    dt : float, optional
        timestep in the original time array
    dp : int, optional
        discontinuity permission, number of timesteps where discontinuity is allowed
    
    Returns
    -----------
    tmini : np.array,
        starting time of the largest time portion where the kernel remains bounded
    tmaxi : np.array,
        end time of the largest time portion where the kernel remains bounded
    max_len : length of the largest time portion that keeps the calculated kernel bounded    
    """
    if dt == 0.0: #time step not passed so it has to be estimated
        dt = t_th[1] - t_th[0]
    max_len = 0
    local_bunch = []
    for i in range(1,len(t_th)):    
        if ( (t_th[i] - t_th[i-1]) < (dt*dp) ):
            local_bunch.append(t_th[i])
            print(len(local_bunch))
        else:            
            print('reinitializing local bunch')
            local_len = len(local_bunch)
            if local_len > max_len:
                max_len = local_len
                tmini = local_bunch[0]
                tmaxi = local_bunch[-1]
            local_bunch = []  #reinitialize local_bunch
        if i == (len(t_th) -1):
            local_len = len(local_bunch)
            if local_len > max_len:
                max_len = local_len
                tmini = local_bunch[0]
                tmaxi = local_bunch[-1]                
    return tmini, tmaxi, max_len

def crop_ftw(tw, ftw, tmini, tmaxi, omega = False):
    """this function returns a function and its dependent variable within a region defined (where the integration kernel remains bounded)
    
    Parameters
    -----------
    tw : np.array
        independent variable, either time or frequency
    ftw : np.array
        function to be cropped (within a smaller region where integration kernel is bounded)
    tmini : float
        starting time of the largest portion in time where the integration kernel is well behaved
    tmaxi: float
        final time of the largest portion in time where the integration kernel is well behaved
    omega : boolean, optional
        if True it indicates that the function passed depends on frequency (instead of time)
    
    Returns
    -----------
    x_crop : np.array
        trace of the dependent variable in the largest region where viscoelastic kernel is well behaved
    fx_crop : np.array
        trace of the function in the largest region where viscoelastic kernel is well behaved    
    """    
    if omega: #the independent variable is frequency in radians
        omega_maxi = 1.0/tmini*2.0*np.pi
        omega_mini = 1.0/tmaxi*2.0*np.pi
        x_crop = tw[ (tw > omega_mini) & (tw< omega_maxi) ]
        fx_crop = ftw[(tw > omega_mini) & (tw< omega_maxi) ]      
    else:
        x_crop = tw[ (tw>tmini) & (tw<tmaxi)]
        fx_crop = ftw[(tw>tmini) & (tw<tmaxi)]
    return x_crop, fx_crop


def noise(y):
    """Noise estimation assuming the noise is gaussian stoichastic
    
    cite as: Czesla, S., T. Molle, and J. H. M. M. Schmitt. "A posteriori noise estimation in variable data sets." arXiv preprint arXiv:1712.02226 (2017).
    
    Parameters
    ----------
    y : np.array
        signal whose noise level is to be estimated
    
    Returns
    ---------
    noise : float
        estimated noise
    noise_std : float
        standard deviation of the noise (uncertainty)   
    
    Documentation available in: https://www.hs.uni-hamburg.de/DE/Ins/Per/Czesla/PyA/PyA/pyaslDoc/aslDoc/estimateSNR.html
    """    
    beq = pyasl.BSEqSamp() # Estimate noise using robust estimate
    N = 1 # Define order of approximation (use larger values such as 2,3, or 4 for # faster varying or less well sampled data sets; also 0 is a valid order)
    j = 1 # Define 'jump parameter' (use larger values such as 2,3, or 4 if correlation between adjacent data point is suspected)
    noise, noise_std = beq.betaSigma(y, N, j, returnMAD=True) # Estimate noise assuming equidistant sampling (often a good approximation even if data are not strictly equidistant) and robust estimation (often advantageous in working with real data)
    return noise, noise_std


def inicon(time, indentation, force, R=10.0e-9, arms = 0, nn=1):
    """this functions calculates suitable initial conditions to be passed to nls optimization
    
    make sure only the repulsive approach portion of the curve is passed
    
    Parameters
    ----------
    time : np.array
        time trace of the repulsive portion of the approach curve
    indentation : np.array
        indentation trace
    force : np.array
        tip-sample force trace, deflection x k(stiffness)
    R : float, optional
        tip radius
    arms : int, optional
        number of terms in the Prony series of the viscoelastic material
    nn : int, optional
        number of retardation times per decade of logarithmic scale
    
    Returns
    ----------
    Jg_i : float
        glassy compliance
    J_i : np.array
        array with values of compliances in the generalized Voigt model
    tau_i : np.array
        array with values of retardation times 
    N : int
        number of arms in the generalized voigt model, either passed or automatically calculated if not passed
    """
    alfa = 16.0/3*np.sqrt(R)    
    tip_rep = indentation[indentation>0] #Making sure only positive values of indentation are considered:
    force_rep = force[indentation>0]  #Making sure only positive values of indentation are considered:   
    time_rep = time[indentation>0]
    #linear regression to get modulus (if info arrays are not empty)
    if len(tip_rep) > 0:
        G, intercept, r_value, p_value, std_err = stats.linregress(tip_rep**1.5, force_rep/alfa)
    else:
        G = -1.0e6  #this negative value of -1MPa is a flag inndicating that the force curve is unphysical
        print('warning, unphysical FD curve')
    if np.isnan(G):
        G = -1.0e6  #this negative value of -1MPa is a flag indicating that the force curve is unphysical
        print('warning, unphysical FD curve')
    if arms == 1:  #only one arm, the SLS model
        Jg_i = 1.0/G/10.0
        J_i = 1.0/G
        tau_i = time_rep[-1]/10.0
        J_i = np.array([J_i])
        tau_i = np.array([tau_i])
        N = arms
    elif arms == 0: #number of arms is not passed
        N = int(  np.round( np.log10( time_rep[-1] ) ) - np.round( np.log10( time_rep[0] )    )    )  #number of arms
        N *= nn
        y = np.linspace(np.log10(time_rep[0]), np.log10(time_rep[-1] ), N)
        tau_i = np.abs(10**y)  #retardation times
        dJ = 1.0/N
        J_i = np.linspace(dJ,dJ+dJ*(N-1),N)*(1.0/G)
        Jg_i = J_i[0]/10.0
    else:
        y = np.linspace(np.log10(time_rep[0]), np.log10(time_rep[-1] ), arms)
        tau_i = np.abs(10**y)  #retardation times
        dJ = 1.0/arms
        J_i = np.linspace(dJ,dJ+dJ*(arms-1),arms)*(1.0/G)
        Jg_i = J_i[0]/10.0   
        N = arms
        
    return Jg_i, J_i, tau_i, N

def snr(t, ft, snr_min=5, sf=10):
    """This function gets the signal to noise ratio and determines the minimum point in time at wich the signal overtakes certain noise treshold
    
    Parameters
    ----------
    t : np.array
        time trace related to the noisy signal
    ft : np.array
        trace of the noisy signal
    snr_min : int, optional
        minimum signal to noise ration accepted
    sf : int, optional
        smooth factor, how many point to be taken in the moving average to calculate strength of signal
        
    Returns
    ----------
    t_min : float
        time at which the signal overcame certain criterion on signal to noise   
    """
    noise_ft, _ = noise(ft)
    sm_ft = moving_av(ft,sf)
    snr_t = sm_ft/noise_ft
    t_min = t[0]
    flag = 0  #flag that points out if t_min has been already found
    for i in range(len(snr_t)):
        if (snr_t[i] > snr_min) and (flag == 0):   #finding t_min
            t_min = t[i]
            flag = 1
        if (flag == 1 ) and (snr_t[i] < snr_min): #problem, the snr became lower after surpassing the t_min
            flag = 0
    return t_min       
        
    
def snr_roi(t, f_t, t_min):
    """This function returns the portion of a function that is considered to be above the noise level
    
    it is needed to pass t_min which is found by snr function defined above, the input is normally passed in logarithmic scale   
    
    Parameters
    ----------
    t : np.array
        time trace related to quantity in a wide time window to be reduced to roi
    f_t : np.array
        trace of the quantity in a wide time window to be reduced to roi
    t_min : float
        time at which the signal overcame certain criterion on signal to noise (found by snr function) 
        
    Returns
    ----------
    t[mini:] : np.array
        time in the window of interest above required snr
    f_t[mini:] : np.array
        calculated quantity in the window of interest above required snr   
    """
    if t[1] > t_min:
        print('error, t_min should be higher than the current time resolution')
    mini = np.argmin(np.abs(t-t_min))
    return t[mini:], f_t[mini:]

def snr_fd(t, d, ind, snr_min):
    """function to get minimum time at which the both deflection and indentation surpass certain signal to noise ratio
    
    Parameters:
    ---------- 
    t : np.array
        time trace
    d : np.array
        deflection trace
    ind : np.array
        indentation trace
    snr_min : int
        minimum signal to noise ratio permissible
    
    Returns:
    ---------- 
    tmin : float
        minimum point in time when both deflection and indentation surpass certain signal to noise ratio    
    """
    tmin1 = snr(t, d, snr_min)
    tmin2 = snr(t, ind, snr_min)
    if tmin2 < tmin1:
        tmin = tmin2
    else:
        tmin = tmin1
    return tmin

def downsample(x, y,factor):
    """downsampling algorithm based in scipy.signal.decimation routine
    
    Parameters
    ----------
    x : np.array
        dependent variable, often time
    y : np.array
        signal to be downsampled
    factor: int
        downsampling factor
        
    Returns
    ----------
    x[0::factor] : np.array
        dependent variable (often time array) corresponding to the downsampled version of y
    y_ds : np.array
        downsampled version of y    
    """
    y_ds = signal.decimate(y,factor)
    return x[0::factor], y_ds

def moving_av(y,window):
    """function that performs a moving average by means of the convolution integral with a rectangular function
    
    Parameters
    ----------
    y : np.array
        funtion to be smoothened
    window : int
        number of points to be used in the rolling average (i.e.,width of the rectangle window used in the convolution)
    
    Returns
    ---------
    y_avg : smoothened function by means of the rolling averaging (moving mean)
    """
    avg_mask = np.concatenate( (np.ones(window)/window, np.zeros(len(y)-window) ) )
    y_avg = np.convolve(y, avg_mask, mode='full')
    y_avg = y_avg[range(len(avg_mask))]
    return y_avg

def moving_std(y,window):
    """function to perform the moving standard deviation
    
    Parameters
    ----------
    y : np.array
        funtion to be smoothened
    window : int
        number of points to be used in the rolling average (i.e.,width of the rectangle window used in the convolution)
    
    Returns
    ---------
    mov_std : np.array
        the rolling standard deviation    
    """
    y_pd = pd.Series(y)
    mov_std = y_pd.rolling(window).std() #Now calculating the moving standard deviation
    return mov_std

def xiny(x, y, ret=2, op=all):
    """
    Description: this is an auxiliary function to the loadibw function
    ret=0: return True, False array
    ret=1: return True index array
    ret=2: return True data array
    op=all: and operation
    op=any: or operation
    """
    if type(x)==type(()) or type(x)==type([]) or type(x)==type(np.array(0)):
        if len(x)==0:
            a = np.array( [True]*len(y) )
        else:
            a = np.array( [op([j in i for j in x]) for i in y] )
    else:
        a = np.array([str(x) in i for i in map(str,y)])
    if ret==0: return a
    if ret==1: return np.arange(a.size)[a]
    if ret==2: return np.array(y)[a]

def findstr(src, soi):
    """
    Description: this is an auxiliary function to the loadibw function
    src: string source
    soi: string of interest
    """
    if len(soi)==0:
        return src
    else:
        if type(soi[0])==str:
            return map(str, xiny(soi, src))
        else:
            a = []
            for i in soi:
                a += map(str, xiny(i, src))
            return a
        

def loadibw(f,cal=0):
    """
    x,y,z,ch,note = loadibw(f)    
    """
    datatype = {
        0:np.dtype('a1'),
        2:np.float32,
        4:np.float64,
        8:np.int8,
        16:np.int16,
        32:np.int32,
        72:np.uint8,
        80:np.uint16,
        96:np.uint32
    }
    # igor header
    fp = open(f,'rb');
    fp.seek(  0); ih  = []
    fp.seek(  0); ih += [('version', np.fromfile(fp,np.int16,1)[0])]
    fp.seek(  8); ih += [('fmsize' , np.fromfile(fp,np.int32,1)[0])]
    fp.seek( 12); ih += [('ntsize' , np.fromfile(fp,np.int32,1)[0])]
    fp.seek( 68); ih += [('cdate'  , np.fromfile(fp,np.uint32,1)[0])]
    fp.seek( 72); ih += [('mdate'  , np.fromfile(fp,np.uint32,1)[0])]
    fp.seek( 76); ih += [('dsize'  , np.fromfile(fp,np.int32,1)[0])]
    fp.seek( 80); ih += [('dtype'  , np.fromfile(fp,np.uint16,1)[0])]
    fp.seek(132); ih += [('shape'  , np.fromfile(fp,np.int32,4))]
    fp.seek(132); ih += [('ndim'   , (ih[-1][1]>0).sum())]
    fp.seek(148); ih += [('sfa'    , np.fromfile(fp,np.float64,4))]
    fp.seek(180); ih += [('sfb'    , np.fromfile(fp,np.float64,4))]
    fp.seek(212); ih += [('dunit'  , np.fromfile(fp,np.dtype('a4'),1)[0])]
    fp.seek(216); ih += [('dimunit', np.fromfile(fp,np.dtype('a4'),4))]
    ih  = dict(ih)
    ih['shape'] = ih['shape'][:ih['ndim']][::-1]
    # images data
    fp.seek(384); z = np.fromfile(fp,datatype[ih['dtype']],ih['dsize']).reshape(ih['shape'])
    fp.seek(ih['fmsize'],1)
    ah = np.fromfile(fp,np.dtype('a%d'%ih['ntsize']),1)[0].split('\r')
    # asylum note
    note = []
    for i in ah:
        if i.find(':')>0:
            j = i.split(':',1)
            try:
                note += [(j[0],float(j[1]))]
            except:
                note += [(j[0],j[1].replace(' ','',1))]
    note = dict(note)
    if type(note['ScanRate']) == type(''):
        note['ScanRate'] = float(note['ScanRate'].split('@')[0])
    # channel & type
    fp.seek(-10,2)
    ch = fp.read()
    fp.seek(-int(ch[:4]),2)
    if ch[-5:] == 'MFP3D':
        ch = findstr(fp.readline().split(';'),'List')[0].split(':')[1].split(',')
        x = np.linspace(-note['FastScanSize']/2.,note['FastScanSize']/2.,note['ScanPoints']) + note['XOffset']
        y = np.linspace(-note['SlowScanSize']/2.,note['SlowScanSize']/2.,note['ScanLines']) + note['YOffset']
    elif ch[-5:] == 'Force':
        ch = findstr(fp.readline().split(';'),'Types')[0].split(':')[1].split(',')[:-1]
        ch.insert(0,ch.pop())
        x = y = []
    else:
        ch = []
        x = y = []
    if ch[-1] == '':
        ch = ch[:-1]
    fp.close()
    
    if cal and (x!=[]):
        x -= x[0]
        y -= y[0]
        x *= 1e6
        y *= 1e6
        for i,j in enumerate(ch):
            if ('Height' in j)or('ZSensor' in j)or('Amplitude' in j)or('Current' in j):
                z[i] *= 1e9
    
    return x,y,z,ch,note

 
def offsety(d):
    """This function receives the approach deflection in meters and gets rid of relative static deflection far from sample
    
    The function makes sure that deflection is zero when the tip is far from the sample (i.e., gets rid of offset deflection)
    
    Parameters:
    ----------
    d : numpy.ndarray
        array containing the approach deflection of the force spectroscopy curve
    
    Returns:
    ----------
    defi : numpy.ndarray
        array with the corrected static deflection being zero when tip is far from the sample
    offset_y : float
        average static deflection to be offseted
    """
    mini = d.argmin()
    if mini > 0:
        defl_at = d[0:int(mini*0.5)]
    else:
        defl_at = d[0]
    offset_y = np.mean(defl_at)
    defi = d - offset_y
    return defi, offset_y
    
def offsetx(defl):
    """This function receives the approach deflection in meters and shifts
    
    The function finds array element where deflection is zero, this is important to offset z-sensor so it becomes zero when first snaps into the sample
    
    Parameters:
    ----------
    defl : numpy.ndarray
        array containing the approach deflection of the force spectroscopy curve
    
    Returns:
    ----------
    zero : int
        array position corresponding to snap-in event
    """
    b,a = signal.butter(10, 0.1, "low")
    df_smooth = signal.filtfilt(b,a,defl)
    zero = df_smooth.argmin()   #index where the smoothed deflection has its minimum point
    return zero


def fd_align(f,invols=0.0, k=0.0, gauss_filt=False, smooth=True, window=10, length = 0):
    """This function receives an igor file containing a force distance curve and returns the curve with appropriate offsets
    
    It applies an offset on z-sensor so it becomes zero when the tip first contacts the sample
    It also applies an offset to deflection so it becomes zero in the approach portion when the tip is far from the sample.
    
    Parameters:
    ----------
    f : str
        string containing the name of the ibw file
    invols : float, optional
        inverse optical lever sensitivity, if not passed it will be read from the ibw file channels
    k : float, optional
        spring constant, if not passed the value will be assigned to the stored value in the igor file
    gauss_filt : boolean, optional
        boolean flag indicating if gaussian filtering is needed to find the snap-in position
    
    Returns:
    ----------
    t : numpy.ndarray
        time trace of the experiment centered at time of contact
    defl : numpy.ndarray
        array with the values of deflection properly offseted to have zero deflection when tip is far from the sample
    zs : numpy.ndarray
        array containing the z-sensor position properly offseted to have zs=0 at the time of contact   
    [approach_vel*1.0e9, retract_vel*1.0e9, inv_ols, stiffness]] : list
        list containing two floats with the approach and retract velocity in m/s (derivative of z-sensor with respect to time), invols in m/V and stiffness in N/m    
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
        
    
    #getting only approach portion which is useful to get the offsets
    max_z = zs.argmax()
    defl_ap = defl[:max_z]
        
    #offseting FD curves in y axis, i.e., correcting deflection to make it zero when far from sample
    _, offset_y = offsety(defl_ap)
    defl_os = defl*1.0 - offset_y  #approach and retract offseted deflection (zero when far from sample)        
        
    #offseting FD curves in x axis
    if not gauss_filt: #no gaussian filtering
        min_defl = defl_ap.argmin()
    else: 
        min_defl = offsetx(defl_ap)        
    zs_os = zs*1.0 - zs[min_defl] + defl_os[min_defl] #making indentation (zs - defl) zero when tip first snaps into the sample
    t -= t[min_defl]   #making time zero when tip first snaps into the sample       
    
    #Getting experimental Parameters
    maxi = zs.argmax()
    approach_vel,_,_,_,_ = linregress(t[0:maxi], zs[0:maxi])
    retract_vel,_,_,_,_ = linregress(t[maxi:maxi+int(len(t)*0.1)], zs[maxi:maxi+int(len(t)*0.1)])   
    
    if (np.isnan(defl_os).sum()>0 ) or (np.isnan(zs_os).sum()>0):
        print('nan')
        return np.nan, np.nan, np.nan, [np.nan, np.nan, np.nan, np.nan, np.nan]
    else:
        if length!= 0:  #downsampling is required
            t_roi, z_roi, d_roi, ind_roi = fdroi(t, zs_os, defl_os, gauss_filt, 50.0) 
            factor = int( round (  len(d_roi)/(length*1.0),0 ) )
            print('ds_factor:%d'%factor)
            
            if factor > 1:
                t_dec, defl_os = downsample(t, defl_os, factor)
                _, zs_os = downsample(t, zs_os, factor)
                t = t_dec
                
        if smooth: #smoothing by means of moving average
            defl_os = moving_av(defl_os,window)
            zs_os = moving_av(zs_os, window)   
      
        return t, defl_os, zs_os, [approach_vel, retract_vel, inv, k, fs]


def avfd_time(files, invols=0.0, k=0.0, fvar = 0.0, savefig = False, gauss_filt =True):
    """this function averages raw force distance curves in ibw files (acquired at SAME SAMPLING FREQUENCY) that will be offseted by fd_align function
    
    The function will average only fd curves that remain within a treshold of mean z-sensor velocity
    
    Parameters:
    ----------
    files : list
        list of strings containing the names of the ibw files (f = glob('*ibw'))
    invols : float, optional
        inverse optical lever sensitivity, if not passed it will be read from the ibw file channels
    k : float, optional
        spring constant, if not passed the value will be assigned to the stored value in the igor file
    fvar = float, optional
        treshold for caculation of average z-sensor velocity
    savefig: boolean, optional
        flag indicating if one wants to save the plot        
    gauss_filt : boolean, optional
        boolean flag indicating if gaussian filtering is needed to find the snap-in position
      
    Returns:
    ----------
    T : numpy.ndarray
        new axis of time trace for the averaged quantity
    Z : numpy.ndarray
        new axis with z-sensor position
    D : numpy.ndarray
        new axis with deflection position
    [zdot_ap, zdot_ret, inv_ols, stiffness, sampling_freq] : list
        list containing the values of: approachand retract z-sensor velocities, inverse optical lever sensitivity (invols), 
        cantilever stiffness (this is helpful in case the values are not passed to the function) and sampling frequency for
        each curve passed to the function (advise: check that all have same sampling frequency).
    """
    zdot_ap = []  #initialyzing list that will contain the zs approach velocities
    zdot_ret = []  #initialyzing list that will contain the zs retract velocities
    sf = []  #list containing the sampling frequencies at which the fd curves were acquired
    k_list = [] #list containing the spring constant stored in the ibw exeriment for each curve
    invols_list = [] #list containing the invols value for each ibw curve passed
    t_min = -100.0
    t_max = 100.0   #initialization of a variable that will contain the maximum time to be used in the definition of new time axis for averaging
    for f in files:  #this for loop is just to calculate the average of fdot (force slope in time) within the acquired curves
        t, d, z, params = fd_align(f, invols, k, gauss_filt)
        if t[len(t)-1] < t_max:
            t_max = t[len(t)-1]
        if t[0] > t_min:
            t_min = t[0]        
        zdot_ap.append(params[0])   
        zdot_ret.append(params[1])
        invols_list.append(params[2])
        k_list.append(params[3])
        sf.append(params[4]) #appending sampling frequency
        
    zdot_ap = np.array(zdot_ap)
    zdot_ret = np.array(zdot_ret)
    sampling_freq = np.array(sf)
    invols_a = np.array(invols_list)
    k_a = np.array(k_list)
    zap_mean = np.mean(zdot_ap)
    zret_mean = np.mean(zdot_ret)
    N = 0.0
    if fvar == 0.0: #if not passed an allowed variation of 5% will be assumed
        fvar = 5.0     
        
    #DEFINING NEW AXES FOR AVERAGING
    dt = 1.0/sf[0]   #timestep defined by sampling frequency
    T = np.arange(t_min, t_max+dt, dt)   #new time axis
    Z, D = np.zeros(len(T)), np.zeros(len(T))  #new z-sensor and deflection axes
    
    m = 0  #counter of total number of curves passed
    for f in files: #this for loop caculates the averaged z-sensor and deflection
        m +=1
        x,y,z,ch,note = loadibw(f)
        note =  {k.lower(): v for k, v in note.items()}  #making the dictionary to have all keys in lowercase
        ch = [u.lower() for u in ch]  #making channel to have all elements in lowercase
        if (invols == 0.0 or k == 0.0):  #extracting k and invols value if not passed              
            if k == 0.0:
                k = note['springconstant']  
            if invols == 0.0:
                invols = note['invols']
        sf = note['numptspersec']    #sampling frequency
        dt = 1.0/sf
        t, d, z, params = fd_align(f)
        defl = d[(t>=t_min) & (t<=t_max)]   #trimmed deflection that will be included in the averaging (falling in roi)
        zs = z[(t>=t_min) & (t<=t_max)]  #trimmed z-sensor to be included in the averaging (fallingin roi)
        if (len(defl) < len(T)):
            defl = np.insert(defl, 0, np.zeros(len(T)-len(defl)) )
        if (len(defl) > len(T)):
            defl = np.delete(defl, np.arange(len(defl)-len(T)))
        if (len(zs) < len(T)):
            zs = np.insert(zs, 0, np.zeros(len(T)-len(zs)))
        if (len(zs) > len(T)):
            defl = np.delete(zs, np.arange(len(zs)-len(T))) 
        if ( ( np.abs(  (zap_mean - params[0])/zap_mean  )*100 ) and ( np.abs(  (zret_mean - params[1])/zret_mean  )*100  ) ) < fvar:
            plt.figure(1, figsize=(7,4))
            plt.plot(T, defl)
            plt.title('deflection curves included in the averaging')
            plt.xlabel(r'$time, \, s$', fontsize='20',fontweight='bold')
            plt.ylabel(r'$deflection, \, m$', fontsize='20',fontweight='bold')
            if savefig:
                plt.savefig('DeflectionCurves.png', bboxinches='tight')
            
            plt.figure(2, figsize=(7,4))
            plt.plot(T, zs)
            plt.title('z-sensor curves included in the averaging')
            plt.xlabel(r'$time, \, s$', fontsize='20',fontweight='bold')
            plt.ylabel(r'$z-sensor, m$', fontsize='20',fontweight='bold')
            if savefig:
                plt.savefig('ZsensorCurves.png', bboxinches='tight')
                
            N += 1.0
            for i in range(len(T)):
                Z[i] += zs[i]
                D[i] += defl[i]
    
    Z /= N
    D /= N
    print('A total of %d curves were passed but only %d are averaged for quality of data'%(m,N))
    
    plt.figure(3, figsize=(7,4))
    plt.plot(-(Z-D)*1.0e9, D*1.0e9)
    plt.xlabel(r'$Tip \, Position, nm$', fontsize='20',fontweight='bold')
    plt.ylabel(r'$Deflection, nm$', fontsize='20',fontweight='bold')
    plt.title('Average FD curve')
    tip_label = -(Z-D)
    minim= tip_label.argmin()
    plt.xlim(tip_label[minim]*1.0e9*1.2, -1.0*tip_label[minim]*1.0e9*2.0)
    if savefig:
        plt.savefig('Average_FDcurve.png', bboxinches='tight')
    return T, Z, D, [zdot_ap, zdot_ret, invols_a, k_a, sampling_freq]

def fdroi(t, zs, defl, gauss_filt=False, percent=50.0, figure = False):
    """this function receives either average fd curve or single curve (containing approach and retract) and returns a region of interest
    
    if average function is to be sent, call previously avfd_time function
    if single fd curve is to be sent, call before fd_align function
    The output of this function (with percent >0) is appropriate to be passed to the NLS fit for non-linear load assumption
        
    Parameters:
    ----------
    t : numpy.ndarray
        time trace of the FD curve
    zs : numpy.ndarray
        trace containing z-sensor position
    defl : numpy.ndarray
        trace containing deflection in FD curve    
    gauss_filt : boolean, optional
        boolean flag indicating if gaussian filtering is needed to find the snap-in position
    percent : float, optional
        percentage of approach length that one wants to observe in the retract (50% is passed by default). In the plot
        generated you can see if this value satisfies your need
    figure : boolean, optional
        if True, a figure will be drawn
    
    Returns:
    ----------
    t_roi : numpy.ndarray
        time trace of the FD curve region of interest
    zs_roi : numpy.ndarray
        trace containing z-sensor position in the region of interest
    defl_roi : numpy.ndarray
        trace containing deflection in FD curve for the region of interest
    indent_roi: numpy.ndarray   
        indentation history (in the region of interest of the FD curve)    
    """            
    d_ap = defl[:zs.argmax()]   #approach portion of deflection to calculate snap to contact point  
    if gauss_filt:  #if a single noisy curve is passed, then some filtering may be done to determine the position of minimum
        offset_pos = offsetx(d_ap)
    else: #no Gauss filtering needed the simple argmin method can be used to get the minimum
        offset_pos = d_ap.argmin() #calculating snap to contact point
    maxi = zs.argmax()
    l_ap = maxi - offset_pos   #length of the repulsive portion of approach
    N = int(l_ap*(1.0+percent/100.0))   #total length of the array containing region of interest
    
    delta_t = np.mean(np.diff(t))
    t_roi = np.linspace(0,(N-1)*delta_t,N) #new axis for time when tip is indenting sample (roi)
    z_roi = zs[offset_pos:N+offset_pos] #new axis for time when tip indents into the sample (roi)
    defl_roi = defl[offset_pos:N+offset_pos] #- defl[offset_pos]  #applying deflection offset
    
    #applying offset of deflection at time zero to make it positive in repulsive portion: applying DMT assumption
    #DMT assumption means that adhesive force is constant during the whole contact portion
    defl_roi = defl_roi - defl_roi[0]
    #defl_roi -= defl_roi[0]  #this line is unstable unlike previous one (still don't understand why)
    z_roi = z_roi - z_roi[0]
    indent_roi = -(defl_roi - z_roi)  #indentation in respulsive portion of FD curve
    if figure:
        plt.figure(1)
        plt.plot(t_roi, indent_roi*1.0e9)
        plt.xlabel('time, s')
        plt.ylabel('indentation, nm')
        plt.title('Indentation in region of interest')
    return t_roi, z_roi, defl_roi, indent_roi 
    
def average_FDcurves(files, k = 1, invols = 1, max_indent = 1):
    """This function averages approach portion of FD curves based on position (interpolation is performed)
    
    Updated on May 22nd 2018
    
    """
    
    df, z_s, tip, tim, k_a, inv_a, vel, sr   =[], [], [], [],[],[],[], []
    inflim_zs = -1.0
    suplim_zs = 1.0
    max_len = 1
    for f in files:
        x,y,z,ch,note = loadibw(f) 
        note =  {k.lower(): v for k, v in note.items()}  #making the dictionary to have all keys in lowercase
        ch = [u.lower() for u in ch]
        if k == 1:
            k = note['springconstant']  
        inv = note['invols']#['Invols']
        if invols ==1:   #no correction needed
            correction = 1.0
        else: #correction needed due to late calibration
            correction = invols/inv
        defl = z[ch.index('defl')]
        defl = defl*correction
        zs = z[ch.index('zsnsr')]
        fs = note['numptspersec']    #sampling frequency
        t = np.arange(len(defl))/fs   #retrieving time array with the aid of sampling frequency
                
        #attaching values to lists containing stiffnesses and invols values
        k_a.append(k)
        inv_a.append(inv)
            
        #GETTING ONLY APPROACH PORTION
        maxi = zs.argmax()
        zs = zs[:maxi]
        defl = defl[:maxi]
        t = t[:maxi]
        
        #APPLYING OFFSETS TO MAKE CURVES FALL ONE OVER EACH OTHER     
        defl_oy,_ = offsety(defl)  
        zero = offsetx(defl_oy)
        zs = zs - zs[zero] + defl_oy[zero]
        
        #FINDING LIMITS OF THE INTERPOLATION, THIS HAS TO FALL WITHIN THE SHORTEST LIMITS OF Z SENSOR AMONG ALL CURVES, OTHERWISE IT WOULD PERFOR EXTRAPOLATION FOR SOME CURVES
        if zs[0] > inflim_zs:
            inflim_zs = zs[0]
        if zs[len(zs)-1] < suplim_zs:
            suplim_zs = zs[len(zs)-1]
        if len(zs) > max_len:   #finding the masimum length of z sensor among the curves
            max_len = len(zs)
                
        plt.figure(1)
        plt.plot(zs)
        plt.xlabel('Z_sensor, nm', fontsize=10)
        plt.ylabel('Deflection, nm', fontsize=10)
        #plt.xlim(-zs[len(zs)-1], zs[len(zs)-1])
        df.append(defl)
        z_s.append(zs)
        tip.append(defl-zs)
        tim.append(t)
        vel.append(str(note['approachvelocity']*1.0e6))  #adding to array containing approach velocity of the force curve (in um/s)
        sr.append(str(note['numptspersec']/1.0e3))  #sampling rate (in kHz)
    
    Z = np.linspace(inflim_zs, suplim_zs, max_len)
    D = np.zeros(len(Z)) #Deflection to be corrected
    T = np.zeros(len(Z)) #Time
    N = 0 #counter for the number of FD curves
    
    for i in range(len(files)):
        #interpolation of deflection with respect to new axis zs
        tck = splrep(z_s[i],df[i],k=1) #this returns a tuple with the vector of knots, the B-spline coefficients, and the degree of the spline
        #at this point occurs the reduction of number of points, from original size to the size of Z (default 1000 points)
        dfl = splev(Z,tck) #this returns an array of values representing the spline function evaluated at the points in Z 
        D += dfl
        tck = splrep(z_s[i],tim[i],k=1)  #interpolation of time with respect to new axis zs
        ti = splev(Z,tck) #at this point occurs the reduction of number of points, from original size to the size of Z (default 1000 points)
        T += ti
        N += 1   #this counter is useful for the final averaging, this counts how many FD curves are being averaged.
        
    # Average
    D /= N
    T /= N
    
    plt.figure(2)
    plt.plot(Z-D,D)
    plt.xlabel('Averaged Tip Position, nm', fontsize=10)
    plt.ylabel('Averaged Deflection, nm', fontsize=10)
    plt.xlim( - ( Z[len(Z) -1] - D[len(Z) - 1]  ),  (Z[len(Z) -1] - D[len(Z) - 1]) )
    
    plt.figure(3)
    plt.plot(Z,D)
    plt.xlabel('Averaged z-sensor, nm', fontsize=10)
    plt.ylabel('Averaged Deflection, nm', fontsize=10)
    
    return T, Z, D, Z-D, k_a, inv_a, vel, sr    
    

def dmt_moduli(ruta, k=0.0, invols=0.0, r_min=0.99, R=10.0e-9, max_indent = 0.0, nu=0.5, figures =False, gauss_filt=False):
    """This function gets material's Young modulus from force-distance curves in ibw files assuming DMT theory
    
    A path is passed (ruta) where the ibw files with the force curves are located
    It will generate and save a main file containing the values of moduli: 'Stiffness_positive.txt'
    
    Parameters:
    ----------
    ruta : string
        pathwhere the ibw files with the force curves are located
    k : float, optional
        cantilever stifness in N/m, if not passed it will acquire the value stored in the ibw file
    invols : float, optional
        inverse optical lever sensitivity value in m/V, if not passed it will acquire the value stored in the ibw file
    r_min : float, optional
        a parameter to disregard the 'bad' force curves, the closest to one, the more strict is the quality control
    R : float, optional
        tip radius of curvature
    max_indent : float, optional
        value of maximum indentation to be used in the calculation of modulus, it is advised to be lower than tip radius
    nu = float, optional
        material's Poisson ratio, by default it is assumed to be an incompressible material: nu = 0.5  
    figures = boolean, optional
        if True then it will generate and save some plots but slows down significantly the figure, default is false    
    gauss_filt : boolean, optional
        boolean flag indicating if gaussian filtering is needed to find the snap-in position
    
    Returns:
    ----------
    df : pandas.DataFrame
        dataframe containing the information on the calculated moduli for the curves that passed the quality check
    """
    fn = os.path.basename(os.path.normpath(ruta))  #GETTING NAME OF FOLDER WHERE FILES ARE LOCATED
    os.chdir(ruta)
    files = glob('*.ibw')    
    slope_p = []
    fname_p = []
    fname_b = []
    slope_b = []
    approach_velp =  []
    retract_velp = []
    max_xlim_b = 0.0
    max_xlim_p = 0.0
    max_ind = 0.0
    for f in files:
        t, defl, zs, params = fd_align(f, invols, k, gauss_filt)
                           
        #attaching values to lists containing stiffnesses and invols values
        if k == 0.0:
            k = params[3]
        if invols == 0.0:
            invols = params[2]
                
        #GETTING REPULSIVE PORTION TO CALCULATE STIFFNESS
        t_r, zs_r, defl_r, tip_r = fdroi(t, zs, defl, gauss_filt, 0.0)
                
        if max_indent != 0.0: #Criterion was given of maximum indentation allowed
            tip_r = tip_r[tip_r<max_indent]
            defl_r = defl_r[:len(tip_r)]
            zs_r = zs_r[:len(tip_r)]
       
        if tip_r[len(tip_r)-1] > max_ind: #finding maximum indentation regardless it is passed or not
            max_ind = tip_r[len(tip_r)-1]  #this will only be useful for the s-limits of a plot
        
        #Making sure only positive values of indentation are considered:
        tip_rep = tip_r[tip_r>0]
        defl_rep = defl_r[tip_r>0]       
        
        #linear regression to get modulus (if info arrays are not empty)
        if (len(tip_r) and len(defl_r)) > 0:
            sl, _, _, _, _ = stats.linregress(tip_r, defl_r)
        else:
            sl = -1.0  #this negative value means that the force curve is unphysical

        if sl > 0.0: #the curve might be a good one
            _, _, r, _, _ = stats.linregress(zs_r, defl_r)
            alfa = 4.0/3*np.sqrt(R)/(1.0-nu**2)
            slope, intercept, r_value, p_value, std_err = stats.linregress(tip_rep**1.5, defl_rep*k/alfa)
            if r**2 < r_min:  #The quality of the raw force spectroscopy curve is bad
                slope_b.append(slope/1.0e6) #result in MPa
                fname_b.append(f)
                if zs[-1] > max_xlim_b:
                    max_xlim_b = zs[-1]
                if figures:
                    plt.figure(1)
                    plt.plot(zs*1.0e9, defl*1.0e9)
                    plt.xlim(-max_xlim_b*1.0e9*2.0, max_xlim_b*1.0e9*1.5)
                    plt.xlabel('Z_sensor, nm', fontsize=10)
                    plt.ylabel('Deflection, nm', fontsize=10)
                    plt.title('Deflection vs Z-sensor Bad Ones')
                    plt.savefig('Defl_Zs_bad_%s.png'%fn)
            else: #the quality of the raw force spectroscopy curve is satisfactory
                slope_p.append(slope/1.0e6)
                fname_p.append(f)                 
                approach_velp.append(params[0]) #attaching approach velocities
                retract_velp.append(params[1]) #attaching retract velocities
                if zs[-1] > max_xlim_p:
                    max_xlim_p = zs[-1]
                if figures:
                    plt.figure(3)
                    plt.plot(zs, defl)
                    plt.xlim(-max_xlim_p*2.0, max_xlim_p*1.5)
                    plt.xlabel('Z_sensor, nm', fontsize=10)
                    plt.ylabel('Deflection, nm', fontsize=10)
                    plt.title('Deflection vs Z-sensor Good Ones')
                    plt.savefig('Defl_Zs_good_%s.png'%fn)

                    plt.figure(4)
                    plt.plot( (tip_rep)*1.0e9, defl_rep*k*1.0e9)
                    plt.xlabel('Tip position, nm', fontsize=10)
                    plt.ylabel('Force, nN', fontsize=10)
                    plt.xlim(-max_ind*2.0*1.0e9, max_ind*2.0*1.0e9)
                    plt.title('FD curves Good Ones')
                    plt.savefig('FvsD_good_%s.png'%fn)
        else: # the slope is negative, then attaching to the file containing the bad curves
            slope_b.append(sl/1.0e6) #modulus in MPa
            fname_b.append(f)  
        
    #Creating DataFrames with the moduli
    if len(fname_p) > 0:
        zipped = list(zip(fname_p, slope_p, approach_velp, retract_velp))
        df = pd.DataFrame(zipped)
        df.columns = ['file_name','Stiffness, MPa','Approach_vel, m/s','Retract_vel, m/s'] #assigning names to columns in dataFrame
        df.to_csv('Stiffness_positive_%s.txt'%fn, sep='\t')
    if len(fname_b) > 0:
        zipped_b = list(zip(fname_b, slope_b))
        df_b = pd.DataFrame(zipped_b)
        df_b.columns = ['file_name','Stiffness, MPa'] #assigning names to columns in dataFrame
        df_b.to_csv('Stiffness_bad_%s.txt'%fn, sep='\t')
    return df

def histo_boxplot(ruta, nb, mini_hist, maxi_hist, tolerance=10.0, vel_info =False, plotit = False):
    """This program walks through the directories of the path given (ruta) and gets histograms of material properties
    
    It will generate histogram plots with moduli and boxplot to summarize data statistical distribution
    Previously you have to run the dmt_moduli function to generate txt files containing the materials' moduli
    
    Parameters:
    ----------
    ruta : str
        path where located the folders that contain the txt files with the moduli 
    nb : int
        number of bins to be used in the histograms
    mini_hist : float
        minimum value in MPa of modulus to be shown in histogram
    maxi_hist : float
        maximum value in MPa of modulus to be shown in histogram
    tolerance : float, optional
        how far the approach and retract velocities can be from the average in percentage
    vel_info : boolean, optional
        if True the labels in the boxplot will correspond to the average approach velocity of the force curves
        
    Returns:
    ----------
    df : pandas.DataFrame
        dataframe containing the information on the calculated moduli for the curves that passed the quality check
    """
    stiffness_a = []
    name_a = []
    name_b = []
    i = 0
    for pt, sf, fn in os.walk(ruta):
        if sf == []:  #no more subfolder in folder
            os.chdir(pt)
            files = glob('*.txt')
            for f in files:
                approach_vel = []
                retract_vel = []
                stiffness = []
                os.chdir(pt)        
  
                if 'Stiffness_positive' in f:
                    res = np.loadtxt(f, delimiter='\t', skiprows=1, dtype='str')
                    approach_vel = res[:,3].astype(np.float)
                    retract_vel = res[:,4].astype(np.float)
                    ap_vel_av = np.mean(approach_vel)
                    ret_vel_av = np.mean(retract_vel)  
                    stiffness = res[:,2].astype(np.float)
                    print('Total of %d force curves processed'%np.size(stiffness))
                    stiffness_rest = stiffness[ (approach_vel > ap_vel_av*(1.0-tolerance/100.0) ) & (approach_vel < ap_vel_av*(1.0+tolerance/100.0))  & ( np.abs(retract_vel) > np.abs(ret_vel_av)*(1.0-tolerance/100.0)) & (np.abs(retract_vel) < np.abs(ret_vel_av)*(1.0+tolerance/100.0))  ]
                    print('But only %d force curves passed the tolerance criterion'%np.size(stiffness_rest))
                    stiffness_a.append(stiffness_rest)
                    name_a.append(os.path.basename(os.path.normpath(os.getcwd())))                    
                    name_b.append('vel %2.3f um/s'%(ap_vel_av*1.0e6))
                    if plotit:
                        plt.figure(i)
                        plt.hist(stiffness_rest, bins=nb, range=(mini_hist,maxi_hist), alpha=0.5, ec='black')
                        plt.title("Histogram %s, vel %2.2f um/s"%(name_a[i], ap_vel_av*1.0e6))
                        plt.xlabel('Stiffness, MPa', fontsize =15)
                        string = os.getcwd()
                        string = string.replace(os.path.basename(os.path.normpath(os.getcwd())),'')
                        os.chdir(string)
                        f = f.split('.',1)[0]
                        plt.savefig("Histogram_%s.png"%name_a[i], bbox_inches='tight')                    
                        i += 1
    df = pd.DataFrame(stiffness_a)
    df = df.transpose()
    if vel_info:
        df.columns = name_b
    else:
        df.columns = name_a
    if plotit:
        df.plot(kind='box')
        os.chdir(ruta)
        plt.ylim(mini_hist, maxi_hist)
        plt.ylabel('Stiffness, MPa', fontsize =15)
        plt.savefig("Boxplot.png", bbox_inches='tight')    
    return df

def fd_zoomin(t, zs, defl, av=True, percent=50.0, figure = False):
    """this function receives either average fd curve or single curve (containing approach and retract) and returns a zoomed in version that can be used to calculate dissipation
    if average function is to be sent, call previously avfd_time function
    if single fd curve is to be sent, call before fd_align function
        
    Parameters:
    ----------
    t : numpy.ndarray
        time trace of the FD curve
    zs : numpy.ndarray
        trace containing z-sensor position
    defl : numpy.ndarray
        trace containing deflection in FD curve
    av : optional, boolean
        indicates if the passed fd is averaged (smooth), by default is True
    percent : float, optional
        percentage of approach length that one wants to observe in the retract (50% is passed by default). In the plot
        generated you can see if this value satisfies your need
    figure : boolean, optional
        if True, a figure will be drawn
    
    Returns:
    ----------
    t_zoom : numpy.ndarray
        time trace of the FD curve region of interest
    zs_zoom : numpy.ndarray
        trace containing z-sensor position in the region of interest
    defl_zoom : numpy.ndarray
        trace containing deflection in FD curve for the region of interest
    indent_zoom: numpy.ndarray   
        indentation history (in the region of interest of the FD curve)    
    """            
    d_ap = defl[:zs.argmax()]   #approach portion of deflection to calculate snap to contact point  
    if not av:  #if a single noisy curve is passed, then some filtering has to be done to determine the position of minimum
        offset_pos = offsetx(d_ap)
    else: #an average fd curve is passed and the simple argmin methon can be used to get the minimum
        offset_pos = d_ap.argmin() #calculating snap to contact point
    maxi = zs.argmax()
    mini = offset_pos - (maxi-offset_pos)
    l_ap = maxi - mini   #length of the repulsive portion of approach
    N = int(l_ap*(1.0+percent/100.0))   #total length of the array containing region of interest
    
    delta_t = np.mean(np.diff(t))
    t_zoom = np.linspace(0,(N-1)*delta_t,N) #new axis for time when tip is indenting sample (roi)
    z_zoom = zs[mini:N+mini] #new axis for time when tip indents into the sample (roi)
    defl_zoom = defl[mini:N+mini] #- defl[offset_pos]  #applying deflection offset
    
    if figure:
        plt.figure(1)
        plt.plot(defl_zoom - z_zoom, defl_zoom)
        plt.xlabel('tip position, m')
        plt.ylabel('deflection, m')
        plt.title('FD curve in region of interest')
    return t_zoom, z_zoom, defl_zoom


def diss_fd(tip, f_ts):
    """This function calculates the tip-sample dissipation per static force spectroscopy experiment    
    
    Parameters
    ----------
    defl : numpy.ndarray
        tip deflection
    f_ts : numpy.ndarray
        tip-sample interacting force
    dt : float
        simulation timestep
    fo1 : float
        eigenmode resonance frequency
    
    Returns
    -------    
    energy_diss/number_of_periods : float
        total dissipated energy per oscillating period       
    """
    energy_diss = 0.0
    for i in range(1, len(tip) - 1):
        # based on integral of f_ts*dz/dt*dt, dz/dt=(defl[i+1]-defl[i-1])/(2.0*dt) Central difference approx
        energy_diss -= f_ts[i] * (tip[i + 1] - tip[i - 1]) / 2.0 
    return energy_diss #/number_of_periods
    
def max_ind(ruta, k=0.0, invols=0.0, r_min=0.99, gauss_filt = False):
    """This function gets the maximum indentation inverse for a set of files given, the inverse of maximum stiffness is expected to be proportional to sample's stiffness
    
    A path is passed (ruta) where the ibw files with the force curves are located
    It will generate and save a main file containing the values of moduli: 'Stiffness_positive.txt'
    
    Parameters:
    ----------
    ruta : string
        pathwhere the ibw files with the force curves are located
    k : float, optional
        cantilever stifness in N/m, if not passed it will acquire the value stored in the ibw file
    invols : float, optional
        inverse optical lever sensitivity value in m/V, if not passed it will acquire the value stored in the ibw file
    r_min : float, optional
        a parameter to disregard the 'bad' force curves, the closest to one, the more strict is the quality control
    gauss_filt : boolean, optional
        boolean flag indicating if gaussian filtering is needed to find the snap-in position
    
    Returns:
    ----------
    df : pandas.DataFrame
        dataframe containing the information on the calculated moduli for the curves that passed the quality check
    """
    fn = os.path.basename(os.path.normpath(ruta))  #GETTING NAME OF FOLDER WHERE FILES ARE LOCATED
    os.chdir(ruta)
    files = glob('*.ibw')   
    min_tip_inverse = []
    fname = []
    approach_velp =  []
    retract_velp = []
    for f in files:
        t, defl, zs, params = fd_align(f, invols, k, gauss_filt)
                           
        #attaching values to lists containing stiffnesses and invols values
        if k == 0.0:
            k = params[3]
        if invols == 0.0:
            invols = params[2]
                
        #GETTING REPULSIVE PORTION OF APPROACH CURVE TO CALCULATE MAXIMUM INDENTATION
        t_r, zs_r, defl_r, tip_r = fdroi(t, zs, defl, gauss_filt, 0.0)                
          
        
        #linear regression to get modulus (if info arrays are not empty)
        if (len(tip_r) and len(defl_r)) > 0:
            sl, _, _, _, _ = stats.linregress(tip_r, defl_r)
        else:
            sl = -1.0  #this negative value means that the force curve is unphysical

        if sl > 0.0: #the curve might be a good one
            _, _, r, _, _ = stats.linregress(zs_r, defl_r)
                         
            if r**2 > r_min: #the quality of the raw force spectroscopy curve is satisfactory
                min_tip = max(tip_r)
                min_tip_inverse.append(1.0/min_tip)
                
                fname.append(f)
                approach_velp.append(params[0]) #attaching approach velocities
                retract_velp.append(params[1]) #attaching retract velocities

        
    #Creating DataFrames with the moduli
    if len(fname) > 0:
        zipped = list(zip(fname, min_tip_inverse, approach_velp, retract_velp))
        df = pd.DataFrame(zipped)
        df.columns = ['file_name','Inverse Max indent, 1/nm','Approach_vel, m/s','Retract_vel, m/s'] #assigning names to columns in dataFrame
        df.to_csv('Max_indent_%s.txt'%fn, sep='\t')
    
    return df


def stats_maxindent(ruta, nb, mini_hist, maxi_hist, tolerance=10.0, vel_info =False):
    """This program walks through the directories of the path given (ruta) and gets histograms of inverse of maximum indentation of force curves
    
    It will generate histogram plots and boxplot of inverse of maximum indentation to summarize data statistical distribution
    Previously you have to run the max_indent function to generate txt files containing the information of the inverse of maximum indentation
    
    Parameters:
    ----------
    ruta : str
        path where located the folders that contain the txt files with the moduli 
    nb : int
        number of bins to be used in the histograms
    mini_hist : float
        minimum value in MPa of modulus to be shown in histogram
    maxi_hist : float
        maximum value in MPa of modulus to be shown in histogram
    tolerance : float, optional
        how far the approach and retract velocities can be from the average in percentage
    vel_info : boolean, optional
        if True the labels in the boxplot will correspond to the average approach velocity of the force curves
    
    Returns:
    ----------
    df : pandas.DataFrame
        dataframe containing the information on the calculated moduli for the curves that passed the quality check
    """
    invindent_a = []
    name_a = []
    name_b = []
    i = 0
    for pt, sf, fn in os.walk(ruta):
        if sf == []:  #no more subfolder in folder
            os.chdir(pt)
            files = glob('*.txt')
            for f in files:
                os.chdir(pt)    
                if 'Max' in f:
                    res = np.loadtxt(f, delimiter='\t', skiprows=1, dtype='str')
                    approach_vel = res[:,3].astype(np.float)
                    retract_vel = res[:,4].astype(np.float)
                    ap_vel_av = np.mean(approach_vel)
                    ret_vel_av = np.mean(retract_vel)  
                    inverse_indent = res[:,2].astype(np.float)
                    print('Total of %d force curves processed'%np.size(inverse_indent))
                    indent_rest = inverse_indent[ (approach_vel > ap_vel_av*(1.0-tolerance/100.0) ) & (approach_vel < ap_vel_av*(1.0+tolerance/100.0))  & ( np.abs(retract_vel) > np.abs(ret_vel_av)*(1.0-tolerance/100.0)) & (np.abs(retract_vel) < np.abs(ret_vel_av)*(1.0+tolerance/100.0))  ]
                    print('But only %d force curves passed the tolerance criterion'%np.size(indent_rest))
                    invindent_a.append(indent_rest)
                    name_a.append(os.path.basename(os.path.normpath(os.getcwd())))                    
                    name_b.append('vel %2.3f um/s'%(ap_vel_av*1.0e6))
                    plt.figure(i)
                    plt.hist(indent_rest, bins=nb, range=(mini_hist,maxi_hist), alpha=0.5, ec='black')
                    plt.title("Histogram %s, vel %2.2f um/s"%(name_a[i], ap_vel_av*1.0e6))
                    plt.xlabel('Inverse of max indentation, 1/m', fontsize =15)
                    string = os.getcwd()
                    string = string.replace(os.path.basename(os.path.normpath(os.getcwd())),'')
                    os.chdir(string)
                    f = f.split('.',1)[0]
                    plt.savefig("Histogram_MaxIndent_%s.png"%name_a[i], bbox_inches='tight')                    
                    i += 1
    df = pd.DataFrame(invindent_a)
    df = df.transpose()
    if vel_info:
        df.columns = name_b
    else:
        df.columns = name_a
    df.plot(kind='box')
    os.chdir(ruta)
    plt.ylim(mini_hist, maxi_hist)
    plt.ylabel('Inverse of max indentation, 1/m', fontsize =15)
    plt.savefig("Boxplot_maxindent.png", bbox_inches='tight')    
    return df    

def ediss_multi(ruta, k=0.0, invols=0.0, r_min=0.99, R=10.0e-9, nu=0.5, figures =False):
    """This function gets energy dissipation per force-distance curve contained in ibw files
    
    A path is passed (ruta) where the ibw files with the force curves are located
    It will generate and save a main file containing the values of dissipated energy: 'ediss.txt'
    
    Parameters:
    ----------
    ruta : string
        pathwhere the ibw files with the force curves are located
    k : float, optional
        cantilever stifness in N/m, if not passed it will acquire the value stored in the ibw file
    invols : float, optional
        inverse optical lever sensitivity value in m/V, if not passed it will acquire the value stored in the ibw file
    r_min : float, optional
        a parameter to disregard the 'bad' force curves, the closest to one, the more strict is the quality control
    R : float, optional
        tip radius of curvature
    nu = float, optional
        material's Poisson ratio, by default it is assumed to be an incompressible material: nu = 0.5  
    figures = boolean, optional
        if True then it will generate and save some plots but slows down significantly the figure, default is false
    
    Returns:
    ----------
    df : pandas.DataFrame
        dataframe containing the information on the calculated moduli for the curves that passed the quality check
    """
    fn = os.path.basename(os.path.normpath(ruta))  #GETTING NAME OF FOLDER WHERE FILES ARE LOCATED
    os.chdir(ruta)
    files = glob('*.ibw')    
    ediss = []
    fname = []
    approach_vel =  []
    retract_vel = []
    max_xlim = 0.0
    max_ind = 0.0
    for f in files:
        t, defl, zs, params = fd_align(f)                           
        #attaching values to lists containing stiffnesses and invols values
        if k == 0.0:
            k = params[3]
        if invols == 0.0:
            invols = params[2]
                
        #GETTING ZOOMED-IN PORTION TO CALCULATE DISSIPATED ENERGY
        t_r, zs_r, defl_r = fd_zoomin(t, zs, defl, False, 100.0) 
        tip_r = defl_r - zs_r 
        t_roi, zs_roi, defl_roi, tip_roi = fdroi(t, zs, defl, False, 0.0) #only repulsive portion

    
        if tip_roi[len(tip_roi)-1] > max_ind: #finding maximum indentation regardless it is passed or not
            max_ind = tip_roi[len(tip_roi)-1]           
        
        #linear regression to get modulus (if info arrays are not empty)
        if (len(tip_roi) and len(defl_roi)) > 0:
            sl, _, _, _, _ = stats.linregress(tip_roi, defl_roi)
            #print(sl)
        else:
            sl = -1.0  #this negative value means that the force curve is unphysical

        if sl > 0.0: #the curve might be a good one
            _, _, r, _, _ = stats.linregress(zs_roi, defl_roi)
            #print(r)
            if r**2 > r_min:  #the quality of the raw force spectroscopy curve is satisfactory
                fname.append(f)
                diss = diss_fd(tip_r, defl_r*k)
                ediss.append((diss*1.0e18))
                approach_vel.append(params[0]) #attaching approach velocities
                retract_vel.append(params[1]) #attaching retract velocities
                if zs[-1] > max_xlim:
                    max_xlim = zs[-1]
                if figures:
                    plt.figure(1)
                    plt.plot(zs_r, defl_r)
                    #plt.xlim(-max_xlim*2.0, max_xlim*1.5)
                    plt.xlabel('Z_sensor, nm', fontsize=10)
                    plt.ylabel('Deflection, nm', fontsize=10)
                    plt.title('Deflection vs Z-sensor ap_ret')
                    plt.savefig('Defl_Zs_apret_%s.png'%fn)

                    plt.figure(2)
                    plt.plot( (tip_r)*1.0e9, defl_r*k*1.0e9)
                    plt.xlabel('Tip position, nm', fontsize=10)
                    plt.ylabel('Force, nN', fontsize=10)
                    plt.xlim(-max_ind*2.0*1.0e9, max_ind*2.0*1.0e9)
                    plt.title('FDcurves_hysteresis')
                    plt.savefig('FvsD_hyst_%s.png'%fn) 
        
    #Creating DataFrames with the moduli
    if len(fname) > 0:
        zipped = list(zip(fname, ediss, approach_vel, retract_vel))
        df = pd.DataFrame(zipped)
        df.columns = ['file_name','Ediss, aJ','Approach_vel, m/s','Retract_vel, m/s'] #assigning names to columns in dataFrame
        df.to_csv('Ediss_%s.txt'%fn, sep='\t')
    return df

def ediss_stats(ruta, nb, mini_hist, maxi_hist, tolerance=10.0, vel_info =False):
    """This program walks through the directories of the path given (ruta) and gets histograms of material properties
    
    It will generate histogram plots with moduli and boxplot to summarize data statistical distribution
    Previously you have to run the dmt_moduli function to generate txt files containing the materials' moduli
    
    Parameters:
    ----------
    ruta : str
        path where located the folders that contain the txt files with the moduli 
    nb : int
        number of bins to be used in the histograms
    mini_hist : float
        minimum value in MPa of modulus to be shown in histogram
    maxi_hist : float
        maximum value in MPa of modulus to be shown in histogram
    tolerance : float, optional
        how far the approach and retract velocities can be from the average in percentage
    vel_info : boolean, optional
        if True the labels in the boxplot will correspond to the average approach velocity of the force curves
    
    Returns:
    ----------
    df : pandas.DataFrame
        dataframe containing the information on the calculated moduli for the curves that passed the quality check
    """
    ediss_a = []
    name_a = []
    name_b = []
    i = 0
    for pt, sf, fn in os.walk(ruta):
        if sf == []:  #no more subfolder in folder
            os.chdir(pt)
            files = glob('*.txt')
            for f in files:
                approach_vel = []
                retract_vel = []
                ediss = []
                os.chdir(pt)        
  
                if 'Ediss' in f:
                    res = np.loadtxt(f, delimiter='\t', skiprows=1, dtype='str')
                    approach_vel = res[:,3].astype(np.float)
                    retract_vel = res[:,4].astype(np.float)
                    ap_vel_av = np.mean(approach_vel)
                    ret_vel_av = np.mean(retract_vel)  
                    ediss = res[:,2].astype(np.float)
                    print('Total of %d force curves processed'%np.size(ediss))
                    ediss_rest = ediss[ (approach_vel > ap_vel_av*(1.0-tolerance/100.0) ) & (approach_vel < ap_vel_av*(1.0+tolerance/100.0))  & ( np.abs(retract_vel) > np.abs(ret_vel_av)*(1.0-tolerance/100.0)) & (np.abs(retract_vel) < np.abs(ret_vel_av)*(1.0+tolerance/100.0))  ]
                    print('But only %d force curves passed the tolerance criterion'%np.size(ediss_rest))
                    ediss_a.append(ediss_rest)
                    name_a.append(os.path.basename(os.path.normpath(os.getcwd())))                    
                    name_b.append('vel %2.3f um/s'%(ap_vel_av*1.0e6))
                    plt.figure(i)
                    plt.hist(ediss_rest, bins=nb, range=(mini_hist,maxi_hist), alpha=0.5, ec='black')
                    plt.title("Ediss Histogram %s, vel %2.2f um/s"%(name_a[i], ap_vel_av*1.0e6))
                    plt.xlabel('Ediss, aJ', fontsize =15)
                    string = os.getcwd()
                    string = string.replace(os.path.basename(os.path.normpath(os.getcwd())),'')
                    os.chdir(string)
                    f = f.split('.',1)[0]
                    plt.savefig("Ediss_Histogram_%s.png"%name_a[i], bbox_inches='tight')                    
                    i += 1
    df = pd.DataFrame(ediss_a)
    df = df.transpose()
    if vel_info:
        df.columns = name_b
    else:
        df.columns = name_a
    df.plot(kind='box')
    os.chdir(ruta)
    plt.ylim(mini_hist, maxi_hist)
    plt.ylabel('Ediss, aJ', fontsize =15)
    plt.savefig("Ediss_Boxplot.png", bbox_inches='tight')    
    return df