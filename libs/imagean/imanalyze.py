# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 14:03:54 2018

@author: Enrique Alejandro
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from inspect import getargspec
from matplotlib import cm
import os

import sys
#syspath = 'C:/Users/enrique/Desktop/QuiqueTemp/ModelFree_LossAngle'
syspath = 'e:/github/modelfree_lossangle'
sys.path.append(syspath)
from libs.forcespec.fdanalysis import loadibw


fplane    = lambda x,a,b,c: a*x[0] + b*x[1]+c
fplanex3   = lambda x,a,b1,b2,b3,b4: a*x[1] + np.polyval([b1,b2,b3,b4],x[0])
fplaney3   = lambda x,a,b1,b2,b3,b4: a*x[0] + np.polyval([b1,b2,b3,b4],x[1])


def Fit(func, X, y, p=0, de=0, roi=[0]):
    """
    Fit(func, X, y, p=0, de=0)
    <Least square fit>
    func: function fitted
    x: axis value (X or (X,Y,...))
    y: 1D or 2D array data
    p: list, parameter preset
    de: 0=fitdata, 1=detrend, 2=param return
    roi: fitting range [start,end]
    """
    X = np.array(X)
    if len(roi)==1:
        fy = y
        fx = X
    elif len(roi)==2:
        fy = y[roi[0]:roi[1]]
        if np.array(X).ndim == 1:
            fx = X[roi[0]:roi[1]]
        else:
            fx = X[:,roi[0]:roi[1]]
    else:
        roi = np.array(roi,dtype=bool)
        fy = y[roi]
        if np.array(X).ndim == 1:
            fx = X[roi]
        else:
            fx = [X[i][roi] for i in range(len(X))]
    if np.array(fx).ndim > 2:
        fx = np.array( map(np.ravel, fx) )
    if p==0:
        pn = len(getargspec(func)[0])-1
        p = np.ones(pn)
    err = lambda p,x,y,f: y-f(x,*p)
    p = leastsq(err, p, args=(fx, fy.ravel(), func))[0]
    f = func(X,*p)
    if de==0: return f
    elif de==1: return y-f
    else: return np.array(p)

def Detrend(y, ord=1, de=1, roi=[0], axis=1):
    """
    Detrend(y, ord=1, de=1, roi=[0], axis=1)
    <Polynomial fit>
    y: 1D or 2D array data
    ord: polynomial order
    de: 0=fitdata, 1=detrend, 2=param return
    roi: fitting range [start,end]
    axis: 0=y axis, 1=x axis
    """
    if y.ndim==1:
        x = np.arange(y.size)
        if len(roi)==1:
            fx = x
            fy = y
        elif len(roi)==2:
            fx = x[roi[0]:roi[1]]
            fy = y[roi[0]:roi[1]]
        else:
            roi = np.array(roi,dtype=bool)
            fx = x[roi]
            fy = y[roi]
        p = np.polyfit(fx,fy,ord)
        f = np.polyval(p,x)
    else:
        if axis==1:
            y = y.T
            if len(roi)>2:
                roi = roi.T
        x = np.arange(y.shape[0])
        if len(roi)==1:
            fx = x
            fy = y
            p = np.polyfit(fx,fy,ord).T
        elif len(roi)==2:
            fx = x[roi[0]:roi[1]]
            fy = y[roi[0]:roi[1]]
            p = np.polyfit(fx,fy,ord).T
        else:
            roi = np.array(roi,dtype=bool)
            fx = [x[roi[:,i]] for i in range(y.shape[1])]
            fy = [y[:,i][roi[:,i]] for i in range(y.shape[1])]
            p = np.array([np.polyfit(fx[i],fy[i],ord) for i in range(y.shape[1])])
        X = np.tile(x,y.shape[1]).reshape((y.shape[1],x.size))
        f = np.array( map(np.polyval,p,X) ).T
        if axis==1:
            y = y.T
            f = f.T
    if de==0: return f
    if de==1: return y-f
    else: return p



def ImageFlatten(z, type='line', axis=1):
    """
    z: 2D array
    type:   'line' = line by line linear flatten
            'plane' = plane flatten
            'plane3' = plane flatten
            'ptr' = ptr image flatten
            'vpr' = ptr vpr image flatten
    """
    if type=='line':
        z = Detrend(z,axis=axis)
    elif type=='plane':
        XY = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))
        z = Fit(fplane, XY, z, de=1)
    elif type=='plane3':
        XY = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))
        if axis:
            z = Fit(fplaney3, XY, z, de=1)
        else:
            z = Fit(fplanex3, XY, z, de=1)
    elif type=='ptr':
        z -= z.mean(0)
        XY = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))
        z = Fit(fplane, XY, z, de=1, roi=[450,1024])
    elif type=='vpr':
        z -= z[0:128].mean(0)
        XY = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))
        z = Fit(fplane, XY, z, de=1, roi=[0,128])
    return z


def zamphi(f, path='', ampinvols=0.0):
    """This function returns information to perform 2D draws of topography, amplitude and phase from tapping mode images
    """
    if path != '':
        os.chdir(path)
        print(path)
    x,y,z,ch,note = loadibw(f) #loading ibw file with Hanaul's magic function

    note =  {k.lower(): v for k, v in note.items()}  #making the dictionary to have all keys in lowercase
    ch = [u.lower() for u in ch]  #making channel to have all elements in lowercase
    if ampinvols == 0.0:
        ampinvols = note['ampinvols']
    scansize = note['scansize']
    return z[0]*1.0e9, z[1]*ampinvols*1.0e9, z[2], z[3]*1.0e9, scansize

def map_ediss(f, size, lines, points, Q, invols=0.0, k=0.0, R=10.0e-9, nu=0.5, max_ind=0.0, fig=False):
    """This function draws a 2D map of dissipated energy from a raw .ibw tapping mode image
    
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
        if True a 2d map will be drawn
        
    Returns:
    ----------
    Ediss : np.array (2 dimensional)
        2D map of dissipated energy from the tapping mode image
    """
    x,y,z,ch,note = loadibw(f) #loading ibw file with Hanaul's magic function

    note =  {k.lower(): v for k, v in note.items()}  #making the dictionary to have all keys in lowercase
    ch = [u.lower() for u in ch]  #making channel to have all elements in lowercase
    if k == 0.0:
        k = note['springconstant']  
    if invols == 0.0:
        inv = note['invols']

 
    w01 = note['resfreq1']*2*np.pi     
    A0 = note['freeairamplitude']*inv
    Amp = z[1]*inv
    Phase = z[2]
    w, h = size(Amp[0]), size(Amp[0])
    Ediss = np.zeros((w,h))
   
    Ediss[:,:] = 1/2.*k*Amp[:,:]**2.0/Q*w01*(A0/Amp[:,:]*np.sin(Phase[:,:]*np.pi/180)-1.0)
    
    if fig:
        plt.imshow(Ediss, cm.bgo, origin = ';lower')
        plt.colorbar()
    return Ediss



def Subplot(rowcol,cell=(0,0),hr=1,vr=1,hp=0.1,vp=0.1,lp=0.125,rp=0.06,tp=0.1,bp=0.125):
    """
    Subplot(rowcol,cell=(0,0),hr=1,vr=1,hp=0.1,vp=0.1,lp=0.125,rp=0.06,tp=0.1,bp=0.125):
    
    rowcol: NumrowNumcol
    cell  : Cell number 'n' or index '(r,c) or ((r1,c1),(r2,c2))'
    hr,vr : 1 or (r1,r2,...)
    hp,vp,lp,rp,tp,bp : padding ratio
    sty = (1,1,0.1,0.1,0.125,0.06,0.1,0.125)
    """
    rn = rowcol/10
    cn = rowcol%10
    w0 = 1-((cn-1)*hp+rp+lp)
    h0 = 1-((rn-1)*vp+tp+bp)
    if hr==1: hr = np.ones(cn)
    else:     hr = np.array(hr)
    if vr==1: vr = np.ones(rn)
    else:     vr = np.array(vr)
    hr = float(w0)*hr/hr.sum()
    vr = float(h0)*vr/vr.sum()
    if type(cell) == int:
        cell = ((cell-1)/cn,(cell-1)%cn)
    if type(cell[0]) == int:
        cell = (cell,cell)
    
    x1 = cell[0][1]
    y1 = cell[0][0]
    x2 = cell[1][1]
    y2 = cell[1][0]
    
    x = lp
    y = bp
    w = hr[x1]
    h = vr[y1]
    for i in range(x1):
        x += hr[i]+hp
    for i in range(rn-1,y2,-1):
        y += vr[i]+vp
    for i in range(x2-x1):
        w += hr[x1+i+1]+hp
    for i in range(y2-y1):
        h += vr[y1+i+1]+vp
    
    return plt.axes([x,y,w,h])

def figannotate(text='a',fs=8,ff='helvetica',fw='bold',pos=(-0.02,1),ax=0,ha='right',va='top'):
    if ax==0:
        ax = plt.gcf().axes
    for i in range(len(ax)):
        ax[i].text(pos[0],pos[1],text,ha=ha,va=va,fontsize=fs,family=ff,weight=fw,transform=ax[i].transAxes)
        if len(text)>2:
            text = text[0]+'%s'%(chr(ord(text[1])+1))+text[2:]
        else:
            text = '%s'%(chr(ord(text[0])+1))+text[1:]
            

def df_np_boxplot(df_summary):
    """This function plots a multiple axis boxplot in a matplotlib fashion from a pandas dataframe"""
    df_nparray = df_summary.values
    # Filter data using np.isnan, to get rid of nan values
    mask = ~np.isnan(df_nparray)
    df_filtnan = [d[m] for d, m in zip(df_nparray.T, mask.T)]
    # Multiple box plots on one Axes
    fig, axes = plt.subplots()
    axes.boxplot(df_filtnan)
    
def multihisto(files, mini_hist, maxi_hist, nb, tolerance=10.0, plotit= False):
    """This function is usefult to plot two or more histograms in the same figure"""
    vels = []
    stiff = []
    for f in files:
        res = np.loadtxt(f, delimiter='\t', skiprows=1, dtype='str')
        approach_vel = res[:,3].astype(np.float)
        retract_vel = res[:,4].astype(np.float)
        ap_vel_av = np.mean(approach_vel)
        ret_vel_av = np.mean(retract_vel)  
        stiffness = res[:,2].astype(np.float)
        print('Total of %d force curves processed'%np.size(stiffness))
        stiffness_rest = stiffness[ (approach_vel > ap_vel_av*(1.0-tolerance/100.0) ) & (approach_vel < ap_vel_av*(1.0+tolerance/100.0))  & ( np.abs(retract_vel) > np.abs(ret_vel_av)*(1.0-tolerance/100.0)) & (np.abs(retract_vel) < np.abs(ret_vel_av)*(1.0+tolerance/100.0))  ]
        print('But only %d force curves passed the tolerance criterion'%np.size(stiffness_rest))
        #name_a = os.path.basename(os.path.normpath(os.getcwd()))                  
        name_b = 'vel %2.3f um/s'%(ap_vel_av*1.0e6)
        vels.append((ap_vel_av*1.0e6))  
        stiff.append(stiffness_rest)
        if plotit:
            plt.hist(stiffness_rest, bins=nb, range=(mini_hist,maxi_hist), alpha=0.5, ec='black', label=name_b)
            plt.legend(loc='best')
    return np.array(stiff), np.array(vels)

def arr2d_stats(arr2d, path='None', legend ='Data', units='Units', nb=100, mini_hist=0.0, maxi_hist=0.0):
    """This function receives a 2d array and returns a histogram and boxplot that are saved in a specific path
    
    Parameters:
    ----------
    arr2d : list of np.2darrays
        numpy 2D array that contains the data to be processed statistically
    path : str, optional
        path where the plots will be saved
    legend : str, optional
        legend to appear in the histogram and boxplot
    units : str, optional
        measurement performed (e.g., Young Modulus, MPa), this will appear in labels and name of files to be saved
    nb : int, optional
        number of bins
    mini_hist : float, optional
        minimum value in the x range limit of the histogram and boxplot
    maxi_hist : float, optional
        maximum value in the x range of the histogram and boxplot    
    """
    if len(arr2d) > 1:
        arr1d = []
        for arr in arr2d:
            arr1dim = arr.flatten()
            arr1d.append(arr1dim)
        arr1d = np.array(arr1d)
        arr1d = arr1d.flatten()
    else:
        arr1d = arr2d[0].flatten()        
    
    arr1d = arr1d[~np.isnan(arr1d)]  #getting rid of nan values
    
    Q3 = np.nanpercentile(arr1d, 75)
    Q1 = np.nanpercentile(arr1d, 25)
    IQR =  Q3 - Q1
    if mini_hist == 0.0:
        mini_hist = Q1 - 1.8*IQR
        
    if maxi_hist == 0.0:
        maxi_hist = Q3 + 1.8*IQR  
    
    plt.hist(arr1d, bins=nb, range=(mini_hist,maxi_hist), alpha=0.5, ec='black', label=legend)
    plt.legend(loc='best')
    plt.xlabel(units)
    plt.ylabel('Counts')
    if path != 'None':
        os.chdir(path)
    plt.savefig('Histogram %s.png'%units, bbox_inches='tight')
    plt.close()
    
    plt.boxplot(arr1d)
    plt.ylabel(units)
    plt.ylim(mini_hist, maxi_hist)
    plt.savefig('Boxplot %s.png'%units, bbox_inches='tight')
    plt.close()