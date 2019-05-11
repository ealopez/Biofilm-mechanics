# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 09:40:27 2019

@author: Enrique Alejandro
"""

import numpy as np
from glob import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
ruta = os.getcwd()

import sys
syspath = 'e:/github/modelfree_lossangle'
sys.path.append(syspath)
from libs.forcespec.fdanalysis import loadibw


def cpd(f, path='', ampinvols=0.0):
    """This function returns information to perform 2D draws of contact potential
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
    
    return z[ch.index('nappotentialretrace')], z[ch.index('heightretrace')], z[ch.index('amplituderetrace')], z[ch.index('phaseretrace')], scansize#z[0]*1.0e9, z[1]*ampinvols*1.0e9, z[2], z[3]*1.0e9, scansize




def cpdav_t(path):
    #This function receives a path where images of KPFM are stored and returns the average CPD in time
    
    os.chdir(path)
    files = glob('*.ibw')
    files.sort()
    cpd_av = []
    cpd_std = []
    time_a = []
    i = 0
    for f in files:
        x,y,z,ch,note = loadibw(f) #loading ibw file with Hanaul's magic function
        note =  {k.lower(): v for k, v in note.items()}  #making the dictionary to have all keys in lowercase
        ch = [u.lower() for u in ch]  #making channel to have all elements in lowercase
        if i == 0:  #set initial time
            time_0 = note['date']+' '+note['time']
            time_0 = pd.datetime.strptime(time_0,'%Y-%m-%d %I:%M:%S%p')
        
        CPD = z[ch.index('nappotentialretrace')].flatten()
        time = note['date']+' '+note['time']
        time = pd.datetime.strptime(time,'%Y-%m-%d %I:%M:%S%p')
        t = time - time_0
        time_a.append( t.total_seconds() )
        cpd_av.append(np.mean(CPD))
        
        cpd_std.append(np.std(CPD))
        i +=1
        
    cpd_av = np.array(cpd_av)
    cpd_std = np.array(cpd_std)
    ts = np.array(time_a)
    return cpd_av, cpd_std, ts, files


  
    