# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 17:18:14 2018

@author: Enrique Alejandro

This program makes figure ViscoMap of the manuscript:
Acquisition of high time-spatial resolution of biofilm mechanical properties     
"""


import os
#PATHS
ruta = os.path.realpath(__file__)
savefig_path = os.path.dirname(ruta)
rutafm = 'E:/Github/Biofilm_TimeFreq_MechProperties/Manuscript_Figures/RawData_Figures/SingleCell/FMsc1ums00'   #route of the force map where viscoelastic analysis will be performed
rutafm2 = ''  #map where only elastic analysis will be performed
ruta3dtopo = 'E:/Github/Biofilm_TimeFreq_MechProperties/Manuscript_Figures/RawData_Figures/SingleCell'   #route for 3D tapping topography
f1a = 'beforeFMsc1ums.ibw'   #file figure tapping topography



#importing generic libraries
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
from mayavi import mlab
from glob import glob

#importing customized libraries
import sys
syspath = 'e:/github/biofilm_timefreq_mechproperties'
sys.path.append(syspath)
from libs.forcespec.maps import map_viscoparams2, map_Jinv_theta2
from libs.imagean.imanalyze import zamphi, ImageFlatten
from libs.simulation.rheology import j_storage, j_loss
from libs.forcespec.maps import map2d_multiphys



###PARAMETERS FOR MATERIAL PROPERTY CALCULATION###
scansize =  500.0e-9  #image size in m, area where the force map was performed
max_indent = 5.0e-9
k = 2.98
invols = 89.48e-9
line = 64
point = 64
R = 10.0e-9
nu=0.5
gauss_filt = True
smooth = False #True
arms = 1
window = 10  #this should be proportional to the number of points in the T, Z, D arrays in the raw curves
snr_min = 100 #getting t_min at which signal to noise exceed certain criterion
length = 300   #approximate length of indentation and deflection arrays in the fitting region regardless the velocity
percentroi = 0.0
######END OF PARAMETERS INPUT######################

"""
#################################FIGURE CONTACT MODE TOPOGRAPHY and SHEAR MODULUS AT VEL VISCO######################################
nu=0.5
_, map_ym1, _,_, z0_1, _, _, _, velfs1 = map2d_multiphys(rutafm, line, point, 'None', invols, k, R, nu, max_indent)
map_ym1 *= 1.0e6  #in Pascals
z0f = ImageFlatten(z0_1, 'line')   #flattening contact mode image
map_shearMod = map_ym1/(2.0*(1.0+nu))

os.chdir(savefig_path)
norm_factor = np.nanpercentile(map_shearMod,98)
np.savetxt('NORM_FACTOR_STIFFNESS.txt', [norm_factor], header = 'NormalizationFactor(Pa)')


plt.figure()  #2D contact mode topography image
#im = plt.imshow(z0f, origin = 'lower', interpolation='spline16') #flattened
im = plt.imshow(z0_1, origin = 'lower', interpolation='nearest') #not flattened
plt.colorbar(im)
plt.axis('off')
plt.savefig('2DcontactTopo_%.2f ums_1.png'%(velfs1*1.0e6))
np.savetxt('map_shearMod_%.2f ums_1.txt'%(velfs1*1.0e6), map_shearMod)


plt.figure()  #normalized shear modulus image
map_shearMod /= norm_factor
smq1 = np.nanpercentile(map_shearMod,5)
smq3 = np.nanpercentile(map_shearMod,90)
imsm = plt.imshow(map_shearMod, vmin = smq1, vmax = smq3, cmap=cm.viridis, origin = 'lower', interpolation='nearest')
plt.axis('off')
plt.colorbar(imsm)
plt.savefig('ShearModulus_%.2fums_1.png'%(velfs1*1.0e6))



#################################FIGURE CONTACT MODE TOPOGRAPHY and SHEAR MODULUS AT HIGH VEL######################################
nu=0.5
if rutafm2 != '':  #if ruta fm2 is passed
    _, map_ym2, _,_, z0_2, _, _, _, velfs2 = map2d_multiphys(rutafm2, line, point, 'None', invols, k, R, nu, max_indent)
    map_ym2 *= 1.0e6  #in Pascals
    z0f2 = ImageFlatten(z0_2, 'line')   #flattening contact mode image
    map_shearMod2 = map_ym2/(2.0*(1.0+nu))
    
    os.chdir(savefig_path)
    
    plt.figure()  #2D contact mode topography image
    #im = plt.imshow(z0f2, origin = 'lower', interpolation='spline16') #flattened
    im = plt.imshow(z0_2, origin = 'lower', interpolation='nearest') #not flattened
    plt.colorbar(im)
    plt.axis('off')
    plt.savefig('2DcontactTopo_%.2f ums_1.png'%(velfs2*1.0e6))
    np.savetxt('map_shearMod_%.2f ums_1.txt'%(velfs2*1.0e6), map_shearMod)
    
    map_shearMod2 /= norm_factor
    plt.figure()  #shear modulus image
    imsm = plt.imshow(map_shearMod2, vmin = smq1, vmax = smq3, cmap=cm.viridis, origin = 'lower', interpolation='nearest')
    plt.axis('off')
    plt.colorbar(imsm)
    plt.savefig('ShearModulus_%.2fums_1.png'%(velfs2*1.0e6))


#################################################FIGURE 3D TAPPING###################################
os.chdir(ruta3dtopo)
topo1a,ampi1a, phi1a, zsen1a, scansize1a = zamphi(f1a)
topoc1a = topo1a-ampi1a
topocf1a = ImageFlatten(topoc1a, 'line')

quantile1_topo = np.percentile(topoc1a, 5)
quantile3_topo = np.percentile(topoc1a, 95)
imsize = scansize*1.0e9   #imnage size in nm
x, y = np.linspace(0.0,imsize,256), np.linspace(0.0,imsize,256)
X,Y = np.meshgrid(x,y)


mlab.figure(figure=1, bgcolor =(1,1,1))
mlab.mesh(Y,X, topoc1a,vmin = quantile1_topo, vmax = quantile3_topo, colormap='plasma')  #flattened
mlab.colorbar(orientation='vertical')
os.chdir(savefig_path)
mlab.savefig('3DtappingTopo.png')




#########################################Getting 2D map of viscoelastic parameters##################################
map_viscoel, vel, tmin, tmax = map_viscoparams2(rutafm, line, point, invols, k, R, nu) 

#saving arrays of Jg, tau and J
rows = line
columns = point
Jg_a, tau_a, J_a = np.zeros((rows,columns)), np.zeros((rows,columns)), np.zeros((rows,columns))
for i in range(rows):
        for j in range(columns):
            Jg_a[i,j] = map_viscoel[i][j][0]
            tau_a[i,j] = map_viscoel[i][j][1]
            J_a[i,j] = map_viscoel[i][j][2]

os.chdir(savefig_path)
np.savetxt('Jg_array.txt',Jg_a)
np.savetxt('tau_array.txt',tau_a)
np.savetxt('J_array.txt',J_a)


fig_params = np.array([[np.round(quantile1_topo,4)], [np.round(quantile3_topo,4)], [np.round(vel*1.0e6,4)], [tmin],[tmax], [ruta3dtopo]])
np.savetxt('2DVISCO_IMAGE_PARAMS.txt', (fig_params).T, delimiter = '\t', header='minTopo(nm)\tmaxTopo(nm)\tApVel(ums-1)\tTmin(s)\tTmax(s)\tDataPath', fmt='%s')
"""



##################################START WORKING HERE IF YOU HAVE SAVED BEFORE THE NLS PARAMETERS###########################
os.chdir(savefig_path)
Jg_a = np.loadtxt('Jg_array.txt')
tau_a = np.loadtxt('tau_array.txt')
J_a = np.loadtxt('J_array.txt')

figpar = np.genfromtxt('2DVISCO_IMAGE_PARAMS.txt', dtype='str')
tmin = float(figpar[3])
tmax = float(figpar[4])

N = 5  #number of points in time and frequency domain
y = np.linspace(np.log10(tmin), np.log10(tmax), N)
t_log = 10**y
omega_log = 2.0*np.pi/t_log
omega_log = omega_log[::-1]
 
#evaluating the viscoelastic parameters for specific time 
#and frequencies to calculate viscoelastic functions
jinv, angle = map_Jinv_theta2(t_log, Jg_a, J_a, tau_a)
#SAVING MAPS IN DIFFERENT POINTS IN THE TIME AND FREQUENCY DOMAIN
for i in range(N):
    np.savetxt('Jinv_t%d.txt'%i, jinv[i])
    np.savetxt('Theta_t%d.txt'%i, angle[i])
    
np.savetxt('TIME_FREQ_ARRS.txt', np.array( (t_log*1.0e3, omega_log/(2.0*np.pi))  ).T, header='time_log(ms)\tFreq_log(Hz)' )  

#plotting loss angle for the (0,0) pixel
i=0  #x-coordinate of the pixel whose loss angle you want to calculate
j=0 #y-coordinate of the pixel whose loss angle you want to calculate
y_pixel = np.linspace(np.log10(tmin), np.log10(tmax), 1000)
tlog_pixel = 10**y_pixel
omega_pixel = 2.0*np.pi/tlog_pixel
omega_pixel=omega_pixel[::-1]
J_prime_pixel = j_storage(omega_pixel, Jg_a[i,j], J_a[i,j], tau_a[i,j])
J_biprime_pixel = j_loss(omega_pixel, Jg_a[i,j], J_a[i,j], tau_a[i,j])
loss_angle_pixel = np.arctan(J_biprime_pixel/J_prime_pixel)*180.0/np.pi

J_prime = j_storage(omega_log, Jg_a[i,j], J_a[i,j], tau_a[i,j])
J_biprime = j_loss(omega_log, Jg_a[i,j], J_a[i,j], tau_a[i,j])
loss_angle = np.arctan(J_biprime/J_prime)*180.0/np.pi


plt.figure(figsize=(6,1.6))
plt.plot(omega_pixel/(2.0*np.pi),loss_angle_pixel)
plt.plot(omega_log/(2.0*np.pi),loss_angle, marker='s', alpha=0.8, markersize=10, color='y', lw=0)
plt.xscale('log')
plt.xlim(min(omega_pixel)/(2.0*np.pi)/10.0, max(omega_pixel)/(2.0*np.pi)*10.0)
plt.ylabel(r'$\theta (\omega), deg$', fontsize = 25)
plt.xlabel(r'$\omega /2 \pi, Hz$', fontsize = 25)
plt.savefig('LossAngle.png', bbox_inches='tight')


######################################START HERE IF YOU HAVE SAVED THE Jinv and Theta maps#########################

######################################GETTING QUANTILE AVERAGES OF JINV AND LOSS ANGLE FOR 5 FRAMES######################################
N = 5   #this is the number of Jinv text files, double check it

mini_Jinv = []
maxi_Jinv = []
jinverse_a = []

for i in range(0,N):
    jinverse = np.loadtxt('Jinv_t%d.txt'%i)
    jinverse_a.append(jinverse)
    mini_Jinv.append(np.nanpercentile(jinverse, 5))
    maxi_Jinv.append(np.nanpercentile(jinverse, 95))
miniav = np.mean(np.array(mini_Jinv))
maxiav = np.mean(np.array(maxi_Jinv))


#PLOTTING AND SAVING MAPS OF LOSS ANGLE
mini_theta = []
maxi_theta = []
theta_a = []
for i in range(0,N):
    theta = np.loadtxt('Theta_t%d.txt'%i)
    theta_a.append(theta)
    mini_theta.append(np.nanpercentile(theta, 5))
    maxi_theta.append(np.nanpercentile(theta, 95))
miniav_theta = np.mean(np.array(mini_theta))
maxiav_theta = np.mean(np.array(maxi_theta))

    
os.chdir(savefig_path)
quantile_params = np.array([[np.round(miniav,4)], [np.round(maxiav,4)], [np.round(miniav_theta,4)], [np.round(maxiav_theta,4)], [ruta3dtopo]])
np.savetxt('QUANTILE_PARAMS.txt', (quantile_params).T, delimiter = '\t', header='minJinv(Pa)\tmaxJinv(Pa)\tminiTheta(deg)\tmaxiTheta(deg)\tDataPath', fmt='%s')




#########################################START HERE IF EVERYTHING IS CALCULATED ALREADY###################################
topoflat = False
norm_fact = ''#0.9237683708993273*1.0e9
shearmin = 0.05
shearmax = 0.8
minJ = 0.1
maxJ = 0.95
degree_sign= u'\N{DEGREE SIGN}'


os.chdir(savefig_path)
files = glob('*.txt')

f_shear = []
for f in files:
    if 'map_shear' in f:
        f_shear.append(f)

vel = []
for k in range(len(f_shear)):
    velo = f_shear[k].strip('map_shearMod')
    velo = velo.replace(' ums_1.txt','')
    vel.append(velo)



####################################################PLOTTTING SHEAR MODULUS MULTIVEL################################
os.chdir(ruta3dtopo)
topo1a,ampi1a, phi1a, zsen1a, scansize1a = zamphi(f1a)
topoc1a = topo1a-ampi1a

if topoflat:
    topocf1a = ImageFlatten(topoc1a, 'line')    
else:
    topocf1a = topoc1a
quantile1_topo = np.percentile(topocf1a, 5)
quantile3_topo = np.percentile(topocf1a, 98)
    

os.chdir(savefig_path)

#GETTING NORMALIZATION FACTOR FOR STIFFNESS
if norm_fact == '':
    norm = np.loadtxt(f_shear[0])
    norm_fact = np.percentile(norm.flatten(),98)*1.5
fig, axes = plt.subplots(nrows=1, ncols= len(f_shear)+1, figsize=(12,4))
fig.subplots_adjust(wspace=0.05, hspace=0.2)
i=0
flag = 0
for i in range(len(f_shear)+1):
    if flag == 0:
        topo = axes[i].imshow(topocf1a, vmin = quantile1_topo, vmax = quantile3_topo, cmap=cm.plasma, origin = 'lower', interpolation='nearest')#, aspect='auto')
        axes[i].axis('off')
        axes[i].set_title(r'$Tapping \, Mode \, Height$', fontsize=15, fontweight='bold')
        flag = 1     
    else:
        shear = np.loadtxt(f_shear[i-1])
        shear /= norm_fact
        imb_shear = axes[i].imshow(shear, vmin = shearmin, vmax = shearmax, cmap=cm.viridis, origin = 'lower', interpolation='nearest')#, aspect='auto')
        axes[i].axis('off')
        axes[i].set_title(r'$Stiffness \, Map \, \, at \,  %s \, \mu m/s$'%(vel[i-1]), fontsize=15, fontweight='bold')


cb_ax = fig.add_axes([0.92, 0.16, 0.02, 0.7])
cbar = fig.colorbar(imb_shear, cax=cb_ax)
cb_ax.yaxis.set_ticks(np.arange(0, 1.1, 1.0))
cb_ax.set_yticklabels([str(shearmin), str(shearmax)], fontweight='bold')


cb_ax2 = fig.add_axes([0.08, 0.16, 0.02, 0.7])
cbar2 = fig.colorbar(topo, cax=cb_ax2)
cb_ax2.yaxis.set_ticks(np.arange(0, 1.1, 1.0))
cb_ax2.yaxis.set_ticks_position('left')
maxi = round(np.abs(quantile3_topo - quantile1_topo),0)
maxi = str(maxi) + ' nm'
cb_ax2.set_yticklabels(['0 nm', maxi], fontweight='bold')

plt.savefig('ShearMod_multivel.png', bbox_inches='tight')


###########################################################PLOTTING JIV AND LOSS ANGLE############################
os.chdir(savefig_path)

tf = np.loadtxt('TIME_FREQ_ARRS.txt')
tlog= tf[:,0]
flog=tf[::-1,1]   #from high freq to low freq

N = 5
os.chdir(savefig_path)
quantile_par = np.genfromtxt('QUANTILE_PARAMS.txt', dtype='str')
minJinv = float(quantile_par[0])/norm_fact
maxJinv = float(quantile_par[1])/norm_fact
minTheta = float(quantile_par[2])
maxTheta = float(quantile_par[3])


fig, axes = plt.subplots(nrows=2, ncols= N, figsize=(17,7))
fig.subplots_adjust(wspace=0.05, hspace=0.2)

#PLOTTING JINVERSE
flag = 0
for i in range(0,N):
    jinverse = np.loadtxt('Jinv_t%d.txt'%i)
    jinverse /= norm_fact
    imb_jinv = axes[0,i].imshow(jinverse, vmin = minJ, vmax = maxJ, cmap=cm.viridis, origin = 'lower', interpolation='nearest', aspect='auto')
    axes[0,i].axis('off')
    axes[0,i].set_title('%.2f ms'%tlog[i], fontsize=20, fontweight='bold')

# add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.37
#see details in: https://jdhao.github.io/2017/06/11/mpl_multiplot_one_colorbar/
cb_ax1 = fig.add_axes([0.92, 0.54, 0.02, 0.365])
cbar1 = fig.colorbar(imb_jinv, cax=cb_ax1)
cb_ax1.yaxis.set_ticks(np.arange(0, 1.1, 1.0))
cb_ax1.set_yticklabels([str(minJ), str(maxJ)], fontweight='bold')


#PLOTTING THETA
flag = 0
for i in range(0,N):
    theta = np.loadtxt('Theta_t%d.txt'%((N-1)-i))
    imb_theta = axes[1,i].imshow(theta, vmin = minTheta, vmax = maxTheta, cmap=cm.coolwarm, origin = 'lower', interpolation='nearest', aspect='auto')
    axes[1,i].axis('off')
    axes[1,i].set_title('%.0f Hz'%flog[i], fontsize=20, fontweight='bold')

# add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.37
#see details in: https://jdhao.github.io/2017/06/11/mpl_multiplot_one_colorbar/
cb_ax2 = fig.add_axes([0.92, 0.1, 0.02, 0.365])
cbar2 = fig.colorbar(imb_theta, cax=cb_ax2)
cb_ax2.yaxis.set_ticks(np.arange(0, 1.1, 1.0))
cb_ax2.set_yticklabels([str(int(round(minTheta)))+degree_sign, str(int(round(maxTheta)))+degree_sign], fontweight='bold')  

plt.savefig('MultiVel_Jinv_theta.png', bbox_inches='tight')


############################################3D TOPOGRAPHY, SAVE MANUALLY#########################################

imsize = scansize*1.0e9   #imnage size in nm
x, y = np.linspace(0.0,imsize,256), np.linspace(0.0,imsize,256)
X,Y = np.meshgrid(x,y)
mlab.figure(figure=1, bgcolor =(1,1,1))
mlab.mesh(X,Y, topocf1a,vmin = quantile1_topo, vmax = quantile3_topo, colormap='plasma')







