#!/usr/bin/env python
# coding: UTF-8
import numpy as np
import scipy.io
import h5py
import astropy.constants as const
from tqdm import tqdm
from scipy import interpolate
from lib_rmclv import *
from PyAstronomy import pyasl
import scipy.interpolate as sci

dir_data            = '/data/genstore1/skn/Projects/WASP-33b/CAHA/stev/n1/raw/'
dir_result          = '/data/genstore1/skn/Projects/WASP-33b/CAHA/stev/model/'
# inputfile           = "wasp33b-n1-vis-21july.h5"
# inputfile           = "wasp33b-n2-vis-7july.h5"
inputfile           = "wasp33b-n1-nir-6aug.h5"
outputfile          = str(inputfile[:-3])+"_SME_model.h5"
input_stellarmodel  = '/data/genstore1/skn/DopplerShadow/SME_model/wasp33b-car-vis.out'

# orderrange          = np.arange(0,21,1)
orderrange          = np.arange(0,50,1)

#Planetary parameters
#WASP-33b
Rp_Rs = 1.679*const.R_jup.value/1.509/const.R_sun.value
a_R   = 3.69
vsys  = -3.69

#from MCMC
vsini = 80.83633601
lambd_= -112.09224949
ip_   =  90.05682678

P=1.2198675
T0=2458792.636403

#KELT-20b
# vsini = 113.0
# Rp_Rs = 0.1175
# a_R   = 7.66
# lambd_= 0
# ip_   = np.arccos(0.481/7.660)/np.pi*180
# vsys  = -22.

#mascara 1b
# vsini = 109.0
# Rp_Rs = 1.5*const.R_jup.value/(2.1*const.R_sun.value)
# a_R   = 0.043*const.au.value/(2.1*const.R_sun.value)
# lambd_= 69.5
# b=0.2
# ip_   = np.arccos(b/a_R)/np.pi*180
# vsys  = -22.

# #kelt-9b
# vsini = 111.4
# Rp_Rs = 1.891*const.R_jup.value/(2.362*const.R_sun.value)
# a_R   = 0.03462*const.au.value/(2.362*const.R_sun.value)
# lambd_= -84.8
# b     = 0.177
# ip_   = np.arccos(b/a_R)/np.pi*180
# vsys  = -20.567


########################

pixsize=0.002

x           = np.arange(-1.,1.+pixsize,pixsize)
y           = np.arange(-1.,1.+pixsize,pixsize)

xy          = np.array(cir_mask(x,y))[1:]
mu_sample   = np.sqrt(1.-xy[:,0]**2-xy[:,1]**2)
vsini_sample= xy[:,0]*vsini


#Loads SME model and interpolates it
stellar_spec    = scipy.io.readsav(input_stellarmodel)
wv_stellar      = vac2air(stellar_spec['sme']['wint'][0])*(1.+vsys*1000./const.c.value)
mu_list         = stellar_spec['sme']['mu'][0]
spec_           = stellar_spec['sme']['sint'][0]
ind_mu          = np.argsort(mu_list)
mu_mesh,wv_mesh = np.meshgrid(mu_list[ind_mu],wv_stellar)
mu_specfunc     = scipy.interpolate.interp2d(wv_stellar,mu_list[ind_mu],spec_[ind_mu],
                                       kind='linear',bounds_error=False)

#Loads observational data
h5f_raw  = h5py.File(str(dir_data)+str(inputfile), 'r')
# phase    = h5f_raw['phase'][:]
rvcor    = h5f_raw['rvcor'][:]

#Loads observational data
h5f_raw  = h5py.File(str(dir_data)+str(inputfile), 'r')
# phase    = h5f_raw['phase'][:]
rvcor    = h5f_raw['rvcor'][:]
bjd      = h5f_raw['bjd'][:]


ph=(np.array(bjd)-(T0-2400000.))/(P) % 1
phase=[]
for p in range(len(ph)):
    if ph[p] >0.5:
        phase.append(ph[p]-1.)
    else:
        phase.append(ph[p])
phase=np.array(phase)  

vp_mid ,x_p_mid ,y_p_mid ,z_p_mid ,r_p_mid = v_dopshadow(a_R, lambd_, vsini, ip_, phase)

with h5py.File(str(dir_result)+str(outputfile), 'w') as h5f_stelldisk:
    for order in tqdm(orderrange):
        print (order)
        wvdata              = h5f_raw["wv-order-"+str(order)][:]
        wvsample            = np.arange(wvdata[0]-3.,wvdata[-1]+3.,0.015)
        wvmask              = (wv_stellar>wvsample[0]-5.)*(wv_stellar<wvsample[-1]+5.)
        stellar_specmodel   = np.zeros((len(mu_sample),len(wvsample)))
        for i in tqdm(range(len(mu_sample))):
            stellar_specmodel[i]=spec_stelpix(mu_specfunc, mu_sample[i], wv_stellar, vsini_sample[i], wvsample, wvmask)
        specfin=np.dot(stellar_specmodel.transpose(), np.ones(len(mu_sample)))

        model=np.zeros((len(phase),len(wvsample)))
        for i in tqdm(range(len(phase))):  

            x,y=x_p_mid[i], y_p_mid[i]
            maskplan = planet_mask(xy,x,y,Rp_Rs)
            
        
            model[i]= specfin-np.dot(stellar_specmodel[~maskplan].transpose(), np.ones(len(mu_sample[~maskplan])))

        h5f_stelldisk.create_dataset('sme_order_'+str(order),data=model,compression="gzip", compression_opts=9)
        h5f_stelldisk.create_dataset('wv_order_'+str(order),data=wvsample,compression="gzip", compression_opts=9)
        
        model=0
        specfin=0
        stellar_specmodel=0
