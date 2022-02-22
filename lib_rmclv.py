import numpy as np
import math
from numba import njit
import astropy.constants as const
import scipy

def vac2air(wv_vac):
    s = 10.**4 /wv_vac
    n = 1.+0.0000834254+0.02406147/(130.-s**2)+0.00015998/(38.9-s**2)
    wv_air=wv_vac/n
    return wv_air

@njit #mapping the pixel inside the stellar disk
def cir_mask(x,y):
    n=[]
    for i in range(len(x)):
        for j in range(len(y)):
            if x[i]**2+y[j]**2-1.<=1e-53:
                n.append(np.array([x[i],y[j]]))
    return n

#calculating the 
def v_dopshadow(a_R,lambd_,vsini,ip_,phase):
    ip=ip_*np.pi/180.
    lambd=lambd_*np.pi/180.
    xp= a_R*np.sin(2.*np.pi*phase)
    yp= -a_R*np.cos(2.*np.pi*phase)*np.cos(ip)

    x_= xp*np.cos(lambd)-yp*np.sin(lambd)
    y_= xp*np.sin(lambd)+yp*np.cos(lambd)

    z_= np.sqrt(1.-x_**2-y_**2)
    r_=np.sqrt(x_**2+y_**2)
    vp= x_*vsini
    return vp,x_,y_,z_,r_
def line_funct(x,x1,y1,x2,y2):
    y=(y2-y1)/(x2-x1)*(x-x1)+y1
    return y

def dx_(Rp_Rs,x1,y1,x2,y2):
    return  np.abs(Rp_Rs*(y2-y1)/np.sqrt((x2-x1)**2+(y2-y1)**2))

def dy_(dx,x1,y1,x2,y2):
    return np.abs(dx*(y2-y1)/(x2-x1))
  
def time_exp_mask(xy,x1,y1,x2,y2,Rp_Rs):
    
    dx=np.abs(dx_(Rp_Rs,x1,y1,x2,y2))
    dy=np.abs(dy_(dx,x1,y1,x2,y2))

    x1u,y1u= x1-dx,y1+dy
    x1d,y1d= x1+dx,y1-dy
    x2u,y2u= x2-dx,y2+dy
    x2d,y2d= x2+dx,y2-dy  
    
    mask=np.ones(len(xy))>0
    for i in range(len(xy)):
        if xy[i][1]<=line_funct(xy[i][0],x1u,y1u,x2u,y2u) and \
           xy[i][1]>=line_funct(xy[i][0],x1d,y1d,x2d,y2d) and \
           xy[i][1]>=line_funct(xy[i][0],x1u,y1u,x1d,y1d) and \
           xy[i][1]<=line_funct(xy[i][0],x2u,y2u,x2d,y2d):
            mask[i]=1<0
    return mask

@njit #masking the pixel overlapped with the planetary disk
def planet_mask(xy,xp,yp,Rp_Rs):
    mask=np.ones(len(xy))>0
    for i in range(len(xy)):
        if np.sqrt((xy[i][0]-xp)**2+(xy[i][1]-yp)**2) <= Rp_Rs:
            mask[i]=1<0
    return mask

def spec_stelpix(mu_specfunc,mu,wv_stellar,vpix,wvdata,wvmask):
    wv_shift=wv_stellar[wvmask]*(1.0 + (vpix)*1000./const.c.value)
    return scipy.interpolate.interp1d(wv_shift,mu_specfunc(wv_stellar[wvmask],mu),
                                      kind='linear',bounds_error=False)(wvdata)
