#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 16:59:57 2018

@author: shaos@MPIBR
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.misc
#from scipy.optimize import fmin #for numerically finding minimum
from scipy.special import erfc #do not forget to divide by 2, see: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.special.erfc.html
from scipy.integrate import quad as quad
import itertools #in order to do nested loop: for i, j in itertools.product(range(x), range(y)):

from joblib import Parallel, delayed #for parallelizing loops, https://pythonhosted.org/joblib/parallel.html
import os#creating directories https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
eps= np.finfo(float).eps #machine precision
import time
starttime = time.time()

cpus = 32 #number of cpus used

N = 2     #number of cells

Nmax1 = 10.
Nmax2 = 10.

spon_fire1 = 0.2
spon_fire2 = 0.0

sig1 = 0.5
sig2 = 0.1


def P(k,n): #Poisson distribution
    return np.exp(-n)*n**k/scipy.math.factorial(k)

sig_s = 1

def P_s(s): #distribution of stimulus
    return 1./np.sqrt(2*np.pi)/sig_s*np.exp(-s*s/2./sig_s/sig_s)
    #return 1./(2.*sig_s)*np.exp(-abs(s)/sig_s)
    

#Ntrap=...  #number of steps for numerical integration, adapted to sigma, see below
xmin = -8 #integration borders (6 should be save as long as sigma and max threshold are both not greater than 1 (erfc(4)=1.5e-08)
xmax = 8
Ntrap = np.int(3/max(0.1, min(sig1,sig2))*1e2)  #number of points in numerical integral
ss=np.linspace(xmin,xmax,Ntrap) #numerical integral


def Pk2(k1, k2, R1, R2, t, sig1, sig2, ss): #marginilized probability for spike count k
    return np.trapz(psk2(k1, k2, R1, R2, t, sig1, sig2, ss) * P_s(ss), ss)


def psk2(k1, k2, R1, R2, t, sig1, sig2, s):
    r1 = spon_fire1 * R1
    r2 = spon_fire2 * R2
    if sig1 == 0:
        Hs1 = 0.5*(np.sign(s-t[0])+1.0)
    else:
        Hs1=0.5*erfc((t[0]-s)/sig1/np.sqrt(2))
    if sig2 == 0:
        Hs2 = 0.5*(np.sign(s-t[1])+1.0)
    else:
        Hs2=0.5*erfc((t[1]-s)/sig2/np.sqrt(2))
    return (Hs1*P(k1,R1)+(1-Hs1)*P(k1,r1)) * (Hs2*P(k2,R2)+(1-Hs2)*P(k2,r2))

def I2ON(t, sig1, sig2, R1, R2, ss):
    K = 2*max(R1, R2) + 10 #spike count over which is summed
    I2k = 0.
    for k1,k2 in itertools.product(np.arange(0,K), np.arange(0,K)):
        plogp = np.zeros(len(ss))
        for i in range(0, len(ss)):
            p_s = psk2(k1, k2, R1, R2, t, sig1, sig2, ss[i])
            if p_s > eps:
                plogp[i] = p_s * np.log(p_s)
        PlogP = 0.
        P_k = Pk2(k1, k2, R1, R2, t, sig1, sig2, ss)
        if P_k > eps:
            PlogP = P_k * np.log(P_k)
        I2k = I2k - PlogP + np.trapz(P_s(ss) * plogp, ss)
    return I2k

def mean_firing(t, sig1, sig2, R1, R2, ss):
    r1 = spon_fire1 * R1
    r2 = spon_fire2 * R2
    if sig1 == 0:
        Hs1 = 0.5*(np.sign(ss-t[0])+1.0)
    else:
        Hs1=0.5*erfc((t[0]-ss)/sig1/np.sqrt(2))
    if sig2 == 0:
        Hs2 = 0.5*(np.sign(ss-t[1])+1.0)
    else:
        Hs2=0.5*erfc((t[1]-ss)/sig2/np.sqrt(2))
    return np.trapz(P_s(ss) * (Hs1*R1 + (1-Hs1)*r1 + Hs2*R2 + (1-Hs2)*r2), ss)


tsteps = 61
trealm = 1.5

def calc_contour_info(j1, j2, sig1, sig2, R1, R2, ss):
    t = np.linspace(-trealm, trealm, tsteps)
    t1 = [t[j1], t[j2]]
    print "j1 = ",j1,", j2 = ",j2
    return I2ON(t1, sig1, sig2, R1, R2, ss)

Icontour = np.zeros((tsteps, tsteps))
avg_frate = np.zeros((tsteps, tsteps))

for j1 in range(0, tsteps):
    a = Parallel(n_jobs=cpus)(delayed(calc_contour_info)(j1, j2, sig1, sig2, Nmax1, Nmax2, ss) for j2 in range(0, tsteps))
    for j2 in range(0, tsteps):
        Icontour[j1,j2] = a[j2]
        t=np.linspace(-trealm, trealm, tsteps)
        avg_frate[j1,j2] = mean_firing([t[j1],t[j2]], sig1, sig2, Nmax1, Nmax2, ss)

print "done calculating"


endtime = time.time()
np.savez(str(endtime), sig1=sig1, sig2=sig2, Nmax1=Nmax1, Nmax2=Nmax2, 
         N=N, spon_fire1=spon_fire1, spon_fire2=spon_fire2, 
         Icontour=Icontour, avg_frate=avg_frate)
timerun =  endtime - starttime
timeprint = "time run:  " + str(timerun/60.) +" min;     " + str(timerun/3600.)+" hours;   "+ str(timerun/3600./24.)+" days"
print
print timeprint  
        
        
        