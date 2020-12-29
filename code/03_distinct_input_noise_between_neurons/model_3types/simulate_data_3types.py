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
from scipy.optimize import fmin #for numerically finding minimum
from scipy.special import erfc #do not forget to divide by 2, see: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.special.erfc.html
from scipy.integrate import quad as quad
import itertools #in order to do nested loop: for i, j in itertools.product(range(x), range(y)):

from joblib import Parallel, delayed #for parallelizing loops, https://pythonhosted.org/joblib/parallel.html
import os#creating directories https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
eps= np.finfo(float).eps #machine precision
import time
starttime = time.time()

cpus = 64 #number of cpus used

N = 3     #number of cells

Nmax1 = 13.8
Nmax2 = 13.8
Nmax3 = 13.8

spon_fire1 = 0.180
spon_fire2 = 0.087
spon_fire3 = 0.019

sig1 = 0.328
sig2 = 0.398
sig3 = 0.571


def P(k,n): #Poisson distribution
    return np.exp(-n)*n**k/scipy.math.factorial(k)

sig_s = 1

def P_s(s): #distribution of stimulus
    return 1./np.sqrt(2*np.pi)/sig_s*np.exp(-s*s/2./sig_s/sig_s)
    #return 1./(2.*sig_s)*np.exp(-abs(s)/sig_s)
    

#Ntrap=...  #number of steps for numerical integration, adapted to sigma, see below
xmin = -8 #integration borders (6 should be save as long as sigma and max threshold are both not greater than 1 (erfc(4)=1.5e-08)
xmax = 8
Ntrap = np.int(3/min(sig1,sig2)*1e2)  #number of points in numerical integral
ss=np.linspace(xmin,xmax,Ntrap) #numerical integral


def Pk3(k1, k2, k3, R1, R2, R3, t, sig1, sig2, sig3, ss): #marginilized probability for spike count k
    return np.trapz(psk3(k1, k2, k3, R1, R2, R3, t, sig1, sig2, sig3, ss) * P_s(ss), ss)


def psk3(k1, k2, k3, R1, R2, R3, t, sig1, sig2, sig3, s):
    r1 = spon_fire1 * R1
    r2 = spon_fire2 * R2
    r3 = spon_fire3 * R3
    Hs1 = 0.5 * erfc((t[0]-s)/sig1/np.sqrt(2))
    Hs2 = 0.5 * erfc((t[1]-s)/sig2/np.sqrt(2))
    Hs3 = 0.5 * erfc((t[2]-s)/sig3/np.sqrt(2))
    return (Hs1*P(k1,R1)+(1-Hs1)*P(k1,r1)) * (Hs2*P(k2,R2)+(1-Hs2)*P(k2,r2)) * (Hs3*P(k3,R3)+(1-Hs3)*P(k3,r3))

def I3ON(t, sig1, sig2, sig3, R1, R2, R3, ss):
    K = 2*max(R1, R2, R3) + 10 #spike count over which is summed
    I3k = 0.
    for k1,k2,k3 in itertools.product(np.arange(0,K), np.arange(0,K), np.arange(0,K)):
        plogp = np.zeros(len(ss))
        for i in range(0, len(ss)):
            p_s = psk3(k1, k2, k3, R1, R2, R3, t, sig1, sig2, sig3, ss[i])
            if p_s > eps:
                plogp[i] = p_s * np.log(p_s)
        PlogP = 0.
        P_k = Pk3(k1, k2, k3, R1, R2, R3, t, sig1, sig2, sig3, ss)
        if P_k > eps:
            PlogP = P_k * np.log(P_k)
        I3k = I3k - PlogP + np.trapz(P_s(ss) * plogp, ss)
    return I3k

def mean_firing(t, sig1, sig2, sig3, R1, R2, R3, ss):
    r1 = spon_fire1 * R1
    r2 = spon_fire2 * R2
    r3 = spon_fire3 * R3
    Hs1 = 0.5 * erfc((t[0]-ss)/sig1/np.sqrt(2))
    Hs2 = 0.5 * erfc((t[1]-ss)/sig2/np.sqrt(2))
    Hs3 = 0.5 * erfc((t[2]-ss)/sig3/np.sqrt(2))
    return np.trapz(P_s(ss) * (Hs1*R1 + (1-Hs1)*r1 + Hs2*R2 + (1-Hs2)*r2 + Hs3*R3 + (1-Hs3)*r3), ss)


tsteps = 5
trealm = 1.5

def calc_contour_info(j, sig1, sig2, sig3, R1, R2, R3, ss):
    t = np.linspace(-trealm, trealm, tsteps)
    j3 = int(np.ceil((j+1.0)/tsteps/tsteps)) - 1
    j2 = int(np.ceil((j+1.0 - j3*tsteps*tsteps)/tsteps)) - 1
    j1 = j - j3*tsteps*tsteps - j2*tsteps
    t1 = [t[j1], t[j2], t[j3]]
    print "j = ",j
    return I3ON(t1, sig1, sig2, sig3, R1, R2, R3, ss)

Icontour = np.zeros((tsteps, tsteps, tsteps))
avg_frate = np.zeros((tsteps, tsteps, tsteps))

a = Parallel(n_jobs=cpus)(delayed(calc_contour_info)(j, sig1, sig2, sig3, Nmax1, Nmax2, Nmax3, ss) for j in range(0, pow(tsteps, 3)))
for j in range(0, pow(tsteps, 3)):
   t=np.linspace(-trealm, trealm, tsteps)
   j3 = int(np.ceil((j+1.0)/tsteps/tsteps)) - 1
   j2 = int(np.ceil((j+1.0 - j3*tsteps*tsteps)/tsteps)) - 1
   j1 = j - j3*tsteps*tsteps - j2*tsteps
   Icontour[j1,j2,j3] = a[j]
   avg_frate[j1,j2,j3] = mean_firing([t[j1],t[j2],t[j3]], sig1, sig2, sig3, Nmax1, Nmax2, Nmax3, ss)

print "done calculating"


endtime = time.time()
np.savez("simulate_data_1", sig1=sig1, sig2=sig2, sig3=sig3, Nmax1=Nmax1, Nmax2=Nmax2, Nmax3=Nmax3,
         N=N, spon_fire1=spon_fire1, spon_fire2=spon_fire2, spon_fire3=spon_fire3, 
         Icontour=Icontour, avg_frate=avg_frate)
timerun =  endtime - starttime
timeprint = "time run:  " + str(timerun/60.) +" min;     " + str(timerun/3600.)+" hours;   "+ str(timerun/3600./24.)+" days"
print
print timeprint      
        
        
