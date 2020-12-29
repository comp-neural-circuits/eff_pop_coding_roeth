# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 15:44:22 2020

@author: shaos
"""

import numpy as np
import matplotlib.pyplot as plt

output_dir = "figures/"
filename = "simulate_data_1"
#filename = "example_3types"

tsteps = 5
trealm = 1.5
t = np.linspace(-trealm,trealm,tsteps)

data = np.load(filename+".npz")
Icontour = data["Icontour"]
avg_frate = data["avg_frate"]
Inorm = Icontour/avg_frate

max_j1 = []
max_j2 = []
max_j3 = []


for j1 in range(1, tsteps-1):
    for j2 in range(1, tsteps-1):
        for j3 in range(1, tsteps-1):
            if Icontour[j1,j2,j3] >= Icontour[j1-1,j2,j3] and \
                Icontour[j1,j2,j3] >= Icontour[j1+1,j2,j3] and \
                Icontour[j1,j2,j3] >= Icontour[j1,j2-1,j3] and \
                Icontour[j1,j2,j3] >= Icontour[j1,j2+1,j3] and \
                Icontour[j1,j2,j3] >= Icontour[j1,j2,j3-1] and \
                Icontour[j1,j2,j3] >= Icontour[j1,j2,j3+1]:
                
                max_j1.append(j1)
                max_j2.append(j2)
                max_j3.append(j3)
                
                
I_all = np.zeros(len(max_j1))
avg_frate_all = np.zeros(len(max_j1))
I_norm_all = np.zeros(len(max_j1))
t_all = np.zeros((len(max_j1), 3)) 
 
for i in range(len(max_j1)):
    I_all[i] = Icontour[max_j1[i], max_j2[i], max_j3[i]]
    avg_frate_all[i] = avg_frate[max_j1[i], max_j2[i], max_j3[i]]
    I_norm_all[i] = I_all[i]/avg_frate_all[i]
    t_all[i, :] = [t[max_j1[i]], t[max_j2[i]], t[max_j3[i]]]
    

                
                