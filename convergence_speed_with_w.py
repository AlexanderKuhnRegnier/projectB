# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 17:18:12 2016

@author: Alexander Kuhn-Regnier
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from time import clock
from AMR_EDM import create_EDM_system,create_EDM_system_from_grid
from AMR_system import AMR_system,Grid
import os
from copy import deepcopy

factor = 30
k = 1.
#with k=1, and a dust size of 0.1 mm (100e-6 m),
#the factor has to be at least 300 in order to be able to
#depict the dust accurately!
system,dust_size,position = create_EDM_system((26*factor,3*factor),k,
                         size=(260*(1+2*k),30*(1+2*k)),
                         small_sources=True,dust_pos=350.,
                         dust_size=1e-1)
                         
def convergence_time(system,w,tol=1e2,max_time=60,max_iter=100000):
    system.SOR(w=w,tol=tol,max_time=max_time,max_iter=max_iter)    
    return system.times[-1]

w_array = np.linspace(1.01,1.99,30)
times = []    
for w in w_array:
    times.append(convergence_time(deepcopy(system),w,tol=1e2))


plt.figure()
plt.plot(w_array,times,linestyle='',marker='o')
plt.margins(0.1)
plt.show()