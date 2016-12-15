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

filename = 'results.csv'

def append_to_file(data,filename=filename):
    data_str = map(str,data)
    to_write = ','.join(data_str)
    to_write += '\n'
    with open(filename,'a') as f:
        f.write(to_write)
                         
def convergence_time(system,w,tol=1e2,max_time=60,max_iter=100000):
    system.SOR(w=w,tol=tol,max_time=max_time,max_iter=max_iter,verbose=False)    
    return system.times[-1]

def sor_parameter_loop(iterations = 20, factor = 50, max_time = 60.):
    k = 1.
    #with k=1, and a dust size of 0.1 mm (100e-6 m),
    #the factor has to be at least 300 in order to be able to
    #depict the dust accurately!
    system,dust_size,position = create_EDM_system((26*factor,3*factor),k,
                             size=(260*(1+2*k),30*(1+2*k)),
                             small_sources=True,dust_pos=350.,
                             dust_size=1e-1)    
    w_array = np.linspace(1.05,1.98,iterations)
    times = []    
    last_iterations = []
    tol = 0 
    for i,w in enumerate(w_array):
        if i == 0:
            #at low w, the time is expected to be the highest, therefore 
            #set the tolerance for ALL the runs based on the lowest tolerance 
            #achieved during the *max_time* set above.
            sys_copy = deepcopy(system)
            times.append(convergence_time(sys_copy,w,tol=tol,max_time=max_time))
            tol = sys_copy.errors[-2]
        else:
            sys_copy = deepcopy(system)
            times.append(convergence_time(sys_copy,w,tol=tol))
        last_iterations.append(sys_copy.last_iteration)
            
    min_index = np.argmin(times)
    return (w_array[min_index],times[min_index],tol,last_iterations[min_index],
            factor)

def execute_main_loop(iterations = 20, max_factor = 200,
                      max_time = 100.):
    while True:
        results = sor_parameter_loop(iterations = iterations, 
                                     factor = np.random.randint(5,max_factor),
                                     max_time = np.random.random()*max_time)
        append_to_file(results)
        
def read_results(filename=filename):
    results = np.loadtxt(filename=filename,delimiter=',')
    