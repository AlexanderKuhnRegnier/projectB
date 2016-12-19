# -*- coding: utf-8 -*-
"""
Created on Dec 15

@author: Alexander Kuhn-Regnier
"""
import numpy as np
from time import clock
from copy import deepcopy
from AMR_system import AMR_system,Grid,build_from_segments
import matplotlib.pyplot as plt
import pandas as pd

def create_sources(Ns):
    xh,yh = build_from_segments(((1,Ns[0]),),((1,Ns[1]),))
    grid = Grid(xh,yh)
    grid.rectangle(1,(0.5,0.5),0.4,0.4)
    grid.rectangle(-1,(0.2,0.4),0.02,0.02)
    return grid
    
def precondition(orig_Ns,precon_Ns,tol,precon_tol,max_time=20,
                 create_sources=create_sources):

    times = []
    grid = create_sources(orig_Ns)
    #grid.show(color=(0,0,0,0.1))
    test = AMR_system(grid)
    
    raw = deepcopy(test)
    
    start = clock()
    
    raw.SOR(max_time=max_time,tol=tol,verbose=False)
    
    rawtime = clock()-start
    times.append(rawtime)
    ######
    #raw.show(title='raw result')
    #####
    
    #####
    #test.show(title='unsolved before')
    #####
    start = clock()
    
    precon_grid = create_sources(precon_Ns)
    precon_system = AMR_system(precon_grid)
    precon_system.SOR(max_time=max_time,tol=precon_tol,verbose=False)
    test.interpolate(precon_system)
    
    precondtime = clock()-start
    times.append(precondtime)
    
    #####
#    test.show(title='after preconditioning')
    test.show()
    #####
    test.SOR(max_time=max_time,tol=tol,verbose=False)
    #####
    
    time2 = clock()-start
    times.append(time2)
    
    #test.show(title='result with precon')
    test.show()
    #####
    
    
    plt.figure()
    plt.imshow(np.abs(raw.potentials-test.potentials).T,origin='lower')
    plt.colorbar()
#    plt.title('raw - test')
    plt.tight_layout()
    plt.show()
    
    print 'Ns',orig_Ns
    print 'precon Ns',precon_Ns
    print 'tol',tol
    print 'precon tol',precon_tol
    print 'raw time', times[0]
    print 'preconditioning time:', times[1]
    print 'total time with preconditioning:', times[2]

    return orig_Ns,precon_Ns,tol,precon_tol,times[0]-times[2]


#Ns = (500,500)
#tol = 1e1
#number = 10
#precon_tols = np.linspace(1e-1,1e5,number)
#precon_Ns_x = np.linspace(10,400,number)
#precon_Ns_y = np.linspace(10,400,number)
#precon_Ns_list = [(i,j) for i,j in zip(precon_Ns_x,precon_Ns_y)]
#
#results = []                  
#for precon_Ns in precon_Ns_list:
#    for precon_tol in precon_tols:
#        results.append(precondition(Ns,precon_Ns,tol,precon_tol))
#    
#results_frame = pd.DataFrame(results)