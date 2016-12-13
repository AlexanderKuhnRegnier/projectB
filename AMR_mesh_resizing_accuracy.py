# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 18:52:11 2016

@author: Alexander Kuhn-Régnier
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from time import clock
from copy import deepcopy
import os
plt.ioff()
from AMR_EDM import create_EDM_system_from_grid
from AMR_system import gradient,build_from_segments,Grid,AMR_system

k = 0.9 #factor determines amount of padding
size = (260*(1+2*k),30*(1+2*k)) #size in mm - these units will be used

low_res_step = 0.005
high_res_step = low_res_step/8.

#set up fine grained system, where everything is computed on the 
#finest possible grid spacing
#+++++++++++++++++++++++++++++++++++Fine Grid Setup++++++++++++++++++++++++++++
xh,yh = build_from_segments(x=((1.,high_res_step),),
                            y=((0.1,high_res_step),))
print('Fine',xh.shape,yh.shape)                            
fine_grid = Grid(xh,yh,aspect_ratio=3./26,size=size)

fine_sys,fine_dust_size,fine_dust_pos = (
         create_EDM_system_from_grid(fine_grid,k,
                                     size=size,
                                     small_sources=True,
                                     dust_pos=300,
                                     dust_size=0.05))

fine_sys.SOR(w=1.9,tol=1e-13,max_time=20,verbose=False)  #solve
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#set up the Adaptive mesh, and solve it again for the same system as above
#+++++++++++++++++++++++++++++++++++AMR Grid Setup+++++++++++++++++++++++++++++
xh,yh = build_from_segments(x=((0.3,low_res_step),(0.7,high_res_step),
                               (1.,low_res_step)),
                            y=((0.02,low_res_step),(0.08,high_res_step),
                               (0.1,low_res_step)))
print('AMR',xh.shape,yh.shape)
AMR_grid = Grid(xh,yh,aspect_ratio=3./26,size = size)

AMR_sys,AMR_dust_size,AMR_dust_pos = (
         create_EDM_system_from_grid(AMR_grid,k,
                                     size=size,
                                     small_sources=True,
                                     dust_pos=300,
                                     dust_size=0.05))

AMR_sys.SOR(w=1.8,tol=1e-13,max_time=20,verbose=False)   #solve
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#compare grids
print('Fine Grid')
print('Nsx, Nsy',fine_sys.Nsx,fine_sys.Nsy)
print('dust size:',fine_dust_size)
print('dust pos :',fine_dust_pos)
print('final error:',fine_sys.errors[-1])
print('time taken in solver loop:',fine_sys.times[-1])

print('AMR Grid')
print('Nsx, Nsy',AMR_sys.Nsx,AMR_sys.Nsy)
print('dust size:',AMR_dust_size)
print('dust pos :',AMR_dust_pos)
print('final error:',AMR_sys.errors[-1])
print('time taken in solver loop:',AMR_sys.times[-1])

fine_sys.show()
AMR_sys.show()

#%%
#compare grids by analysing the difference in potential values achieved
#with the two different grids, by selecting fewer datapoints from the
#fine grid where appropriate - only selecting those points which lie at
#the same location as the AMR grid. The AMR grid is assumed to be no
#more detailed, but only less detailed - ie with a larger grid spacing
#in certain regions

#get the row indices to select out of the fine grid
row_indices = ((np.abs(fine_sys.grid.x-
                       AMR_sys.grid.x.reshape(-1,1))).argmin(axis=1))
column_indices = ((np.abs(fine_sys.grid.y-
                          AMR_sys.grid.y.reshape(-1,1)))).argmin(axis=1)
selection_index = (row_indices.reshape(-1,1),column_indices)
matched_potentials = fine_sys.potentials[selection_index]
extracted_system = AMR_system(AMR_sys.grid,create_matrix = False)
extracted_system.potentials = matched_potentials

difference = extracted_system.potentials - AMR_sys.potentials

dummy_system = AMR_system(AMR_sys.grid,create_matrix = False)
dummy_system.potentials = difference
dummy_system.show(title='difference')

#%%