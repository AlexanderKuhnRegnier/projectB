# -*- coding: utf-8 -*-
"""
Created on Dec 7 2016

@author: Alexander Kuhn-Regnier

Calculating the homogeneity of the electric field generated for a 
given configuration of the sources.
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from time import clock
import os
plt.ioff()
from AMR_EDM import create_EDM_system
from AMR_system import gradient

factor = 40
k = 0.9
system,dust_size = create_EDM_system((26*factor,3*factor),k,
                                     size=(260*(1+2*k),30*(1+2*k)),
                                     small_sources=True,
                                     dust_pos=300,
                                     dust_size=2.)
#system.show_setup()
system.SOR(w=1.2,tol=1e-14,max_time=1)
#system.show()
print('dust size:',dust_size)

#calculate Electric field and electric field magnitude
E_field = system.E_field
E_field_mag = system.E_field_mag

#plt.figure()
#plt.imshow(E_field_mag.T,origin='lower',interpolation='none')
#plt.title('Electric Field Magnitude')
#plt.xlabel('x gridpoints')
#plt.ylabel('y gridpoints')
#plt.colorbar()
#plt.tight_layout()
#plt.show()

#get central value by dividing shape by 2 and using that as an index
#for non-ideal arrangements, could get two central rows and average them
#but since the gradient along the direction  of the electron beam
#is relatively small, this should not have a large effect, especially
#as the grid spacing is decreased

central_value = E_field_mag[tuple((np.asarray(E_field_mag.shape))/2)]
                      
beam_path = E_field_mag[:,system.potentials.shape[1]/2]
plt.figure()
plt.plot(np.linspace(0,system.grid.size[0],len(beam_path)),beam_path)
plt.title('Electric Field Magnitude along Electron Beam Path')
plt.xlabel('x gridpoints')
plt.ylabel('electric field magnitude')
plt.autoscale(enable=True, axis='x', tight=True)
plt.show()


#%%
#tolerances = [0.01,1e-11]
tolerances = np.linspace(1e-13,1,200)
print('total number of points along x:',beam_path.size)

longest = []
for tol in tolerances:
    match = ((beam_path < central_value*(1+tol)) &
             (beam_path > central_value*(1-tol)))
    #identify contiguous intervals!
    indices = np.arange(len(beam_path))[match]
#    print('nr of matches:',indices.size)
#    print('max index:',max(indices))
    diffs = np.diff(indices)
    breaks = indices[np.where(diffs!=1)[0]+1] #this is where new section starts
#    print('breaks:',breaks)
    sections = np.zeros(len(breaks)+2,dtype=np.int64)
    sections[0] = indices[0]
    sections[-1] = indices[-1]+1    #slice notation is non-inclusive
    sections[1:-1] = breaks
    
#    print('sections:',sections)
    
    section_starts = sections[0:-1]
    section_ends = sections[1:]
    section_lengths = (system.grid.x[section_ends-1]-
                       system.grid.x[section_starts])*system.grid.size[0]
#    print ('section lengths:',section_lengths)                       
    longest.append(max(section_lengths))
#    print('tol:',tol)
#    print('longest section length:',max(section_lengths))
#    for i,j in zip(section_starts,section_ends):
#        within_bounds = beam_path[i:j]
#        indices_within_bounds = np.arange(i,j)
#        x_position = indices_within_bounds*system.h*system.size[0]
#        fig,ax = plt.subplots()
#        ax.get_yaxis().get_major_formatter().set_useOffset(False)
#        ax.get_yaxis().get_major_formatter().set_powerlimits((-1,1))
#        ax.plot(x_position,within_bounds)
#        plt.title('Electric Field Values within tolerance: {}'.format(tol))
#        plt.xlabel('x [mm]')
#        plt.ylabel('electric field magnitude')
#        plt.autoscale(enable=True, axis='x', tight=True)
#        plt.show()

plt.figure()
plt.plot(tolerances,longest)
plt.xlabel('tol')
plt.ylabel('length')
plt.title('Length of longest contiguous region')
plt.show()
#%%
