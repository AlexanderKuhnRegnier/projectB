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
from matplotlib.ticker import FuncFormatter
factor = 200
k = 0.9
size = (260*(1+2*k),30*(1+2*k))
dust_positions = [280,340]
longest_list = []
for dust_pos in dust_positions:
    system,dust_size,pos = create_EDM_system((26*factor,3*factor),k,
                                         size=size,
                                         small_sources=True,
                                         dust_pos=dust_pos,
                                         dust_size=0.05)
    #system.show_setup()
    system.SOR(tol=1e-16,max_time=1000)
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
    
    #%%
    central_value = E_field_mag[tuple((np.asarray(E_field_mag.shape))/2)]
    distances = system.grid.x*system.grid.distance_factor
    beam_path = E_field_mag[:,system.potentials.shape[1]/2]
    
    plt.figure()
    plt.plot(distances,beam_path)
    plt.xlabel(r'$\mathrm{x\ (m)}$',fontsize=16)
    plt.ylabel(r'$\mathrm{| E |\ (V\ m^{-1})}$',fontsize=16)
    plt.autoscale(enable=True, axis='x', tight=True)
    
    plt.gca().tick_params(axis='both',which='major', labelsize=16)
    plt.gca().tick_params(axis='both',which='minor', labelsize=16)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda value,pos:'$\mathrm{%.1f}$'%value))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda value,pos:'$\mathrm{%.1e}$'%value))
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()
    
    plt.savefig('EDM_e_mag_cross_section.pdf',bbox_inches='tight')
    
    mask = (distances<0.453) & (distances>0.274)
    plt.figure()
    plt.plot(distances[mask],beam_path[mask])
    plt.xlabel(r'$\mathrm{x\ (m)}$',fontsize=16)
    plt.ylabel(r'$\mathrm{| E |\ (V\ m^{-1})}$',fontsize=16)
    plt.autoscale(enable=True, axis='x', tight=True)
    
    plt.gca().tick_params(axis='both',which='major', labelsize=16)
    plt.gca().tick_params(axis='both',which='minor', labelsize=16)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda value,pos:'$\mathrm{%.1e}$'%value))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda value,pos:'$\mathrm{%.3e}$'%value))
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()
    
    plt.savefig('EDM_e_mag_cross_zoom.pdf',bbox_inches='tight')
    #%%
    
    #%%
    #tolerances = [0.01,1e-11]
    tolerances = np.linspace(1e-13,1,50000)
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
                           system.grid.x[section_starts])*system.grid.distance_factor
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
    longest_list.append(longest)
#%%

#%%
plt.figure()

#colours =  ['#e41a1c','#377eb8','#4daf4a']
colours =  ['#e41a1c','#377eb8']
#linestyles = [':','--','-.']
linestyles = ['','']
#markers = ['>','^','D']
markers = ['>','D']
markevery = 0.1     
ms = 12
lw = 5

for longest,c,marker,ls in zip(longest_list,colours,markers,linestyles):
    plt.plot(tolerances,longest,c=c,ls=ls,marker=marker,markevery=markevery)
    plt.xlabel(r'$\mathrm{tolerance}$',fontsize=16)
    plt.ylabel(r'$\mathrm{homogeneous\ section\ length\ (m)}$',fontsize=16)
    
plt.show()
#%%
print('dust size:',dust_size)