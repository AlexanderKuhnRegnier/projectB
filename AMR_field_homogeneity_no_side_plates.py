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
import cPickle as pickle
factor = 280
k = 0.9
size = (260*(1+2*k),30*(1+2*k))
side_sources = [True,False]
files = ['potentials2','potentials_no_side']
#%%
systems = []
for filename,side_source in zip(files,side_sources):
    system,dust_size,pos = create_EDM_system((26*factor,3*factor),k,
                                         size=size,
                                         small_sources=side_source,
                                         dust_pos=None,
                                         dust_size=None)
    #preload previously calculated potentials for a similar system
    
#    with open(filename,'rb') as f:    
#        system.potentials = pickle.load(f)

    #system.show_setup()
#    system.SOR(tol=1e-16,max_time=1)
    system.SOR(w=1.,tol=1e-16,max_time=1000)
    #system.show()
    
#    with open(filename,'wb') as f:
#        pickle.dump(system.potentials,f,protocol=2)

    systems.append(system)
#%%
longest_list = []
for system,side_source in zip(systems,side_sources):
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
    distances = system.grid.x*system.grid.distance_factor
    beam_path = E_field_mag[:,system.potentials.shape[1]/2]
    
    plt.figure()
    plt.plot(distances,beam_path,lw=4,c='#377eb8')
    plt.xlabel(r'$\mathrm{x\ (m)}$',fontsize=20)
    plt.ylabel(r'$\mathrm{| E |\ (V\ m^{-1})}$',fontsize=20)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.margins(y=0.05)
    plt.gca().tick_params(axis='both',which='major', labelsize=16)
    plt.gca().tick_params(axis='both',which='minor', labelsize=16)
    plt.gca().ticklabel_format(style='sci',scilimits=(0,0),
                            useoffset=False)      
#    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda value,pos:'$\mathrm{%.1f}$'%value))
#    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda value,pos:'$\mathrm{%.1e}$'%value))
    plt.minorticks_on()
    plt.tight_layout()
    plt.grid()
    
    ax = plt.gca()
    ax.annotate(s=r'$\mathrm{central\ plate}$',
                xy=(0.35,1.55e6),
                xytext=(0.06,1.4e6),
                arrowprops={'arrowstyle':'simple',
                            'color':'k',
                            'lw':1},
                xycoords='data',
                fontsize=19)
    if side_source:
        side_height = 4.5e5
        ax.annotate(s=r'$\mathrm{side\ plates}$',
                    xy=(0.23,side_height),
                    xytext=(0.06,7e5),
                    arrowprops={'arrowstyle':'simple',
                                'color':'k',
                                'lw':1},
                    xycoords='data',
                    fontsize=19)
        
        ax.annotate(s='',
                    xy=(0.47,side_height),
                    xytext=(0.22,7e5),
                    arrowprops={'arrowstyle':'simple',
                                'color':'k',
                                'lw':1},
                    xycoords='data',
                    fontsize=19)
    plt.show()
    
#    plt.savefig('EDM_e_mag_cross_section.pdf',bbox_inches='tight')
    
    
#    mask = (distances<0.453) & (distances>0.274)
#    plt.figure()
#    plt.plot(distances[mask],beam_path[mask])
#    plt.xlabel(r'$\mathrm{x\ (m)}$',fontsize=16)
#    plt.ylabel(r'$\mathrm{| E |\ (V\ m^{-1})}$',fontsize=16)
#    plt.autoscale(enable=True, axis='x', tight=True)
#    
#    plt.gca().tick_params(axis='both',which='major', labelsize=16)
#    plt.gca().tick_params(axis='both',which='minor', labelsize=16)
#    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda value,pos:'$\mathrm{%.1e}$'%value))
#    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda value,pos:'$\mathrm{%.3e}$'%value))
#    plt.minorticks_on()
#    plt.tight_layout()
#    plt.show()
    
#    plt.savefig('EDM_e_mag_cross_zoom.pdf',bbox_inches='tight')

    

    #tolerances = [0.01,1e-11]
    tolerances = np.linspace(1e-5,1e-3,80000)
    print('total number of points along x:',beam_path.size)
    
    longest = []
    for tol in tolerances:
        match = ((beam_path < central_value*(1+tol)) &
                 (beam_path > central_value*(1-tol)))
        #identify contiguous intervals!
        indices = np.arange(len(beam_path))[match]
#        print('tol',tol)
#        print('nr of matches:',indices.size)
    #    print('max index:',max(indices))
        diffs = np.diff(indices)
        breaks_after = indices[np.where(diffs!=1)[0]+1] #this is where new section starts
        breaks_before = indices[np.where(diffs!=1)[0]]+1  #this is end of old section
#        print('breaks:',breaks_after)
#        print('before:',breaks_before)
        sections = []
        sections.append(indices[0])
        for break_before,break_after in zip(breaks_before,breaks_after):
            sections.extend([break_before,break_after])
        sections.append(indices[-1]+1)    #slice notation is non-inclusive
        sections = np.array(sections)
#        print('sections:',sections)
        
        section_starts = sections[0::2]
        section_ends = sections[1::2]
        section_lengths = (system.grid.x[section_ends-1]-
                           system.grid.x[section_starts])*system.grid.distance_factor
#        print ('section lengths:',section_lengths)                       
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
overview = True #if not, then detailed view with a separate subplot for each
                #source
#colours =  ['#e41a1c','#377eb8','#4daf4a']
colours =  ['#e41a1c','#377eb8']
#linestyles = [':','--','-.']
linestyles = ['-','-']
#markers = ['>','^','D']
markers = ['>','D']
markevery = 0.1     
ms = 12
lw = 5
labels = [r'$\mathrm{with\ side\ plates}$',r'$\mathrm{no\ side\ plates}$']
plt.rc('font', size=15)   
print(max(tolerances),'max tol')       
if overview:
    plt.figure()
    for longest,c,marker,ls,label in zip(longest_list,colours,
                                         markers,linestyles,labels):
        plt.plot(tolerances[:600]*100,longest[:600],c=c,ls=ls,marker=marker,markevery=markevery,
                 label=label,ms=ms,lw=lw)
        plt.xlabel(r'$\mathrm{tolerance\ (\%)}$',fontsize=20)
        plt.ylabel(r'$\mathrm{homogeneous\ section\ length\ (m)}$',fontsize=20)
        
    leg = plt.legend(loc='best')
    ltext  = leg.get_texts()  # all the text.Text instance in the legend
    llines = leg.get_lines()  # all the lines.Line2D instance in the legend
    plt.setp(ltext, fontsize=20)    # the legend text fontsize
    plt.setp(llines, linewidth=1.5)      # the legend linewidth
    plt.margins(0.05)
    
    plt.gca().tick_params(axis='both', labelsize=16)
    plt.gca().ticklabel_format(style='sci',scilimits=(0,0),
                               useoffset=False)         
#    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda value,pos:r'$\mathrm{%.0e}$'%value))
#    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda value,pos:r'$\mathrm{%.1e}$'%value))
    plt.minorticks_on()
    plt.grid()
    plt.tight_layout()
    plt.show()
    
else:
    fig,axes = plt.subplots(2,1,sharex=True,sharey=False)
    for ax,longest,c,marker,ls,label in zip(axes,longest_list,colours,
                                         markers,linestyles,labels):
        ax.plot(tolerances*100,longest,c=c,ls=ls,
                 label=r'$\mathrm{%s}$' % label,lw=lw)
        ax.set_ylabel(r'$\mathrm{L_{h}\ (m)}$',fontsize=20)
        ax.minorticks_on()
        ax.margins(0.05)
        ax.set_title(r'$\mathrm{%s}$' % label,fontsize=20)
        ax.grid()
        ax.tick_params(axis='both', labelsize=16)
        ax.ticklabel_format(style='sci',scilimits=(0,0),
                            useoffset=False)         
        ax.axvline(tolerances[0]*100,c='k',lw=1.5)
        ax.text(0.09,0.84,s=r'$\mathrm{tol=%.2e}$'%(tolerances[0]*100),
                    transform=ax.transAxes,fontsize=16,
                    bbox=dict(facecolor='white', edgecolor='white'))
        ax.arrow(0.09,0.87,-0.027,0,transform=ax.transAxes,fc='k', ec='k',
                 head_width=0.032, head_length=0.012,lw=1.7)
        
#    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value,pos:r'$\mathrm{%.2e}$'%value))
#    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value,pos:r'$\mathrm{%.2e}$'%value if pos%2 else ''))
    axes[-1].set_xlabel(r'$\mathrm{tolerance\ (\%)}$',fontsize=20)
#    axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda value,pos:r'$\mathrm{%.2e}$'%value))
    
    plt.tight_layout()
    plt.show()
    
#%%