# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 00:15:10 2016

@author: Alexander Kuhn-Regnier
"""
from __future__ import print_function
from AMR_system import Grid,build_from_segments
import numpy as np
import matplotlib.pyplot as plt
from AMR_coaxial_cable import Cable
from matplotlib.ticker import FuncFormatter

plt.ioff()

save = False
max_time = 600
tol = 1e-16
Ns_array = np.array([10,20,50,600])

side_length = 20.
grid = Grid(*build_from_segments(((1,Ns_array[-1]-1),)),size=(60.,60.),
            units='mm',potential_scaling=10.)
grid.square(1,(30.,30.),side_length)
reference_system = Cable(grid)
reference_system.SOR(tol=tol,max_iter=1e6,max_time=max_time,verbose=False)   
#reference_system.show()
#this system will not be used to solve, but just to compare
reference_system_copy = Cable(grid,create_matrix=False)

diff_norms = []
errors = []
last_iters = []
ws = []
cross_sections = []
grid_positions_list = []
grid_spacings = []

for Ns in Ns_array:
    grid = Grid(*build_from_segments(((1,Ns-1),)),size=(60.,60.),
                units='mm',potential_scaling=10.)
    grid.square(1,(30.,30.),side_length)
    cable = Cable(grid)
    #solve system
    cable.SOR(tol=tol,max_iter=1e7,max_time=max_time,verbose=False)   
    #generate figures
#    cable.cross_section_potential(side_length=side_length)
#    if save:
#        plt.savefig('{:03d}_Ns_cable_potential_cross.pdf'.format(Ns),bbox_inches='tight')    
#    cable.show(quiver=False,every=12)
#    if save:
#        plt.savefig('{:03d}_Ns_cable_overview.pdf'.format(Ns),bbox_inches='tight')
#    cable.cross_section(side_length=side_length,fit=None)
#    if save:
#        plt.savefig('{:03d}_Ns_cable_e_field_cross.pdf'.format(Ns),bbox_inches='tight')
    
    #compare to reference system!
    reference_system_copy.interpolate(cable)
#    reference_system_copy.show()
    potential_diffs_norm = np.linalg.norm(reference_system.potentials
                                          -reference_system_copy.potentials)
    diff_norms.append(potential_diffs_norm)
    errors.append(cable.errors[-1])
    last_iters.append(cable.last_iteration)
#    print('norm:',np.linalg.norm(cable.potentials)/np.sqrt(Ns**2))
    ws.append(cable.w)
    
    mid_row_index = np.abs(cable.grid.x-1/2.).argmin()
    grid_positions = cable.grid.grid[0][:,0]*cable.grid.distance_factor
    grid_positions_list.append(grid_positions)
    cross_section = cable.potentials[mid_row_index]*cable.grid.potential_scaling
    cross_sections.append(cross_section)
    grid_spacings.append(cable.grid.x_h[0]*cable.grid.distance_factor)
  
spacing = 12
print('{1:>{0:d}s}{2:>{0:d}s}{3:>{0:d}s}{4:>{0:d}s}{5:>{0:d}s}'.format(spacing,'Ns',
                                      'Diffs Norm','Last Error','Iterations','w'))
for Ns,diff_norm,error,iterations,w in zip(Ns_array,diff_norms,errors,last_iters,ws):
    print('{1:>{0:d}.2e}{2:>{0:d}.2e}{3:>{0:d}.2e}{4:>{0:d}.2e}{5:>{0:d}.2e}'.
          format(spacing,Ns,diff_norm,error,iterations,w))    
    
#%%
plt.figure()
linestyles = [':','--','-.','-']
markers = ['>','<','^','D']
markevery = 0.1
colours =  ['#e41a1c',
            '#377eb8',
            '#4daf4a',
            '#984ea3']
ms = 10     
lw = 4    
zorders = [4,3,2,1]
for Ns,cross_section,grid_positions,ls,marker,c,zorder,gs in zip(Ns_array,
                                                              cross_sections,
                                              grid_positions_list,linestyles,
                                              markers,colours,zorders,
                                              grid_spacings):
    plt.plot(grid_positions,cross_section,label=r'$\mathrm{h:%0.2e\ m}$'%gs,
             lw=lw,ls=ls,marker=marker,markevery=markevery,c=c,
             ms = ms,zorder=zorder)
    plt.xlabel('$\mathrm{x\ (m)}$',
                          fontsize=17)
    plt.ylabel(r'$\mathrm{Potential\ (V)}$',
                          fontsize=17)  
plt.legend(loc='best')
plt.margins(y=0.1,x=0)

#adapted in part from:
#http://matplotlib.org/1.2.1/examples/pylab_examples/legend_demo.html
# set some legend properties.  All the code below is optional.  The
# defaults are usually sensible but if you need more control, this
# shows you how
leg = plt.gca().get_legend()
ltext  = leg.get_texts()  # all the text.Text instance in the legend
llines = leg.get_lines()  # all the lines.Line2D instance in the legend
frame  = leg.get_frame()  # the patch.Rectangle instance surrounding the legend

# see text.Text, lines.Line2D, and patches.Rectangle for more info on
# the settable properties of lines, text, and rectangles
#frame.set_facecolor('0.80')      # set the frame face color to light gray
plt.setp(ltext, fontsize=17)    # the legend text fontsize
plt.setp(llines, linewidth=1.5)      # the legend linewidth
#leg.draw_frame(False)           # don't draw the legend frame

plt.gca().tick_params(axis='both', labelsize=16)
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda value,pos:'$\mathrm{%0.2f}$'%value))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda value,pos:'$\mathrm{%d}$'%value))
plt.minorticks_on()
plt.grid()
plt.show()

plt.savefig('cable_diff_Ns_pot.pdf',bbox_inches='tight')
#
#repeat above for a zoomed in section of the plot!
#

plt.figure()
linestyles = [':','--','-.','-']
markers = ['>','<','^','D']
markevery = 0.4
colours =  ['#e41a1c',
            '#377eb8',
            '#4daf4a',
            '#984ea3']
ms = 12    
lw = 5  
zorders = [4,3,2,1]
for Ns,cross_section,grid_positions,ls,marker,c,zorder,gs in zip(Ns_array,
                                                              cross_sections,
                                              grid_positions_list,linestyles,
                                              markers,colours,zorders,
                                              grid_spacings):
    plt.plot(grid_positions,cross_section,label=r'$\mathrm{h:%0.2e\ m}$'%gs,
             lw=lw,ls=ls,marker=marker,markevery=markevery,c=c,
             ms = ms,zorder=zorder)
    plt.xlabel('$\mathrm{x\ (m)}$',
                          fontsize=17)
    plt.ylabel(r'$\mathrm{Potential\ (V)}$',
                          fontsize=17)  
plt.legend(loc='best')

#adapted in part from:
#http://matplotlib.org/1.2.1/examples/pylab_examples/legend_demo.html
# set some legend properties.  All the code below is optional.  The
# defaults are usually sensible but if you need more control, this
# shows you how
leg = plt.gca().get_legend()
ltext  = leg.get_texts()  # all the text.Text instance in the legend
llines = leg.get_lines()  # all the lines.Line2D instance in the legend
frame  = leg.get_frame()  # the patch.Rectangle instance surrounding the legend

# see text.Text, lines.Line2D, and patches.Rectangle for more info on
# the settable properties of lines, text, and rectangles
#frame.set_facecolor('0.80')      # set the frame face color to light gray
plt.setp(ltext, fontsize=17)    # the legend text fontsize
plt.setp(llines, linewidth=1.5)      # the legend linewidth
#leg.draw_frame(False)           # don't draw the legend frame

plt.gca().tick_params(axis='both', labelsize=16)
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda value,pos:'$\mathrm{%0.e}$'%value))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda value,pos:'$\mathrm{%0.1f}$'%value))
plt.minorticks_on()
plt.grid()

plt.gca().set_xlim(0,5.5e-3)
plt.gca().set_ylim(0,2.64)

plt.show()

plt.savefig('cable_diff_Ns_pot_zoom.pdf',bbox_inches='tight')

#%%