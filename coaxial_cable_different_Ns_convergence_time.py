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
from matplotlib.ticker import FormatStrFormatter
plt.ioff()

def format_sci(number,exp,dec_places=1):
    format_dict = {'scaled_number':number/(10**exp),
              'dec_places':dec_places,
              'dec_places_2':dec_places+2,
              'exp':exp}
    formatted = ('{scaled_number:0={dec_places_2:d}.{dec_places:d}f}e{exp:+d}'.
             format(**format_dict))
    return formatted
#print(format_sci(0.213982,1))
#print(format_sci(0.0213982,-1))

save = False
max_time = 600
tolerances = [1e1,1e-7]
Ns_array = np.linspace(10,800,40,dtype=np.int64)
side_length = 20.
tot_last_iters = []

for tol in tolerances:
    grid_spacings = []
    errors = []
    last_iters = []
    ws = []
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
        
        errors.append(cable.errors[-1])
        last_iters.append(cable.last_iteration)
    #    print('norm:',np.linalg.norm(cable.potentials)/np.sqrt(Ns**2))
        ws.append(cable.w)
        grid_spacings.append(cable.grid.x_h[0]*cable.grid.distance_factor)
    
    spacing = 12
    print('{1:>{0:d}s}{2:>{0:d}s}{3:>{0:d}s}{4:>{0:d}s}'.format(spacing,'Ns',
                                          'Last Error','Iterations','w'))
    for Ns,error,iterations,w in zip(Ns_array,errors,last_iters,ws):
        print('{1:>{0:d}.2e}{2:>{0:d}.2e}{3:>{0:d}.2e}{4:>{0:d}.2e}'.
              format(spacing,Ns,error,iterations,w))
    tot_last_iters.append(last_iters)
    
         
#%%
plt.figure()
#colours =  ['#e41a1c','#377eb8','#4daf4a']
colours =  ['#e41a1c','#377eb8']
#linestyles = [':','--','-.']
linestyles = ['','']
#markers = ['>','^','D']
markers = ['>','D']
markevery = 1       
ms = 12
lw = 5
positions = ((0.6,0.08),(0.1,0.58))
for tol,last_iters,c,marker,ls,pos in zip(tolerances,tot_last_iters,colours,markers,
                                   linestyles,positions):
    plt.plot(Ns_array,last_iters,lw=lw,c=c,marker=marker,markevery=markevery,
             ls=ls,ms=ms,label='$\mathrm{tol=%.1e}$'%tol,
            markeredgecolor='k',markeredgewidth=1.2)
    #do linear fit for each dataset
    p,V = np.polyfit(Ns_array,last_iters,deg=1,cov=True)
    stds = np.sqrt(np.diag(V))
    print('parameters:',p)
    print('errors    :',stds)
    N = np.linspace(0,max(Ns_array)*1.05,1000)
    fitted = np.polyval(p,N)
    plt.plot(N,fitted,ls ='--', c='k',lw=4)
    text = plt.text(pos[0],pos[1],(r'$\mathrm{I=m \times N+c,}$'+'\n'+
        r'$\mathrm{m= %.2f \pm %.2f}$'+'\n'+r'$\mathrm{c= %.2f \pm %.2f}$')%
        (p[0],stds[0],p[1],stds[1]),fontdict={'fontsize':16},
        bbox=dict(facecolor='white', edgecolor='black'),
        transform = plt.gca().transAxes)

plt.xlabel('$\mathrm{\sqrt{ Grid\ Points},\ \sqrt{N}}$',
                      fontsize=17)
plt.ylabel(r'$\mathrm{Number\ of\ Iterations,\ I}$',
                      fontsize=17)  

#plt.axis('tight')
plt.margins(0.05)

leg = plt.legend(loc='best')
ltext  = leg.get_texts()  # all the text.Text instance in the legend
llines = leg.get_lines()  # all the lines.Line2D instance in the legend
plt.setp(ltext, fontsize=17)    # the legend text fontsize
plt.setp(llines, linewidth=1.5)      # the legend linewidth

plt.gca().tick_params(axis='both', labelsize=16)
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda value,pos:'$\mathrm{%d}$'%value))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda value,pos:'$\mathrm{%d}$'%value))
plt.minorticks_on()
plt.grid()
plt.show()

#plt.savefig('iter_vs_Ns.pdf',bbox_inches='tight')

#%%
###
###Now plot the dependence of iterations to grid spacing - power law
###
#%%
plt.figure()
#colours =  ['#e41a1c','#377eb8','#4daf4a']
colours =  ['#e41a1c','#377eb8']
#linestyles = [':','--','-.']
linestyles = ['','']
#markers = ['>','^','D']
markers = ['>','D']
markevery = 1       
ms = 15
lw = 5
positions = ((0.1,0.3),(0.7,0.6))
for tol,last_iters,c,marker,ls,pos in zip(tolerances,tot_last_iters,colours,markers,
                                   linestyles,positions):
    plt.plot(grid_spacings,last_iters,lw=lw,c=c,marker=marker,markevery=markevery,
             ls=ls,ms=ms,label='$\mathrm{tol=%.1e}$'%tol,
             markeredgecolor='k',markeredgewidth=1.2)
    
    #do linear fit for each dataset
    p,V = np.polyfit(np.log(grid_spacings),np.log(last_iters),deg=1,cov=True)
    stds = np.sqrt(np.diag(V))
    print('parameters:',p)
    print('errors    :',stds)
    ax = plt.gca()
    fitted = np.polyval(p,np.log(grid_spacings))
    plt.plot(grid_spacings,np.exp(fitted),ls ='--', c='k',lw=4,zorder=2)
    text = plt.text(pos[0],pos[1],(r'$\mathrm{I=k \times h^{d},}$'+'\n'+
        r'$\mathrm{d= %.2f \pm %.2f}$'+'\n'+r'$\mathrm{k= %.2f \pm %.2f}$')%
        (p[0],stds[0],p[1],stds[1]),
        transform=ax.transAxes,
        fontdict={'fontsize':16},
        bbox=dict(facecolor='white', edgecolor='black'))

plt.xlabel('$\mathrm{Grid\ Spacing,\ h \ (m)}$',
                      fontsize=17)
plt.ylabel(r'$\mathrm{Number\ of\ Iterations,\ I}$',
                      fontsize=17)  

plt.yscale('log',base='e')
plt.xscale('log',base='e')
#plt.axis('tight')
plt.margins(0.05)
leg = plt.legend(loc='best')
ltext  = leg.get_texts()  # all the text.Text instance in the legend
llines = leg.get_lines()  # all the lines.Line2D instance in the legend
plt.setp(ltext, fontsize=17)    # the legend text fontsize
plt.setp(llines, linewidth=1.5)      # the legend linewidth

plt.gca().tick_params(axis='both',which='major', labelsize=16)
plt.gca().tick_params(axis='both',which='minor', labelsize=16)
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda value,pos:'$\mathrm{%.0e}$'%value))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda value,pos:'$\mathrm{%.0e}$'%value))
plt.minorticks_on()
#plt.tick_params(axis='y', which='minor')
def minor_x_formatter(value,pos):
#    print(value,pos)
    lead = int(format(value,'e')[0])
    if lead in [3,6]:
        return '$\mathrm{%.0e}$'%value
    else:
        return ''
plt.gca().xaxis.set_minor_formatter(FuncFormatter(minor_x_formatter))
plt.grid()
plt.tight_layout()
plt.show()

#plt.savefig('iter_vs_h.pdf',bbox_inches='tight')
#%%


#
#Analyse the dependence on number of grid points
#
#%%
plt.figure()
#colours =  ['#e41a1c','#377eb8','#4daf4a']
colours =  ['#e41a1c','#377eb8']
#linestyles = [':','--','-.']
linestyles = ['','']
#markers = ['>','^','D']
markers = ['>','D']
markevery = 1       
ms = 12
lw = 5
dec_places = 4
positions = ((0.59,0.09),(0.04,0.6))
for tol,last_iters,c,marker,ls,pos in zip(tolerances,tot_last_iters,colours,markers,
                                   linestyles,positions):
    plt.plot(Ns_array**2,last_iters,lw=lw,c=c,marker=marker,markevery=markevery,
             ls=ls,ms=ms,label='$\mathrm{tol=%.1e}$'%tol,
            markeredgecolor='k',markeredgewidth=1.2)
    #do linear fit for each dataset
    p,V = np.polyfit(Ns_array[-Ns_array.size/2:]**2,
                     last_iters[-Ns_array.size/2:],deg=1,cov=True)
    stds = np.sqrt(np.diag(V))
    print('parameters:',p)
    print('stds      :',stds)
    
    exponent0 = int(format(p[0],'.1e')[-3:])
    exponent1= int(format(p[1],'.1e')[-3:])
    p_strings = [format_sci(p[0],exponent0,dec_places=1),
                 format_sci(p[1],exponent1,dec_places=1)]
                 
    std_strings = [format_sci(stds[0],exponent0,dec_places=1),
                   format_sci(stds[1],exponent1,dec_places=1)]
                 
    print('p   ',p_strings)
    print('stds',std_strings)
                   
    N = np.linspace(0,max(Ns_array)*1.01,1000)**2
    fitted = np.polyval(p,N)
    plt.plot(N,fitted,ls ='--', c='k',lw=4)
    text = plt.text(pos[0],pos[1],(r'$\mathrm{I=m \times N+c,}$'+'\n'+
        r'$\mathrm{m= %s \pm %s}$'+'\n'+r'$\mathrm{c= %s \pm %s}$')%
        (p_strings[0],
         std_strings[0],
         p_strings[1],
         std_strings[1]),fontdict={'fontsize':16},
        bbox=dict(facecolor='white', edgecolor='black'),
        transform=plt.gca().transAxes)

plt.xlabel('$\mathrm{Grid\ Points,\ N}$',
                      fontsize=17)
plt.ylabel(r'$\mathrm{Number\ of\ Iterations,\ I}$',
                      fontsize=17)  

#plt.axis('tight')
plt.margins(0.05)

leg = plt.legend(loc='best')
ltext  = leg.get_texts()  # all the text.Text instance in the legend
llines = leg.get_lines()  # all the lines.Line2D instance in the legend
plt.setp(ltext, fontsize=17)    # the legend text fontsize
plt.setp(llines, linewidth=1.5)      # the legend linewidth

plt.gca().tick_params(axis='both', labelsize=16)
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda value,pos:'$\mathrm{%.0e}$'%value))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda value,pos:'$\mathrm{%.0e}$'%value))
plt.minorticks_on()
plt.grid()
plt.show()
plt.tight_layout()
#plt.savefig('iter_vs_Ns.pdf',bbox_inches='tight')

#%%