# -*- coding: utf-8 -*-
"""
Created on Dec 15

@author: Alexander Kuhn-Regnier

Simulating an infinitely long square coaxial cable,
which as a 2cm square cross section. The cable is held
inside of a grounded 6cm-square tube.

0V outside
 -------------------
|       +++++       | 
|       +10V+       |
|       +++++       |
 -------------------       
 2 cm | 2 cm | 2 cm

<-----   6 cm  ----->

Scaled to the natural grid units which range from
0 to 1 along each axis,
    2 cm -> 1/3
    6 cm -> 1 (as required)
    ie. scaling factor = (6 cm)^-1
    
Potential is also scaled to natural units, with the 
scaling factor being the inverse of the largest 
magntiude potential in the system to be modelled,
in this case 10 V
    ie. scaling factor = (10 V)^-1
"""
from __future__ import print_function
from scipy.optimize import curve_fit
from AMR_system import AMR_system,Grid,build_from_segments
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FormatStrFormatter
plt.ioff()

#augment original system class by adding new methods which can plot the 
#potential and electric field across the diagonal of the system
class Cable(AMR_system):
    def cross_section_potential(self,side_length=0,show=True,savepath=''):
        '''
        now, plot a cross section of the potential across the center.
        Ideally, the number of grid points should be an ODD number for this
        to work ideally - due to the symmetry of the problem.
        '''
        #use middle row!
        mid_row_index = int((self.Nsx-1)/2.)
        cross_section = self.potentials[mid_row_index]*self.grid.potential_scaling
        
        #use diagonal!
#        cross_section = np.diag(self.potentials)
        plt.figure()
        plt.title(r'$\epsilon = %.1e,\ \mathrm{Nsx} = %.1e,\ \mathrm{Nsy} = %.1e,\ \mathrm{L} = %.1e$' %
                  (self.errors[-1],self.Nsx,self.Nsy,side_length),fontsize=18)                  
                                     
        grid_positions = self.grid.grid[0][:,0]*self.grid.distance_factor
        plt.plot(grid_positions,cross_section,label='potential')
        plt.xlabel(r'$\mathrm{x\ (m)}$',
                              fontsize=16)
        plt.ylabel(r'$\mathrm{Potential\ (V)}$',
                              fontsize=16)
    #    plt.legend()
        ymin,ymax = plt.ylim()
        plt.ylim(ymax=ymax*1.08)
        plt.tight_layout()
        
        if savepath:
            plt.savefig(savepath,bbox_inches='tight',dpi=200) 
            plt.close('all') 
        else:
            if show:
                plt.show()
            else:
                plt.close('all')
        return cross_section
 
    def cross_section(self,side_length=0,show=True,savepath='',fit='inverse'):
        '''
        now, plot a cross section of the electric field magnitude across the 
        center. Ideally, the number of grid points should be an ODD number 
        for this to work ideally - due to the symmetry of the problem.
        '''
        assert self.Nsy == self.Nsx,'Needs square grid! (uses diagonal)'        
        #use middle row!
        mid_row_index = np.abs(self.grid.x-1/2.).argmin()
        #skip ahead to be clear of the source
        #using pythagoras to get distance from centre of square to the
        #vertices
        #side_length is in the same units as the system size,
        #but internally, everythin is scaled to have a maximum value of 1,
        #so this must be done in order to find the right position on the 
        #grid.
        if fit == 'inverse':
            radius = (2*(side_length/(2.*self.grid.size[0]))**2)**0.5
            print('radius:',radius)
            radius *= 1.1   #move further away from the source in order to be able
                            #to compare the electric field to that of a free
                            #infinite source
            skip = np.abs(self.grid.y-(max(self.grid.y)/2.)-
                                       radius).argmin()
            print('mid row index:',mid_row_index)
            print('skip:',skip)
            
        elif fit == 'linear':
            skip = np.abs(self.grid.y-(max(self.grid.y)/2.)-
                                side_length/(2.*self.grid.size[0])).argmin()
                                
        cross_section = self.E_field_mag[mid_row_index,skip:]                                
        #use diagonal!
#        cross_section = np.diag(E_field_mag)

        plt.figure()
        
#        plt.yscale('log')
#        plt.xscale('log')
#        plt.minorticks_on()
#        plt.gca().xaxis.set_minor_formatter(FormatStrFormatter("%.2f"))        
        
#        plt.title('1-D Cross-Section of the Electric Field Magnitude\n'
#                  +'tol = {:.2e}, Nsx = {:.2e}, Nsy = {:.2e}, side length = {:.3e}'.
#                  format(self.tol,self.Nsx,self.Nsy,side_length))

        plt.title(r'$\epsilon = %.1e,\ \mathrm{Nsx} = %.1e,\ \mathrm{Nsy} = %.1e,\ \mathrm{L} = %.1e$' %
                  (self.errors[-1],self.Nsx,self.Nsy,side_length),fontsize=18)                  
                                   
        grid_positions = self.grid.grid[0][:,0][skip:]*self.grid.distance_factor
        print('grid positions',grid_positions.shape)
        print('cross section',cross_section.shape)
        plt.plot(grid_positions,cross_section,label='electric field magnitude')
        plt.xlabel(r'$\mathrm{x\ (m)}$',
                              fontsize=16)
        plt.ylabel(r'$\mathrm{Electric\ Field\ Magnitude\ (V\ m^{-1})}$',
                              fontsize=16)
        
        if fit=='inverse':
            #the electric field should vary as 1/r, where r is the distance
            #from the vertex. Therefore try to fit a curve like k/r to one side
            #of the data, where k is a constant.
            func = lambda r,k,s:k/(r+s)
            popt,pcov = curve_fit(func,grid_positions,cross_section)
            print('popt')
            print(popt)
            print('stds')
            stds = np.sqrt(np.diag(pcov))
            print(stds)
            ratio = stds/popt
            results = np.vstack((popt,stds,ratio))
            float_format = lambda s : '{:0.2e}'.format(s)
            formatted = np.array([float_format(i) for i in results.flatten()]).reshape(3,-1)
            plt.plot(grid_positions,[func(i,*popt) for i in grid_positions],
                                     label=str(formatted),linestyle='--')
        elif fit == 'linear':
            #do a linear fit, if the cable is very close to the border
            func = lambda r,m,c:m*r+c
            popt,pcov = curve_fit(func,grid_positions,cross_section)
            print('popt')
            print(popt)
            print('stds')
            stds = np.sqrt(np.diag(pcov))
            print(stds)
            ratio = stds/popt
            results = np.vstack((popt,stds,ratio))
            float_format = lambda s : '{:0.2e}'.format(s)
            formatted = np.array([float_format(i) for i in results.flatten()]).reshape(3,-1)
            plt.plot(grid_positions,[func(i,*popt) for i in grid_positions],
                                     label=str(formatted),linestyle='--')
        plt.legend()
        #ymin,ymax = plt.ylim()
        #plt.ylim(ymax=ymax*1.1)
        plt.tight_layout()
        plt.axis('tight')        
        if savepath:
            plt.savefig(savepath,bbox_inches='tight',dpi=200) 
            plt.close('all') 
        else:
            if show:
                plt.show()
            else:
                plt.close('all')
        return cross_section


def name_folder(folder):
    '''
    Input: folder (string)
    Returns a new folder name string with a number appended to it, so as to
        make it unique in its directory.
    '''
    for i in range(1000):
        new = folder+str(i)
        if not os.path.isdir(new):
            return new
 
if __name__ == '__main__':
    '''
    show - to test out shapes
    '''
    show = True
    
    if show:
        Ns = 100
        side_length = 20
        grid = Grid(*build_from_segments(((1,Ns-1),)),size=(60.,60.))
        grid.square(1,(30.,30.),side_length)
        cable = Cable(grid)
        cable.SOR(tol=1e-8,max_iter=10000000,max_time=2,verbose=True)   
        cable.cross_section(side_length=side_length)
        
    picture_folder = name_folder(os.path.join(os.getcwd(),'potential_cross_sections'))            
    #import time
    #start = time.clock()
    #anything below 250 makes the electric field look too discrete, and also 
    #uneven when comparing the electric field at the two opposing vertices
    Ns = 400
    #print('time: {:.3f}'.format(time.clock()-start))
    #cable.show_setup(interpolation='none')
    tol = 1e-14
    max_iter = 500000
    max_time = 200
    solve = False
    
    side_lengths = np.linspace(1e-2,2e-1,10)
    
    if solve:
        os.mkdir(picture_folder)
        for i,side_length in enumerate(side_lengths):
            grid = Grid(*build_from_segments(((1,Ns-1),)),size=(1.,1.))
            grid.square(1,(0.5,0.5),side_length)
            cable = Cable(grid)  
            '''
            solve the system
            '''
            cable.SOR(tol=tol,max_iter=max_iter,max_time=max_time)
            '''
            now, plot a cross section of the potential across the central row.
            Ideally, the number of grid points should be an ODD number for this
            to work ideally - due to the symmetry of the problem
            '''
            savepath = os.path.join(picture_folder,str(i))
            cross_section = cable.cross_section(side_length,savepath=savepath)
