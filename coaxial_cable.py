# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 20:26:11 2016

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
from system import Shape,System
from AMR_system import gradient
import numpy as np
import matplotlib.pyplot as plt
from time import clock
import os
plt.ioff()

#augment original system class by adding new methods which can plot the 
#potential and electric field across the diagonal of the system
class Cable(System):
    def cross_section_potential(self,side_length=0,show=True,savepath=''):
        '''
        now, plot a cross section of the potential across the center.
        Ideally, the number of grid points should be an ODD number for this
        to work ideally - due to the symmetry of the problem.
        '''
        #use middle row!
        mid_row_index = int((self.Nsx-1)/2.)
        cross_section = self.potentials[mid_row_index]
        
        #use diagonal!
#        cross_section = np.diag(self.potentials)
        plt.figure()
        plt.title('1-D Cross-Section of the Potential across the cross section\n'
                  +'tol = {:.2e}, Nsx = {:.2e}, Nsy = {:.2e}, side length = {:.3e}'.
                  format(self.tol,self.Nsx,self.Nsy,side_length))
        grid_positions = self.grid[0][:,0]
        plt.plot(grid_positions,cross_section,label='potential')
        plt.xlabel('Distance from left wall (natural units)')
        plt.ylabel('Potential (scaled V)')
    #    plt.legend()
        ymin,ymax = plt.ylim()
        plt.ylim(ymax=ymax*1.1)
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
 
    def cross_section(self,side_length=0,show=True,savepath=''):
        '''
        now, plot a cross section of the electric field magnitude across the 
        center. Ideally, the number of grid points should be an ODD number 
        for this to work ideally - due to the symmetry of the problem.
        '''
        assert self.Nsy == self.Nsx,'Needs square grid! (uses diagonal)'
        E_field = gradient(self.potentials,[self.h]*(self.Nsx-1))
        E_field_mag = np.sqrt(E_field[0]**2+E_field[1]**2)
        
        #use middle row!
        mid_row_index = int((self.Nsx-1)/2.)
        skip = int((side_length/self.h)) #skip ahead to be claer of the source
        skip += int(self.Nsx/2.)
        cross_section = E_field_mag[mid_row_index,skip:]
        
        #use diagonal!
#        cross_section = np.diag(E_field_mag)

        plt.figure()
        plt.title('1-D Cross-Section of the Electric Field Magnitude cross section\n'
                  +'tol = {:.2e}, Nsx = {:.2e}, Nsy = {:.2e}, side length = {:.3e}'.
                  format(self.tol,self.Nsx,self.Nsy,side_length))
        grid_positions = self.grid[0][:,0][skip:]
        print('skip',skip)
        print('grid positions',grid_positions.shape)
        print('cross section',cross_section.shape)
        plt.plot(grid_positions,cross_section,label='electric field magnitude')
        plt.xlabel('Distance from left wall (natural units)')
        plt.ylabel('Electric Field Magnitude (scaled)')
        ymin,ymax = plt.ylim()
        plt.ylim(ymax=ymax*1.1)
        plt.tight_layout()
        
        #the electric field should vary as 1/r, where r is the distance
        #from the vertex. Therefore try to fit a curve like k/r to one side
        #of the data, where k is a constant.
        
        func = lambda r,k,s:k/(r+s)
        x = np.arange(0,cross_section.size,dtype=np.float64)
        popt,pcov = curve_fit(func,x,cross_section,bounds=((0.1,1),(200,200)))
        print('popt')
        print(popt)
        print('stds')
        stds = np.sqrt(np.diag(pcov))
        print(stds)
        ratio = stds/popt
        results = np.vstack((popt,stds,ratio))
        float_format = lambda s : '{:0.2e}'.format(s)
        formatted = np.array([float_format(i) for i in results.flatten()]).reshape(3,-1)
        plt.plot(grid_positions,[func(i,*popt) for i in x],label=str(formatted))
        plt.legend()
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
 
'''
show - to test out shapes
'''
show = False

if show:
    Ns = 300
    side_length = 0.45
    square_coaxial_cable = Shape(Ns,1,(0.5,0.5),side_length,shape='square',
                                 filled=True)
    cable = Cable(Ns)
    cable.add(square_coaxial_cable)
    cable.SOR_single(tol=1e-16,max_iter=10000000,max_time=5,verbose=False)   
    cable.cross_section(side_length=side_length)
    
picture_folder = name_folder(os.path.join(os.getcwd(),'potential_cross_sections'))            
#import time
#start = time.clock()
#anything below 250 makes the electric field look to discrete, and also 
#uneven when comparing the electric field at the two opposing vertices
Ns = 400
#print('time: {:.3f}'.format(time.clock()-start))
#cable.show_setup(interpolation='none')
tol = 1e-14
max_iter = 500000
max_time = 200
solve = True

side_lengths = np.linspace(1e-2,2e-1,10)

if solve:
    os.mkdir(picture_folder)
    for i,side_length in enumerate(side_lengths):
        square_coaxial_cable = Shape(Ns,1,(0.5,0.5),side_length,shape='square')
        cable = Cable(Ns)
        cable.add(square_coaxial_cable)    
        '''
        solve the system
        '''
        cable.SOR_single(tol=tol,max_iter=max_iter,max_time=max_time)
        '''
        now, plot a cross section of the potential across the central row.
        Ideally, the number of grid points should be an ODD number for this
        to work ideally - due to the symmetry of the problem
        '''
        savepath = os.path.join(picture_folder,str(i))
        cross_section = cable.cross_section(side_length,savepath=savepath)
