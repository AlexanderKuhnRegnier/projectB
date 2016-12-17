# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 12:12:03 2016

@author: ahfku
"""
from __future__ import print_function
from AMR_system import Grid,build_from_segments
import numpy as np
import matplotlib.pyplot as plt
from AMR_coaxial_cable import Cable
plt.ioff()


save = True
max_time = 100
#Plot two different systems with different side length of the 
#source

def normal_source():
    #plot the larger source first, as required by the lab script
    Ns = 400
    side_length = 20.
    grid = Grid(*build_from_segments(((1,Ns-1),)),size=(60.,60.),
                units='mm',potential_scaling=10.)
    grid.square(1,(30.,30.),side_length)
    cable = Cable(grid)
    cable.SOR(tol=1e-14,max_iter=1e6,max_time=max_time,verbose=True)   
    #generate figures
    cable.cross_section_potential(side_length=side_length)
    if save:
        plt.savefig('normal_tube_coaxial_cable_potential_cross.pdf',bbox_inches='tight')    
    cable.show(quiver=True,every=12)
    if save:
        plt.savefig('normal_coaxial_cable_overview.pdf',bbox_inches='tight')
    cable.cross_section(side_length=side_length)
    if save:
        plt.savefig('normal_coaxial_cable_e_field_cross.pdf',bbox_inches='tight')

def small_tube():
    #then decrease the size of the outer tube
    Ns = 400
    side_length = 20.
    grid2 = Grid(*build_from_segments(((1,Ns-1),)),size=(30.,30.),
                units='mm',potential_scaling=10.)
    grid2.square(1,(15.,15.),side_length)
    cable2 = Cable(grid2)
    
    cable2.SOR(tol=1e-14,max_iter=1e6,max_time=max_time,verbose=True)   
    #generate figures
    cable2.cross_section_potential(side_length=side_length)
    if save:
        plt.savefig('small_tube_coaxial_cable_potential_cross.pdf',bbox_inches='tight')
    cable2.show(quiver=True,every=12)
    if save:
        plt.savefig('small_tube_coaxial_cable_overview.pdf',bbox_inches='tight')
    cable2.cross_section(side_length=side_length,fit='linear')
    if save:
        plt.savefig('small_tube_coaxial_cable_e_field_cross.pdf',bbox_inches='tight')

def large_tube():
    #now increase the size of the outer tube
    Ns = 400
    side_length = 20.
    grid3 = Grid(*build_from_segments(((1,Ns-1),)),size=(480.,480.),
                units='mm',potential_scaling=10.)
    grid3.square(1,(240.,240.),side_length)
    cable3 = Cable(grid3)
    cable3.SOR(tol=1e-14,max_iter=1e6,max_time=max_time,verbose=True)   
    #generate figures
    cable3.cross_section_potential(side_length=side_length) 
    if save:
        plt.savefig('large_tube_coaxial_cable_potential_cross.pdf',bbox_inches='tight')        

    cable3.show(quiver=True,every=10)
    if save:
        plt.savefig('large_tube_coaxial_cable_overview.pdf',bbox_inches='tight')
    cable3.cross_section(side_length=side_length)
    if save:
        plt.savefig('large_tube_coaxial_cable_e_field_cross.pdf',bbox_inches='tight')

def execute_all():
    normal_source()
    small_tube()
    large_tube()