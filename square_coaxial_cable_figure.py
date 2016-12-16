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


#Plot two different systems with different side length of the 
#source

#plot the larger source first, as required by the lab script
Ns = 300
side_length = 20.
grid = Grid(*build_from_segments(((1,Ns-1),)),size=(60.,60.),
            units='mm',potential_scaling=10.)
grid.square(1,(30.,30.),side_length)
cable = Cable(grid)
cable.SOR(tol=1e-14,max_iter=1e6,max_time=200,verbose=True)   

#%%
cable.cross_section(side_length=side_length)
cable.show(quiver=True,every=10)
#%%