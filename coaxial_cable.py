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
from system import Shape,System
import numpy as np
import matplotlib.pyplot as plt
from time import clock
import os
plt.ioff()
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
show = True

if show:
    Ns = 100
    square_coaxial_cable = Shape(Ns,1,(0.5,0.5),(1./3),shape='square',filled=True)
    cable = System(Ns)
    cable.add(square_coaxial_cable)
    start = clock()
    cable.SOR(tol=1e-12,max_iter=1000)   
    print('time:',clock()-start)
    cable.show(interpolation='none')  

picture_folder = name_folder(os.path.join(os.getcwd(),'potential_cross_sections'))            
#import time
#start = time.clock()
Ns = 50
#print('time: {:.3f}'.format(time.clock()-start))
#cable.show_setup(interpolation='none')
tol = 1e-8
max_iter = 500000

solve = False

steps = 10

if solve:
    os.mkdir(picture_folder)
    for i,side_length in enumerate(np.linspace(4e-1,5.3e-1,steps)):
        square_coaxial_cable = Shape(Ns,1,(0.5,0.5),side_length,shape='square')
        cable = System(Ns)
        cable.add(square_coaxial_cable)    
        '''
        solve the system
        '''
        cable.SOR(tol=tol,max_iter=max_iter)
    #    cable.show(title='Square Coaxial Cable, SOR',interpolation='none')
        '''
        now, plot a cross section of the potential across the central row.
        Ideally, the number of grid points should be an ODD number for this
        to work ideally - due to the symmetry of the problem
        '''
        savepath = os.path.join(picture_folder,str(i))
        cross_section = cable.cross_section(side_length,savepath=savepath)
