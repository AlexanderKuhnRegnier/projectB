# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 14:20:14 2016

@author: Alexander Kuhn-Regnier
"""
from __future__ import print_function
from system import Shape,System
import numpy as np
import matplotlib.pyplot as plt
from time import clock
import os

def create_EDM_system(Ns,kx,ky=None):
    '''
    Create the shapes needed to model the EDM experiment and then add
    these shapes to a system, both with the given grid size.
    kx and ky describe the space left between the end of the experimental
    setup and the boundary, which will be held at 0V.
    This distance is described as a multiple of the length of the 
    experimental setup in that axis.
    '''
    if ky == None:
        ky = kx
    if not hasattr(Ns,'__iter__'):
        Ns = (Ns,Ns)
    A = Ns[1]/float(Ns[0])
    hx = 1./(26*(2*kx+1))
    hy = A/(3*(2*ky+1))
    Sx = kx * 26*hx
    Sy = ky * 3*hy
    shapes = []
    shapes.append(Shape(Ns,0.25,(Sx+hx,    Sy+5.*hy/2.),2*hx,hy,
                        shape='rectangle', filled=True))
    shapes.append(Shape(Ns,1.,  (Sx+13.*hx,Sy+5.*hy/2.),20*hx,hy,
                        shape='rectangle', filled=True))
    shapes.append(Shape(Ns,0.25,(Sx+25.*hx,Sy+5.*hy/2.),2*hx,hy,
                        shape='rectangle', filled=True))
    
    shapes.append(Shape(Ns,-0.25,(Sx+hx,    Sy+hy/2.),2*hx,hy,
                        shape='rectangle', filled=True))
    shapes.append(Shape(Ns,-1.,  (Sx+13.*hx,Sy+hy/2.),20*hx,hy,
                        shape='rectangle', filled=True))
    shapes.append(Shape(Ns,-0.25,(Sx+25.*hx,Sy+hy/2.),2*hx,hy,
                        shape='rectangle', filled=True))
    
    system = System(Ns)
    for shape in shapes:
        system.add(shape)
    return system
    
if __name__ == '__main__':
    factor = 150
    test = create_EDM_system((26*factor,3*factor),0.9)
    test.show_setup()
    test.SOR_single(w=1.5,tol=1e-10,max_time=60)
    test.show(quiver=False)