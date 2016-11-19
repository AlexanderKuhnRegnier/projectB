# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:54:06 2016

@author: Alexander Kuhn-Regnier

Solving Laplace's equation in order to calculate the electric potential for a 
variety of different source conditions. The field can then be calculated 
from this potential.

Elliptic Equation
Solved by relaxation method
Dirichlet boundary conditions
Equation solved in 2D - grid spacing h

Pictorial operator arises from 2nd order central finite difference
(divided by h^2). Applied to every gridpoint, calculates next iteration
using the value at the current gridpoint as well as the four closest
gridpoints around it (includes boundary conditions if needed).
Can be combined into a matrix so that the entire system can be solved 
using the relaxation method.


"""
import numpy as np
from numba import jit
import matplotlib.pyplot as plt

class system:
    def __init__(self,Ns):
        '''
        'Ns' is the number of gridpoints along an axis, so the
        grid spacing h is the inverse of this, since the axes range from 0 to 1
        '''
        assert type(Ns)==int,'Ns should be an integer'
        self.Ns = Ns
        self.h = 1./Ns
    def add_source(self,shape,origin,coord1,coord2):
        '''
        In 2D, a regular shape can be specified by 3 parameters, the centre
        (origin) of the shape, and two additional coordinates (could be more
        if irregular shapes would be implemented).
        The center of the shapes described below is specified by the 
        'origin' argument, with the lower left corner of the simulated space
        defined as (0,0), and the top right corner as (1,1).
        implemented here:
            'circle'
                
            'rectangle' (with square as special case (coord1 = coord2))
                (would be orientated with edges vertical & horizontal)
            '
            
        '''
        pass
    def show_setup(self):
        pass
    def show(self):
        pass
    def solve(self,method):
        pass
    
test = system()