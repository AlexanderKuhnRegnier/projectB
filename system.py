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
from collections import Counter

class system:
    def __init__(self,Ns):
        '''
        'Ns' is the number of gridpoints along an axis, so the
        grid spacing h is the inverse of this, since the axes range from 0 to 1
        '''
        assert type(Ns)==int,'Ns should be an integer'
        self.Ns = Ns
        self.h = 1./Ns
        self.grid = np.mgrid[0:1:complex(0,self.Ns),0:1:complex(0,self.Ns)]
        self.xx,self.yy = self.grid                             
    def find_closest_gridpoint(self,coords):
        '''
        Find closest point on grid to coords
        '''
        coords = np.array(coords).reshape(-1,)
                        #make sure coords have correct shape for next step
                        #really necessary (could have assert)?
        coords = coords[(slice(None),)+(None,)*coords.shape[0]]
                        #reshape to (2,1,1) or (3,1,1,1) etc. for correct
                        #broadcasting
        abs_diff = np.abs(self.grid - coords)
        diff_list = []
        for i in range(self.grid.shape[0]):
            #print "grid"
            #print self.grid[i]
            #print "abs diff"
            #print abs_diff[i]
            '''
            find row number of minimum (absolute) differences in
            grid[0] which stores the x coordinates, and find the
            min column for grid[1], which stores the y coords
            '''
            diff = np.where(abs_diff[i]==np.min(abs_diff[i]))
            #print "diff"
            #print diff
            #print np.vstack(diff)   
            diff_list.append(np.vstack(diff))
        #print "diff list"
        #print diff_list
        #print np.hstack(diff_list)
        diffs = np.hstack(diff_list).T
            #.T so the rows can be iterrated over in order to find the 
            #duplicated indices - which will then reveal where the overlap is
        diff_tuples = map(tuple,diffs)
        counts = Counter(diff_tuples).most_common()
        match = counts[0][0]
            #desired coordinates!
        #print "match", match
        return match
        
    def add_source(self,potential,origin,*args,**kwargs):
        '''
        *args: coord1[,coord2]
        **kwargs: 'shape':'circle' or 'rectangle'
        
        In 2D, a regular shape can be specified by 3 parameters, the centre
        (origin) of the shape, and two additional coordinates (could be more
        if irregular shapes would be implemented).
        The centre of the shapes described below is specified by the 
        'origin' argument, with the lower left corner of the simulated space
        defined as (0,0), and the top right corner as (1,1).
        If only origin is given (ie. no non-keyword args), then a point source
        at the grid point closest to the origin is created. 
        implemented here:
            'rectangle':origin is the centre of the rectangle
                        coord1 is lower left vertex
                (square as special case (coord1 = coord2),
                 would be orientated with edges vertical & horizontal)              
            'circle':   origin is the centre of the circle
                        coord1 is the radius
        '''
        if 'shape' in kwargs:
            '''
            shape specified explicitly
            '''
            shape = kwargs['shape']
        else:
            shape = 'rectangle'
        
        if shape == 'rectangle':
            if not args: #if only origin is specified
                source_coords = [self.find_closest_gridpoint(origin)]
                #print "Source at:",source_coords
                #print self.grid
                #print self.grid[(slice(None),)+source_coords[0]]
            elif len(args)==1:
                print "1 arg"
            else:
                print "2 or more"
        if shape == 'circle':
            if not args: #if only origin is specified
                source_coords = [self.find_closest_gridpoint(origin)]            
                
    def show_setup(self):
        pass
    def show(self):
        pass
    def solve(self,method):
        pass
    
test = system(6)
test.add_source(1,(0.5,0.58))