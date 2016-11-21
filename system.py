# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:54:06 2016

@author: Alexander Kuhn-Regnier

Solving Laplace's equation in order to calculate the electric potential for a 
variety of different source conditions. The field can then be calculated 
from this potential.

Elliptic Equation
Solved by iterative successive over-relaxation method (SOR)
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

#np.set_printoptions(threshold=np.inf)

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
        self.potentials = np.zeros(self.grid.shape[1:])      
        self.sources = np.zeros(self.potentials.shape,dtype=np.bool)                   
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
        #print counts
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
                
        for coords in source_coords:
            #print "coords",coords
            self.potentials[coords] = potential
            self.sources[coords] = True
            
    def show_setup(self):
        pass
    def show(self):
        pass

    def create_method_matrix(self):
        N = self.Ns**2        
        self.A = np.zeros((N,N))
        boundary_conditions = []
        for i in range(N):
            boundaries_row = []
            #print "i", i
            coord1 = int(float(i)/self.Ns)
            coord2 = i%self.Ns
            self.A[i,i] = -4
            for c1,c2 in zip([coord1,coord1,coord1-1,coord1+1],
                             [coord2-1,coord2+1,coord2,coord2]):
                #print '{:02d} {:02d} {:02d} {:02d}'.format(coord1, coord2, c1,c2)
                try:
                    if c1==-1 or c2==-1 or c1>self.Ns-1 or c2>self.Ns-1:
                        raise IndexError
                    elif c1 == coord1-1: 
                        '''
                        row has changed, need to move 'cell'
                        column cannot have changed, so move 
                        by Ns along row
                        '''
                        self.A[i,i-self.Ns] = 1
                    elif c1 == coord1+1: 
                        self.A[i,i+self.Ns] = 1
                    elif c2 == coord2-1:
                        self.A[i,i-1]=1
                    elif c2 == coord2+1:
                        self.A[i,i+1]=1
                    else:
                        print "error",c1,c2
                except IndexError:
                    boundaries_row.append((c1,c2))
                #print self.A
            boundary_conditions.append(boundaries_row)
        #print self.A
        #print boundary_conditions,len(boundary_conditions)
        self.boundary_conditions = boundary_conditions
        
    def jacobi(self,tol=1e-4,max_iter=5000):
        N = self.Ns**2  
        self.create_method_matrix()
        b = np.zeros(N)
        
        #get diagonal, D
        D = np.diag(np.diag(self.A)) #but these are all just -4
        L = np.tril(self.A,k=-1)
        U = np.triu(self.A,k=1)
        x = self.potentials.reshape(-1,)
        orig_x = x.copy()
        sources = self.sources.reshape(-1,)
        D_inv = np.linalg.inv(D)
        L_U = L+U
        T = - np.dot(D_inv,L_U)
        D_inv_b = np.dot(D_inv,b).reshape(-1,)
        #print "before\n",x.reshape(self.Ns,-1)
        for i in range(max_iter):
            initial_norm = np.linalg.norm(x)
            x = np.dot(T,x).reshape(-1,) + D_inv_b
            #print "after\n",x.reshape(self.Ns,-1)
            x[sources] = orig_x[sources]
            final_norm = np.linalg.norm(x)
            diff = np.abs(initial_norm-final_norm)
            print "i,diff:",i,diff
            if diff<tol:
                break
        #print "done\n",x.reshape(self.Ns,-1)
        self.potentials = x.reshape(self.Ns,-1)
    def SOR(self,w=1.5):
        '''
        A = L + D + U
        A x = b - b are the boundary conditions
        
        x is arranged like:
            u_1,1
            u_1,2
            u_2,1
            u_2,2
        
        D is of length N^2, every element is -4, N is the number of gridpoints 
        '''
        N = self.Ns**2  
        w = float(w)
        #create array (matrix) A
        self.create_method_matrix()
        b = np.zeros(N)
        
        #get diagonal, D
        D = np.diagonal(self.A) #but these are all just -4
        L = np.tril(self.A,k=-1)
        U = np.triu(self.A,k=1)
        #print D
        #print L
        #print U
        x = self.potentials.reshape(-1,)
        sources = self.sources.reshape(-1,)
        for i in range(2):
            #print "before\n",x.reshape(self.Ns,-1)
            for k in range(N):
                if sources[k]:
                    #print "source at:",k
                    continue
                #print k,N
                s1 = 0
                s2 = 0
                for j in range(0,k):
                    #print "j1:",j,L[k,j],x[j]
                    s1 += L[k,j]*x[j]
                for j in range(k,N):
                    #print "j2:",j,U[k,j],x[j]
                    s2 += U[k,j]*x[j]
                #print L[k]
                #print U[k]
                #print D[k],s1,s2,b[k]
                #print ''
                x[k] += (w/D[k])*(-s1 -s2 + b[k])
            #print "after\n",x.reshape(self.Ns,-1)
            #print ''
        self.potentials = x.reshape(self.Ns,-1)
test = system(100)
test.add_source(1,(0.01,0.01))
test.add_source(2,(0.3,0.4))
test.add_source(2,(0.6,0.9))
test.add_source(1,(0.1,0.9))
r = 0.4
c = (0.5,0.5)
v = 1.6
for theta in np.linspace(0,2*np.pi,200):
    test.add_source(v,(c[0]+r*np.sin(theta),c[1]+r*np.cos(theta)))
    
test.add_source(-1,(0.5,0.5))
#print test.potentials
test.jacobi(tol=1e-2)
#print test.potentials
plt.figure()
plt.imshow(test.potentials)
plt.tight_layout()