# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 18:11:34 2016

@author: Alexander Kuhn-Regnier
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from scipy import sparse
from time import clock
from matplotlib.collections import LineCollection
#np.set_printoptions(threshold=np.inf)

class Grid:
    '''
    Variable mesh sizing along the rows and column in order to 
    increase resolution locally around a specific point
    '''    
    def __init__(self,x_h,y_h,aspect_ratio=1.):
        '''
        Pass in arrays containing the stepsizes along the x and y 
        direction.
        These will be scaled so that the sum of the entries in *x_h*
        is 1. The sum of the entries in *y_h* will be *aspect_ratio*.
        [aspect_ratio=1.].
        '''
        assert (0 not in x_h) and (0 not in y_h),('Step sizes have to be'+
                                                  ' greater than 0!')
        x_sum = np.sum(x_h)
        self.x_h = x_h / float(x_sum)
        '''
        extend x_h for later computations, since two 
        additional step sizes are needed to 'connect' the 
        last potential values and the boundary conditions.
        these are taken to be the first and last step sizes given.
        This could be changed by taking the first and last
        step sizes given for this purpose - this would decrease
        the calculated number of potentials by 2 along each axis.
        '''  
        self.x_h_extended = np.hstack((self.x_h[0],self.x_h,
                                       self.x_h[-1]))
        
        self.x = np.append([0.],np.cumsum(self.x_h))
        y_sum = np.sum(y_h)
        self.y_h = y_h / (y_sum*(1./aspect_ratio))
        
        self.y_h_extended = np.hstack((self.y_h[0],self.y_h,
                                       self.y_h[-1]))
        
        self.y = np.append([0.],np.cumsum(self.y_h))
        self.grid = np.meshgrid(self.x,self.y,indexing='ij')
        self.x_indices = np.arange(len(self.x))
        self.y_indices = np.arange(len(self.y))
        self.source_potentials = np.zeros(self.grid[0].shape)
    def square(self,potential,origin,length):
        self.rectangle(potential,origin,length,length)
    def rectangle(self,potential,origin,width,height):
        x = (origin[0]-width/2.,origin[0]+width/2.)
        y = (origin[1]-height/2.,origin[1]+height/2.)
        x_indices = (np.abs(self.x-x[0]).argmin(),
                     np.abs(self.x-x[1]).argmin())
        y_indices = (np.abs(self.y-y[0]).argmin(),
                     np.abs(self.y-y[1]).argmin())        
        
        print('x:',self.x)
        print('x,y indices:',x_indices,y_indices)
        
        mask = ((self.x_indices == x_indices[0]),
                (self.y_indices <= y_indices[1]) &
                (self.y_indices >= y_indices[0]))
        self.source_potentials[mask] = potential

        mask = ((self.x_indices == x_indices[1]),
                (self.y_indices <= y_indices[1]) &
                (self.y_indices >= y_indices[0]))
        self.source_potentials[mask] = potential
        
        mask = ((self.x_indices <= x_indices[1]) &
                (self.x_indices >= x_indices[0]),
                (self.y_indices == y_indices[0]))
        self.source_potentials[mask] = potential

        mask = ((self.x_indices <= x_indices[1]) &
                (self.x_indices >= x_indices[0]),
                (self.y_indices == y_indices[1]))
        self.source_potentials[mask] = potential      
        
    def filled_square(self,potential,origin,length):
        self.filled_rectangle(potential,origin,length,length)
    
    def filled_rectangle(self,potential,origin,width,height):
        mask = ((self.x < origin[0]+width/2.)  & (self.x > origin[0]-width/2.),
                (self.y < origin[1]+height/2.) & (self.y > origin[1]-height/2.) 
               )
        #use np.ix_ to broadcast the boolean 'mask' arrays into the correct
        #shape for use in indexing as below. The same could have been
        #achieved by using *self.grid[0]* instead of self.x (and likewise 
        #for y) as self.grid has been created from self.x and self.y using
        #np.meshgrid
        self.source_potentials[np.ix_(*mask)] = potential
        
    def circle(self,potential,origin,radius):
        '''
        At present, the thickness of the perimeter is approx. as 
        thick as the largest stepsize used in the entire grid.
        Thus, this method is not recommended for creating fine
        structures.
        '''
        mask_whole = ((((self.grid[0]-origin[0])**2) + 
                       ((self.grid[1]-origin[1])**2)) < radius**2)
        mask_inner = ((((self.grid[0]-origin[0])**2) + 
                       ((self.grid[1]-origin[1])**2)) < (radius-np.max(np.append(self.x_h,self.y_h)))**2)
        self.source_potentials[mask_whole & ~mask_inner] = potential
        
    def filled_circle(self,potential,origin,radius):
        mask = ((((self.grid[0]-origin[0])**2) + 
                 ((self.grid[1]-origin[1])**2)) < radius**2)
        self.source_potentials[mask] = potential
    def create_grid_line_collection(self,lw=2,color=(0,0,0,0.1)):
        x_lines = np.zeros((len(self.x),2,2))
        x_lines[:,:,0] = self.x[:,None]     #all the same x
        x_lines[:,:,1] = [0.,np.max(self.y)]

        y_lines = np.zeros((len(self.y),2,2))
        y_lines[:,:,1] = self.y[:,None] #change dim to broadcast
        y_lines[:,:,0] = [0.,1.]
        
        lines = np.vstack((x_lines,y_lines))
        grid_lines = LineCollection(lines,lw=lw,color=color)
        return grid_lines
               
    def show(self,**kwargs):
        '''
        Need to supply coordinates of 'bounding' boxes for the assigned
        potentials, ie.
           h0  h1  ...          hn
          +---+---+--+-+------+-----+
        |   |   |   | |    |     |      |
        
        Need to determine locations of the *|* above -> is that an accurate
        representation?
        '''
        '''
        Creat list, where the first x_bound is interpolated backwards from x0
        by a step half the size of the first step size h0. The remainder of
        the bounds are calculated by adding half of the step size to the
        corresponding x-positions, where the last step size is duplicated in
        order to have the right number of stepsizes for the operation.
        '''
        x_bounds = np.append([-self.x_h[0]/2.],
                             self.x+np.append(self.x_h,[self.x_h[-1]])/2.)
        '''
        Repeat for the y axis
        '''
        y_bounds = np.append([-self.y_h[0]/2.],
                             self.y+np.append(self.y_h,[self.y_h[-1]])/2.)        
        fig,ax = plt.subplots()
        plot = ax.pcolorfast(x_bounds,y_bounds,self.source_potentials.T)        
        grid_lines = self.create_grid_line_collection(**kwargs)
        ax.add_collection(grid_lines) 
        fig.colorbar(plot)
        
class AMR_system:
    '''
    Variable mesh sizing along the rows and column in order to 
    increase resolution locally around a specific point
    '''
    def __init__(self,grid):
        '''
        Pass a Grid instance
        '''
        self.grid = grid
        self.potentials = self.grid.source_potentials.copy()
        #makes for faster truth testing, eg in a loop
        self.sources = np.array(self.grid.source_potentials,
                                dtype=np.bool)
        self.Nsx = self.potentials.shape[0]
        self.Nsy = self.potentials.shape[1]
        self.create_matrix()
        
    def create_matrix(self):
        N = self.Nsx*self.Nsy   #works for rectangular setup as well
        indices = np.arange(0,N,dtype=np.int64)
        coord1 = np.array((np.arange(0,N,dtype=np.float64)/self.Nsy),
                          dtype=np.int64) #which row
        coord2 = indices%self.Nsy   #which column  
        '''
        Define step size arrays which will be used below
        coord1 + 1 retrieves h_(x,i),
        where coord1 retrieves h_(x,i-1)    
        '''
        h_x_i = self.grid.x_h_extended[coord1+1]
        h_x_i_1 = self.grid.x_h_extended[coord1]
        x_divisor = h_x_i_1*h_x_i**2
        h_y_j = self.grid.y_h_extended[coord2+1]
        h_y_j_1 = self.grid.y_h_extended[coord2]
        y_divisor = h_y_j_1*h_y_j**2
        '''
        Fill in diagonal, according to the formula
        Involves both the step sizes in the y and the x direction
        matrix *A* should have shape (N,N)
        '''
        '''
        Contribution to diagonal from x results from
        (h_(x,i)+h_(x,i-1)) * u_(i,j) term.
        In the array holding the potentials, the first coordinate,
        ie. the rows - corresponds to x, and the second coordinate,
        the columns - corresponds to y.
        So moving 'down' a column corresponds to translating in x,
        and moving 'right' along a row corresponds to translating
        in y.
        When reshaping the potentials, rows will be continguous,
        so as the index increases from 0 to N (N = Nsx*Nsy),
        the columns will be traversed first, before the rows.
        So i, the first coordiante will increase every time the 
        index passes Nsy, ie. given by coord1
        '''
        x_diag_contribution = -(h_x_i + h_x_i_1)/x_divisor
        '''
        Repeat the above for the y diagonal contribution
        '''
        y_diag_contribution = -(h_y_j + h_y_j_1)/y_divisor
        self.A = sparse.diags(x_diag_contribution+y_diag_contribution,
                              format='lil')

        '''
        row has decreased (i-1) (contribution to u_(i-1,j))
        column cannot have changed (j is still coord2), so move
        by -Nsy along row of matrix A (to get to previous new row)
        '''
        mask = (coord1-1) >= 0
        row_indices = indices[mask]
        column_indices = indices[mask]-self.Nsy
        
        self.A[tuple(row_indices),tuple(column_indices)] = (h_x_i[mask]/
                                                            x_divisor[mask])
        '''
        Increase row, move by +Nsy
        (contribution to u_(i+1,j), j is still coord2)
        '''
        mask = (coord1+1) < (self.Nsx)
        row_indices = indices[mask]
        column_indices = indices[mask]+self.Nsy
        self.A[tuple(row_indices),tuple(column_indices)] = (h_x_i_1[mask]/
                                                            x_divisor[mask])
        '''
        Change column now, so move by -1 along row of matrix A to
        adjacent cell
        (contribution to u_(i,j-1), i is coord1)
        Since we are changing the y coordinate, the y step sizes are relevant
        now
        '''
        mask = (coord2-1) >= 0
        row_indices = indices[mask]
        column_indices = indices[mask]-1
        self.A[tuple(row_indices),tuple(column_indices)] = (h_y_j[mask]/
                                                            y_divisor[mask])     
        '''
        Move to next adjacent cell, now in + direction
        (contribution to u_(i,j+1))
        '''
        mask = (coord2+1) < (self.Nsy)
        row_indices = indices[mask]
        column_indices = indices[mask]+1
        self.A[tuple(row_indices),tuple(column_indices)] = (h_y_j_1[mask]/
                                                            y_divisor[mask])      
        self.A = self.A.tocsc()
        
    def jacobi(self, tol=1e-2, max_iter=10000, max_time=10, verbose=True):
        #get diagonal, D
        D = sparse.diags(self.A.diagonal(),format='csc')
        L = sparse.tril(self.A,k=-1,format='csc')
        U = sparse.triu(self.A,k=1,format='csc')
        
        D_inv = sparse.linalg.inv(D)
        L_U = L+U
        T = - D_inv.dot(L_U)
#        print('T',type(T))
#        D_inv_b = D_inv.dot(b).reshape(-1,) #just 0s anyway
        print("Jacobi: finished creating matrices")
        
        x = self.potentials.reshape(-1,)
        orig_x = x.copy()
        sources = self.sources.reshape(-1,)
        inv_sources = ~sources
        #randomise starting potential
        x = np.random.random(x.shape)
        x[sources] = orig_x[sources]    
        #randomise starting potential
        
        start = clock()
        for i in range(max_iter):
            x = T.dot(x).reshape(-1,) # + D_inv_b all 0s
            x[sources] = orig_x[sources]
            error = np.mean(np.abs(self.A.dot(x))[inv_sources])  
            #similar computational effort as 2xnorm
            if verbose:
                print("i, error:",i,error)
            if error < tol:
                print('Error in potential lower than tolerance')
                break
            if (clock()-start) > max_time:
                print('Time limit exceed')
                break
        self.potentials = x.reshape(self.Nsx,-1)        
        
    def gauss_seidel(self, tol=1e-2, max_iter=10000, max_time=10, verbose=True):
        #get diagonal, D
        D = sparse.diags(self.A.diagonal(),format='csc')
        L = sparse.tril(self.A,k=-1,format='csc')
        U = sparse.triu(self.A,k=1,format='csc')
        L_D_inv = sparse.linalg.inv(L+D)
#        L_D_inv_b = np.dot(L_D_inv,b)
        T = -L_D_inv.dot(U)
        print("Gauss Seidel: finished creating matrices")
        x = self.potentials.reshape(-1,)
        orig_x = x.copy()
        sources = self.sources.reshape(-1,)
        inv_sources = ~sources
        #randomise starting potential
        x = np.random.random(x.shape)
        x[sources] = orig_x[sources]    
        #randomise starting potential
#        print('T',type(T))
        start = clock()        
        for i in range(max_iter):
            x = T.dot(x).reshape(-1,) # + L_D_inv_b
            x[sources] = orig_x[sources]
            error = np.mean(np.abs(self.A.dot(x))[inv_sources])   
            if verbose:
                print("i, error:",i,error)
            if error < tol:
                print('Error in potential lower than tolerance')
                break
            if (clock()-start) > max_time:
                print('Time limit exceed')
                break
        self.potentials = x.reshape(self.Nsx, -1)        
        
    def SOR(self):
        pass
    def show(self,**kwargs):
        '''
        Need to supply coordinates of 'bounding' boxes for the assigned
        potentials, ie.
           h0  h1  ...          hn
          +---+---+--+-+------+-----+
        |   |   |   | |    |     |      |
        
        Need to determine locations of the *|* above -> is that an accurate
        representation?
        '''
        '''
        Creat list, where the first x_bound is interpolated backwards from x0
        by a step half the size of the first step size h0. The remainder of
        the bounds are calculated by adding half of the step size to the
        corresponding x-positions, where the last step size is duplicated in
        order to have the right number of stepsizes for the operation.
        '''
        x_bounds = np.append([-self.grid.x_h[0]/2.],
                             self.grid.x+np.append(self.grid.x_h,
                                                   [self.grid.x_h[-1]])/2.)
        '''
        Repeat for the y axis
        '''
        y_bounds = np.append([-self.grid.y_h[0]/2.],
                             self.grid.y+np.append(self.grid.y_h,
                                                   [self.grid.y_h[-1]])/2.)        
        fig,ax = plt.subplots()
        plot = ax.pcolorfast(x_bounds,y_bounds,self.potentials.T)      
        grid_lines = self.grid.create_grid_line_collection(**kwargs)
        ax.add_collection(grid_lines) 
        fig.colorbar(plot)
    def show_setup(self):
        self.grid.show()
    
xh = np.ones(100)
#xh[12:45]=0.1
yh = np.ones(50)
#yh[6:40]=0.1
test = Grid(xh,yh)
test.filled_rectangle(1,(0.5,0.5),0.4,0.7)
test.filled_circle(1,(0.2,0.4),0.01)
#test.show(color=(0,0,0,0.1))

system = AMR_system(test)
system.jacobi(max_iter=1000000,max_time=15,tol=1e-10)
system.show()