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
plt.rcParams['image.cmap'] = 'viridis'  #set default colormap to something
                                        #better than jet

def gradient(data,*args):
    '''
    Data:
        data where gradient is to be taken, assume data has shape 
        (N1,) for 1D, (N1,N2) for 2D, (N1,N2,N3) for 3D and so on...
    args:
        Can optionally supply grid spacings. Grid spacings should have length
        Ni-1, where i refers to the dimension (axis) of interest.
        If no args are given, a uniform grid spacing of 1 is assumed.
        Either one can supply 1 additional argument, which will be taken
        for all axes, otherwise an array of grid spacings must be 
        supplied for every axis.
    Second order accurate central differences are taken every where but 
    at the edges, where first order accurate forward or backward differences
    are calculated, based on the location.
    '''
    D = data.ndim
    gradients = []
    n = len(args)
    if n == D:
        spacings = list(args)
    elif n == 1:
        #only works if square, will throw error otherwise!
        spacings = list(args)*D
    else:
        spacings = [np.array([1.0]*(N-1)) for N in data.shape]
    for i in range(D):
        spacings[i] = np.asarray(spacings[i],dtype=np.float64)
    select1 = [slice(None)]*D
    select2 = [slice(None)]*D
    select3 = [slice(None)]*D
    spacing_reshape = [None]*D
    for m in range(D):
        gradient = np.empty_like(data,dtype=np.float64)    
        spacing_reshape[m] = slice(None) 
#        Take central differences, therefore exclude last along axis
#        and first along axis
        select1[m] = slice(1,-1) #select central values along axis
        select2[m] = slice(2,None) #select last values along axis
        select3[m] = slice(None,-2) #select first values along axis
#        print('m',m)
#        print(select1,'\n',select2,'\n',select3)
#        print(data[select1],'\n',data[select2],'\n',data[select3],'\n',spacings[m])
        gradient[select1] = (((data[select2] - data[select1])/
                                spacings[m][1:][spacing_reshape]
                        +(data[select1] - data[select3])/
                            spacings[m][:-1][spacing_reshape])
                            /2.)
#        now do the forward difference
        select1[m] = 0 #forward difference will fill in values here
        select2[m] = 1 #these values are used in the gradient calculation
        gradient[select1] = (data[select2] - data[select1])/spacings[m][0]
#        now do the backward difference
        select1[m] = -1 #where to put gradients
        select2[m] = -2 #used to calculate backwards difference
        gradient[select1] = (data[select1] - data[select2])/spacings[m][-1]

#        reset the selection objects for the next axis
        select1 = [slice(None)]*D
        select2 = [slice(None)]*D
        select3 = [slice(None)]*D        
        spacing_reshape = [None]*D
        gradients.append(gradient)
    return gradients
        
class Grid:
    '''
    Variable mesh sizing along the rows and column in order to 
    increase resolution locally around a specific point
    '''    
    def __init__(self,x_h,y_h,aspect_ratio=1.,size=(1.,1.),units='mm',
                 potential_scaling = 1.):
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
        
        self.aspect_ratio = size[1]/size[0]        
        y_sum = np.sum(y_h)
        self.y_h = y_h / (y_sum*(1./self.aspect_ratio))
        
        self.y_h_extended = np.hstack((self.y_h[0],self.y_h,
                                       self.y_h[-1]))        
        self.y = np.append([0.],np.cumsum(self.y_h))
        
        self.grid = np.meshgrid(self.x,self.y,indexing='ij')
        self.x_indices = np.arange(len(self.x))
        self.y_indices = np.arange(len(self.y))
        self.source_potentials = np.zeros(self.grid[0].shape)
        if not hasattr(size,'__iter__'):
            self.size = (size,size) 
        else:
            self.size = size   
        self.units = units
        self.distance_units_SI = {'mm':1e-3,'cm':1e-2,'m':1.,'um':1e-6}
        self.distance_factor = self.size[0]*self.distance_units_SI[self.units]
        self.potential_scaling = potential_scaling
            
    def __str__(self):
        a='''This is a test string
             for testing purposes
             .'''
        return a
            
    def hollow_square(self,potential,origin,length):
        self.hollow_rectangle(potential,origin,length,length)
        
    def hollow_rectangle(self,potential,origin,width,height):
        width /= self.size[0]
        height /= self.size[0]
        origin = (origin[0]/self.size[0],origin[1]/self.size[0])        
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
        
    def square(self,potential,origin,length):
        self.rectangle(potential,origin,length,length)
    
    def rectangle(self,potential,origin,width,height):
        width /= self.size[0]
        height /= self.size[0]
        origin = (origin[0]/self.size[0],origin[1]/self.size[0])   
        mask = ((self.x < origin[0]+width/2.)  & (self.x >= origin[0]-width/2.),
                (self.y < origin[1]+height/2.) & (self.y >= origin[1]-height/2.) 
               )
        #use np.ix_ to broadcast the boolean 'mask' arrays into the correct
        #shape for use in indexing as below. The same could have been
        #achieved by using *self.grid[0]* instead of self.x (and likewise 
        #for y) as self.grid has been created from self.x and self.y using
        #np.meshgrid
        if not (np.sum(mask[0]) and (np.sum(mask[1]))):
            #if no gridpoint has been found by the above method, ie.
            #no gridpoint is included within the limits, as the entire
            #shape happens to be located within two gridpoints, but not
            #overlapping any grid lines.
            mask = (np.zeros(self.x.shape,dtype=np.bool),
                    np.zeros(self.y.shape,dtype=np.bool))
            #take the closest gridpoint!
            mask[0][np.abs(self.x-origin[0]).argmin()] = True
            mask[1][np.abs(self.y-origin[1]).argmin()] = True
            
        self.source_potentials[np.ix_(*mask)] = potential
        
    def hollow_circle(self,potential,origin,radius):
        '''
        At present, the thickness of the perimeter is approx. as 
        thick as the largest stepsize used in the entire grid.
        Thus, this method is not recommended for creating fine
        structures.
        '''
        radius /= self.size[0]
        origin = (origin[0]/self.size[0],origin[1]/self.size[0])           
        mask_whole = ((((self.grid[0]-origin[0])**2) + 
                       ((self.grid[1]-origin[1])**2)) < radius**2)
        mask_inner = ((((self.grid[0]-origin[0])**2) + 
                       ((self.grid[1]-origin[1])**2)) < (radius-np.max(np.append(self.x_h,self.y_h)))**2)
        self.source_potentials[mask_whole & ~mask_inner] = potential
        
    def circle(self,potential,origin,radius):
        radius /= self.size[0]
        origin = (origin[0]/self.size[0],origin[1]/self.size[0])           
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
        
        lines = np.vstack((x_lines,y_lines))*self.distance_factor
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
        x_bounds *= self.distance_factor
        y_bounds *= self.distance_factor
        print(x_bounds)
        fig,ax = plt.subplots()
        plot = ax.pcolorfast(x_bounds,y_bounds,
                             self.source_potentials.T*self.potential_scaling)        
        grid_lines = self.create_grid_line_collection(**kwargs)
        ax.add_collection(grid_lines) 
        cb = fig.colorbar(plot)
        plt.axis('square')
        plt.autoscale()
        cb.set_label(r'$\mathrm{Potential\ (V)}$',fontsize=16)
        ax.set_xlabel(r'$\mathrm{x\ (m)}$',fontsize=16)
        ax.set_ylabel(r'$\mathrm{y\ (m)}$',fontsize=16)        
        plt.show()
        
class AMR_system(object):
    '''
    Variable mesh sizing along the rows and column in order to 
    increase resolution locally around a specific point
    '''
    def __init__(self,grid,create_matrix = True):
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
        #we don't need to create this matrix if we are not going to do
        #any solving with the class, ie. if we just want to use its 
        #plotting methods. Could also be implemented with multiple 
        #ihneritance.
        if create_matrix:   
            self.create_matrix()

    #use properties to get the electric field and its magnitude,
    #since the electric field would only have to be re-calculated
    #if the potential has changed
    def __get_E_field(self):
        if hasattr(self,'_AMR_system__past_potentials'):
            if not np.all(self.__past_potentials == self.potentials):
                self.calculate_E_field()
                self.__past_potentials = self.potentials
            return self._E_field
        else:
            self.calculate_E_field()
            self.__past_potentials = self.potentials
            return self._E_field
            
    E_field = property(fget=__get_E_field)
    
    def __get_E_field_mag(self):
        if hasattr(self,'_AMR_system__past_potentials'):
            if not np.all(self.__past_potentials == self.potentials):
                self.calculate_E_field()
                self.__past_potentials = self.potentials
            return self._E_field_magnitude
        else:
            self.calculate_E_field()
            self.__past_potentials = self.potentials
            return self._E_field_magnitude       
    
    E_field_mag = property(fget=__get_E_field_mag)
    
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
        start = clock()
        #get diagonal, D
        D = sparse.diags(self.A.diagonal(),format='csc')
        L = sparse.tril(self.A,k=-1,format='csc')
        U = sparse.triu(self.A,k=1,format='csc')
        
        D_inv = sparse.linalg.inv(D)
        L_U = L+U
        T = - D_inv.dot(L_U)
#        print('T',type(T))
#        D_inv_b = D_inv.dot(b).reshape(-1,) #just 0s anyway
        if verbose:
            print("Jacobi: finished creating matrices")
            print('Time taken:',clock()-start)
        
        x = self.potentials.reshape(-1,)
        orig_x = x.copy()
        sources = self.sources.reshape(-1,)
        inv_sources = ~sources
        #randomise starting potential
        x = np.random.random(x.shape)
        x[sources] = orig_x[sources]    
        #randomise starting potential
        
        start = clock()
        max_iter = int(max_iter)
        for i in range(max_iter):
            x = T.dot(x).reshape(-1,) # + D_inv_b all 0s
            x[sources] = orig_x[sources]
            error = np.linalg.norm(self.A.dot(x)[inv_sources])  
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
        
    def iterative_jacobi(self, tol=1e-2, max_iter=10000, max_time=10, verbose=True):
        sources = self.sources
        inv_source_mask = ~(sources.reshape(-1,1))
        '''
        Create array to contain the potential, including a boundary -
        the boundary conditions, which are never altered during the program's
        execution.
        Then 'fill in' the potential at the center of this matrix
        
        Initialising the potentials with 0s is not really necessary, since 
        this is done at instance creation anyway, plus this overwrites 
        later changes to the potential with preconditioning, for example
        '''        
        x = np.zeros((self.Nsx+2,self.Nsy+2))
        x[1:-1,1:-1] = self.potentials
        #get diagonal, D
        D = self.A.diagonal()
        start = clock()
        max_iter = int(max_iter)
        for iteration in xrange(max_iter):
            #need to use 'extended' step size array, since the 
            #boundary conditions are being used here as well
            x = self.jacobi_sub_func(x,
                                  np.array([self.Nsx,self.Nsy],dtype=np.int64),
                                  sources,D,self.grid.x_h_extended,
                                  self.grid.y_h_extended)
            x[1:-1,1:-1][sources] = self.grid.source_potentials[sources]
            error = np.linalg.norm(self.A.dot(x[1:-1,1:-1].reshape(-1,1))
                                                       [inv_source_mask])
#            plt.figure()
#            plt.imshow(x.T,origin='lower',interpolation='none')
#            plt.colorbar()
#            plt.show()
            if verbose:
                print("iteration, error:",iteration,error)
            if error < tol:
                print('Error in potential lower than tolerance')
                break
            if (clock()-start) > max_time:
                print('Time limit exceed')
                break
        self.potentials = x[1:-1,1:-1] 
        
    @staticmethod
    @jit(nopython=True,cache=True)
    def jacobi_sub_func(x,Ns,sources,D,x_h,y_h):
        out = np.zeros(x.shape)
        count = -1
        for i in range(0,Ns[0]):
            i_1 = i+1
            x_h_i = x_h[i]
            x_h_i_1 = x_h[i_1]
            x_h_divisor = x_h_i*x_h_i_1**2.
#            x_h_divisor = 1.
            for j in range(0,Ns[1]):
                count += 1
                j_1 = j+1
                y_h_j = y_h[j]
                y_h_j_1 = y_h[j_1]
                y_h_divisor = y_h_j*y_h_j_1**2. 
#                y_h_divisor = 1.
                if sources[i,j]:
                    continue
                '''
                Need:
                    i_1
                    i_1-1 -> i
                    i_1+1
                    j_1
                    j_1-1 -> j
                    j_1+1
                combinations:
                    i_1,j_1+1
                    i_1,j_1-1
                    i_1+1,j_1
                    i_1-1,j_1
                    these are transformed as above (needs fewer operations)
                '''
                out[i_1,j_1] = -((1./(D[count])) *(
                                 ((y_h_j*  x[i_1,j_1+1]+
                                   y_h_j_1*x[i_1,j])
                                     /y_h_divisor)+
                                 ((x_h_i*  x[i_1+1,j_1]+
                                   x_h_i_1*x[i,    j_1])
                                     /x_h_divisor)
                                              )
                             )
        return out
        
    def gauss_seidel(self, tol=1e-2, max_iter=10000, max_time=10, verbose=True):
        start = clock()
        #get diagonal, D
        D = sparse.diags(self.A.diagonal(),format='csc')
        L = sparse.tril(self.A,k=-1,format='csc')
        U = sparse.triu(self.A,k=1,format='csc')
        L_D_inv = sparse.linalg.inv(L+D)
#        L_D_inv_b = np.dot(L_D_inv,b)
        T = -L_D_inv.dot(U)
        if verbose:
            print("Gauss Seidel: finished creating matrices")
            print('Time taken:',clock()-start)
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
        max_iter = int(max_iter)
        for i in xrange(max_iter):
            x = T.dot(x).reshape(-1,) # + L_D_inv_b
            x[sources] = orig_x[sources]
            error = np.linalg.norm(self.A.dot(x)[inv_sources])   
            if verbose:
                print("i, error:",i,error)
            if error < tol:
                print('Error in potential lower than tolerance')
                break
            if (clock()-start) > max_time:
                print('Time limit exceed')
                break
        self.potentials = x.reshape(self.Nsx, -1)        
        
    def SOR(self, w=None, tol=1e-2, max_iter=1000000, max_time=10, verbose=True):
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
        if not w:
            avg_Ns = (self.Nsx+self.Nsy)/2.
            avg_h = 1./avg_Ns
            #assign ideal relaxation parameter
            w = 2./(1+np.sin(np.pi*avg_h))
            #however, this only holds for a square grid
            if self.Nsx != self.Nsy:
                #better to overestimate so increase
                w *= 1.05
                #do not increase too much, otherwise the SOR method
                #might become unstable
                if w>1.99:
                    w = 1.99
        
        self.w = w
        max_iter = int(max_iter)
        self.tol = tol
        sources = self.sources
        inv_source_mask = ~(sources.reshape(-1,1))
        '''
        Create array to contain the potential, including a boundary -
        the boundary conditions, which are never altered during the program's
        execution.
        Then 'fill in' the potential at the center of this matrix
        
        Initialising the potentials with 0s is not really necessary, since 
        this is done at instance creation anyway, plus this overwrites 
        later changes to the potential with preconditioning, for example
        '''        
        x = np.zeros((self.Nsx+2,self.Nsy+2))
        x[1:-1,1:-1] = self.potentials
        #get diagonal, D
        D = self.A.diagonal()
        Ns_array = np.array([self.Nsx,self.Nsy],dtype=np.int64)
        errors = np.zeros(max_iter,dtype=np.float64)
        times  = np.zeros(max_iter,dtype=np.float64)
        start = clock()
        increasing_error_count = -1 #will catch first time,
        #since -1th element is initialised to 0
        for iteration in xrange(max_iter):
            #need to use 'extended' step size array, since the 
            #boundary conditions are being used here as well
            x = self.SOR_sub_func(x, Ns_array,
                                  sources,w,D,self.grid.x_h_extended,
                                  self.grid.y_h_extended)
            error = np.linalg.norm(self.A.dot(x[1:-1,1:-1].reshape(-1,1))
                                                       [inv_source_mask])
            
            errors[iteration] = error
            time_diff = clock()-start
            times[iteration]  = time_diff
            if verbose:
                print("iteration, error:",iteration,error)
            if error < tol:
                print('Error in potential lower than tolerance')
                break
            if errors[iteration-1] < error:
                increasing_error_count += 1
                if increasing_error_count == 50:
                    print('Error has increased 50 times')
                    break
            if time_diff > max_time:
                print('Time limit exceed')
                break
        self.potentials = x[1:-1,1:-1] 
        self.errors = errors[:iteration+1]
        self.times = times[:iteration+1] 
        self.last_iteration = iteration
        
    @staticmethod
    @jit(nopython=True,cache=True)
    def SOR_sub_func(x,Ns,sources,w,D,x_h,y_h):
        w_1 = (1.-w)
#        initial_norm = np.linalg.norm(x)
        count = -1
        for i in range(0,Ns[0]):
            i_1 = i+1
            x_h_i = x_h[i]
            x_h_i_1 = x_h[i_1]
            x_h_divisor = x_h_i*x_h_i_1**2
            for j in range(0,Ns[1]):
                count += 1
                j_1 = j+1
                y_h_i = y_h[j]
                y_h_i_1 = y_h[j_1]
                y_h_divisor = y_h_i*y_h_i_1**2 
                if sources[i,j]:
                    continue
                '''
                Need:
                    i_1
                    i_1-1 -> i
                    i_1+1
                    j_1
                    j_1-1 -> j
                    j_1+1
                combinations:
                    i_1,j_1+1
                    i_1,j_1-1
                    i_1+1,j_1
                    i_1-1,j_1
                    these are transformed as above (needs fewer operations)
                '''
                x[i_1,j_1] = ( w_1*x[i_1,j_1] - 
                              (w/(D[count])) *(
                                 ((y_h_i*  x[i_1,j_1+1]+
                                   y_h_i_1*x[i_1,j])/y_h_divisor )+
                                 ((x_h_i*  x[i_1+1,j_1]+
                                   x_h_i_1*x[i,j_1])/x_h_divisor)
                                              )
                             )
        return x
        
    def interpolate(self,other):
        #calculate gradients between grid points of the other system
        #these gradients will then be used to interpolate the other potentials
        #to the current potentials
        gradients = gradient(other.potentials,other.grid.x_h,other.grid.y_h)
        #take difference between own x positions and other x positions
        x_diff = other.grid.x - self.grid.x.reshape(-1,1)
        #mask these differences, since we only want to find points 
        #where our own x positions are larger than the other x positions
#        print(x_diff)
        x_masked = np.ma.array(x_diff,mask=x_diff>1e-12)
#        print(x_masked)
        #find the closest grid point from the other x coordinates, which
        #came closest before our own x coordinates
        #find maximum here, since the differences are negative, and 
        #we wish to find the point closest to 0
        x_diff_maxs = np.max(x_masked,axis=1).reshape(-1,1)
        #get the indices of where this happens
        x_indices = np.where(x_diff==x_diff_maxs)[1]
        x_distances = np.abs(x_diff_maxs.flatten())
        
        #repeat the above for the y coordinates
        y_diff = other.grid.y - self.grid.y.reshape(-1,1)
        y_masked = np.ma.array(y_diff,mask=y_diff>1e-12)
        y_diff_maxs = np.max(y_masked,axis=1).reshape(-1,1)
        y_indices = np.where(y_diff == y_diff_maxs)[1]
        y_distances = np.abs(y_diff_maxs.flatten())
#        print(x_indices)
#        print(y_indices)
#        
        #create a 'dummy row' for the gradients at the very end, which would
        #then be used in the calculation for the new potential for those 
        #grid points which lie exactly on the last row or column.
        #Since the distances there are 0, the value of the gradient does
        #not matter there.
        x_gradients = np.zeros((gradients[0].shape[0]+1,gradients[0].shape[1]))
        y_gradients = np.zeros((gradients[1].shape[0],gradients[1].shape[1]+1))
        
        x_gradients[:-1] = gradients[0]
        y_gradients[:,:-1] = gradients[1]
        
        x_potentials = (other.potentials[x_indices]+
                        x_distances.reshape(-1,1)*x_gradients[x_indices])
        #use y indices to select resulting potentials,
        #effectively reshaping array once again to fit new, desired, shape
        x_potentials = x_potentials[:,y_indices]
        
#        print(other.potentials.shape)
#        print(y_indices.shape)
#        print(y_distances.shape)
#        print(y_gradients.shape)        
#        
        y_potentials = (other.potentials[:,y_indices]+
                        y_distances.reshape(1,-1)*y_gradients[:,y_indices])
        y_potentials = y_potentials[x_indices]
        
        self.potentials = (x_potentials+y_potentials)/2.
        self.potentials[self.sources] = self.grid.source_potentials[self.sources]
        
    def show(self,title='',grid_lines=True,
             quiver=False,every=1,**kwargs):
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
        Create list, where the first x_bound is interpolated backwards from x0
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
        x_bounds *= self.grid.distance_factor
        y_bounds *= self.grid.distance_factor
        fig,ax = plt.subplots()
        print('shapes:',x_bounds.shape,y_bounds.shape,self.potentials.T.shape)
        plot = ax.pcolorfast(x_bounds,y_bounds,
                             self.potentials.T*self.grid.potential_scaling)      
        if grid_lines:
            ax.add_collection(self.grid.create_grid_line_collection(**kwargs)) 
        if quiver:
            U,V = self.E_field
            mags = self.E_field_mag
#            U = -U.T
#            V = -V.T
#            X,Y = np.meshgrid(self.grid.x*self.grid.distance_factor,
#                              self.grid.y*self.grid.distance_factor)   

            X,Y = (self.grid.grid[0]*self.grid.distance_factor,
                   self.grid.grid[1]*self.grid.distance_factor)
            plt.quiver(X[::every,::every],Y[::every,::every],
                       U[::every,::every],V[::every,::every])
        cb = fig.colorbar(plot)
        plt.axis('square')
        plt.autoscale('tight')
        if title:
            plt.title(title)
        cb.set_label(r'$\mathrm{Potential\ (V)}$',fontsize=16)
        ax.set_xlabel(r'$\mathrm{x\ (m)}$',fontsize=16)
        ax.set_ylabel(r'$\mathrm{y\ (m)}$',fontsize=16)
        plt.show()
        
    def streamplot(self,title='',**fargs):
        plt.figure()
        U,V = self.E_field
        fields = self.E_field_mag
        U = -U.T
        V = -V.T        
        X,Y = np.meshgrid(self.grid.x*self.grid.size[0],
                          self.grid.y*self.grid.size[0])
        lw = 8*fields.T/np.max(fields)
        plt.streamplot(X, Y, U, V,
                       density = [1,1],
                       color = self.potentials.T,
                       linewidth = lw)
        if title:
            plt.title(title)
        plt.tight_layout() 
        plt.axis('tight')
        cb = plt.colorbar() 
        ax = plt.gca()
        cb.set_label(r'$\mathrm{Potential\ (V)}$',fontsize=16)
        ax.set_xlabel(r'\mathrm{x\ (m)}',fontsize=16)
        ax.set_ylabel(r'\mathrm{y\ (m)}',fontsize=16)          
        plt.show()        
        
    def show_setup(self):
        self.grid.show()
        
    def calculate_E_field(self):
        print('calculating e field')
        self._E_field = gradient(self.potentials*self.grid.potential_scaling,
                                 self.grid.x_h*self.grid.distance_factor,
                                 self.grid.y_h*self.grid.distance_factor)
        self._E_field[0] *= -1
        self._E_field[1] *= -1
        self._E_field_magnitude = np.sqrt(self._E_field[0]**2 + 
                                         self._E_field[1]**2)
    
def build_from_segments(x=None,y=None,Ns=None):
    """
    Create arrays containing step sizes to be used to initialise a 
    Grid instance
    *Ns*:
        Number of grid points, can be a tuple (see below)
    *x*,*y*:
        need to be iterable, a series of parameters specifying the start,end
        and either the grid spacing in the interval, or the number of
        intervals therein.
    To be used like so:
        xh,yh = build_from_segments((100,50),
                                    x=((end1,steps1),(end2,steps2)),
                                    y=((end1,steps1),(end2,steps2)))
        *end*: float
            ranging from 0 to 1, describing position along the axis
        *steps*: int
            number of grid points between *start* and *end*
            the sum of all steps must add up to the number of steps given in
            *Ns* for each axis.
        can then create Grid instance:
            grid_instance = Grid(xh,yh)
        and then use that to initialise an AMR_system instance:
            system = AMR_system(test)
    Instead of the above, *steps* can be replaced by a grid-spacing.
    Thus, if one wishes to specify grid spacings, the input should be < 1
    and of type float
    """
    outputs = []
    for args in [x,y]:
        if not args:
            continue
        ends,params = zip(*args)
        starts = (0.,)+ends[:-1]
#        print(starts,ends,params)
        assert starts[1:] == ends[:-1],'Starts should match ends'
        if np.all(np.array(params)>=1):
            #assume step numbers are specified
            starts = np.array(starts,dtype=np.float64)
            ends = np.array(ends,dtype=np.float64)
            steps = np.array(params,dtype=np.int64)
            steps_cumulative = np.cumsum(steps)
#            print('cumulative',steps_cumulative)
            grid_spacings = (ends-starts)/steps
#            print('stepsizes',grid_spacings)
            h = np.zeros(np.sum(steps))
            for i in range(steps.size):
                if not i:
                    start = 0
                else:
                    start = steps_cumulative[i-1]
                end = steps_cumulative[i]
                print('start end',start,end)
                h[start:end] = grid_spacings[i]
            outputs.append(h)
        elif np.all(np.array(params)<1):
#            print('spacings')
            #assume grid spacings are specified
            starts = np.array(starts,dtype=np.float64)
            ends = np.array(ends,dtype=np.float64)
            grid_spacings = np.array(params,dtype=np.float64)    
#            print('spacings',grid_spacings)
            steps = (ends-starts)/grid_spacings
            int_steps = np.array(np.round(steps),dtype=np.int64)
            assert np.sum(steps - int_steps)<1e-10,(
                    'Grid spacings should match starts and ends!')
            steps = int_steps
            steps_cumulative = np.cumsum(steps)
            grid_spacings = (ends-starts)/steps     #adjust for whole numbers
            h = np.zeros(np.sum(steps))
#            print('steps',steps)
            for i in range(steps.size):
                if not i:
                    start = 0
                else:
                    start = steps_cumulative[i-1]
                end = steps_cumulative[i]
#                print('start end',start,end)
                h[start:end] = grid_spacings[i]
            outputs.append(h)            
        else:
            print('Non-homogeneous parameter specifications!')
    if len(outputs)==1:
        outputs = [outputs[0],outputs[0]]
    if Ns:
        if hasattr(Ns,'__iter__'):
            Nsx,Nsy = Ns
        else:
            Nsx = Ns
            Nsy = Ns            
        if (len(outputs[0]) != (Nsx-1)) or (len(outputs[1]) != (Nsy-1)):
            print('Number of stepsizes output do not agree with number input!')
    return outputs
    
if __name__ == '__main__':
#    xh,yh = build_from_segments(((0.1,0.01),(0.25,0.005),(1,0.01)),
#                                ((0.35,0.01),(0.45,0.005),(1,0.01))
#                               )
    xh,yh = build_from_segments(((1,50),),((1,200),))
    test = Grid(xh,yh)
    test.rectangle(1,(0.5,0.5),0.4,0.7)
    test.rectangle(1,(0.2,0.4),0.02,0.02)
    test.show(color=(0,0,0,0.1))
    system = AMR_system(test)

#    system.SOR(max_iter=10000,max_time=1,tol=1e-10,verbose=True)
#    system.iterative_jacobi()
    system.SOR(max_time=1,verbose=False)
#    system.gauss_seidel()
    system.show(quiver=True,every=5)
