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
from __future__ import print_function
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
# np.set_printoptions(threshold=np.inf)

class Shape:
    def __init__(self,Ns,potential,origin,*args,**kwargs):
        '''
        'Ns' is the number of gridpoints along an axis, so the
        grid spacing h is the inverse of this, since the axes range from 0 to 1
        '''
        assert type(Ns)==int, 'Ns should be an integer'
        self.Ns = Ns
        self.h = 1./Ns
        self.potential = potential
        self.grid = np.mgrid[0:1:complex(0, self.Ns), 0:1:complex(0, self.Ns)]
        self.potentials = np.zeros(self.grid.shape[1:])
        self.sources = np.zeros(self.potentials.shape,dtype=np.bool)
        self.source_potentials = np.zeros(self.sources.shape)
        self.add_source(origin,*args,**kwargs)

    def find_closest_gridpoint(self,coords):
        '''
        Find closest point on grid to coords
        '''
        coords = np.array(coords).reshape(-1,)
#        print('coords:',coords)
        # make sure coords have correct shape for next step
        # really necessary (could have assert)?
        coords = coords[(slice(None),)+(None,)*coords.shape[0]]
        # reshape to (2,1,1) or (3,1,1,1) etc. for correct
        # broadcasting
        abs_diff = np.abs(self.grid - coords)
        match = []
        N_dim = self.grid.shape[0]
        for i in range(N_dim):
            index = np.where(abs_diff[i]==np.min(abs_diff[i]))[i][0]
            match.append(int(index))
#        print('match:',match)
        return match
        
    def add_source(self,origin,*args,**kwargs):
        '''
        *args: coord1[,coord2]
        **kwargs: 'shape':'circle' or 'rectangle'

        In 2D, a regular shape can be specified by 3 parameters, the centre
        (origin) of the shape, and two additional coordinates (could be more
        if irregular shapes would be implemented).
        The centre of the shapes described below is specified by the
        'origin' argument, with the top left corner of the simulated space
        defined as (0,0), and the bottom left corner as (1,1).
        (Note: These positions only pertain to the position in say, a matrix,
        where the x and y positions increase from the top left to the
        bottom right - this is also apparent in visualisation generated using
        imshow, for example.)
        If only origin is given (ie. no non-keyword args), then a point source
        at the grid point closest to the origin is created.
        implemented here:
            'rectangle':origin is the centre of the rectangle
                        coord1 is width
                        coord2 is height
                (square as special case (coord1 = coord2),
                 would be orientated with edges vertical & horizontal)
            'circle':   origin is the centre of the circle
                        coord1 is the radius
        '''
        if not args: #if only origin is specified
            #need to convert list to tuple below so that the right kind of
            #indexing is triggered
            source_coords = tuple(self.find_closest_gridpoint(origin))
#            print('source coords:',source_coords)
            self.potentials[source_coords] = self.potential
            self.sources[source_coords] = True
            self.source_potentials[source_coords] = self.potential
            # terminate function here, since the necessary action has been
            # performed.
            return None
        # shape selection
        if 'shape' in kwargs:
            # shape specified explicitly
            shape = kwargs['shape']
        else:
            shape = 'rectangle'
        # select whether the shape should be filled or not
        if 'filled' in kwargs:
            filled = kwargs['filled']
        else:
            filled = True

        not_implemented_message = (
"""{:} args not implemented for shape: {:}
No shape has been added, please refer to function
documentation""".format(len(args),shape))

        if shape == 'rectangle' or shape == 'square':
            if 1 <= len(args) <= 2:
                width = args[0]
                width_grid = int(round(args[0]/self.h))
                half_width_grid = int(round(args[0]/(2*self.h)))
                if len(args) == 1:
                    height = width
                    height_grid = width_grid
                    half_height_grid = half_width_grid
                else:
                    height = args[1]
                    height_grid = int(round(args[1]/self.h))
                    half_height_grid = int(round(args[1]/(2*self.h)))
                    
                print ("Adding {:} centred at {:} with "
                       "width: {:.3f}, height: {:.3f}".format(shape,origin,
                                                        width,height))

                origin_grid = self.find_closest_gridpoint(origin)
                min_x = origin_grid[0] - half_width_grid
                max_x = min_x + width_grid
                min_y = origin_grid[1] - half_height_grid
                max_y = min_y + height_grid
                def limiter(coord):
                    if coord < 0:
                        return 0
                    elif coord >= self.Ns:
                        return self.Ns-1
                    else:
                        return coord
                starts  =    map(limiter,[min_y,min_x,max_y,max_x])
                targets =    map(limiter,[max_y,max_x,min_y,min_x])
                keep_fixed = map(limiter,[min_x,max_y,max_x,min_y])
                changing =   [1,    0,    1,    0    ] #0 for x, 1 for y
                const = [1,0]
#                print(origin_grid)
#                print(starts)
#                print(targets)
#                print(keep_fixed)
#                print(self.sources)
                for start,target,fixed,change in zip(starts,targets,
                                                     keep_fixed,changing):
#                    print('')
#                    print(start,target,fixed,change)
                    '''
                    Need +1 on end, since the direction of traversal 
                    changes ie. always increases (although this could be 
                    changed), and therefore a grid point may be missed out,
                    since the clockwise 'path' is not traced out directly.
                    '''
                    for variable in range(min(start,target),
                                          max(start,target)+1):
                        r = [0,0]
                        r[change] = variable
                        r[const[change]] = fixed
#                        print('r',r)
                        r = tuple(r) #need tuple for proper type of indexing!
                        self.potentials[r] = self.potential
                        self.sources[r] = True
                        self.source_potentials[r] = self.potential
#                print(self.sources)
                if filled:
#                    print('filling')
                    self.fill()
            else:
                print(not_implemented_message)
        if shape == 'circle':
            if len(args) == 1:
                print("Adding circle centred at {:} with radius {:}".format(
                origin,args[0]))
                # interval of angles calculated so that every grid point
                # should be covered ~1-2 times for a given radius, to make
                # sure that every grid point is covered
                r = args[0]
                d_theta = self.h/(2*r)
                for theta in np.arange(0,2*np.pi,d_theta):
                    self.add_source((origin[0]+r*np.sin(theta),
                                     origin[1]+r*np.cos(theta)))
                if filled:
                    self.fill()
            else:
                print(not_implemented_message)
    def fill(self):
        '''
        fill shape row-wise, assigning the same potential throughout
        if 2 or more (should just be 2 maximum) grid points are marked
        as being a source, mark the grid points in between these as
        sources too, with the same potential
        '''
        for i,row in enumerate(self.sources):
            indices = np.where(row==True)[0]
            if indices.shape[0]>1:
                min_index = indices[0]
                max_index = indices[-1]
                for index in range(min_index+1,max_index):
                    self.potentials[i,index] = self.potential
                    self.sources[i,index] = True
                    self.source_potentials[i,index] = self.potential

class System:
    def __init__(self,Ns):
        '''
        'Ns' is the number of gridpoints along an axis, so the
        grid spacing h is the inverse of this, since the axes range from 0 to 1
        '''
        assert type(Ns)==int, 'Ns should be an integer'
        self.Ns = Ns
        self.h = 1./Ns
        self.grid = np.mgrid[0:1:complex(0, self.Ns), 0:1:complex(0, self.Ns)]
        self.potentials = np.zeros(self.grid.shape[1:])
        self.sources = np.zeros(self.potentials.shape,dtype=np.bool)
        self.source_potentials = np.zeros(self.sources.shape)
    def add(self,shape_instance):
        '''
        Add sources to the system using instances of the 'Shape' class.
        Note: Potentials of overlapping shapes are added. Once a grid point
            has been assigned to a source, its potential will remain fixed
            throughout.
        '''
        assert shape_instance.Ns == self.Ns, 'Grids should be the same'
        self.potentials += shape_instance.potentials
        self.sources += shape_instance.sources
        self.source_potentials += shape_instance.source_potentials

    def cross_section(self,side_length,show=True,savepath=''):
        '''
        now, plot a cross section of the potential across the central row.
        Ideally, the number of grid points should be an ODD number for this
        to work ideally - due to the symmetry of the problem
        '''
        mid_row_index = int((self.Ns-1)/2.)
        cross_section = self.potentials[mid_row_index]
        plt.figure()
        plt.title('1-D Cross-Section of the Potential across the System\n'
                  +'tol = {:.2e}, Ns = {:.2e}, side length = {:.3e}'.
                  format(self.tol,self.Ns,side_length))
        grid_positions = self.grid[0][:,0]
        plt.plot(grid_positions,cross_section,label='potential')
        plt.xlabel('Distance from left wall (natural units)')
        plt.ylabel('Potential (V)')
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
    def show_setup(self,title='',**fargs):
        '''
        Show the sources in the system
        '''
        plt.figure()
        plt.title('Sources')
        plt.imshow(self.source_potentials.T,origin='lower',**fargs)
        plt.colorbar()
        plt.tight_layout()
        if title:
            plt.title(title)        
        plt.show()
    def show(self,every=1,title='',**fargs):
        '''
        Show the calculated potential
        '''
        plt.figure()
        plt.title('Potential')
        plt.imshow(self.potentials.T,origin='lower',**fargs)
        U,V = np.gradient(self.potentials)
        plt.colorbar()
#        U = U[::every,::every]
#        V = V[::every,::every]
        X,Y = np.meshgrid(np.arange(self.Ns),np.arange(self.Ns))
        U = U.T
        V = V.T
        plt.quiver(X[::every,::every],Y[::every,::every],
                   U[::every,::every],V[::every,::every])
        plt.tight_layout()
        if title:
            plt.title(title)
        plt.show()
    def create_method_matrix(self):
        N = self.Ns**2
        self.A = np.zeros((N, N))
        boundary_conditions = []
        for i in range(N):
            boundaries_row = []
            coord1 = int(float(i)/self.Ns)
            coord2 = i%self.Ns
            self.A[i,i] = -4
            for c1,c2 in zip([coord1,coord1,coord1-1,coord1+1],
                             [coord2-1,coord2+1,coord2,coord2]):
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
                        print("error",c1,c2)
                except IndexError:
                    boundaries_row.append((c1,c2))
            boundary_conditions.append(boundaries_row)
        self.boundary_conditions = boundary_conditions
    def jacobi(self, tol=1e-3, max_iter=5000, verbose=True):
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
        #randomise starting potential
        x = np.random.random(x.shape)
        x[sources] = orig_x[sources]    
        #randomise starting potential
        D_inv = np.linalg.inv(D)
        L_U = L+U
        T = - np.dot(D_inv, L_U)
        D_inv_b = np.dot(D_inv, b).reshape(-1,)
        print("Jacobi: finished creating matrices")
        for i in range(max_iter):
            initial_norm = np.linalg.norm(x)
            x = np.dot(T,x).reshape(-1,) + D_inv_b
            x[sources] = orig_x[sources]
            final_norm = np.linalg.norm(x)
            diff = np.abs(initial_norm-final_norm)
            if verbose:
                print("i,diff:",i,diff)
            if diff < tol:
                break
        self.potentials = x.reshape(self.Ns,-1)
    def gauss_seidel(self, tol=1e-3, max_iter=5000, verbose=True):
        N = self.Ns**2
        #create array (matrix) A
        self.create_method_matrix()
        b = np.zeros(N)

        #get diagonal, D
        D = np.diag(np.diag(self.A)) #but these are all just -4
        L = np.tril(self.A,k=-1)
        U = np.triu(self.A,k=1)
        L_D_inv = np.linalg.inv(L+D)
        L_D_inv_b = np.dot(L_D_inv,b)
        T = -np.dot(L_D_inv,U)
        print("Gauss Seidel: finished creating matrices")
        x = self.potentials.reshape(-1,)
        orig_x = x.copy()
        sources = self.sources.reshape(-1,)
        #randomise starting potential
        x = np.random.random(x.shape)
        x[sources] = orig_x[sources]    
        #randomise starting potential

        for i in range(max_iter):
            #print "before\n",x.reshape(self.Ns,-1)
            initial_norm = np.linalg.norm(x)
            x = np.dot(T,x).reshape(-1,) + L_D_inv_b
            x[sources] = orig_x[sources]
            #print "sources",x[sources]
            final_norm = np.linalg.norm(x)
            diff = np.abs(initial_norm-final_norm)
            #print "after\n",x.reshape(self.Ns,-1)
            if verbose:
                print("i,diff:",i,diff)
            if diff < tol:
                break
            #print ''
        self.potentials = x.reshape(self.Ns, -1)
    def SOR(self, w=1.2, tol=1e-3, max_iter=5000, verbose=True):
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
        self.tol = tol
        self.w = w
        Ns = self.Ns
        w = float(w)
        sources = self.sources
        '''
        Create array to contain the potential, including a boundary -
        the boundary conditions, which are never altered during the program's
        execution.
        Then 'fill in' the potential at the center of this matrix
        '''
        x = np.zeros((Ns+2,Ns+2))      
        x[1:-1,1:-1] = self.source_potentials 
        '''
        better choice than random initial state needs to be found!
        could use pre-conditioning with coarse grid, which is initialised
        with
        '''
        x = self.SOR_sub_func(max_iter,x,Ns,sources,w,tol,verbose)
        self.potentials = x[1:-1,1:-1]
        
    @staticmethod
    @jit(nopython=True,cache=True)
    def SOR_sub_func(max_iter,x,Ns,sources,w,tol,verbose):
        w_1 = (1.-w)
        w_4 = (w/(4.))
        for iteration in range(max_iter):
            initial_norm = np.linalg.norm(x)
            for i in range(0,Ns):
                i_1 = i+1
                for j in range(0,Ns):
                    j_1 = j+1
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
                    x[i_1,j_1] = w_1*x[i_1,j_1] + w_4 *(x[i_1,j_1+1]+
                                                        x[i_1,j]+
                                                        x[i_1+1,j_1]+
                                                        x[i,j_1])

            final_norm = np.linalg.norm(x)
            diff = np.abs(initial_norm-final_norm)
            
            if verbose:
                print("iteration, diff:",iteration,diff)
            
            if diff < tol:
                break  
        return x

    def SOR_anim(self, w=1.5, tol=1e-3, max_iter=5000, verbose=True):
        '''
        Equivalent to SOR except for the fact that it returns an array
        of all the potentials calculated along the way, for later
        plotting.
        '''
        Ns = self.Ns
        w = float(w)
        sources = self.sources
        '''
        Create array to contain the potential, including a boundary -
        the boundary conditions, which are never altered during the program's
        execution.
        Then 'fill in' the potential at the center of this matrix
        '''
        x = np.zeros((Ns+2,Ns+2))
        #randomise starting potential
        x_seed = np.random.random(self.potentials.shape)
        x_seed[sources] = self.source_potentials[sources]    
        #randomise starting potential        
        x[1:-1,1:-1] = x_seed     
        '''
        better choice than random initial state needs to be found!
        could use pre-conditioning with coarse grid, which is initialised
        with
        '''
        x, all_potentials = self.SOR_sub_func_anim(max_iter,x,Ns,sources,w,tol,verbose)
        self.potentials = x[1:-1,1:-1]
        return all_potentials
        
    @staticmethod 
    @jit(nopython=True)
    def SOR_sub_func_anim(max_iter,x,Ns,sources,w,tol,verbose):
        all_potentials = np.zeros((max_iter, Ns, Ns))
        for iteration in range(max_iter):
            all_potentials[iteration] = x[1:-1,1:-1]
            initial_norm = np.linalg.norm(x)
            for i in range(0,Ns):
                i_1 = i+1
                for j in range(0,Ns):
                    j_1 = j+1
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
                    x[i_1,j_1] = (1.-w)*x[i_1,j_1] + (w/(4.)) *(x[i_1,j_1+1]+
                                                                x[i_1,j]+
                                                                x[i_1+1,j_1]+
                                                                x[i,j_1])
            final_norm = np.linalg.norm(x)
            diff = np.abs(initial_norm-final_norm)
            if verbose:
                print("iteration, diff:",iteration,diff)
            if diff < tol:
                break              
        return x,all_potentials[:iteration+1,...]
        
if __name__ == '__main__':        
    Ns = 20
    test = System(Ns)
    '''
    #used for 'grid size case study' folder images
    test.add(Shape(Ns,-4,(0.9,0.9),0.1,shape='square'))
    test.add(Shape(Ns,-1.3,(0.5,0.5),0.18,shape='circle',filled=False))
    test.add(Shape(Ns,1.8,(0.5,0.5),0.1,shape='circle',filled=False))
    test.add(Shape(Ns,2,(0.5,0.5),0.3,shape='circle',filled=False))
    test.add(Shape(Ns,-1,(0.354,0.506),0.03,shape='circle',filled=False))
    test.add(Shape(Ns,1,(0.37,0.49),0.03,shape='circle',filled=False))
    '''
#    test.add(Shape(Ns,1,(0.3,0.5),0.01,0.5))
#    test.add(Shape(Ns,-1,(0.7,0.5),0.01,0.5))
    test.add(Shape(Ns,-5,(0.5,0.5)))
    
    calc = 1
    tol = 1e-6
    max_iter = 5000
    show = True
    #methods = [test.SOR,test.jacobi,test.gauss_seidel]
    methods = [test.SOR]
    names = [f.__name__ for f in methods]
    if calc:
        for name,f in zip(names,methods):
            print(name)
            f(tol=tol,max_iter=max_iter)
            if show:
                test.show(title=name,interpolation='none',every=1)
