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


26/11/16:
    Implemented different numbers of grid points along the two axes,
    where the x-axis is to have a length of 1, regardless of the proportion
    of the two grid point numbers. 
    When specifying Ns = (20,40), for example, the grid point at the top right
    would be at distance r = (1,2) from the origin in the lower left corner.
    The stepsize is therefore always defined with respect to the number of 
    grid points along the x axis.

"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
# np.set_printoptions(threshold=np.inf)

class Shape:
    def __init__(self,Ns,potential,origin,*args,**kwargs):
        '''
        'Ns' is the number of gridpoints along an axis, so the
        grid spacing h is the inverse of this, since the axes range from 0 to 1
        By giving a tuple of (Nsx,Nsy) values, the number of gridpoints along
        the x and y axis may be specified, resulting in a rectangular grid.
        The stepsize between the grid points will be kept constant along each
        axis.
        The 'length' of the grid in the x direction will be defined as 1,
        with the 'length' of the grid in the y direction depending on the
        ratio between the Nsy and Nsx given.
        If only a single value is given for the 'Ns' argument, the grid will
        default to being a square grid with Ns gridpoints along each axis.
        '''
        if hasattr(Ns,'__iter__'):
            Nsx,Nsy = Ns
        else:
            Nsx = Ns
            Nsy = Nsx
#        assert type(Nsx)==int, 'Ns should be an integer'
#        assert type(Nsy)==int, 'Ns should be an integer'
        self.Nsx = int(Nsx)
        self.Nsy = int(Nsy)
        self.aspect_ratio = float(Nsy)/Nsx
        self.h = 1./Nsx
        self.potential = potential
        self.grid = np.mgrid[0:1:complex(0, self.Nsx), 
                             0:self.aspect_ratio:complex(0, self.Nsy)]
        self.potentials = np.zeros(self.grid.shape[1:])
        self.sources = np.zeros(self.potentials.shape,dtype=np.bool)
        self.source_potentials = np.zeros(self.sources.shape)
        self.shape_creation_args = []        
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
#        print('abs diff:\n',abs_diff)
        match = []
        N_dim = self.grid.shape[0]
        for i in range(N_dim):
            index = np.where(abs_diff[i]==np.min(abs_diff[i]))[i][0]
            match.append(int(index))
#        print('match:',match)
        source_coords = tuple(match)
        return source_coords        
    def add_closest_gridpoint(self,coords):
        '''
        Add closest point on grid to coords
        '''
        source_coords = self.find_closest_gridpoint(coords)
        self.potentials[source_coords] = self.potential
        self.sources[source_coords] = True
        self.source_potentials[source_coords] = self.potential        

    def limiter(self,coord,axis=None):
        '''
        Used so that shapes are 'clipped' at the edges of the
        grid
        '''
        if axis=='x':
            Ns = self.Nsx
        elif axis=='y':
            Ns = self.Nsy
        else:
            raise Exception("Please select an axis ('x','y')")
        if coord < 0:
            return 0
        elif coord >= Ns:
            return Ns-1
        else:
            return coord
             
    def add_source(self,origin,*args,**kwargs):
        '''
        *args: coord1[,coord2]
        **kwargs: 'shape':'circle', 'rectangle' or 'point'

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
        self.shape_creation_args.append([(self.Nsx,self.Nsy),self.potential,
                                         origin,args,kwargs])

        # shape selection
        if 'shape' in kwargs:
            # shape specified explicitly
            shape = kwargs['shape'].lower()
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

        if shape == 'point': #if only origin is specified, 'point' source
            print("Adding grid-point source at {:}".format(origin))
            #need to convert list to tuple below so that the right kind of
            #indexing is triggered
            self.add_closest_gridpoint(origin)
        
        elif shape == 'rectangle' or shape == 'square':
            if 1 <= len(args) <= 2:
                width = args[0]
                width_grid = int(round(args[0]/self.h))
                if len(args) == 1:
                    height = width
                    height_grid = width_grid
                else:
                    height = args[1]
                    height_grid = int(round(args[1]/self.h))
                    
                print ("Adding {:} centred at {:} with "
                       "width: {:.3f}, height: {:.3f}".format(shape,origin,
                                                        width,height))

                min_x = int(round((origin[0]-width/2.)/self.h))
                min_y = int(round((origin[1]-height/2.)/self.h))
                max_x = min_x + width_grid
                max_y = min_y + height_grid
                
                min_x = self.limiter(min_x,axis='x')
                max_x = self.limiter(max_x,axis='x')
                min_y = self.limiter(min_y,axis='y')
                max_y = self.limiter(max_y,axis='y')              
                
                starts  =    [min_y,min_x,max_y,max_x]
                targets =    [max_y,max_x,min_y,min_x]
                keep_fixed = [min_x,max_y,max_x,min_y]
                changing =   [1,    0,    1,    0    ] #0 for x, 1 for y
                const = [1,0]
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
        elif shape == 'circle':
            if len(args) == 1:
                if filled:
                    print("Adding filled circle centred at {:} with radius {:}"
                            .format(origin,args[0])) 
                    mask = ((self.grid[0]-origin[0])**2 + 
                            (self.grid[1]-origin[1])**2) < args[0]**2
                    self.potentials[mask] = self.potential
                    self.sources[mask] = True
                    self.source_potentials[mask] = self.potential
                else:
                    print("Adding circle centred at {:} with radius {:}".format(
                    origin,args[0]))
                    origin_coords = (int(origin[0]/self.h),int(origin[1]/self.h))
                    '''
                    # interval of angles calculated so that every grid point
                    # should be covered ~1-2 times for a given radius, to make
                    # sure that every grid point is covered
                    r = args[0]
                    d_theta = self.h/(2*r)
                    for theta in np.arange(0,2*np.pi,d_theta):
                        self.add_closest_gridpoint((origin[0]+r*np.sin(theta),
                                                    origin[1]+r*np.cos(theta)))
                    '''
                    '''
                    Calculate for first quadrant, which should then map onto
                    all other coordinates, as they are symmetrical
                    First octant 'ends' when two consecutive intervals 
                    feature a change in both x and y grid points, as 
                    before, only the y grid point will change with every iteration,
                    and the x coordinate only occasionally.
                    The first octant also ends when x=y, assuming that 
                    one starts from y = 0 on the right upper quadrant.
                    '''
                    x = int(args[0]/self.h)  
                    y = 0
                    err = (5./4)-x  #initialise with 5/4 - r - Mid point algorithm
                    indices = [[],[]]
                    while (x >= y):
                        r = (origin_coords[0]+x,origin_coords[1]+y)
                        indices[0].append(r[0])
                        indices[1].append(r[1])
                        
                        r = (origin_coords[0]+x,origin_coords[1]-y)
                        indices[0].append(r[0])
                        indices[1].append(r[1])
                        
                        r = (origin_coords[0]+y,origin_coords[1]+x)
                        indices[0].append(r[0])
                        indices[1].append(r[1])
                        
                        r = (origin_coords[0]+y,origin_coords[1]-x)
                        indices[0].append(r[0])
                        indices[1].append(r[1])
                                                                
                        r = (origin_coords[0]-x,origin_coords[1]+y)
                        indices[0].append(r[0])
                        indices[1].append(r[1])
                        
                        r = (origin_coords[0]-x,origin_coords[1]-y)
                        indices[0].append(r[0])
                        indices[1].append(r[1])
                        
                        r = (origin_coords[0]-y,origin_coords[1]+x)
                        indices[0].append(r[0])
                        indices[1].append(r[1])
                        
                        r = (origin_coords[0]-y,origin_coords[1]-x)
                        indices[0].append(r[0])
                        indices[1].append(r[1])
                        
                        y += 1
                        err += 1 + 2*y
                        if (2*(err-x)) + 1 > 0:
                            x -= 1
                            err += 1 - 2*x
                            
#                    tuple_x,tuple_y = tuple(indices[0]),tuple(indices[1])
                    #comparably small loss in speed due to below
                    tuple_x = tuple([self.limiter(i,axis='x') for i in indices[0]])
                    tuple_y = tuple([self.limiter(i,axis='y') for i in indices[1]])
                    self.potentials[tuple_x,tuple_y] = self.potential
                    self.sources[tuple_x,tuple_y] = True
                    self.source_potentials[tuple_x,tuple_y] = self.potential
            else:
                print(not_implemented_message)
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
        By giving a tuple of (Nsx,Nsy) values, the number of gridpoints along
        the x and y axis may be specified, resulting in a rectangular grid.
        The stepsize between the grid points will be kept constant along each
        axis.
        The 'length' of the grid in the x direction will be defined as 1,
        with the 'length' of the grid in the y direction depending on the
        ratio between the Nsy and Nsx given.
        If only a single value is given for the 'Ns' argument, the grid will
        default to being a square grid with Ns gridpoints along each axis.
        '''
        if hasattr(Ns,'__iter__'):
            Nsx,Nsy = Ns
        else:
            Nsx = Ns
            Nsy = Nsx
#        assert type(Nsx)==int, 'Ns should be an integer'
#        assert type(Nsy)==int, 'Ns should be an integer'
        self.Nsx = int(Nsx)
        self.Nsy = int(Nsy)
        self.aspect_ratio = float(Nsy)/Nsx
        self.h = 1./Nsx
        self.grid = np.mgrid[0:1:complex(0, self.Nsx), 
                             0:self.aspect_ratio:complex(0, self.Nsy)]
        self.potentials = np.zeros(self.grid.shape[1:])
        self.sources = np.zeros(self.potentials.shape,dtype=np.bool)
        self.source_potentials = np.zeros(self.sources.shape)
        self.shape_creation_args = []    
#        self.shapes = []
        
    def add(self,shape_instance):
        '''
        Add sources to the system using instances of the 'Shape' class.
        Note: Potentials of overlapping shapes are added. Once a grid point
            has been assigned to a source, its potential will remain fixed
            throughout.
        '''
        assert shape_instance.Nsx == self.Nsx, 'Grids should be the same'
        assert shape_instance.Nsy == self.Nsy, 'Grids should be the same'        
        self.potentials += shape_instance.potentials
        self.sources += shape_instance.sources
        self.source_potentials += shape_instance.source_potentials
#        don't store the shape instance, as this uses a lot of RAM and is
#        not really used throughout the code
#        self.shapes.append(shape_instance) 
        
        self.shape_creation_args.extend(shape_instance.shape_creation_args)
           
    def sampling(self,Ns_new):
        '''
        Tuple Ns_new
        
        Interpolate the known potentials ('potentials_old') based on the 
        current grid sizing ('Ns_old') and use this to determine the 
        potentials on a new grid sizing given by 'Ns_new'
        '''
        U,V = np.gradient(self.potentials)
        new_aspect_ratio = float(Ns_new[1])/Ns_new[0]
        aspect_ratio_ratio = self.aspect_ratio/new_aspect_ratio
        '''
        U gives gradient at each grid point going "down" the columns,
        V gives the gradient at each grid point going "right" along the rows
        '''
        new_potentials = self.sampling_sub_func(np.array([self.Nsx,self.Nsy],
                                                         dtype=np.int64),
                                                np.array(Ns_new,
                                                         dtype=np.int64),
                                                self.potentials,U,V,
                                                aspect_ratio_ratio)
        return new_potentials
        
    @staticmethod
    @jit(nopython=True,cache=True)
    def sampling_sub_func(Ns_old,Ns_new,potentials_old,U,V,
                          aspect_ratio_ratio):
        '''
        Array Ns_old, Ns_new
        
        First go down the columns, and create an array of new potentials based
        on this.
        Select the column with index j
        Divide 'new_h_values' by the old grid spacing 'h_old',
        and take the integer value of the result in order to determine
        which gridpoint came before - this previous gridpoint's potential 
        and gradient will be used to estimate the desired potential.
        
        The new points will lie in between two old rows & columns in general,
        so the proportion of those should be taken into account in the
        calculation below, however, it should be good enough to simply
        follow the route of taking the integer part of the ratio between
        the new and the old step-sizes, since this approximation will get
        better the more gridpoints there are. Also, assuming that the variation
        between two adjacents columns / rows will be small will become more
        valid as the physical spacing between grid points decreases, as well.
        '''        
        h_old = 1./Ns_old[0]
        h_new = 1./Ns_new[0]
        new_h_values_x = np.arange(Ns_new[0])*h_new
        new_h_values_y = np.arange(Ns_new[1])*h_new*aspect_ratio_ratio
        
        point_on_old_x = (new_h_values_x/h_old)
        point_on_old_int_x = np.zeros(point_on_old_x.shape[0],dtype=np.int64)
        for i in range(point_on_old_x.shape[0]):
            point_on_old_int_x[i] = int(point_on_old_x[i])
        distances_x = point_on_old_x - point_on_old_int_x
        
        point_on_old_y = (new_h_values_y/h_old)
        point_on_old_int_y = np.zeros(point_on_old_y.shape[0],dtype=np.int64)
        for i in range(point_on_old_y.shape[0]):
            point_on_old_int_y[i] = int(point_on_old_y[i])
        distances_y = point_on_old_y - point_on_old_int_y     
    
#        print('ns new',Ns_new)
#        print('ns old',Ns_old)
#        print('new h values x',new_h_values_x)
#        print('point on old x',point_on_old_x)
#        print('point on old x int',point_on_old_int_x)
#        print('distances x',distances_x)        
#        
#        print('new h values y',new_h_values_y)
#        print('point on old y',point_on_old_y)
#        print('point on old y int',point_on_old_int_y) 
#        print('distances y:',distances_y)
#        
#        print('U shape',U.shape)
#        print('V shape',V.shape)
        
        
        new_potentials_rowwise = np.zeros((Ns_new[0],Ns_new[1]))
        for j_new in range(Ns_new[1]):
            j_old = point_on_old_int_y[j_new]
            column_potentials_old = potentials_old[:,j_old]
            gradients = U[:,j_old]
            for i_new in range(Ns_new[0]):
                i_old = point_on_old_int_x[i_new]
                new_potentials_rowwise[i_new,j_new] = (
                                                column_potentials_old[i_old]+
                                                gradients[i_old]*
                                                distances_x[i_new])
        '''
        Now repeat the above, just for the columns, and then average the
        results in order to get the final estimate.
        '''  
        new_potentials_columnwise = np.zeros((Ns_new[0],Ns_new[1]))        
        for i_new in range(Ns_new[0]):
            i_old = point_on_old_int_x[i_new]
            row_potentials_old = potentials_old[i_old,:]
            gradients = V[i_old,:]
            for j_new in range(Ns_new[1]):
                j_old = point_on_old_int_y[j_new]
                new_potentials_columnwise[i_new,j_new] = (
                                                 row_potentials_old[j_old]+
                                                 gradients[j_old]*
                                                 distances_y[j_new])
        new_potentials = (new_potentials_rowwise+new_potentials_columnwise)/2.
        return new_potentials            
        
    def precondition(self,Ns,w=1.2,max_iter=20000,tol=1e-3,verbose=True):
        '''
        Solve the desired system using a lower-resolution mesh, and then
        start solving the system for the mesh resolution requested afterwards,
        using an interpolation of the previously determined potentials 
        on the lower resolution grid as a starting point. If needed, this
        procedure can be repeated several times by repeatedly calling this
        function.
        '''
        if not hasattr(Ns,'__iter__'):
            Ns = (Ns,Ns)        
        preconditioning_system = System(Ns)
        '''
        Now need to add the desired shapes to the preconditioning_system,
        using the 'source_creation_args' that records the arguments that
        previously went into creating all the sources. The only difference
        will be the Ns argument, since there will be a different grid.
        Shapes will therefore be created using the original potentials,
        new Ns values and the old shape creation arguments.
        '''
        for creation_args in self.shape_creation_args:
            new_args = creation_args[:] #copy so original is not changed!
            new_args[0] = Ns #replace old grid resolution with new resolution!
            preconditioning_system.add(Shape(new_args[0],new_args[1],
                                             new_args[2],*new_args[3],
                                             **new_args[4]))

        '''
        Need to translate the current potentials to the grid, use the 
        sampling function to this using the gradients between the
        grid points.
        Uses current potential, since this may have been updated by
        previous preconditioning (and would therefore not simply be
        all 0s anymore).
        '''                                  
    
#        preconditioning_system.show_setup(title='precon setup')
        
        preconditioning_system.potentials = self.sampling(Ns)
        '''
        Set the source terms back to their proper values based on the assigned
        shapes, since the averaging in the sampling function will 
        corrupt this
        '''
#        preconditioning_system.show(title='precon1')
        '''
        Especially with differing aspect ratios, this often interferes with
        the proper placement of the source. The same effect can be observed 
        with identical aspect ratios, since the placement of the source(s)
        will differ from the expected placement on the 'original' fine grid,
        due to the discreteness of the grid.
        Need to change the position of the sources according to the ratio 
        between the aspect ratios, only the y position would have to be 
        changed, which would be then when creating the shapes in the 
        first place above.
        This can be ignored if the ratio of aspect ratios is small, since
        the error assocaited with this would then be smaller than the
        discretisation error which exists regardless.
        '''
        
        preconditioning_system.potentials[preconditioning_system.sources] = (
                                     preconditioning_system.source_potentials[
                                     preconditioning_system.sources])
        
#        preconditioning_system.show(title='precon2')
        
        preconditioning_system.SOR(w=w,tol=tol,max_iter=max_iter,verbose=verbose)
        
#        preconditioning_system.show(title='precon3')
        
        new_potenials = preconditioning_system.sampling((self.Nsx,self.Nsy))
        '''
        Reset the source terms once again after assigning them back to self
        '''
        self.potentials = new_potenials
        self.potentials[self.sources] = self.source_potentials[self.sources]
        
#        self.show(every=10,quiver=False)
#        preconditioning_system.show(title='preconditioning system',quiver=False)
    
    def cross_section(self,side_length=0,show=True,savepath=''):
        '''
        now, plot a cross section of the potential across the central row.
        Ideally, the number of grid points should be an ODD number for this
        to work ideally - due to the symmetry of the problem
        '''
        mid_row_index = int((self.Nsx-1)/2.)
        cross_section = self.potentials[mid_row_index]
        plt.figure()
        plt.title('1-D Cross-Section of the Potential across the System\n'
                  +'tol = {:.2e}, Nsx = {:.2e}, Nsy = {:.2e}, side length = {:.3e}'.
                  format(self.tol,self.Nsx,self.Nsy,side_length))
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
    def show_setup(self,title='',interpolation='none',**fargs):
        '''
        Show the sources in the system
        '''
        plt.figure()
        plt.title('Sources')
        plt.imshow(self.source_potentials.T,origin='lower',
                   interpolation='none',**fargs)
        plt.colorbar()
        plt.tight_layout()
        if title:
            plt.title(title)        
        plt.show()
    def streamplot(self,title='',**frags):
        plt.figure()
        U,V = np.gradient(self.potentials)
        fields = np.sqrt(U**2+V**2)
        U = -U.T
        V = -V.T        
        X,Y = np.meshgrid(np.arange(self.Nsx),np.arange(self.Nsy))
        lw = 8*fields/np.max(fields)
        plt.streamplot(X, Y, U, V,
                       density = [1,1],
                       color = self.potentials.T,
                       linewidth = lw)
        if title:
            plt.title(title)
        plt.tight_layout()            
        plt.show()
    def show(self,every=1,title='',interpolation='none',quiver=False,**fargs):
        '''
        Show the calculated potential
        '''
        plt.figure()
        plt.title('Potential')
        plt.imshow(self.potentials.T,origin='lower',interpolation='none',
                   **fargs)
        plt.colorbar()
        if quiver:
            U,V = np.gradient(self.potentials)
    #        U = U[::every,::every]
    #        V = V[::every,::every]
            X,Y = np.meshgrid(np.arange(self.Nsx),np.arange(self.Nsy))
            '''
            Have to take transpose of potential gradients, but not of the 
            positions, due to the different ways that mgrid and meshgrid
            arange their outputs.
            '''
            U = -U.T
            V = -V.T
            plt.quiver(X[::every,::every],Y[::every,::every],
                       U[::every,::every],V[::every,::every])
        plt.tight_layout()
        if title:
            plt.title(title)
        plt.show()
    def create_method_matrix(self):
        N = self.Nsx**2
        self.A = np.zeros((N, N))
        boundary_conditions = []
        for i in range(N):
            boundaries_row = []
            coord1 = int(float(i)/self.Nsx)
            coord2 = i%self.Nsx
            self.A[i,i] = -4
            for c1,c2 in zip([coord1,coord1,coord1-1,coord1+1],
                             [coord2-1,coord2+1,coord2,coord2]):
                try:
                    if c1==-1 or c2==-1 or c1>self.Nsx-1 or c2>self.Nsx-1:
                        raise IndexError
                    elif c1 == coord1-1:
                        '''
                        row has changed, need to move 'cell'
                        column cannot have changed, so move
                        by Ns along row
                        '''
                        self.A[i,i-self.Nsx] = 1
                    elif c1 == coord1+1:
                        self.A[i,i+self.Nsx] = 1
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
        assert self.Nsx == self.Nsy,'Jacobi method requires a square grid'
        N = self.Nsx**2
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
        self.potentials = x.reshape(self.Nsx,-1)
    def gauss_seidel(self, tol=1e-3, max_iter=5000, verbose=True):
        assert self.Nsx == self.Nsy,'Gauss Seidel method requires a square grid'        
        N = self.Nsx**2
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
            #print "before\n",x.reshape(self.Nsx,-1)
            initial_norm = np.linalg.norm(x)
            x = np.dot(T,x).reshape(-1,) + L_D_inv_b
            x[sources] = orig_x[sources]
            #print "sources",x[sources]
            final_norm = np.linalg.norm(x)
            diff = np.abs(initial_norm-final_norm)
            #print "after\n",x.reshape(self.Nsx,-1)
            if verbose:
                print("i,diff:",i,diff)
            if diff < tol:
                break
            #print ''
        self.potentials = x.reshape(self.Nsx, -1)
    def SOR(self, w=1.2, tol=1e-3, max_iter=5000, verbose=True,
            boundary_conditions=None):
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
        w = float(w)
        sources = self.sources
        '''
        Create array to contain the potential, including a boundary -
        the boundary conditions, which are never altered during the program's
        execution.
        Then 'fill in' the potential at the center of this matrix
        
        Initialising the potentials with 0s is not really necessary, since 
        this is done at instance creation anyway, plus this overwrites 
        later changes to the potential with preconditioning, for example
        '''
        
        if boundary_conditions is None:
            print('no boundary conditions!')
            boundary_conditions = np.zeros((self.Nsx+2,self.Nsy+2))
        assert boundary_conditions.shape == (self.Nsx+2,self.Nsy+2),(
                'The array should have shape (Nsx+2,Nsy+2)')   
        
        '''
        print('boundary conditions\n',boundary_conditions)        
        
        plt.figure()
        plt.imshow(boundary_conditions.T,origin='lower',interpolation='none')
        plt.title('boundary conditions +{:}'.format(np.mean(boundary_conditions)))
        plt.colorbar()
        plt.show()
        '''        
        
        x = boundary_conditions
        #x[1:-1,1:-1] = self.source_potentials
        x[1:-1,1:-1] = self.potentials
           
        '''
        plt.figure()
        plt.imshow(boundary_conditions.T,origin='lower',interpolation='none')
        plt.title('boundary conditions w/ new core')
        plt.colorbar()
        plt.show()
        '''        
        
        '''
        better choice than random initial state needs to be found!
        could use pre-conditioning with coarse grid.
        '''
        x = self.SOR_sub_func(max_iter,x,
                              np.array([self.Nsx,self.Nsy],dtype=np.int64),
                              sources,w,tol,verbose)
        self.potentials = x[1:-1,1:-1]
        
    @staticmethod
    @jit(nopython=True,cache=True)
    def SOR_sub_func(max_iter,x,Ns,sources,w,tol,verbose):
        w_1 = (1.-w)
        w_4 = (w/(4.))
        for iteration in range(max_iter):
            initial_norm = np.linalg.norm(x)
            for i in range(0,Ns[0]):
                i_1 = i+1
                for j in range(0,Ns[1]):
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
        self.tol = tol
        self.w = w
        w = float(w)
        sources = self.sources
        '''
        Create array to contain the potential, including a boundary -
        the boundary conditions, which are never altered during the program's
        execution.
        Then 'fill in' the potential at the center of this matrix
        '''
        x = np.zeros((self.Nsx+2,self.Nsy+2))
        x[1:-1,1:-1] = self.potentials     
        '''
        better choice than random initial state needs to be found!
        could use pre-conditioning with coarse grid.
        '''
        x,all_potentials = self.SOR_sub_func_anim(max_iter,x,
                                                  np.array([self.Nsx,self.Nsy],
                                                           dtype=np.int64),
                                                  sources,w,tol,verbose)

        self.potentials = x[1:-1,1:-1]
        return all_potentials
        
    @staticmethod 
    @jit(nopython=True,cache=True)
    def SOR_sub_func_anim(max_iter,x,Ns,sources,w,tol,verbose):
        all_potentials = np.zeros((max_iter, Ns[0], Ns[1]))
        w_1 = (1.-w)
        w_4 = (w/(4.))        
        for iteration in range(max_iter):
            all_potentials[iteration] = x[1:-1,1:-1]
            initial_norm = np.linalg.norm(x)
            for i in range(0,Ns[0]):
                i_1 = i+1
                for j in range(0,Ns[1]):
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
        return x,all_potentials[:iteration+1,...]
    
    def separate_sources(self):
        self.point_sources = [args for args in self.shape_creation_args 
                              if args[-1].get('shape','rectangle') == 'point']
        self.rectangle_sources = [args for args in self.shape_creation_args
                                  if args[-1].get('shape','rectangle') in 
                                  ['square','rectangle']]          
    def AMR_static(self,origin,width_height,zoom,**SOR_args):
        '''
        Only rectangles / point sources supported
        
        Default shape is 'rectangle'
        Default filled status is True
        '''
        origin = np.array(origin)
        assert type(zoom) == int,'Zoom should be an integer'
        if not hasattr(width_height,'__iter__'):
            width_height = np.array([width_height,width_height])
        '''
        Need to leave at least one row/column free in order to create
        boundary conditions 
        '''
        if ( np.any(origin - width_height < 1) or
             np.any(origin + width_height > np.array((self.Nsx-1,self.Nsy-1)))
           ):
            raise Exception(('Need to leave at least 1 row/column gap between'+
                             ' selection window and system'))
        new_Ns = np.array(width_height)*zoom
        new_Ns = np.array(new_Ns,dtype=np.int64)
        print('new Ns:',new_Ns)
        AMR_system = System(new_Ns)               
        '''
        Grab the necessary potentials from the original system, and then
        upsample them to the desired grid resolution.
        Then insert these as the initial conditions into the SOR method
        These new upsampled potentials will have shape (new_Ns+2,new_Ns+2)
        '''
        #Need to evalute original system before selecting boundary conditions
        self.SOR(**SOR_args)
        self.show(title='self')        
        min_row = int(origin[0] - width_height[0]/2.)
        max_row = min_row + width_height[0]
        min_column = int(origin[1] - width_height[1]/2.)
        max_column = min_column + width_height[1]
        potential_selection = self.potentials[min_row-1:max_row+1,
                                              min_column-1:max_column+1]
        print(min_row,max_row,min_column,max_column)
        print('selection shape',potential_selection.shape)
        U,V = np.gradient(potential_selection)
        new_potentials = self.sampling_sub_func(np.array((width_height[0]+2,
                                                          width_height[1]+2),
                                                         dtype=np.int64),
                                                np.array((new_Ns[0]+2*zoom,
                                                          new_Ns[1]+2*zoom),
                                                         dtype=np.int64),
                                                potential_selection,U,V,1.)

        print('new pot.')
        print(new_potentials)
        print(np.max(U),np.min(U))
        print(np.max(V),np.min(V))
        boundaries = new_potentials[zoom-1:-(zoom-1),zoom-1:-(zoom-1)]
        boundaries[1:-1,1:-1] = np.zeros((new_Ns[0],new_Ns[1]))
        print('new potentials shape',new_potentials.shape)
        print('boundaries shape',boundaries.shape)
        '''
        Transform the shape creation arguments from the original system
        to the new 'zoomed in' system, so that the sources can be recreated
        in the new system.
        '''     
        origin_coords = np.array(origin)*self.h
        '''
        
        vertex 2    +--------+   vertex 3
                    |        |
                    |        |
        vertex 1    +--------+   vertex 4
        '''        
        width_height_coords = np.array(width_height)*self.h
        lower_left_coords = origin_coords - width_height_coords/2.
        vertices_coords = (
                   [origin_coords - width_height_coords/2.,
                    origin_coords + np.array([-width_height_coords[0]/2.,
                                              width_height_coords[1]/2.]),
                    origin_coords + width_height_coords/2.,
                    origin_coords + np.array([width_height_coords[0]/2.,
                                              -width_height_coords[1]/2.])])
        
        print('window vertices\n',vertices_coords)
        upper_right_coords = origin_coords + width_height_coords/2.
        width_coords = upper_right_coords[0]-lower_left_coords[0]
        height_coords = upper_right_coords[1] - lower_left_coords[0]

        window_warning = 'Selection window should not extend beyond system'
        assert np.all(lower_left_coords>=np.array([0,0])),window_warning
        assert np.all(upper_right_coords<=np.array([1,1])),window_warning        
        
        self.separate_sources()
        to_plot = []   
                          
        for source in self.point_sources:
            source = source[:]
            source_origin = source[2]
            source_origin = np.array(source_origin)
#            print(source_origin,origin_coords)
            diffs = source_origin - origin_coords
#            print('diffs',diffs)
#            print(width_height_coords/2.)
            if np.all(np.abs(diffs)<width_height_coords/2.):
                '''
                Modify source coordinates to fit the new coordinate system
                given by the selection window.
                '''
                source[0] = new_Ns
#                print('lower left',lower_left_coords)
                subtracted = source[2] - lower_left_coords
                scaled = np.array(subtracted)/np.array(width_coords,
                                                       height_coords)
                source[2] = scaled
#                print('sub',subtracted)
#                print('scaled',scaled)
                to_plot.append(source)
#        print('new ns',new_Ns,new_Ns.dtype)
        
        for source in self.rectangle_sources:
            source = source[:]
            source_origin = np.array(source[2])
            print('args',source)
            width = source[-2][0]
            if source[-1].get('shape','rectangle') == 'square':
                height = width
            else:
                height = source[-2][1]
            '''
            
            vertex 2    +--------+   vertex 3
                        |        |
                        |        |
            vertex 1    +--------+   vertex 4
            '''
            vertices = [np.array([source_origin[0]-width/2.,
                                  source_origin[1]-height/2.]),
                        np.array([source_origin[0]-width/2.,
                                  source_origin[1]+height/2.]),          
                        np.array([source_origin[0]+width/2.,
                                  source_origin[1]+height/2.]),
                        np.array([source_origin[0]+width/2.,
                                  source_origin[1]-height/2.])]
            print('vertices\n',vertices)
            '''
            0 or 1 vertices are expected to overlap into the selection window
            The selection window may also lie in between two vertices            
            '''
            origin_diff = source_origin - origin_coords
            if np.all(np.abs(origin_diff)<(width_height_coords/2.+
                                     np.array([width,height])/2.)):
                print('accepted diff',origin_diff)
                source[0] = new_Ns                
                '''
                Assign new width and height based on the overlap between
                the shape and the window
                '''
                window_x = np.array([vertices_coords[0][0],vertices_coords[-1][0]])
                window_y = np.array([vertices_coords[0][1],vertices_coords[1][1]])
                source_x = np.array([vertices[0][0],vertices[-1][0]])
                source_y = np.array([vertices[0][1],vertices[1][1]])
                low_x =  (source_x[0] if source_x[0] > window_x[0]
                                      else window_x[0])
                high_x = (source_x[1] if source_x[1] < window_x[1]
                                      else window_x[1])
                low_y =  (source_y[0] if source_y[0] > window_y[0]
                                      else window_y[0])
                high_y = (source_y[1] if source_y[1] < window_y[1]
                                      else window_y[1])
        
                new_width = high_x - low_x
                new_height = high_y - low_y
                print('new width, height',new_width,new_height)
                source[3] = (np.array((new_width,new_height))/
                                                    np.array(width_coords,
                                                             height_coords))
                '''
                Assign new origin to the rectangle, based on the 
                dimensions calculated
                '''                     
                new_origin = (np.mean([high_x,low_x]),np.mean([high_y,low_y]))
                subtracted = new_origin - lower_left_coords
                scaled = np.array(subtracted)/np.array(width_coords,
                                                       height_coords)
                source[2] = scaled
                print(low_x,high_x,low_y,high_y)
                print('to append\n',source)
                to_plot.append(source)  
            
        for shape_args in to_plot:
#            print('args',shape_args)
            AMR_system.add(Shape(shape_args[0],shape_args[1],shape_args[2],
                                 *shape_args[3],**shape_args[4]))

        print('final args\n',AMR_system.shape_creation_args)
        AMR_system.show_setup()
        
        '''
        Calculate the potential using the previously determined 
        boundary conditions
        '''
        AMR_system.show(title='AMR not solved')
        print('bound.\n',boundaries)
        AMR_system.SOR(boundary_conditions=boundaries,**SOR_args)
        AMR_system.show(title='AMR solved')
        plt.figure()
        plt.title('AMR test')
         
        ax = plt.gca()
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.imshow(self.potentials.T,origin='lower',interpolation='none',
                   extent=(0,1,0,1))
        ax.imshow(AMR_system.potentials.T,origin='lower',
                  interpolation='none',
                  extent=(lower_left_coords[0],
                          upper_right_coords[0],
                          lower_left_coords[1],
                          upper_right_coords[1]),alpha=0.6)
                               
        
        
if __name__ == '__main__': 
    Ns = 81
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
    test.add(Shape(Ns,1,(0.5,0.5),0.4))
    test.show_setup()
    calc = True
    tol = 1e-6
    max_iter = 100
    show = False
    #methods = [test.SOR,test.jacobi,test.gauss_seidel]
    methods = [test.SOR]
    names = [f.__name__ for f in methods]
    if calc:
        for name,f in zip(names,methods):
            print(name)
            f(tol=tol,max_iter=max_iter)
            if show:
                test.show(title=name,interpolation='none',every=7)
                test.streamplot()
