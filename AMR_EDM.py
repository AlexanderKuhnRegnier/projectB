# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 14:20:14 2016

@author: Alexander Kuhn-Regnier
"""
from __future__ import print_function
from AMR_system import Grid,AMR_system,build_from_segments
import numpy as np
import matplotlib.pyplot as plt
from time import clock
import os

def create_EDM_system(Ns,kx,ky=None,size=(1.,1.),dust_pos=None,
                      dust_size=1e-1,small_sources=True):
    '''
    Create the shapes needed to model the EDM experiment and then add
    these shapes to a system, both with the given grid size.
    kx and ky describe the space left between the end of the experimental
    setup and the boundary, which will be held at 0V.
    This distance is described as a multiple of the length of the 
    experimental setup in that axis.
    '''
    #The grid building function takes as input the number of STEPS, which 
    #is 1 less than the number of grid points, which are specified using *Ns*
    if not hasattr(size,'__iter__'):
        size = (size,size) 
        
    if ky == None:
        ky = kx
    if not hasattr(Ns,'__iter__'):
        Ns = (Ns,Ns)
    #The grid building function takes as input the number of STEPS, which 
    #is 1 less than the number of grid points, which are specified using *Ns*
    xh,yh = build_from_segments(((1,Ns[0]-1),),((1,Ns[1]-1),))
    grid = Grid(xh,yh,aspect_ratio = float(Ns[1])/Ns[0],size=size,
                potential_scaling = 8000)
    #compute parameters for shape creation based on padding selected using
    #*kx* and *ky* as well as the known shape ratios.
    
    return create_EDM_system_from_grid(grid,kx,ky,size=size,dust_pos=dust_pos,
                                       dust_size=dust_size,
                                       small_sources=small_sources)
  
def create_EDM_system_from_grid(grid,kx,ky=None,size=(1.,1.),dust_pos=None,
                      dust_size=1e-1,small_sources=True):
    '''
    Capable of creating the EDM shapes on non_uniform grids specified by the
    passed-in Grid instance.
    Create the shapes needed to model the EDM experiment and then add
    these shapes to a system, both with the given grid size.
    kx and ky describe the space left between the end of the experimental
    setup and the boundary, which will be held at 0V.
    This distance is described as a multiple of the length of the 
    experimental setup in that axis.
    '''
    if ky == None:
        ky = kx

    #compute parameters for shape creation based on padding selected using
    #*kx* and *ky* as well as the known shape ratios.
    A = grid.aspect_ratio
    hx = 1./(26*(2*kx+1))
    hy = A/(3*(2*ky+1))
    Sx = kx * 26*hx
    Sy = ky * 3*hy
    
#    print('hy',hy)
#    print('hx',hx)
#    print('size',size)
#    print('A',A)
    
    grid.rectangle(1., ((Sx+13.*hx)*size[0],
                        (Sy+5.*hy/2.)*size[0]),
                   20*hx*size[0],hy*size[0])
    
    grid.rectangle(-1.,((Sx+13.*hx)*size[0],
                        (Sy+hy/2.)*size[0]),
                        20*hx*size[0],
                        hy*size[0])
    
    if small_sources:    
        grid.rectangle(0.25, ((Sx+hx)*size[0],
                              (Sy+5.*hy/2.)*size[0]),
                                2*hx*size[0],
                                hy*size[0])                           
        grid.rectangle(0.25, ((Sx+25.*hx)*size[0],
                              (Sy+5.*hy/2.)*size[0]),
                                2*hx*size[0],
                                hy*size[0])                                
        grid.rectangle(-0.25,((Sx+hx)*size[0],
                              (Sy+hy/2.)*size[0]),
                                2*hx*size[0],
                                hy*size[0])
        grid.rectangle(-0.25,((Sx+25.*hx)*size[0],
                              (Sy+hy/2.)*size[0]),   
                                2*hx*size[0],
                                hy*size[0])
                     
    system = AMR_system(grid)        
#    system.show_setup()
#    print(system.Nsx,system.Nsy)
    dust_tuple = ()
    position = ()
    if dust_pos:
        if not hasattr(dust_size,'__iter__'):
            dust_size = (dust_size,dust_size)  
        #scale dust size by x-length, since the scaled length
        #in x is always 1, not so for y
        scaled_dust_size = (dust_size[0]/size[0],dust_size[1]/size[0])
        scaled_dust_pos = dust_pos/size[0]
        
        #determine where the dust particle will sit along the 
        #x axis based on the scaled position (assumed to be along x)
        #and the scaled size of the dust particle
        dust_extent_x = (scaled_dust_pos-scaled_dust_size[0]/2.,
                       scaled_dust_pos+scaled_dust_size[0]/2.)
        lowest_x_grid,highest_x_grid = (
            (np.abs(system.grid.x-
                    np.array(dust_extent_x).reshape(-1,1))).argmin(axis=1))
        #lowest 'real' coordinate in scaled coordinates of the
        #top sources, which all lie along one horizontal line
        y_source = Sy+2.*hy
        #translate this to an index on the grid
        #do this by taking the index of the grid point along the y axis 
        #which is closest to *y_source* but also lower than it, since we 
        #are looking for the point below the end of the top source
#        print('y source pos:',y_source)
#        print('max y:',max(system.grid.y))
#        print(system.grid.y-y_source)
        y_source_grid = max((i for (i,j) in enumerate(system.grid.y-y_source) 
                             if j < 0))
#        print('y source grid:',y_source_grid)
        #lowest position on grid of top source should be given by
        #*y_source_grid* above. If not, then no potential will be
        #assigned there, so we should start assigning at this point
        #as opposed to the point below
        if np.any(system.sources[:,y_source_grid]):
            #source already assigned, so the empty space must
            #start at the grid point below this
            #this method won't work if the sources are too close together
            #but in that case it is irrelevant anyway
            top_dust_grid = y_source_grid-1
        else:
            top_dust_grid = y_source_grid
        bottom_dust_scaled_pos = (system.grid.y[top_dust_grid]-
                                  scaled_dust_size[1])
        bottom_dust_grid = (np.abs(system.grid.y-
                                   bottom_dust_scaled_pos)).argmin()
        #grab source potential from above grid point, which makes
        #the method more general, if a 'small' potential were
        #to be investigated, for example. This would have a 
        #potential of +0.25, not +1.
        #We are only interested in placing the dust particle
        #on the top sources, since the setup is symmetric
        
#        print('dust pos',dust_pos)
#        print('scaled dust pos',scaled_dust_pos)
#        print('lowest x grid',lowest_x_grid)
#        print('top x grid',highest_x_grid)
#        print('max x grid',system.grid.x.size)
#        print('top dust grid',top_dust_grid)
#        print('max y grid',system.grid.y.size)
        
        assert system.sources[lowest_x_grid,top_dust_grid+1],(
               'Dust particle should be in contact with source!')
        potential = system.grid.source_potentials[lowest_x_grid,
                                                  top_dust_grid+1]
        for x_grid in range(lowest_x_grid,highest_x_grid+1):
            for y_grid in range(bottom_dust_grid,top_dust_grid+1):
                index = (x_grid,y_grid)
#                print('index:',index)
                system.grid.source_potentials[index] = potential
                system.potentials[index] = potential
                system.sources[index] = True
        #calculate physical size of the particle given the dimensions of
        #the system given in the *size* argument. Take into account the
        #spacings between the grid points as well
        dust_tuple = ((system.grid.x[highest_x_grid]-
                       system.grid.x[lowest_x_grid]
                       +system.grid.x_h[lowest_x_grid-1]/2.
                       +system.grid.x_h[highest_x_grid]/2.)*size[0],
                      (system.grid.y[top_dust_grid]-
                       system.grid.y[bottom_dust_grid]
                       +system.grid.y_h[bottom_dust_grid-1]/2.
                       +system.grid.y_h[top_dust_grid]/2.)*size[0])
        position = (system.grid.grid[0][lowest_x_grid,top_dust_grid],
                              system.grid.grid[1][lowest_x_grid,top_dust_grid])
    return system,dust_tuple,position
  
if __name__ == '__main__':
    factor = 10
    k = 1.
    #with k=1, and a dust size of 0.1 mm (100e-6 m),
    #the factor has to be at least 300 in order to be able to
    #depict the dust accurately!
    test,dust_size,position = create_EDM_system((26*factor,3*factor),k,
                             size=(260*(1+2*k),30*(1+2*k)),
                             small_sources=True,dust_pos=350.,
                             dust_size=1e-1)
    test.show_setup()
    print('Dust Size:',dust_size)
#    test.SOR(w=1.5,tol=1e-10,max_time=10)
#    test.show(quiver=False)