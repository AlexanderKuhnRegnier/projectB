# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 01:12:37 2016

@author: Alexander Kuhn-Regnier

Compare the SOR execution speed of the two modules with and without AMR
"""
from AMR_system import AMR_system,Grid,build_from_segments
from system import System,Shape
import matplotlib.pyplot as plt
from time import clock

Ns = 600
w = 1.

start = clock()

test = Grid(*build_from_segments(((1,Ns),)))
test.rectangle(1,(0.5,0.5),0.4,0.7)
test.rectangle(1,(0.2,0.4),0.02,0.02)

#test.show(color=(0,0,0,0.1))

amr_system = AMR_system(test)
amr_system.SOR(w=w,max_iter=10000,max_time=150,tol=1e-10,verbose=False)

print('amr time:',clock()-start)
#amr_system.show()

start = clock()

system = System(Ns+1)

system.add(Shape(Ns+1,1,(0.5,0.5),0.4,0.7,shape='rectangle'))
system.add(Shape(Ns+1,1,(0.2,0.4),0.02,0.02,shape='rectangle'))

#system.show_setup()

system.SOR_single(w=w,max_iter=10000,max_time=150,tol=1e-10,verbose=False)

print('normal time:',clock()-start)
#system.show()

source_diff = system.source_potentials - amr_system.grid.source_potentials
plt.figure()
plt.title('source diffs')
plt.imshow(source_diff.T,origin='lower',interpolation='none')
plt.colorbar()

potential_diff = system.potentials - amr_system.potentials

plt.figure()
plt.imshow(potential_diff.T,origin='lower',interpolation='none')
plt.title('potential diffs')
plt.colorbar()