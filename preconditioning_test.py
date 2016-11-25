# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 22:29:51 2016

@author: Alexander Kuhn-Regnier
"""
import numpy as np
from time import clock
from copy import deepcopy
from system import System,Shape
import matplotlib.pyplot as plt

Ns = 201
tol = 1e-14
max_iter = 50000

times = []

test = System(Ns)

#test.add(Shape(Ns,-1,(0.5,0.5)))


test.add(Shape(Ns,-4,(0.9,0.9),0.1,shape='square'))
test.add(Shape(Ns,-1.3,(0.2,0.2),0.18,shape='circle',filled=True))
test.add(Shape(Ns,1.8,(0.2,0.2),0.1,shape='circle',filled=False))
test.add(Shape(Ns,2,(0.2,0.2),0.3,shape='circle',filled=False))
#test.add(Shape(Ns,-1,(0.354,0.506),0.03,shape='circle',filled=False))
#test.add(Shape(Ns,1,(0.37,0.49),0.03,shape='circle',filled=False))


raw = deepcopy(test)

start = clock()

raw.SOR(tol=tol,max_iter=max_iter,verbose=False)

rawtime = clock()-start
times.append(rawtime)
######
#raw.show(title='raw result')
#####


#####
#test.show(quiver=False)
#####
start = clock()
#test.precondition(101,tol=1e-2,verbose=False)
#test.precondition(81,tol=1e-1,verbose=False)
test.precondition(71,tol=1e-7,verbose=False)
#test.precondition(251,tol=1e-3,verbose=False)
precondtime = clock()-start
times.append(precondtime)

#####
#test.show(title='after preconditioning')
#####
test.SOR(tol=tol,max_iter=max_iter,verbose=False)
#####

time2 = clock()-start
times.append(time2)

#test.show(title='result with precon')
#####

plt.figure()
plt.imshow(np.abs(raw.potentials-test.potentials).T,origin='lower')
plt.colorbar()
plt.title('raw - test')
plt.tight_layout()
plt.show()


print 'raw time', times[0]
print 'preconditioning time:', times[1]
print 'total time with preconditioning:', times[2]