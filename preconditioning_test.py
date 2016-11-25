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
tol = 5e-4
max_iter = 50000

times = []

test = System(Ns)
test.add(Shape(Ns,-1,(0.5,0.5)))

raw = deepcopy(test)

start = clock()

raw.SOR(tol=tol,max_iter=max_iter,verbose=False)

######
#raw.show(quiver=False,title='raw')
#####
rawtime = clock()-start
times.append(rawtime)

#####
#test.show(quiver=False)
#####
start = clock()
test.precondition(101,tol=0.1e-2,verbose=False)
#test.precondition(151,tol=1e-4,verbose=False)
#test.precondition(751,tol=1e-3,verbose=False)
#test.precondition(851,tol=1e-3,verbose=False)
precondtime = clock()-start
times.append(precondtime)

#####
#test.show(quiver=False)
#####
test.SOR(tol=tol,max_iter=max_iter,verbose=False)
#####
#test.show(quiver=False,title='with precon')
#####

plt.figure()
plt.imshow(np.abs(raw.potentials-test.potentials))
plt.colorbar()
plt.title('raw - test')
plt.tight_layout()
plt.show()

time2 = clock()-start
times.append(time2)

print 'raw time', times[0]
print 'preconditioning time:', times[1]
print 'total time with preconditioning:', times[2]