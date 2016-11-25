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

Ns = 301
tol = 5e-4
max_iter = 50000

times = []

test = System(Ns)

#test.add(Shape(Ns,-1,(0.5,0.5)))


#test.add(Shape(Ns,-4,(0.9,0.9),0.1,shape='square'))
test.add(Shape(Ns,-1.3,(0.5,0.5),0.18,shape='circle',filled=True))
#test.add(Shape(Ns,1.8,(0.5,0.5),0.1,shape='circle',filled=False))
#test.add(Shape(Ns,2,(0.5,0.5),0.3,shape='circle',filled=False))
#test.add(Shape(Ns,-1,(0.354,0.506),0.03,shape='circle',filled=False))
#test.add(Shape(Ns,1,(0.37,0.49),0.03,shape='circle',filled=False))


raw = deepcopy(test)

start = clock()

raw.SOR(tol=tol,max_iter=max_iter,verbose=False)

rawtime = clock()-start
times.append(rawtime)
######
raw.show(quiver=False,title='raw')
#####


#####
#test.show(quiver=False)
#####
start = clock()
#test.precondition(101,tol=1e-2,verbose=False)
test.precondition(11,tol=1e-1,verbose=False)
#test.precondition(301,tol=1e-3,verbose=False)
#test.precondition(851,tol=1e-3,verbose=False)
precondtime = clock()-start
times.append(precondtime)

#####
test.show(quiver=False)
#####
test.SOR(tol=tol,max_iter=max_iter,verbose=False)
#####

time2 = clock()-start
times.append(time2)

test.show(quiver=False,title='with precon')
#####

plt.figure()
plt.imshow(np.abs(raw.potentials-test.potentials))
plt.colorbar()
plt.title('raw - test')
plt.tight_layout()
plt.show()



print 'raw time', times[0]
print 'preconditioning time:', times[1]
print 'total time with preconditioning:', times[2]