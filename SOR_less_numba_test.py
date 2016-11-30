# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 11:13:44 2016

@author: Alexander Kuhn-Regnier
"""

import numpy as np
from time import clock
from copy import deepcopy
from system import System,Shape
import matplotlib.pyplot as plt

plt.close('all')

Ns = 100
test = System(Ns)
shape = Shape(Ns,1,(0.5,0.5),0.3,shape='square')
test.add(shape)
test.show_setup()
start = clock()
test.SOR_single(max_iter=600,tol=1e-12,verbose=False)
print 'single SOR',clock()-start
test.show(title='SOR single')


Ns = 100
test = System(Ns)
shape = Shape(Ns,1,(0.5,0.5),0.3,shape='square')
test.add(shape)
test.show_setup()
start = clock()
test.SOR(max_iter=600,tol=1e-12,verbose=False)
print 'SOR',clock()-start
test.show(title='SOR')