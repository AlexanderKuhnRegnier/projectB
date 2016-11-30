# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 13:54:04 2016

@author: Alexander Kuhn-Regnier
"""

import numpy as np
from time import clock
from copy import deepcopy
from system import System,Shape
import matplotlib.pyplot as plt

start = clock()
Ns = 300
test = System(Ns)
circle = Shape(Ns,1,(0.5,0.5),0.2,shape='circle',filled=False)
test.add(circle)
print 'time:',clock()-start
test.SOR(verbose=False,tol=1e-2)
#test.show_setup()
#test.cross_section()
test.show(quiver=True,every=50)