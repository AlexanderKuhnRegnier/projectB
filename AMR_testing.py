# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 19:15:44 2016

@author: Alexander Kuhn-Regnier
"""
from system import System,Shape
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
Ns = 50

test = System(Ns)
point = Shape(Ns,1,(0.45,0.5),0.05,0.3,shape='rectangle')
test.add(point)
point = Shape(Ns,1,(0.55,0.5),0.05,0.3,shape='rectangle')
test.add(point)
test.show_setup('original system')

test.AMR_static((25,25),(10,20),20,max_iter=200,tol=1e-4,verbose=False)
