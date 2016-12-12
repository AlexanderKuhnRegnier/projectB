# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 18:52:11 2016

@author: Alexander Kuhn-Régnier
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from time import clock
import os
plt.ioff()
from AMR_EDM import create_EDM_system
from AMR_system import gradient

factor = 40
k = 0.9
system,dust_size = create_EDM_system((26*factor,3*factor),k,
                                     size=(260*(1+2*k),30*(1+2*k)),
                                     small_sources=True,
                                     dust_pos=300,
                                     dust_size=2.)
#system.show_setup()
system.SOR(w=1.2,tol=1e-14,max_time=1)
#system.show()
print('dust size:',dust_size)