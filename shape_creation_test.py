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

Ns = 100
test = System(Ns)
square = Shape(Ns,1,(0.5,1.),0.3,shape='square',filled=True)
test.add(square)
test.SOR()
test.show_setup()
test.cross_section()

Ns = 1000
test = System(Ns)
square = Shape(Ns,1,(0.5,1.),0.3,shape='square',filled=True)
test.add(square)
test.show_setup()
test.SOR()
test.cross_section()