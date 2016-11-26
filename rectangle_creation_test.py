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

Ns = 71

test = System(Ns)
test.add(Shape(Ns,1,(0.5,0.5),0.3,shape='square',filled=False))
test.show_setup()