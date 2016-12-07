# -*- coding: utf-8 -*-
"""
Created on Dec 7 2016

@author: Alexander Kuhn-Regnier

Calculating the homogeneity of the electric field generated for a 
given configuration of the sources.
"""
from __future__ import print_function
from system import Shape,System
import numpy as np
import matplotlib.pyplot as plt
from time import clock
import os
plt.ioff()

