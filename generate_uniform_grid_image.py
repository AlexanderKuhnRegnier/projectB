# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:13:51 2016

@author: ahfku
"""
from AMR_system import Grid,build_from_segments
import matplotlib.pyplot as plt
plt.close('all')
xh,yh = build_from_segments(x=((0.5,1),(1,5)),
y=((0.5,1),(1,3)))
test = Grid(xh,yh,units='m')
#test.show()
collection = test.create_grid_line_collection(color=(0,0,0,1))
fig,ax = plt.subplots()
ax.add_collection(collection)
ax.axis('square')
ax.set_ylim(-0.17,1.1)
ax.set_xlim(-0.2,1.1)

ax.set_xlabel(r'$\mathrm{x\ (arbitrary\ units)}$',fontsize=20)
ax.set_ylabel(r'$\mathrm{y\ (arbitrary\ units)}$',fontsize=20)

height_x = -0.03
height_x_text = -0.12

plt.annotate(
    '', xy=(0, height_x), xycoords='data',
    xytext=(0.5,height_x), textcoords='data',
    arrowprops={'arrowstyle': '<->'})
plt.annotate(
    r'$h_{x,0}$', xy=(0.2, height_x_text), xycoords='data',
    xytext=(0, 0), textcoords='offset points',fontsize=22)
plt.annotate(
    '', xy=(0.5, height_x), xycoords='data',
    xytext=(0.6,height_x), textcoords='data',
    arrowprops={'arrowstyle': '<->'})
plt.annotate(
    r'$h_{x,1}$', xy=(0.5, height_x_text), xycoords='data',
    xytext=(0, 0), textcoords='offset points',fontsize=22)

sep_y_text = -0.18

plt.annotate(
    '', xy=(height_x,0), xycoords='data',
    xytext=(height_x,0.5), textcoords='data',
    arrowprops={'arrowstyle': '<->'})
plt.annotate(
    r'$h_{y,0}$', xy=(sep_y_text,0.2), xycoords='data',
    xytext=(0, 10), textcoords='offset points',fontsize=22)
plt.annotate(
    '', xy=(height_x,0.5), xycoords='data',
    xytext=(height_x,test.y[2]), textcoords='data',
    arrowprops={'arrowstyle': '<->'})
plt.annotate(
    r'$h_{y,1}$', xy=(sep_y_text,0.56), xycoords='data',
    xytext=(0, 0), textcoords='offset points',fontsize=22)


plt.show()
fig.savefig('non_uniform_grid.pdf',bbox_inches='tight')