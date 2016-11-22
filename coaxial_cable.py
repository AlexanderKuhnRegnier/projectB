# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 20:26:11 2016

@author: Alexander Kuhn-Regnier

Simulating an infinitely long square coaxial cable,
which as a 2cm square cross section. The cable is held
inside of a grounded 6cm-square tube.

0V outside
 -------------------
|       +++++       | 
|       +10V+       |
|       +++++       |
 -------------------       
 2 cm | 2 cm | 2 cm

<-----   6 cm  ----->

Scaled to the natural grid units which range from
0 to 1 along each axis,
    2 cm -> 1/3
    6 cm -> 1 (as required)
    ie. scaling factor = (6 cm)^-1
    
Potential is also scaled to natural units, with the 
scaling factor being the inverse of the largest 
magntiude potential in the system to be modelled,
in this case 10 V
    ie. scaling factor = (10 V)^-1
"""
from __future__ import print_function
from system import Shape,System
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

Ns = 50
square_coaxial_cable = Shape(Ns,1,(0.5,0.5),1./6,shape='square',filled=True)
cable = System(Ns)
cable.add(square_coaxial_cable)

#cable.show_setup()

#Solve the system using the SOR method

#cable.SOR(w=1.,tol=1e-4,max_iter=100000)
#cable.show()

frames = 150

all_potentials = cable.SOR_anim(tol=0,max_iter=frames)
plt.ioff()
fig,ax = plt.subplots(figsize=(15,15))
image = plt.imshow(all_potentials[0],
                   vmin = np.min(all_potentials), vmax=np.max(all_potentials),
                    interpolation = 'hermite',
                    aspect='equal',extent=None
                    )
iter_text = ax.set_title('')
plt.colorbar()

def update_image(*args):
    print('args:',args,args[0])
    iter_text.set_text(str(args[0]))
    image.set_array(all_potentials[args[0]])
    return image,iter_text

ani = animation.FuncAnimation(fig,update_image,blit=False,frames=frames,
                              interval=200, repeat=False,repeat_delay=500)

save = True

if save:
    import os
    plt.rcParams['animation.ffmpeg_path'] = os.path.join(os.getcwd(),'ffmpeg.exe')
    FFwriter = animation.FFMpegWriter(fps=10,bitrate=3000)
    ani.save('test.mp4',writer=FFwriter,dpi=300)
else:
    plt.show()