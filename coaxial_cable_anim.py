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
import os
plt.ioff()
def name_folder(folder):
    '''
    Input: folder (string)
    Returns a new folder name string with a number appended to it, so as to
        make it unique in its directory.
    '''
    for i in range(1000):
        new = folder+str(i)
        if not os.path.isdir(new):
            return new
 

'''
It 'save = True', then 'anim=True' must be 
set for the animation to be saved.
Note: to save the animation, 'ffmpeg.exe' must
be located in the script's directory.
'''
anim = True
save = True
frames = 1000    
    
if anim:
    Ns = (300,300)
    side_length = (1/3.)
    square_coaxial_cable = Shape(Ns,1.5,(0.5,0.5),3e-1,side_length,shape='square',
                                 filled=True)
    cable = System(Ns)
    cable.add(square_coaxial_cable) 
    
    all_potentials = cable.SOR_anim(w=1.9999999999999999999999999995,tol=0,max_iter=frames)
    plt.ioff()
    fig,ax = plt.subplots(figsize=(15,15))
    image = plt.imshow(all_potentials[0].T,
                       vmin = np.min(all_potentials), vmax=np.max(all_potentials),
                        interpolation = 'none',
                        aspect='equal',extent=None,
                        origin='lower')
    iter_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, 
            size=20,bbox=dict(facecolor='white',alpha=0.5))

    plt.colorbar()
    
    def update_image(*args):
        print('args:',args,args[0])
        iter_text.set_text(str(args[0]))
        image.set_array(all_potentials[args[0]].T)
        return image,iter_text
    
    ani = animation.FuncAnimation(fig,update_image,blit=True,frames=frames,
                                  interval=1e-5, repeat=True,repeat_delay=500)
    
    if save:
        plt.rcParams['animation.ffmpeg_path'] = os.path.join(os.getcwd(),'ffmpeg.exe')
        FFwriter = animation.FFMpegWriter(fps=24,bitrate=20000)
        ani.save('coaxial_cable.mp4',writer=FFwriter,dpi=200)
        plt.close('all')
    else:
        plt.show()
          