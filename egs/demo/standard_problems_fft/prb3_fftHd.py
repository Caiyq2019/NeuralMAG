# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 10:00:00 2023

#########################################
#                                       #
#  muMAG Standard Problem #3            #
#                                       #
#  -- Soft cube energy @ each state     #
#     Ms = 800            emu/cc        #
#     Ku = 4.02e5 (0.1Km) erg/cc        #
#     Ax = 1.0 e-6        erg/cm        #
#     lx = 5              nm            #
#     cube size 20~100    nm            #
#                                       #
#  -- FLower state .vs. vortex state    #
#                                       #
#########################################

"""


import libs.MAG2305 as MAG2305
import numpy as np
import time, os


##########################
# Define a plot function #
##########################
import matplotlib.pyplot as plt

# Font size
parameters = {'axes.labelsize' : 15,
              'axes.titlesize' : 15,
              'xtick.labelsize': 12,
              'ytick.labelsize': 12,
              'legend.fontsize': 15,
              'figure.dpi'     : 200}
plt.rcParams.update(parameters)

def plot_data(path, data, plt_name):
    data = np.array(data)

    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111)

    ax.plot( data[0], data[1])
    ax.scatter( data[0], data[1], s=16, label='flower')
    ax.plot( data[0], data[2])
    ax.scatter( data[0], data[2], s=16, label='vortex')

    ax.set_xlabel(r"$L$ / $l_{ex}$")
    ax.set_ylabel(r"$Eng$ / $2 \pi M_s^2$")
    ax.grid(True, lw=1.0, ls='-.')
    ax.legend()

    plt.savefig(path + plt_name)
    plt.close()

    return None


###############################
# Prepare NeuralMAG2305 model #
###############################
# How many cells in the cube model
cube_size = [i+8 for i in range(33)]

# Size of each cell [nm]
cell = (2.5, 2.5, 2.5)

# Saturation magnetization [emu/cc]
Ms = 800
Km = 2 * np.pi * Ms**2
# Exchange stiffness [erg/cm]
Ax = 1.0e-6
lx = np.sqrt(Ax / Ms**2 /2/np.pi) * 1.0e7  # [nm]
# Uniaxial anisotropy energy [erg/cc]
Ku = 4.02e5
Kvec = (0, 0, 1.0)

# method for calculation
method = 'LLG_RK4'
dtime = 1.0e-13    # dtime for LLG_RK4


###################
# DO Calculations #
###################
path0 = "./data_problem3/"
if not os.path.exists(path0):
    os.mkdir(path0)

# DO calculation for each [cell_size]
eng_rcd = np.array([ [], [], [] ])
for csize in cube_size:
    time_start = time.time()

    # How many cells in each axis
    size = (csize, csize, csize)

    # Total cells
    cell_num = size[0] * size[1] * size[2]

    # Create MM model
    cube0 = MAG2305.mmModel(types='bulk', size=size, cell=cell, 
                            Ms=Ms, Ax=Ax, Ku=Ku, Kvec=Kvec)

    # Initialize demag matrix
    cube0.DemagInit()

    # Initialize spin for FLOWER state
    spin = np.empty( size + (3,) )
    spin[...,:] = [0, 0, 1]
    cube0.SpinInit(spin)

    print('\nBegin flower state relaxing:\n')

    # Do calculation
    _ = cube0.GetStableState(method=method, error_limit=1.0e-5, dtime=dtime)
    energy_flower = cube0.GetEnergy_fromHeff() / cell_num / Km
    print('  Flower state enrgy:')
    print('     Total    - {}\n'.format(energy_flower))


    # Initialize spin for VORTEX state
    spin = np.empty( size + (3,) )
    if csize % 2 ==0:
        spin[:size[0]//2, :, :] = [0, 0, 1]
        spin[size[0]//2:, :, :] = [0, 0,-1]
    else:
        spin[:size[0]//2, :, :] = [0, 0, 1]
        spin[size[0]//2+1:, :, :] = [0, 0,-1]
        spin[size[0]//2, :, :] = [0, -1, 0]
    cube0.SpinInit(spin)

    print('\nBegin vortex state relaxing:\n')

    # Do calculation
    _ = cube0.GetStableState(method=method, error_limit=1.0e-5, dtime=dtime)
    energy_vortex = cube0.GetEnergy_fromHeff() / cell_num / Km
    print('  Vortex state enrgy:')
    print('     Total    - {}\n'.format(energy_vortex))

    time_finish = time.time()
    print('\nEnd case mode_size={}. Time cost: {:.1f}s\n'
            .format(csize, time_finish-time_start) )
    eng_rcd = np.append(eng_rcd, [ [csize*cell[0]/lx], [energy_flower], [energy_vortex] ], axis=1)

    # Save & plot data
    np.save(path0+'energy_data-fftHd', eng_rcd)
    plot_data(path=path0, data=eng_rcd, plt_name='Energy_vs_size-fftHd')
