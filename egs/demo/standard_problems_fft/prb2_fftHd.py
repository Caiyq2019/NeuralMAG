# -*- coding: utf-8 -*-
"""
Created on Sun Jun 04 14:00:00 2023

#########################################
#                                       #
#  muMAG Standard Problem #2            #
#                                       #
#  -- Permalloy film switching (M-H)    #
#  -- 125nm x 25nm x 2.5nm              #
#  -- Ms = 800            emu/cc        #
#  -- Ku = 0.0            erg/cc        #
#  -- Ax = 25 ~ 0.04 e-6  erg/cm        #
#  -- lx = 25 ~ 1         nm            #
#  -- Hext along [111] direction        #
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

def plot_data(path, Hc_data, Mr_data, plt_name):

    Hc_data = np.array(Hc_data)
    Mr_data = np.array(Mr_data)

    fig = plt.figure(figsize=(12,4))
    gs = fig.add_gridspec(100,100)
    ax1 = fig.add_subplot(gs[5:95, 0:26])
    ax2 = fig.add_subplot(gs[5:95, 36:62])
    ax3 = fig.add_subplot(gs[5:95, 72:98])

    ax1.plot(Hc_data[0], Hc_data[1], label='Hc')
    ax2.plot(Mr_data[0], Mr_data[1], label='Mrx')
    ax3.plot(Mr_data[0], Mr_data[2], label='Mry')
    ax1.scatter(Hc_data[0], Hc_data[1], label='Hc')
    ax2.scatter(Mr_data[0], Mr_data[1], label='Mrx')
    ax3.scatter(Mr_data[0], Mr_data[2], label='Mry')

    ax1.set_xlabel(r"$d$ / $l_{ex}$")
    ax1.set_ylabel(r"$H_c$ / $4 \pi M_s$")
    ax1.set_xlim(0, 30)
    ax1.set_ylim(0.04, 0.07)
    ax1.grid(True, lw=1.0, ls='-.')

    ax2.set_xlabel(r"$d$ / $l_{ex}$")
    ax2.set_ylabel(r"$<M_x>$ / $M_s$")
    ax2.set_xlim(0, 30)
    ax2.set_ylim(0.95, 1.0)
    ax2.grid(True, lw=1.0, ls='-.')

    ax3.set_xlabel(r"$d$ / $l_{ex}$")
    ax3.set_ylabel(r"$<M_y>$ / $M_s$")
    ax3.set_xlim(0, 30)
    ax3.set_ylim(0, 0.1)
    ax3.grid(True, lw=1.0, ls='-.')

    plt.savefig(path + plt_name)

    return None


###############################
# Prepare NeuralMAG2305 model #
###############################
# How many cells along each axis
size = (50, 10, 1)

# Size of each cell [nm]
cell = (2.5, 2.5, 2.5)

# Saturation magnetization [emu/cc]
Ms = 800
# Exchange stiffness [erg/cm]
Ax_list = [25, 1, 0.25, 0.2, 0.16, 0.12, 0.09, 0.07, 0.06, 0.05, 0.04]
# Uniaxial anisotropy energy [erg/cc]
Ku = 0.0e3


path0 = "./data_problem2/"
if not os.path.exists(path0):
    os.mkdir(path0)

# Find Hc for all Ax in Ax_list
Mr_rcd = np.array([ [],[],[],[] ])
Hc_rcd = np.array([ [],[] ])
for Ax in Ax_list:
    lx = np.sqrt(Ax*1.0e-6 /Ms**2 /2/np.pi) * 1.0e7

    # Create MM model
    film0 = MAG2305.mmModel(types='bulk', size=size, cell=cell, 
                            Ms=Ms, Ax = Ax*1.0e-6, Ku=Ku)

    # Initialize demag matrix
    film0.DemagInit()

    # Initialize spin state
    film0.SpinInit(Spin_in = MAG2305.get_randspin_2D(size=size, split=2, rand_seed=0))


###################
# M-H calculation #
###################
    print('\nBegin spin updating:\n')

    # method
    method = 'LLG_RK4'
    if Ax <= 1:
        dtime = 1.0e-13
    else:
        dtime = 1.0e-14

    Hext_range = np.linspace(10000, -10000, 20001)
    Hext_vec = np.array([1,1,1]) / np.sqrt(3)

    # Do iteration
    mext = 0
    time_start = time.time()
    for Hext_val in Hext_range:
        Hext = Hext_val * Hext_vec

        spin, spin_sum , *_ = film0.GetStableState( Hext=Hext, method=method, error_limit=1.0e-5, damping=0.1, dtime=dtime)

        if abs(Hext_val) < 0.0001: 
            Mr_rcd = np.append(Mr_rcd, [[size[1]*cell[1]/lx] , [spin_sum[0]], [spin_sum[1]], [spin_sum[2]]], axis=1)

        mext_pre = mext
        mext = np.dot(spin_sum, Hext_vec)

        if mext_pre > 0 and mext <= 0:
            Hc_rcd = np.append(Hc_rcd, [[size[1]*cell[1]/lx] , [abs(Hext_val) / Ms /4/np.pi]], axis=1)
            break

    time_finish = time.time()
    print('\nEnd case Ax={:.3e}. Time cost: {:.1f}s\n'
            .format(Ax, time_finish-time_start) )

    # Save data & plot data
    np.save(path0+'Mr_data-fftHd', Mr_rcd)
    np.save(path0+'Hc_data-fftHd', Hc_rcd)
    plot_data(path=path0, Hc_data=Hc_rcd, Mr_data=Mr_rcd, plt_name='Results-fftHd')


#######################
# Plot with benchmark #
#######################
# Benchmark
bench = [ [], [], [] ]
f0 = open(file='./benchmark/problem-2-Donahue-data.txt', mode='r')
for n, line in enumerate(f0):
    if n > 0:
        read = line.split()
        bench = np.append(bench, [ [float(read[0])], [float(read[2])], [float(read[4])] ], axis=1)
f0.close()

# Plot
fig = plt.figure(figsize=(9,4))
gs = fig.add_gridspec(100,100)
ax1 = fig.add_subplot(gs[5:95, :42])
ax2 = fig.add_subplot(gs[5:95, 58:])

ax1.scatter(bench[0],  bench[1],  label='Donahue', alpha=0.8)
ax1.scatter(Hc_rcd[0], Hc_rcd[1], label='fftHd', marker='*')
ax2.scatter(bench[0],  bench[2],  label='Donahue', alpha=0.8)
ax2.scatter(Mr_rcd[0], Mr_rcd[1], label='fftHd', marker='*')

ax1.grid(True, lw=0.8, ls='-.')
ax1.set_xlabel(r"$d$ / $l_{ex}$")
ax1.set_ylabel(r"$H_c$ / $4 \pi M_s$")
ax1.set_xlim(0, 30)
ax1.set_ylim(0.04, 0.07)
ax1.legend()

ax2.grid(True, lw=0.8, ls='-.')
ax2.set_xlabel(r"$d$ / $l_{ex}$")
ax2.set_ylabel(r"$<M_x>$ / $M_s$")
ax2.set_xlim(0, 30)
ax2.set_ylim(0.95, 1.0)
ax2.legend()

plt.savefig(path0+"Hc_Mr_benchmark-fftHd")
