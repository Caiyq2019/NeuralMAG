# -*- coding: utf-8 -*-
"""
Created on Tue May 16 20:00:00 2023

#########################################
#                                       #
#  muMAG Standard Problem #1            #
#                                       #
#  -- Permalloy film hysteresis         #
#  -- 2um x 1um x 20nm                  #
#  -- Ax = 1.3e-6  erg/cm               #
#  -- Ms = 800     emu/cc               #
#  -- Ku = 5.0e3   erg/cc [//long edge] #
#                                       #
#########################################

"""


import libs.MAG2305 as MAG2305
import numpy as np
import time, os


###############################
# Prepare NeuralMAG2305 model #
###############################
# How many cells along each axis
size = (100, 50, 1)
# size = (200, 100, 2)

# Size of each cell [nm]
cell = (20, 20, 20)
# cell = (10, 10, 10)

# Saturation magnetization [emu/cc]
Ms = 800
# Exchange stiffness [erg/cm]
Ax = 1.3e-6
# Uniaxial anisotropy energy [erg/cc]
Ku = 5.0e3
Kvec = (1.0, 0, 0)

# Create MM model
film0 = MAG2305.mmModel(types="bulk", size=size, cell=cell, 
                        Ms=Ms, Ax=Ax, Ku=Ku, Kvec=Kvec)

# Initialize demag matrix
time_start = time.time()
film0.DemagInit()
time_finish = time.time()
print('Time cost: {:f} s for getting demag matrix \n'.format(time_finish-time_start))


########################
# M-H loop calculation #
########################
path0 = "./data_problem1/"
if not os.path.exists(path0):
    os.mkdir(path0)

print('\nBegin spin updating:\n')

# External field - x
Hext_range = np.linspace(500, -500, 1001)
Hext_vec = np.array([ np.cos(1/180*np.pi), 
                      np.sin(1/180*np.pi), 0.0 ])

# method
method = 'LLG_RK4'

# Initialize spin state
film0.SpinInit(Spin_in = MAG2305.get_randspin_2D(size=size, split=10, rand_seed=0))

MH_rcd = np.array([ [],[] ])
time_start = time.time()
for nloop, Hext_val in enumerate(Hext_range):

    Hext = Hext_val * Hext_vec

    spin, spin_sum, error, iters = film0.GetStableState( Hext=Hext, method=method, error_limit=1.0e-5 )

    MH_rcd = np.append(MH_rcd, [[Hext_val], [np.dot(Hext_vec, spin_sum)]], axis=1)

    if abs(Hext_val) < 0.001:
        spin_Mr = spin[:,:,0,:]

    print(' -----------------------------------------')
    print(' -----------------------------------------\n')

time_finish = time.time()
print('\nEnd M-H calculation. Time cost: {:.1f}s\n'
      .format(time_finish-time_start) )

# Save data
np.save(path0+'MHx_data-fftHd', MH_rcd)
np.save(path0+'Mrx_spin-fftHd', spin_Mr)


##################################
# Plot MH loop & remanence state #
##################################
import matplotlib.pyplot as plt

# Font size
parameters = {'axes.labelsize' : 15,
              'axes.titlesize' : 15,
              'xtick.labelsize': 12,
              'ytick.labelsize': 12,
              'legend.fontsize': 15,
              'figure.dpi'     : 200}
plt.rcParams.update(parameters)


def plot_data(path, mh_name, spin_name, plt_name):

    MH   = np.load(path + mh_name + '.npy')
    spin = np.load(path + spin_name + '.npy')

    fig = plt.figure(figsize=(6,8))
    gs = fig.add_gridspec(10,1)
    ax1 = fig.add_subplot(gs[0:5,0])
    ax2 = fig.add_subplot(gs[5:10,0])

    xlim0 = -1.2*np.abs(MH[0]).max() 
    xlim1 =  1.2*np.abs(MH[0]).max()
    ylim0 = -1.2*np.abs(MH[1]).max() 
    ylim1 =  1.2*np.abs(MH[1]).max()
    xyrat = xlim1 / ylim1
    ax1.plot( [xlim0, xlim1], [0, 0], lw=0.5, linestyle='--', color='grey')
    ax1.plot( [0, 0], [ylim0, ylim1], lw=0.5, linestyle='--', color='grey')
    ax1.plot( MH[0],  MH[1], lw=1.5, color='k', alpha=0.8 )
    ax1.plot(-MH[0], -MH[1], lw=1.5, color='k', alpha=0.8 )
    ax1.set_aspect( xyrat,
            adjustable='box')
    ax1.set_xlim( xlim0, xlim1 )
    ax1.set_ylim( ylim0, ylim1 )
    ax1.set_xlabel(r"$H_{ext}$ [Oe]")
    ax1.set_ylabel(r"$M_{ext}$ / $M_s$")

    ax2.quiver( spin[:,:,0].T, spin[:,:,1].T,
                spin[:,:,2].T, clim=[-0.5, 0.5] )
    # ax2.imshow( spin[:,:,0].T, origin='lower' )
    ax2.set_aspect( "equal", adjustable='box')
    ax2.set_xlabel(r"$x$")
    ax2.set_ylabel(r"$y$")

    plt.tight_layout()
    plt.savefig(path + plt_name)

    return None

plot_data(path=path0, mh_name='MHx_data-fftHd', spin_name='Mrx_spin-fftHd', plt_name='MH_Mr-fftHd')


#######################
# Plot with benchmark #
#######################
# Benchmark data
path = "./benchmark/"
data_name1 = "problem-1-mo96a-data.txt"
data_name2 = "problem-1-pb97a-data.txt"
data_name3 = "problem-1-ts96b-data.txt"

for i, name in enumerate([data_name1, data_name2, data_name3]):
    data_bench = np.array([[],[]])

    with open(path + name) as f:
        for line in f:
            h = float(line.split(",")[0])
            m = float(line.split(",")[1])
            data_bench = np.append( data_bench, [[h],[m]], axis=1 )

    if i == 0:
        data_mo96a = data_bench
    elif i == 1:
        data_pb97a = data_bench
    elif i == 2:
        data_ts96b = data_bench


# NeuralMAG data
path = path0
data_mag = np.load(path + "MHx_data-fftHd.npy")

# Plot
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111)
ax.scatter( data_pb97a[0], data_pb97a[1], label="pb97a", 
            alpha=0.5, marker="*", linewidth=1.0 )
ax.scatter( data_ts96b[0], data_ts96b[1], label="ts96b", 
            alpha=0.5, marker="P", linewidth=1.0 )
ax.scatter( data_mo96a[0], data_mo96a[1], label="mo96a", 
            alpha=0.5, marker="o", linewidth=1.0 )
ax.plot(    data_mag[0], data_mag[1], label="fftHd",
            alpha=0.7, linewidth=2.0, color='k')

ax.set_xlabel(r"$H_{ext}$ [Oe]")
ax.set_ylabel(r"$M_{ext}$ / $M_s$")
ax.set_title(r"$\mu MAG$ Problem#1 Test")
ax.legend()
ax.set_aspect( data_mag[0][0], adjustable='box')

plt.savefig(path + "MH_vs_benchmark-fftHd")
