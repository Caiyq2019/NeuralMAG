# -*- coding: utf-8 -*-
"""
Created on Fri Jun 02 09:00:00 2023

#########################################
#                                       #
#  muMAG Standard Problem #4            #
#                                       #
#  -- Permalloy film LLG dynamic        #
#  -- 500nm x 125nm x 3nm               #
#  -- Ax = 1.3e-6  erg/cm               #
#  -- Ms = 800     emu/cc               #
#  -- Ku = 0.0     erg/cc               #
#  -- alpha = 0.02                      #
#  -- Hext = 250 Oe [170* ccw from +x]  #
#  or Hext = 360 Oe [190* ccw from +x]  #
#                                       #
#########################################

"""


import libs.MAG2305 as MAG2305
import numpy as np
import time, os


#########################
# Define functions here #
#########################
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
    ax = fig.add_subplot(111)

    ax.plot(data[0], data[1], label="<mx>")
    ax.plot(data[0], data[2], label="<my>")
    ax.plot(data[0], data[3], label="<mz>")

    ax.set_xlabel("Time [ns]")
    ax.set_ylabel(r"$<M>$ / $M_s$")
    ax.set_ylim(-1.0, 1.0)
    ax.grid(True, lw=1.0, ls='-.')
    ax.legend()

    plt.savefig(path + plt_name)

    return None

# Plot benchmark
marker_list = ['o', '*', '^', 'P', 'X', 's', 'D' ]
def plot_benchmark(data_list, plot_name):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    for n, data in enumerate(data_list):
        if data[0] == 'fftHd':
            ax.plot( data[1][0], data[1][2], label=data[0],
                     alpha=0.8, linewidth=2.0, color='k')
        else:
            ax.scatter( data[1][0], data[1][1], label=data[0], 
                        alpha=0.5, marker=marker_list[n], s=14)

    ax.set_xlim(0,1)
    ax.set_ylim(-0.8,0.8)
    ax.set_xlabel(r"time [ns]")
    ax.set_ylabel(r"$M_y$ / $M_s$")
    ax.set_title(r"$\mu MAG$ Problem#4 Test")

    ax.legend()
    ax.grid(True, lw=1.0, ls='-.')

    plt.savefig(plot_name)

    return None

# External field function
def assign_external_field(t, cases):

    if cases == 0:
        return max(0, 10000 - t*2.0e12) * np.array([1,1,1])

    elif cases == 1:
        return 250 * np.array([ np.cos(170/180*np.pi),
                                np.sin(170/180*np.pi), 
                                0])
    elif cases == 2:
        return 360 * np.array([ np.cos(190/180*np.pi),
                                np.sin(190/180*np.pi), 
                                0])
    else:
        return 0


###############################
# Prepare NeuralMAG2305 model #
###############################
# How many cells along each axis
size = (200, 50, 1)
cell_num = size[0] * size[1] * size[2]

# Size of each cell [nm]
cell = (2.5, 2.5, 3)

# Saturation magnetization [emu/cc]
Ms = 800
# Exchange stiffness [erg/cm]
Ax = 1.3e-6
# Uniaxial anisotropy energy [erg/cc]
Ku = 0.0e3
Kvec = (1.0, 0, 0)

# Create a test-model
film0 = MAG2305.mmModel(types='bulk', size=size, cell=cell,
                        Ms=Ms, Ax=Ax, Ku=Ku, Kvec=Kvec)

# Initialize demag matrix
time_start = time.time()
film0.DemagInit()
time_finish = time.time()
print('Time cost: {:f} s for getting demag matrix \n'.format(time_finish-time_start))

# Initialize spin state
# Problem #4 requirement : saturated along [1,1,1]
spin = np.ones( tuple(film0.size) + (3,) )
film0.SpinInit(spin)


###################
# LLG calculation #
###################
path0 = "./data_problem4/"
if not os.path.exists(path0):
    os.mkdir(path0)

print('\nBegin spin updating:\n')

# method
method = 'LLG_RK4'

# Total time [s]
time_relax = 8.0e-9
time_llg = 1.0e-9

# Time step [s]
dtime = 1.0e-13

iters_relax = int(time_relax / dtime)
iters_llg = int(time_llg / dtime) + 1

# Damping alpha
damping = 0.02

########################
# Relax from M = [1,1,1]
time_start = time.time()
for iters in range(iters_relax):
    t = iters * dtime
    Hext = assign_external_field(t, 0)

    error = film0.SpinLLG_RK4(Hext=Hext, dtime=dtime, damping=0.1)

    spin = film0.Spin.cpu().numpy()
    spin_sum = np.sum(spin, axis=(0,1,2)) / cell_num

    if iters % 1000 == 0:
        print('  Hext={},  steps={:d}'.format(list(Hext), iters+1))
        print('  M_avg={} \n'.format(list(spin_sum)))

spin_relax = spin
time_finish = time.time()
print('\nEnd state relaxation. Time cost: {:.1f}s\n'
                              .format(time_finish-time_start) )

########################
# Case 1
print('\nBegin case 1 calculation:\n')

MH_rcd = np.array([ [],[],[],[] ])
time_start = time.time()
for iters in range(iters_llg):
    t = iters * dtime

    Hext = assign_external_field(t, 1)

    error = film0.SpinLLG_RK4(Hext=Hext, dtime=dtime, damping=damping)

    spin = film0.Spin.cpu().numpy()
    spin_sum = np.sum(spin, axis=(0,1,2)) / cell_num

    MH_rcd = np.append(MH_rcd, [ [t/1.0e-9], 
                                 [spin_sum[0]], [spin_sum[1]], [spin_sum[2]] ], 
                       axis=1)

    if iters % 1000 == 0:
        print('  Hext={},  steps={:d}'.format(list(Hext), iters+1))
        print('  M_avg={} \n'.format(list(spin_sum)))

time_finish = time.time()
print('\nEnd Case 1 calculation. Time cost: {:.1f}s\n'
                              .format(time_finish-time_start) )

np.save(path0+'Mt_case1-fftHd', MH_rcd)
plot_data(path=path0, data=MH_rcd, plt_name='Mt_case1-fftHd')


########################
# Case 2
film0.SpinInit(spin_relax)
print('\nBegin case 2 calculation:\n')

MH_rcd = np.array([ [],[],[],[] ])
time_start = time.time()
for iters in range(iters_llg):
    t = iters * dtime

    Hext = assign_external_field(t, 2)

    error = film0.SpinLLG_RK4(Hext=Hext, dtime=dtime, damping=damping)

    spin = film0.Spin.cpu().numpy()
    spin_sum = np.sum(spin, axis=(0,1,2)) / cell_num

    MH_rcd = np.append(MH_rcd, [ [t/1.0e-9], 
                                 [spin_sum[0]], [spin_sum[1]], [spin_sum[2]] ], 
                       axis=1)

    if iters % 1000 == 0:
        print('  Hext={},  steps={:d}'.format(list(Hext), iters+1))
        print('  M_avg={} \n'.format(list(spin_sum)))

time_finish = time.time()
print('\nEnd Case 2 calculation. Time cost: {:.1f}s\n'
                              .format(time_finish-time_start) )

np.save(path0+'Mt_case2-fftHd', MH_rcd)
plot_data(path=path0, data=MH_rcd, plt_name='Mt_case2-fftHd')


#######################
# Plot with benchmark #
#######################
# Benchmark data case1
path = "./benchmark/"
name = "problem-4-Donahue_case1.txt"
data_Donahue = np.array([[],[]])
f = open(path + name, mode='r')
for n, line in enumerate(f):
    if n >= 5:
        h = float(line.split()[-1]) / 1.0e-9
        m = float(line.split()[-4])
        data_Donahue = np.append( data_Donahue, [[h],[m]], axis=1 )

name = "problem-4-Rasmus_case1.txt"
data_Rasmus = np.array([[],[]])
f = open(path + name, mode='r')
for n, line in enumerate(f):
    if n >= 1:
        h = float(line.split()[0]) / 1.0e-9
        m = float(line.split()[2])
        data_Rasmus = np.append( data_Rasmus, [[h],[m]], axis=1 )

# NeuralMAG data
data_name = "Mt_case1-fftHd.npy"
data_nnmag = np.load(path0 + data_name)

data_for_plot = [ ["Donahue", data_Donahue],
                  ["Rasmus",  data_Rasmus], 
                  ["fftHd", data_nnmag] ]
plot_benchmark( data_list=data_for_plot, plot_name=path0 + "Mt_case1_benchmark-fftHd" )


# Benchmark data case2
path = "./benchmark/"
name = "problem-4-Donahue_case2.txt"
data_Donahue = np.array([[],[]])
f = open(path + name, mode='r')
for n, line in enumerate(f):
    if n >= 5:
        h = float(line.split()[-1]) / 1.0e-9
        m = float(line.split()[-4])
        data_Donahue = np.append( data_Donahue, [[h],[m]], axis=1 )

name = "problem-4-Rasmus_case2.txt"
data_Rasmus = np.array([[],[]])
f = open(path + name, mode='r')
for n, line in enumerate(f):
    if n >= 1:
        h = float(line.split()[0]) / 1.0e-9
        m = float(line.split()[2])
        data_Rasmus = np.append( data_Rasmus, [[h],[m]], axis=1 )

# NeuralMAG data
data_name = "Mt_case2-fftHd.npy"
data_nnmag = np.load(path0 + data_name)

data_for_plot = [ ["Donahue", data_Donahue],
                  ["Rasmus",  data_Rasmus], 
                  ["fftHd", data_nnmag] ]
plot_benchmark( data_list=data_for_plot, plot_name=path0 + "Mt_case2_benchmark-fftHd" )
