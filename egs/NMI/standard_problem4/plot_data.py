# -*- coding: utf-8 -*-
"""
Created on Mon Jul 01 17:00:00 2024

"""

import numpy as np
import matplotlib.pyplot as plt

# Font size
parameters = {'axes.labelsize' : 13,
              'axes.titlesize' : 13,
              'xtick.labelsize': 11,
              'ytick.labelsize': 11,
              'legend.fontsize': 11,
              'figure.dpi'     : 300}
plt.rcParams.update(parameters)

# Canvas
fig = plt.figure(figsize=(10,10))
subplot_list = [(3,2,1), (3,2,2), (3,2,3), (3,2,4), (3,1,3)]
# index
sn_list = ["(a)", "(b)", "(c)", "(d)", "(e)"]
rp = [[0.07, 0.10], [0.07, 0.10], [0.07, 0.10], 
      [0.07, 0.10], [0.92, 0.82]]

# Plot benchmark
marker_list = ['o', '*', '^', 'P', 'X', 's', 'D' ]
def plot_benchmark(data_list, title, fig, iax):

    ax = fig.add_subplot(*subplot_list[iax])

    for n, data in enumerate(data_list):
        if "Unet" in data[0]:
            ax.plot( data[1][0], data[1][1], label=data[0],
                     alpha=0.7, linewidth=2.0, color='k')
        elif "FFT" in data[0]:
            ax.plot( data[1][0], data[1][1], label=data[0],
                     alpha=0.5, linewidth=2.0, color='g')
        else:
            ax.scatter( data[1][0], data[1][1], label=data[0], 
                        alpha=0.9, marker=marker_list[n], s=12)

    ax.set_xlim(0,1)
    if iax >= 4:
        ax.set_xlim(0,4)

    ylim = data[1][1].max() + 0.1
    ax.set_ylim(-ylim, ylim)

    # if iax in [4]:
    if True:
        ax.set_xlabel(r"time [ns]")
    # if iax in [0,2,4]:
    if True:
        ax.set_ylabel(r"$M_y$ / $M_s$")

    ax.set_title(title)

    ncol = 2 if iax==3 else 1
    ax.legend(ncol=ncol, loc="lower right")
    ax.grid(True, lw=1.0, ls='-.')

    # Add index
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(x = rp[iax][0]*xlim[1], 
            y = rp[iax][1]*(ylim[1]-ylim[0]) + ylim[0],
            s = sn_list[iax], fontsize=16)

    return None


#######################
# Plot with benchmark #
#######################
path0 = "./data_problem4/"
for iax, size in enumerate([32, 64, 128, 512]):
    # NeuralMAG data
    data_name = "Mt_case1-fft_size{}.npy".format(size)
    data_fft = np.load(path0 + data_name)[::2]
    data_name = "Mt_case1-unet_size{}.npy".format(size)
    data_unet = np.load(path0 + data_name)[::2]

    data_for_plot = [ ["FFT/LLG", data_fft],
                      ["Unet/LLG",  data_unet] ]

    title = "{}nm x {}nm x 3nm".format(int(size*1.5), int(size*1.5//4))

    if size >= 512:
        # Benchmark data case1
        path = "./benchmark/"
        name = "problem-4-Donahue_case1.txt"
        data_Donahue = np.array([[],[]])
        f = open(path + name, mode='r')
        for n, line in enumerate(f):
            # if n >= 5:
            if n >= 5 and n%4==0:
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

        data_for_plot.extend( [["Donahue", data_Donahue],
                               ["Rasmus",  data_Rasmus]] )

        # title=r"$\mu MAG$ Problem#4"
        title= "500nm x 125nm x 3nm (Problem #4)"

    plot_benchmark(data_list=data_for_plot, 
                   title=title, fig=fig, iax=iax)

# plot converge
data_name = "Mt_case1-fft_size512_converge.npy"
data_fft = np.load(path0 + data_name)[::2]
data_name = "Mt_case1-unet_size512_converge.npy"
data_unet = np.load(path0 + data_name)[::2]

data_for_plot = [ ["FFT/LLG", data_fft],
                  ["Unet/LLG",  data_unet] ]

title= "500nm x 125nm x 3nm (Problem #4, longer time)"
plot_benchmark(data_list=data_for_plot, 
               title=title, fig=fig, iax=4)


# save figure
plt.tight_layout()
plot_name="Mt_case1_benchmark.jpg"
plt.savefig(plot_name, format="JPG")
plt.close()
