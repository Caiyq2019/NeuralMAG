# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:26:00 2024
"""

import os, re
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as cls
import numpy as np


parser = argparse.ArgumentParser(description='Unet speed test method: LLG_RK4')    
parser.add_argument('--method',      type=str,    default="fft",     help='calculation method (default: fft)')
args = parser.parse_args()

# data list
data_list = np.array([[],[],[],[]])

# dir 
path0 = "./analyze_vortex/{}/".format(args.method)
file = "/summary.txt"

# get data
pattern = r"\d+"
for folder in os.listdir(path0):
    print("folder: " + path0 + folder)
    size = int(re.findall(pattern, folder)[0])

    f = open(file= path0+folder+file, mode="r")
    for line in f:
        if line.find("single") >=0:
            item = line.split(",")
            data = [[size], [float(re.findall(r"\d.\d+", item[0])[0])],
                            [float(re.findall(r"\d.\d+", item[1])[0])],
                            [float(re.findall(r"\d.\d+", item[2])[0])] ]
            data_list = np.append(data_list, data, axis=1)
    f.close()

data_list = data_list.T[data_list[0].argsort()].T

# plot phase diagram
x = data_list[0]
scale = 1000
y = np.linspace(0,scale,scale)
X, Y = np.meshgrid(x,y)

data_plot = np.zeros_like(X.T)
for i in range(len(x)):
    pt1 = int(data_list[1][i] * scale)
    pt2 = int(data_list[2][i] * scale) + pt1
    data_plot[i][:pt1] = 0
    data_plot[i][pt1:pt2] = 1
    data_plot[i][pt2:] = 2


# Font size
parameters = {'axes.labelsize' : 18,
              'axes.titlesize' : 18,
              'xtick.labelsize': 11,
              'ytick.labelsize': 11,
              'legend.fontsize': 15,
              'font.size'      : 15,  # text fontsize
              'figure.dpi'     : 200}
plt.rcParams.update(parameters)

# colormap
# colors = ["#501D8A", "#1C8041", "#E55709"]
# my_color = cls.ListedColormap(colors, name='from_list', N=3)

fig, ax = plt.subplots(figsize=(6,5))

extent = (x[0]-4, x[-1]+4, 0, 1)
ax.imshow(data_plot.T, origin="lower", 
          alpha=0.7, cmap="viridis",
          extent=extent, interpolation="nearest",
          )
ax.set_xlim(extent[0], extent[1])
ax.set_xticks(x, [int(i) for i in x])
ax.set_ylim(extent[2], extent[3])
ax.set_aspect(x[-1]/1.5)

# Add label
ax.set_xlabel("Sample size")
ax.set_ylabel("Probability")
ax.set_title("Ground state - {}".format(args.method))

# Add text
ax.text(x=31, y=0.04, s="single\ndomain", 
        style="italic", c="white")
ax.text(x=70, y=0.45, s="single vortex", 
        style="italic", c="white")
ax.text(x=80, y=0.9, s="multiple vortices", 
        style="italic", c="white")

plt.savefig("phase_diagram_{}".format(args.method))
plt.close()
