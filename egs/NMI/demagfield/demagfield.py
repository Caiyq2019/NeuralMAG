import numpy as np
import argparse, sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.gridspec import GridSpec

import libs.MAG2305 as MAG2305
import torch
from libs.Unet import UNet
from model_utils import *


# Font size
parameters = {'axes.labelsize' : 13,
              'axes.titlesize' : 13,
              'xtick.labelsize': 11,
              'ytick.labelsize': 11,
              'legend.fontsize': 11,
              'figure.dpi'     : 300}
plt.rcParams.update(parameters)


#############
# Fucntions #
#############

def get_args():

    parser = argparse.ArgumentParser(description='plot demag field')

    parser.add_argument('--w',          type=int,       default=32,        help='sample in-plane size (default: 32)')
    parser.add_argument('--cell',       type=float,     default=3.0,       help='mm cell size (default: 3.0)')
    parser.add_argument('--Ms',         type=float,     default=1000,      help='saturation Ms (default: 1000)')

    return parser.parse_args()


def get_state(case, args):

    grid = (args.w, args.w, 2)

    if "square" in case:
        shape = np.ones(grid, dtype=int)

    elif "convex" in case:
        mask = create_random_mask(size=args.w, seed=5)[:,:,:,0]
        shape = mask.repeat(2, axis=2)

    else:
        print("Unknown shape case!")
        sys.exit(0)

    if "uniform" in case:
        spin = np.zeros(grid + (3,))
        spin[...,:] = [1,0,0]

    elif "multidm" in case:
        spin = np.zeros(grid + (3,))
        for ijk in np.ndindex(grid):
            if ijk[0] >= ijk[1] and ijk[0] + ijk[1] >= args.w:
                spin[ijk] = [-1, 0,0]
            elif ijk[0] < ijk[1] and ijk[0] + ijk[1] < args.w:
                spin[ijk] = [ 1, 0,0]
            elif ijk[0] >= ijk[1] and ijk[0] + ijk[1] < args.w:
                spin[ijk] = [ 0,-1,0]
            elif ijk[0] < ijk[1] and ijk[0] + ijk[1] >= args.w:
                spin[ijk] = [ 0, 1,0]
        spin[shape==0] = [0,0,0]

    else:
        print("Unknown shape case!")
        sys.exit(0)

    return shape, spin


def create_sample(args, shape, spin):

    cell = (args.cell, args.cell, args.cell)

    # Create mm model
    film0 = MAG2305.mmModel(types='bulk', cell=cell, model=shape,
                            Ms=args.Ms)

    # Initialize demag matrix
    film0.DemagInit()

    # Initialize spin state
    film0.SpinInit(spin)

    return film0


def plot_spin(ax, data, stride=8):

    pos = [(data.shape[0]//4-1, data.shape[1]//2-1),
           (data.shape[0]//2-1, data.shape[1]//4-1),
           (data.shape[0]//2-1, 3*data.shape[1]//4-1),
           (3*data.shape[0]//4-1, data.shape[1]//2-1) ]

    for ij in pos:
        x = ij[0] + 0.5,
        y = ij[1] + 0.5,
        U = data[ij][0]
        V = data[ij][1]
        ax.quiver(x,y,U,V,color="white", 
                  scale_units="inches", scale=3,
                  width=0.02,
                  pivot="mid")

    data_im = (data + np.array([1,1,1]))/2

    extent = (0, data_im.shape[0], 0, data_im.shape[1])

    ax.imshow(data_im.transpose(1,0,2), origin="lower",
              extent=extent, vmin=0, vmax=1)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(20))

    return None


def plot_field(ax, data, stride=4):

    start = stride//2
    end = start - stride

    x = np.linspace(start, data.shape[0]+end, int(data.shape[0]/stride))
    y = np.linspace(start, data.shape[1]+end, int(data.shape[1]/stride))

    X, Y = np.meshgrid(x,y)

    data_norm = np.linalg.norm(data, axis=-1)
    data_quiv = np.zeros_like(data)

    for l in range(3):
        data_quiv[:,:,l] = data[:,:,l] / data_norm

    ax.quiver(X, Y, 
              data_quiv[start::stride,start::stride,0].T, 
              data_quiv[start::stride,start::stride,1].T,
              color="white",
              headwidth=10, headlength=20,
              headaxislength=20, pivot="mid")

    extent = (0, data_norm.shape[0], 0, data_norm.shape[1])

    ax.imshow(data_norm.T, origin="lower", 
              extent=extent)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(20))

    return None


def plot_diff(ax, data, cmap):

    ax.imshow(data.T, origin="lower", cmap=cmap)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(20))

    return None


def plot_cbar(ax, max, min, label, cmap="viridis"):

    cbar = np.linspace(int(min), int(max), int(max-min))
    cbar = np.array([cbar, cbar])

    ax.imshow(cbar.T, origin="lower", cmap=cmap)

    ax.xaxis.set_visible(False)
    ax.yaxis.tick_right()
    ax.set_ylabel(label)
    ax.yaxis.set_label_position("right")
    ax.set_aspect(30/(max-min))

    return None


def plot_plot(axs, sample):

    spin = sample.Spin.cpu().numpy()[:,:,0]
    plot_spin(axs[0], spin)

    sample.GetHeff_intrinsic()
    Hd_fft = sample.Hd.cpu().numpy()[:,:,0]
    plot_field(axs[1], Hd_fft)

    sample.GetHeff_unetHd()
    Hd_unet = sample.Hd.cpu().numpy()[:,:,0]
    plot_field(axs[2], Hd_unet)

    data_max = np.linalg.norm(Hd_unet, axis=-1).max()
    data_min = np.linalg.norm(Hd_unet, axis=-1).min()
    plot_cbar(axs[3], data_max, data_min, label="Hd [Oe]")

    diff = np.linalg.norm(Hd_fft - Hd_unet, axis=-1)
    plot_diff(axs[4], diff, cmap="cool")

    data_max = diff.max()
    data_min = diff.min()
    plot_cbar(axs[5], data_max, data_min, label="Herr [Oe]", cmap="cool")

    return None

########
# Main #
########

case_list = ["square_uniform", "square_multidm",
             "convex_uniform", "convex_multidm" ]

if __name__ == "__main__":
    # Load Unet model
    krn=16
    device = torch.device("cuda:0")
    inch = 6

    model = UNet(kc=krn, inc=inch, ouc=inch).eval().to(device)
    ckpt = '../ckpt/k{}/model.pt'.format(krn)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    MAG2305.load_model(model)

    # get args
    args = get_args()

    # Canvas
    fig = plt.figure(figsize=(12,11))
    gs = GridSpec(nrows=4, ncols=6, 
                  width_ratios=[2,2,2,1,2,1], 
                  height_ratios=[1,1,1,1])

    axs = []
    for nline, case in enumerate(case_list):

        shape, spin = get_state(case, args)

        sample = create_sample(args=args, shape=shape, spin=spin)

        for i in range(6):
            axs.append(fig.add_subplot(gs[6*nline + i]))

        plot_plot(axs[6*nline:6*(nline+1)], sample)

    # Adjust subplots location
    plt.subplots_adjust(left=0.07, right=0.94, bottom=0.04, top=0.96)

    for nline in range(len(case_list)):

        iax = 1 + nline * 6
        l, b, w, h = axs[iax].get_position().bounds
        axs[iax].set_position([l+0.2*w, b, w, h])

        iax = 2 + nline * 6
        l, b, w, h = axs[iax].get_position().bounds
        axs[iax].set_position([l+0.4*w, b, w, h])

        iax = 3 + nline * 6
        l, b, w, h = axs[iax-1].get_position().bounds
        ll, bb, ww, hh = axs[iax].get_position().bounds
        axs[iax].set_position([ll+0.1*w, b, ww, h])

        iax = 4 + nline * 6
        l, b, w, h = axs[iax].get_position().bounds
        axs[iax].set_position([l+0.5*w, b, w, h])

        iax = 5 + nline * 6
        l, b, w, h = axs[iax-1].get_position().bounds
        ll, bb, ww, hh = axs[iax].get_position().bounds
        axs[iax].set_position([ll+0.2*w, b, ww, h])

    # Add titles
    axs[0].set_title("Spin", fontsize=14)
    axs[1].set_title("Hd-FFT", fontsize=14)
    axs[2].set_title("Hd-Unet", fontsize=14)
    axs[4].set_title("Hd error", fontsize=14)

    # Add index
    xt = -25
    yt = 60
    axs[0].text(x=xt, y=yt, s="(a)", fontsize=15)
    axs[6].text(x=xt, y=yt, s="(b)", fontsize=15)
    axs[12].text(x=xt, y=yt, s="(c)", fontsize=15)
    axs[18].text(x=xt, y=yt, s="(d)", fontsize=15)

    # save figure
    plt.savefig("demag_field.jpg", format="JPG")
    plt.close()
