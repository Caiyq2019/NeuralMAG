# -*- coding: utf-8 -*-
"""
Created on Mon Jul 01 10:00:00 2024

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


import numpy as np
import time, os
import argparse
import matplotlib.pyplot as plt

import libs.MAG2305 as MAG2305
import torch
from libs.Unet import UNet

#########################
# Define functions here #
#########################

def get_args():
    parser = argparse.ArgumentParser(description='Temporal test')
    parser.add_argument('--w',          type=int,       default=32,        help='sample in-plane size (default: 32)')
    parser.add_argument('--cell',       type=float,     default=1.5,       help='mm cell size (default: 1.5)')
    parser.add_argument('--Ms',         type=float,     default=800,       help='saturation Ms (default: 800)')
    parser.add_argument('--Ax',         type=float,     default=1.3e-6,    help='exchange stiffness Ax (default: 1.3e-6)')
    parser.add_argument('--Ku',         type=float,     default=0.0,       help='uniaxial anisotropy Ku (default: 0.0)')
    parser.add_argument('--damping',    type=float,     default=0.02,      help='damping constatn (default: 0.02)')
    parser.add_argument('--dtime',      type=float,     default=1.0e-13,   help='real time step (default: 1.0e-13)')
    parser.add_argument('--converge',   type=bool,      default=False,     help='calculation to convergence (default: False)')

    return parser.parse_args()


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


# Define shape of model
def define_model_shape(args):
    if args.w < 512:
        # width : height = 4 : 1
        w = args.w
        h = w // 4
        model = np.zeros((w,w,2), dtype=int)
        model[:, (w-h)//2:(w+h)//2, :] = 1

    else:
        # standard problem 4
        w = args.w
        model = np.zeros((w,w,2), dtype=int)
        model[1:334, 1:84, :] = 1

    return model


# Run LLG simulation
def run_relax(args, spin_relax=None):
    cell = (args.cell, args.cell, args.cell)
    model = define_model_shape(args)
    cell_num = model[model>0].sum()

    # Create a test-model
    film0 = MAG2305.mmModel(types='bulk', cell=cell, model=model,
                            Ms=args.Ms, Ax=args.Ax, Ku=args.Ku)

    # Initialize demag matrix
    film0.DemagInit()

    # Initialize spin state
    if spin_relax is not None:
        spin_relax = np.array(spin_relax)
        film0.SpinInit(spin_relax)

    else:
        # Problem #4 requirement : saturated along [1,1,1]
        spin = np.ones( tuple(film0.size) + (3,) )
        film0.SpinInit(spin)

        # Relax from M = [1,1,1]
        print('\nBegin spin relaxation (size={}):\n'.format(args.w))
        time_relax = 8.0e-9
        iters_relax = int(time_relax / args.dtime)

        time_start = time.time()
        for iters in range(iters_relax):
            t = iters * args.dtime
            Hext = assign_external_field(t, 0)

            _ = film0.SpinLLG_RK4(Hext=Hext, dtime=args.dtime, damping=0.1)

            spin = film0.Spin.cpu().numpy()
            spin_sum = np.sum(spin, axis=(0,1,2)) / cell_num

            if iters % 1000 == 0:
                print('  Hext={},  steps={:d}'.format(list(Hext), iters+1))
                print('  M_avg={} \n'.format(list(spin_sum)))

        spin_relax = spin
        time_finish = time.time()
        print('\nEnd state relaxation. Time cost: {:.1f}s\n'
                                      .format(time_finish-time_start) )

    return film0, spin_relax


def run_llg(args, nnModel, method="fft"):
    path0 = "./data_problem4/"
    if not os.path.exists(path0):
        os.mkdir(path0)

    cell_num = nnModel.model[nnModel.model>0].sum()

    # Case 1 calculation
    print('\nBegin case 1 calculation (size={}, {}):\n'.format(args.w, method))
    time_llg = 4.0e-9 if args.converge else 1.0e-9
    iters_llg = int(time_llg / args.dtime) + 1
    MH_rcd = np.array([ [],[],[],[] ])
    time_start = time.time()
    for iters in range(iters_llg):
        t = iters * args.dtime

        Hext = assign_external_field(t, 1)

        if method == "unet":
            _ = nnModel.SpinLLG_RK4_unetHd(Hext=Hext, dtime=args.dtime, damping=args.damping)
        else:
            _ = nnModel.SpinLLG_RK4(Hext=Hext, dtime=args.dtime, damping=args.damping)

        spin = nnModel.Spin.cpu().numpy()
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

    file_name = "Mt_case1-{}_size{}".format(method, args.w)
    if args.converge:
        file_name += "_converge"
    np.save(path0+file_name, MH_rcd)
    plot_data(path=path0, data=MH_rcd, plt_name=file_name)

    return None


########
# main #
########

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

    # run test
    model0, spin_relax = run_relax(args=args)
    run_llg(args=args, nnModel=model0, method="fft")

    model1, _ = run_relax(args=args, spin_relax=spin_relax)
    run_llg(args=args, nnModel=model1, method="unet")
