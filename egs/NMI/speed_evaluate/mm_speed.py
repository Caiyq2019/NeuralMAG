# -*- coding: utf-8 -*-

import numpy as np
import time
import argparse
import torch

from libs.misc import Culist, spin_prepare
import libs.MAG2305 as MAG2305
from libs.Unet import UNet


def initialize_models(args, device):
    #Initialize MAG2305 models.
    film1 = MAG2305.mmModel(types='bulk', size=(args.w, args.w, args.layers), cell=(3,3,3), 
                            Ms=args.Ms, Ax=args.Ax, Ku=args.Ku, Kvec=args.Kvec, 
                            device="cuda:" + str(args.gpu)
                            )
    print('Creating {} layer models \n'.format(args.layers))

    # spin initialization cases
    spin_split = np.random.randint(low=2, high=32)
    rand_seed = np.random.randint(low=0, high=100000)
    spin = spin_prepare(spin_split, film1, rand_seed)
    film1.SpinInit(spin)
    print('spin shape',film1.Spin.shape)

    # Initialize demag matrix
    time_start = time.time()
    film1.DemagInit()
    time_finish = time.time()
    print('Time cost: {:f} s for getting demag matrix \n'.format(time_finish-time_start))

    return film1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unet speed test')
    parser.add_argument('--gpu',        type=int,   default=0,         help='GPU ID (default: 0)')
    parser.add_argument('--krn',        type=int,   default=16,        help='unet first layer kernels (default: 16)')

    parser.add_argument('--w',          type=int,    default=32,        help='MAG model size (default: 32)')
    parser.add_argument('--layers',     type=int,    default=1,         help='MAG model layers (default: 1)')

    parser.add_argument('--Ms',         type=float,  default=1000,      help='MAG model Ms (default: 1000)')
    parser.add_argument('--Ax',         type=float,  default=0.5e-6,    help='MAG model Ax (default: 0.5e-6)')
    parser.add_argument('--Ku',         type=float,  default=0.0,       help='MAG model Ku (default: 0.0)')
    parser.add_argument('--Kvec',       type=Culist, default=(0,0,1),   help='MAG model Kvec (default: (0,0,1))')
    parser.add_argument('--damping',    type=float,  default=0.1,       help='MAG model damping (default: 0.1)')
    parser.add_argument('--Hext_val',   type=float,  default=0,         help='external field value (default: 0.0)')
    parser.add_argument('--Hext_vec',   type=Culist, default=(1,0,0),   help='external field vector (default:(1,1,0))')

    parser.add_argument('--dtime',      type=float,  default=1.0e-13,   help='real time step (default: 1.0e-13)')
    parser.add_argument('--n_loop',     type=int,    default=100,       help='loop number (default: 100)')
    args = parser.parse_args()
    
    device = torch.device("cuda:{}".format(args.gpu))

    # Initialize MAG models, prepare films.
    film1 = initialize_models(args, device)

    ######################
    # MAG2305 speed test #
    ######################
    #MAG2305 spin calc speed test
    spin_step_times = torch.zeros(args.n_loop).to(device)
    for i in range(args.n_loop):
        torch.cuda.synchronize()
        start_time = time.time()
        error_mm = film1.SpinLLG_RK4()
        torch.cuda.synchronize()
        end_time = time.time()
        spin_step_times[i] = end_time - start_time

    Spin_speed = torch.mean(spin_step_times[10:]).item()


    # FFT Hd calculation speed test
    hd_calc_times = torch.zeros(args.n_loop).to(device)
    for i in range(args.n_loop):
        torch.cuda.synchronize()
        start_time = time.time()
        film1.Demag()
        torch.cuda.synchronize()
        end_time = time.time()
        hd_calc_times[i] = end_time - start_time

    Hd_speed = torch.mean(hd_calc_times[10:]).item()*4.0

    print(f'||MAG_size: {args.w} || Spin calc speed: {Spin_speed:.1e} s || Hd calc speed: {Hd_speed:.1e} s||')

