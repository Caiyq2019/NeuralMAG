# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:09:00 2024
"""

import torch
import numpy as np
import time, os
from tqdm import tqdm
import argparse

from util.plot import *
from util.vortex_utils import get_winding

from libs.misc import Culist, create_model
import libs.MAG2305 as MAG2305
from libs.Unet import UNet


def load_unet_model(args):
    # load Unet Model
    model = UNet(kc=args.krn, inc=args.layers*3, ouc=args.layers*3).eval().to(device)
    ckpt = '../ckpt/k{}/model.pt'.format(args.krn)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    MAG2305.load_model(model)
    print('Unet model loaded from {}'.format(ckpt))


def initialize_models(args):
    # Model shape and save model
    test_model = create_model(args.w, args.modelshape)
    path0 = "./fft/size{}/".format(args.w)+"InitCore{}/".format(args.InitCore)
    os.makedirs(path0, exist_ok=True)
    np.save(path0 + 'model', test_model[:,:,0])

    #Initialize MAG2305 models.
    model0 = MAG2305.mmModel(types='bulk', size=(args.w, args.w, args.layers), cell=(3,3,3), 
                             Ms=args.Ms, Ax=args.Ax, Ku=args.Ku, Kvec=args.Kvec, 
                             device="cuda:" + str(args.gpu)
                             )
    print('Creating {} layer model \n'.format(args.layers))

    # Initialize demag matrix
    time_start = time.time()
    model0.DemagInit()
    time_finish = time.time()
    print('Time cost: {:f} s for getting demag matrix \n'.format(time_finish-time_start))

    return model0, test_model, path0


def update_spin_state(film0, Hext, args, test_model, path):
    # Do iteration
    rcd_dspin_fft = np.array([[],[]])
    rcd_windabs_fft = np.array([[],[]])
    rcd_windsum_fft = np.array([[],[]])
    fig, ax1, ax2, ax3 = plot_prepare()
    
    nplot = args.nplot

    for iters in range(args.max_iter):
        if iters == 0:
            spin_ini = np.array(film0.Spin[:,:,0].cpu())

        #MAG calculate for spin iteration
        error_fft = film0.SpinLLG_RK4(Hext=Hext, dtime=args.dtime, damping=0.1)
        spin_fft = film0.Spin.cpu().numpy()

        if iters % args.nsave ==0 or error_fft <=args.error_min:
            rcd_dspin_fft = np.append(rcd_dspin_fft, [[iters], [error_fft]], axis=1)

            wind_dens_fft, wind_abs_fft, wind_sum_fft = get_winding(spin_fft[:,:,0],
                                                                    test_model[:,:,0])
            rcd_windabs_fft = np.append(rcd_windabs_fft, 
                                        [[iters], [wind_abs_fft]], axis=1)
            rcd_windsum_fft = np.append(rcd_windsum_fft, 
                                        [[iters], [wind_sum_fft]], axis=1)

        if iters % nplot ==0 or error_fft <=args.error_min:
            plot_spin( spin_fft[:,:,0], ax1, 'fft - iters{}'.format(iters))
            plot_wind( wind_dens_fft, ax2, 'fft-vortices wd[{}]/[{}]'.format(round(wind_abs_fft), round(wind_sum_fft)))
            plot_wind_list( rcd_windabs_fft, ax3 )
            plot_save(path, "spin_iters{}".format(iters))
    
        if error_fft <=args.error_min or iters==args.max_iter-1:
            spin_end_fft = np.array(film0.Spin[:,:,0].cpu())
            plot_close()
            break

    return (rcd_dspin_fft, rcd_windabs_fft, rcd_windsum_fft,
            spin_ini, spin_end_fft)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unet speed test method: LLG_RK4')
    parser.add_argument('--gpu',        type=int,    default=0,         help='GPU ID (default: 0)')
    parser.add_argument('--krn',        type=int,    default=16,        help='unet first layer kernels (default: 16)')

    parser.add_argument('--w',          type=int,    default=32,        help='MAG model size (default: 32)')
    parser.add_argument('--layers',     type=int,    default=2,         help='MAG model layers (default: 2)')
    parser.add_argument('--split',      type=int,    default=1,         help='MAG model split (default: 1)')
    parser.add_argument('--InitCore',   type=int,    default=0,         help='MAG model InitCore (default: 0)')
    parser.add_argument('--modelshape', type=str,    default='square',  help='MAG model shape: square, circle, triangle')
    
    parser.add_argument('--Ms',         type=float,  default=1000,      help='MAG model Ms (default: 1000)')
    parser.add_argument('--Ax',         type=float,  default=0.5e-6,    help='MAG model Ax (default: 0.5e-6)')
    parser.add_argument('--Ku',         type=float,  default=0.0,       help='MAG model Ku (default: 0.0)')
    parser.add_argument('--Kvec',       type=Culist, default=(0,0,1),   help='MAG model Kvec (default: (0,0,1))')
    parser.add_argument('--damping',    type=float,  default=0.1,       help='MAG model damping (default: 0.1)')
    parser.add_argument('--Hext_val',   type=float,  default=0,         help='external field value (default: 0.0)')
    parser.add_argument('--Hext_vec',   type=Culist, default=(1,0,0),   help='external field vector (default:(1,0,0))')

    parser.add_argument('--dtime',      type=float,  default=1.0e-13,   help='real time step (default: 1.0e-13)')
    parser.add_argument('--error_min',  type=float,  default=1.0e-5,    help='min error (default: 1.0e-5)')
    parser.add_argument('--max_iter',   type=int,    default=100000,    help='max iteration number (default: 100000)')
    parser.add_argument('--nsave',      type=int,    default=10,        help='save number (default: 10)')
    parser.add_argument('--nplot',      type=int,    default=2000,      help='plot number (default: 2000)')
    parser.add_argument('--nsamples',   type=int,    default=100,       help='sample number (default: 100)')
    args = parser.parse_args()
    
    device = torch.device("cuda:{}".format(args.gpu))

    # create two film models
    film0, test_model, path0 = initialize_models(args)

    # Random seed list
    seeds_list = list(range(10000, 110000, 100))[:args.nsamples]
    for rand_seed in tqdm(seeds_list):

        # Initialize spin state
        spin0 = MAG2305.get_randspin_2D(size=(args.w, args.w, args.layers),
                                        split=args.split, rand_seed=rand_seed)
        film0.SpinInit(spin0)
        print('initializing spin state \n')
        
        # External field
        Hext = args.Hext_val * np.array(args.Hext_vec)

        # Create directory
        path = path0+"split{}_rand{}/".format(args.split, rand_seed)
        os.makedirs(path, exist_ok=True)
        
        #check any bad cases
        if os.path.exists(os.path.join(path, 'Spin_fft_converge.npy')):
            print('exits and skip: ',path)
            continue # skip this case
        print('do ',path)

        ###########################
        # Spin update calculation #
        ###########################
        print('Begin spin updating:\n')
        (rcd_dspin_fft, rcd_windabs_fft, rcd_windsum_fft,
        spin_ini, spin_end_fft
        ) = update_spin_state(film0, Hext, args, test_model, path)
        
        
        ###################
        # Data processing #
        ###################
        
        np.save(path+'Dspin_fft_max', rcd_dspin_fft)
        np.save(path+'Wind_fft_abs', rcd_windabs_fft)
        np.save(path+'Wind_fft_sum', rcd_windsum_fft)
        
        np.save(path+'Spin_initial',  spin_ini)
        np.save(path+'Spin_fft_converge', spin_end_fft)
