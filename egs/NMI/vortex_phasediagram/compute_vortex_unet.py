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
    unet_size = 2**(int(np.log2(args.w-1))+1)
    test_model = np.ones((unet_size, unet_size, args.layers), dtype=int)
    test_model[: (unet_size-args.w) //2, :, :] = 0
    test_model[(unet_size+args.w) //2 :, :, :] = 0
    test_model[:, : (unet_size-args.w) //2, :] = 0
    test_model[:, (unet_size+args.w) //2 :, :] = 0

    path0 = "./unet/size{}/".format(args.w)+"pre_core{}/".format(args.pre_core)
    os.makedirs(path0, exist_ok=True)
    np.save(path0 + 'model', test_model[:,:,0])

    #Initialize MAG2305 models.
    model0 = MAG2305.mmModel(types='bulk', model=test_model, cell=(3,3,3), 
                             Ms=args.Ms, Ax=args.Ax, Ku=args.Ku, Kvec=args.Kvec, 
                             device="cuda:" + str(args.gpu)
                             )
    print('Creating {} layer model \n'.format(args.layers))

    # Initialize demag matrix
    time_start = time.time()
    model0.DemagInit()
    time_finish = time.time()
    print('Time cost: {:f} s for getting demag matrix \n'.format(time_finish-time_start))

    # load Unet Model
    load_unet_model(args)

    return model0, test_model, path0


def update_spin_state(film0, Hext, args, test_model, path):
    # Do iteration
    rcd_dspin_unet = np.array([[],[]])
    rcd_windabs_unet = np.array([[],[]])
    rcd_windsum_unet = np.array([[],[]])
    fig, ax1, ax2, ax3 = plot_prepare()
    
    wind_abs = 10000
    first_iteration = True
    nplot = args.nplot

    for iters in range(args.max_iter):
        if iters == 0:
            spin_ini = np.array(film0.Spin[:,:,0].cpu())

        #MAG calculate for spin iteration
        if wind_abs > args.pre_core:
            error_unet = film0.SpinLLG_RK4(Hext=Hext, dtime=args.dtime, damping=0.1)
            spin_unet = film0.Spin.cpu().numpy()
            _, wind_abs, _ = get_winding(spin_unet[:,:,0], test_model[:,:,0])
        else:
            #Hd calculate by Unet for spin iteration
            error_unet = film0.SpinLLG_RK4_unetHd(Hext=Hext, dtime=args.dtime, damping=0.1)
            spin_unet = film0.Spin.cpu().numpy()

        if iters % args.nsave ==0 or error_unet <=args.error_min:
            rcd_dspin_unet = np.append(rcd_dspin_unet, [[iters], [error_unet]], axis=1)

            wind_dens_unet, wind_abs_unet, wind_sum_unet = get_winding(spin_unet[:,:,0],
                                                                    test_model[:,:,0])
            rcd_windabs_unet = np.append(rcd_windabs_unet, 
                                        [[iters], [wind_abs_unet]], axis=1)
            rcd_windsum_unet = np.append(rcd_windsum_unet, 
                                        [[iters], [wind_sum_unet]], axis=1)

        if iters % nplot ==0 or error_unet <=args.error_min:
            plot_spin( spin_unet[:,:,0], ax1, 'unet - iters{}'.format(iters))
            plot_wind( wind_dens_unet, ax2, 'unet-vortices wd[{}]/[{}]'.format(round(wind_abs_unet), round(wind_sum_unet)))
            plot_wind_list( rcd_windabs_unet, ax3 )
            plot_save(path, "spin_iters{}".format(iters))
    
        if error_unet <=args.error_min or iters==args.max_iter-1:
            spin_end_unet = np.array(film0.Spin[:,:,0].cpu())
            plot_close()
            break

    return (rcd_dspin_unet, rcd_windabs_unet, rcd_windsum_unet,
            spin_ini, spin_end_unet)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unet speed test method: LLG_RK4')
    parser.add_argument('--gpu',        type=int,    default=0,         help='GPU ID (default: 0)')
    parser.add_argument('--krn',        type=int,    default=16,        help='unet first layer kernels (default: 16)')

    parser.add_argument('--w',          type=int,    default=32,        help='MAG model size (default: 32)')
    parser.add_argument('--layers',     type=int,    default=2,         help='MAG model layers (default: 2)')
    parser.add_argument('--split',      type=int,    default=1,         help='MAG model split (default: 1)')
    parser.add_argument('--pre_core',   type=int,    default=0,         help='MAG model pre_core (default: 0)')
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
        spin1 = np.zeros(test_model.shape + (3,))
        spin1[test_model==1] = spin0.reshape(args.w * args.w * args.layers, 3)
        film0.SpinInit(spin1)
        print('initializing spin state \n')
        
        # External field
        Hext = args.Hext_val * np.array(args.Hext_vec)

        # Create directory
        path = path0+"split{}_rand{}/".format(args.split, rand_seed)
        os.makedirs(path, exist_ok=True)
        
        #check any bad cases
        if os.path.exists(os.path.join(path, 'Spin_unet_converge.npy')):
            print('exits and skip: ',path)
            continue # skip this case
        print('do ',path)

        ###########################
        # Spin update calculation #
        ###########################
        print('Begin spin updating:\n')
        (rcd_dspin_unet, rcd_windabs_unet, rcd_windsum_unet,
        spin_ini, spin_end_unet
        ) = update_spin_state(film0, Hext, args, test_model, path)
        
        
        ###################
        # Data processing #
        ###################
        
        np.save(path+'Dspin_unet_max', rcd_dspin_unet)
        np.save(path+'Wind_unet_abs', rcd_windabs_unet)
        np.save(path+'Wind_unet_sum', rcd_windsum_unet)
        
        np.save(path+'Spin_initial',  spin_ini)
        np.save(path+'Spin_unet_converge', spin_end_unet)
