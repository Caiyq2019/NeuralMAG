# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 10:00:00 2023
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import argparse
import torch

from libs.misc import Culist, MaskTp, spin_prepare, create_trt_model
import libs.MAG2305 as MAG2305
from libs.Unet import UNet



def load_unet_model(args):
    # load Unet Model
    inch = args.layers*3
    model = UNet(kc=args.krn, inc=inch, ouc=inch).eval().to(device)
    ckpt = '../ckpt/k{}/model.pt'.format(args.krn)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    # Creat trt model
    if args.trt=='True':
        model = create_trt_model(model, inch, args.w, torch.float16, device)
        print('Unet model loaded with TensorRT')
    else:
        print('Unet model loaded')

    MAG2305.load_model(model)
    print('Unet model loaded from {}'.format(ckpt))


def initialize_models(args):
    #Initialize MAG2305 models.
    film2 = MAG2305.mmModel(types='bulk', size=(args.w, args.w, args.layers), cell=(3,3,3), 
                            Ms=args.Ms, Ax=args.Ax, Ku=args.Ku, Kvec=args.Kvec, 
                            device="cuda:" + str(args.gpu))
    print('Creating {} layer models \n'.format(args.layers))


    # load Unet Model
    load_unet_model(args)

    return film2


def prepare_spin_state(film2, args):
    """
    Prepare the initial spin state.
    """
    #spin_split = np.random.randint(low=2, high=32)
    #rand_seed  = np.random.randint(low=1000, high=100000)
    spin_split = 8
    rand_seed  = 1234
    spin = spin_prepare(spin_split, film2, rand_seed, mask=args.mask)
    film2.SpinInit(spin)
    cell_count = (np.linalg.norm(spin, axis=-1) > 0).sum()
    return spin_split, rand_seed, cell_count

def update_spin_state(film2, Hext, args):
    """
    Update the spin state of the model.
    """
    error_un = 1.0
    itern = 0
    error_fluc = 1.0
    error2_rcd = np.array([])
    while itern < args.max_iter and error_un > args.error_min:
        # Unet_Hd spin update
        error_un = film2.SpinLLG_RK4_unetHd(Hext=Hext, dtime=args.dtime, damping=0.1)

        error2_rcd = np.append(error2_rcd, error_un)
        
        # fluctation error break condition
        if itern > 20000:
            error_fluc = np.abs(error2_rcd[-2000:].mean() - error2_rcd[-500:].mean()) / error2_rcd[-2000:].mean()
            if error_fluc < 0.02 and error_un <= args.error_min*10:
                print('Unet error not decreasing! Break.')
                break
        # Print iteration info
        if itern % 100 == 0:  # Adjust the frequency of printing as needed
            print(f'Iteration: {itern} \n'
                  f'Error_converge UNet: {error_un:.2e}')
        itern += 1

    return error2_rcd, itern


def plot_results():
    """
    Plot and save the results.
    """
    #plot figures
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('{} layers film size:{}_split{}_seed{}\n \nloop:{} , Hext={}'.format(
                args.layers, args.w, spin_split, rand_seed, nloop, Hext), fontsize=18 )
        
    # Plot spin-un RGB figures
    spin = (spin_un + 1)/2
    axs[0, 0].imshow(spin[:,:,0,:].transpose(1,0,2), alpha=1.0, origin='lower')
    axs[0, 0].set_title('Spin-un steps: [{:d}]'.format(itern), fontsize=18)
    axs[0, 0].set_xlabel('x [nm]')
    axs[0, 0].set_ylabel('y [nm]')

    #MH loop figures
    axs[0, 1].plot(x_plot, y2_plot, lw=1.5, label='un', marker='o', markersize=0, color='red', alpha=0.6)
    axs[0, 1].legend(fontsize=16, loc='upper left')
    axs[0, 1].set_title('M-H data',fontsize=16)
    axs[0, 1].set_xlabel('Hext [Oe]',fontsize=16)
    axs[0, 1].set_ylabel('Mext/Ms',fontsize=16)
    axs[0, 1].set_xlim(min(Hext_range)*1.1, max(Hext_range)*1.1)
    axs[0, 1].set_ylim(-1.1, 1.1)
    axs[0, 1].grid(True, axis='both', lw=0.5, ls='-.')

    #Hd-un rgb figures
    Hd_un_norm = Normalize(vmin=Hd_un[:,:,0,:].min(), vmax=Hd_un[:,:,0,:].max())(Hd_un[:,:,0,:])
    axs[1, 0].imshow(Hd_un_norm.transpose(1,0,2), alpha=1.0, origin='lower')
    axs[1, 0].set_title('Hd_un steps: [{}]'.format(itern), fontsize=18)
    axs[1, 0].set_xlabel('x [nm]')
    axs[1, 0].set_ylabel('y [nm]')

    # Plot error
    x = np.arange(len(error2_rcd))
    axs[1, 1].plot(x, error2_rcd, color='red',  alpha=0.6, label='un')
    axs[1, 1].set_title('Error plot', fontsize=16)
    axs[1, 1].set_xlabel('Iterations', fontsize=16)
    axs[1, 1].set_ylabel('Maximal $\\Delta$m', fontsize=16)
    axs[1, 1].set_yscale('log')
    axs[1, 1].legend(fontsize=16, loc='upper right')
    
    # save img
    plt.tight_layout()
    plt.savefig(filename+'loop_{}.png'.format(nloop))
    plt.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MH Test')
    parser.add_argument('--gpu',         type=int,    default=0,         help='GPU ID (default: 0)')
    parser.add_argument('--krn',         type=int,    default=16,        help='unet first layer kernels (default: 16)')
    parser.add_argument('--trt',         type=str,    default='False',   help='unet with tensorRT (default: False)')

    parser.add_argument('--w',           type=int,    default=32,        help='MAG model size (default: 32)')
    parser.add_argument('--layers',      type=int,    default=2,         help='MAG model layers (default: 1)')

    parser.add_argument('--Ms',          type=float,  default=1000,      help='MAG model Ms (default: 1000)')
    parser.add_argument('--Ax',          type=float,  default=0.5e-6,    help='MAG model Ax (default: 0.5e-6)')
    parser.add_argument('--Ku',          type=float,  default=0.0,       help='MAG model Ku (default: 0.0)')
    parser.add_argument('--Kvec',        type=Culist, default=(0,0,1),   help='MAG model Kvec (default: (0,0,1))')
    parser.add_argument('--damping',     type=float,  default=0.1,       help='MAG model damping (default: 0.1)')
    parser.add_argument('--Hext_val',    type=float,  default=0,         help='external field value (default: 0.0)')

    parser.add_argument('--dtime',       type=float,  default=1.0e-13,   help='real time step (default: 1.0e-13)')
    parser.add_argument('--error_min',   type=float,  default=1.0e-5,    help='min error (default: 1.0e-6)')
    parser.add_argument('--max_iter',    type=int,    default=100000,     help='max iteration number (default: 50000)')
    parser.add_argument('--mask',        type=MaskTp, default=False,     help='mask (default: False)')
    args = parser.parse_args() 
    
    device = torch.device("cuda:{}".format(args.gpu))

    # create two film models
    film2 = initialize_models(args)

    # initialize spin state
    spin_split, rand_seed, cell_count = prepare_spin_state(film2, args)
    
    # create folder
    filename='./figs_k{}/shape_{}/size{}_Ms{}_Ax{}_Ku{}_dtime{}_split{}_seed{}_Layers{}/'.format(
                    args.krn, args.mask, args.w, 
                    args.Ms, args.Ax, args.Ku, 
                    args.dtime, spin_split, rand_seed, args.layers
                    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    
    # get MH data
    x_plot, y2_plot = [],[]

    # Hext range
    Hext_range = np.linspace(1000,-1000,201)
    Hext_vec = np.array([np.cos(0.01), np.sin(0.01), 0.0])
    
    # Main loop
    for nloop, Hext_val in enumerate(Hext_range):
        Hext = Hext_val * Hext_vec
        print('>>>>>loop: {} , Hext: {}'.format(nloop, Hext_val))

        # Update spin state
        error2_rcd, itern = update_spin_state(film2, Hext, args)

        # get spin and Hd data
        spin_un = film2.Spin.detach().cpu().numpy()
        Hd_un = film2.Hd.detach().cpu().numpy()
        
        #MH loop data
        x_plot.append(Hext_val)
        y2_plot.append( np.dot(spin_un.sum(axis=(0,1,2)), Hext_vec)/ cell_count )

        # Plot results
        plot_results()

        # Save MH data
        np.save(filename + "Hext_array", x_plot)
        np.save(filename + "Mext_array_un", y2_plot)
