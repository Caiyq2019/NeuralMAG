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

from libs.misc import Culist, MaskTp, spin_prepare
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
    #Initialize MAG2305 models.
    film1 = MAG2305.mmModel(types='bulk', size=(args.w, args.w, args.layers), cell=(3,3,3), 
                            Ms=args.Ms, Ax=args.Ax, Ku=args.Ku, Kvec=args.Kvec, 
                            device="cuda:" + str(args.gpu))
    
    film2 = MAG2305.mmModel(types='bulk', size=(args.w, args.w, args.layers), cell=(3,3,3), 
                            Ms=args.Ms, Ax=args.Ax, Ku=args.Ku, Kvec=args.Kvec, 
                            device="cuda:" + str(args.gpu))

    print('Creating {} layer models \n'.format(args.layers))

    # Initialize demag matrix
    film1.DemagInit()
    print('initializing demag matrix \n')

    # load Unet Model
    load_unet_model(args)

    return film1, film2

def prepare_spin_state(film1, film2, args):
    """
    Prepare the initial spin state.
    """
    # spin_split = np.random.randint(low=2, high=32)
    # rand_seed  = np.random.randint(low=1000, high=100000)
    spin_split = 8
    rand_seed  = 1234
    spin = spin_prepare(spin_split, film1, rand_seed, mask=args.mask)
    film1.SpinInit(spin)
    film2.SpinInit(spin)
    cell_count = (np.linalg.norm(spin, axis=-1) > 0).sum()
    return spin_split, rand_seed, cell_count

def update_spin_fft(model, Hext, args):
    """
    Update the spin state of the model.
    """
    error = 1.0
    itern = 0
    error_rcd = np.array([])
    while itern < args.max_iter and error > args.error_min:
        # FFT_Hd spin update
        error = model.SpinLLG_RK4(Hext=Hext, dtime=args.dtime, damping=0.1)

        error_rcd = np.append(error_rcd, error)
        
        # Print iteration info
        if error <= args.error_min or itern % 1000 == 0:  # Adjust the frequency of printing as needed
            print(f'Iteration: {itern} \n'
                  f'Error_converge FFT: {error:.2e}')
        itern += 1

    return error_rcd, itern

def update_spin_unet(model, Hext, error_mm, args):
    """
    Update the spin state of the model.
    """
    error = 1.0
    itern = 0
    error_fluc = 1.0
    error_rcd = np.array([])
    while itern < args.max_iter and error > args.error_min:
        # Unet_Hd spin update
        error = model.SpinLLG_RK4_unetHd(Hext=Hext, dtime=args.dtime, damping=0.1)

        error_rcd = np.append(error_rcd, error)
        
        # fluctation error break condition
        if itern > 20000 and error_mm <= 1.0e-5:
            error_fluc = np.abs(error_rcd[-2000:].mean() - error_rcd[-500:].mean()) / error_rcd[-2000:].mean()
            if error_fluc < 0.02 and error < 1.0e-4:
                print('Unet error not decreasing! Break.')
                break
        # Print iteration info
        if error <= args.error_min or itern % 1000 == 0:  # Adjust the frequency of printing as needed
            print(f'Iteration: {itern} \n'
                  f'Error_converge UNet: {error:.2e}')
        itern += 1

    return error_rcd, itern

def plot_results():
    """
    Plot and save the results.
    """
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('{} layers film size:{}_split{}_seed{}\n \nloop:{} , Hext={}'.format(
                args.layers, args.w, spin_split, rand_seed, nloop, Hext), fontsize=18 )
        
    # Plot spin-mm RGB figures
    spin = (spin_mm + 1)/2
    axs[0, 0].imshow(spin[:,:,0,:].transpose(1,0,2), alpha=1.0, origin='lower')
    axs[0, 0].set_title('Spin-mm steps: [{:d}]'.format(itern1), fontsize=18)
    axs[0, 0].set_xlabel('x [nm]')
    axs[0, 0].set_ylabel('y [nm]')

    # Plot spin-mm RGB figures
    spin = (spin_un + 1)/2
    axs[0, 1].imshow(spin[:,:,0,:].transpose(1,0,2), alpha=1.0, origin='lower')
    axs[0, 1].set_title('Spin-un steps: [{:d}]'.format(itern2), fontsize=18)
    axs[0, 1].set_xlabel('x [nm]')
    axs[0, 1].set_ylabel('y [nm]')

    # Plot mse heatmap
    mse = np.square(spin_un - spin_mm).sum(axis=-1)
    mse = mse.transpose((1, 0, 2))[:,:,0]
    im = axs[0, 2].imshow(mse, cmap='hot', origin="lower")
    axs[0, 2].set_title('MSE of spin_mm & spin_un, \nMSE_avg: {:4f}'.format(mse.mean()), fontsize=16)
    cbar1 = fig.colorbar(im, ax=axs[0, 2])

    #MH loop figures
    axs[0, 3].plot(x_plot, y1_plot, lw=1.5, label='mm', marker='o', markersize=0, color='blue',  alpha=0.6)
    axs[0, 3].plot(x_plot, y2_plot, lw=1.5, label='un', marker='o', markersize=0, color='red', alpha=0.6)
    axs[0, 3].legend(fontsize=16, loc='upper left')
    axs[0, 3].set_title('M-H data',fontsize=16)
    axs[0, 3].set_xlabel('Hext [Oe]',fontsize=16)
    axs[0, 3].set_ylabel('Mext/Ms',fontsize=16)
    axs[0, 3].set_xlim(min(Hext_range)*1.1, max(Hext_range)*1.1)
    axs[0, 3].set_ylim(-1.1, 1.1)
    axs[0, 3].grid(True, axis='both', lw=0.5, ls='-.')

    #Hd-mm rgb figures
    Hd_mm_norm = Normalize(vmin=Hd_mm[:,:,0,:].min(), vmax=Hd_mm[:,:,0,:].max())(Hd_mm[:,:,0,:])
    axs[1, 0].imshow(Hd_mm_norm.transpose(1,0,2), alpha=1.0, origin='lower')
    axs[1, 0].set_title('Hd_mm steps: [{}]'.format(itern1), fontsize=18)
    axs[1, 0].set_xlabel('x [nm]')
    axs[1, 0].set_ylabel('y [nm]')

    #Hd-un rgb figures
    Hd_un_norm = Normalize(vmin=Hd_un[:,:,0,:].min(), vmax=Hd_un[:,:,0,:].max())(Hd_un[:,:,0,:])
    axs[1, 1].imshow(Hd_un_norm.transpose(1,0,2), alpha=1.0, origin='lower')
    axs[1, 1].set_title('Hd_un steps: [{}]'.format(itern2), fontsize=18)
    axs[1, 1].set_xlabel('x [nm]')
    axs[1, 1].set_ylabel('y [nm]')

    # Plot mse heatmap
    mse_Hd = np.square(Hd_mm - Hd_un).sum(axis=-1)
    mse_Hd = mse_Hd.transpose((1, 0, 2))[:,:,0]
    im = axs[1, 2].imshow(mse_Hd, cmap='hot', origin="lower")
    axs[1, 2].set_title('MSE of Hd_mm & Hd_un \nMSE_avg{:.1e}'.format(mse_Hd.mean()), fontsize=16)
    cbar2 = fig.colorbar(im, ax=axs[1, 2])

    # Plot error
    x1 = np.arange(len(error1_rcd))
    axs[1, 3].plot(x1, error1_rcd, color='blue', alpha=0.6, label='mm')
    x2 = np.arange(len(error2_rcd))
    axs[1, 3].plot(x2, error2_rcd, color='red',  alpha=0.6, label='un')
    axs[1, 3].set_title('Error plot', fontsize=16)
    axs[1, 3].set_xlabel('Iterations', fontsize=16)
    axs[1, 3].set_ylabel('Maximal $\\Delta$m', fontsize=16)
    axs[1, 3].set_yscale('log')
    axs[1, 3].legend(fontsize=16, loc='upper right')
    
    # save img
    plt.tight_layout()
    plt.savefig(filename+'loop_{}.png'.format(nloop))
    plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MH Test')
    parser.add_argument('--gpu',         type=int,    default=0,         help='GPU ID (default: 0)')
    parser.add_argument('--krn',         type=int,    default=16,        help='unet first layer kernels (default: 16)')
    parser.add_argument('--w',           type=int,    default=32,        help='MAG model size (default: 32)')
    parser.add_argument('--layers',      type=int,    default=2,         help='MAG model layers (default: 2)')

    parser.add_argument('--Ms',          type=float,  default=1000,      help='MAG model Ms (default: 1000)')
    parser.add_argument('--Ax',          type=float,  default=0.5e-6,    help='MAG model Ax (default: 0.5e-6)')
    parser.add_argument('--Ku',          type=float,  default=0.0,       help='MAG model Ku (default: 0.0)')
    parser.add_argument('--Kvec',        type=Culist, default=(0,0,1),   help='MAG model Kvec (default: (0,0,1))')
    parser.add_argument('--damping',     type=float,  default=0.1,       help='MAG model damping (default: 0.1)')
    parser.add_argument('--Hext_val',    type=float,  default=0,         help='external field value (default: 0.0)')

    parser.add_argument('--dtime',       type=float,  default=1.0e-13,   help='real time step (default: 1.0e-13)')
    parser.add_argument('--error_min',   type=float,  default=1.0e-5,    help='min error (default: 1.0e-5)')
    parser.add_argument('--max_iter',    type=int,    default=100000,    help='max iteration number (default: 100000)')
    parser.add_argument('--mask',        type=MaskTp, default=False,     help='mask (default: False)')
    args = parser.parse_args() 
    
    device = torch.device("cuda:{}".format(args.gpu))

    # create two film models
    film1, film2 = initialize_models(args)

    # initialize spin state
    spin_split, rand_seed, cell_count = prepare_spin_state(film1, film2, args)
    
    # create folder
    filename='./figs_k{}/shape_{}/size{}_Ms{}_Ax{}_Ku{}_dtime{}_split{}_seed{}_Layers{}/'.format(
                    args.krn, args.mask, args.w, 
                    args.Ms, args.Ax, args.Ku, 
                    args.dtime, spin_split, rand_seed, args.layers
                    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    
    # get MH data
    x_plot,y1_plot,y2_plot = [],[],[]

    # Hext range
    Hext_range = np.linspace(1000,-1000,201)
    Hext_vec = np.array([np.cos(0.01), np.sin(0.01), 0.0])

    spin_mm = np.array([[[[1]]]])
    spin_un = np.array([[[[1]]]])

    # Main loop
    for nloop, Hext_val in enumerate(Hext_range):
        Hext = Hext_val * Hext_vec
        print('>>>>>loop: {} , Hext: {}'.format(nloop, Hext_val))

        spin0_mm = spin_mm
        spin0_un = spin_un

        # Update spin state
        error1_rcd, itern1 = update_spin_fft(film1, Hext, args)
        error2_rcd, itern2 = update_spin_unet(film2, Hext, error1_rcd[-1], args)

        # get spin and Hd
        spin_mm = film1.Spin.detach().cpu().numpy()
        spin_un = film2.Spin.detach().cpu().numpy()
        Hd_mm = film1.Hd.detach().cpu().numpy()
        Hd_un = film2.Hd.detach().cpu().numpy()
        
        #MH loop data
        x_plot.append(Hext_val)
        y1_plot.append( np.dot(spin_mm.sum(axis=(0,1,2)), Hext_vec)/ cell_count )
        y2_plot.append( np.dot(spin_un.sum(axis=(0,1,2)), Hext_vec)/ cell_count )

        # Plot results
        plot_results()

        # Save MH data
        np.save(filename + "Hext_array", x_plot)
        np.save(filename + "Mext_array_mm", y1_plot)
        np.save(filename + "Mext_array_un", y2_plot)

        # Save Mr
        if Hext_val == 0:
            np.save(filename + "Mr_spin_mm", spin_mm)
            np.save(filename + "Mr_spin_un", spin_un)

        # Save Hc
        Mi = spin0_mm[:,:,:,0].sum()
        Mj = spin_mm[:,:,:,0].sum()
        if Mi > 0 and Mj <= 0:
            np.save(filename + "Hc{}_spin_mm".format(nloop-1), spin0_mm)
            np.save(filename + "Hc{}_spin_mm".format(nloop), spin_mm)

        Mi = spin0_un[:,:,:,0].sum()
        Mj = spin_un[:,:,:,0].sum()
        if Mi > 0 and Mj <= 0:
            np.save(filename + "Hc{}_spin_un".format(nloop-1), spin0_un)
            np.save(filename + "Hc{}_spin_un".format(nloop), spin_un)
