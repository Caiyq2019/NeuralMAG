# -*- coding: utf-8 -*-
import random
import time, os
import numpy as np
from tqdm import tqdm
import argparse
import torch

from libs.misc import Culist, initial_spin_prepare, create_random_mask, error_plot
import libs.MAG2305 as MAG2305

def prepare_model(args):
    film = MAG2305.mmModel(types='bulk', size=(args.w, args.w, args.layers), 
                           cell=(3,3,3), Ms=args.Ms, Ax=args.Ax, Ku=args.Ku, 
                           Kvec=args.Kvec, device="cuda:" + str(args.gpu))
    print('Creating {} layer models \n'.format(args.layers))

    # Initialize demag matrix
    time_start = time.time()
    film.DemagInit()
    time_finish = time.time()
    print('Time cost: {:f} s for initializing demag matrix \n'.format(time_finish-time_start))
    return film

def generate_data(args, film):
    Hext_val = np.random.randn(3) * args.Hext_val
    Hext = Hext_val * args.Hext_vec

    for seed in tqdm(range(0, args.nseeds)):
        path_format = './Dataset/data_Hd{}_Hext{}_mask/seed{}' if args.mask=='True' else './Dataset/data_Hd{}_Hext{}/seed{}'
        save_path = path_format.format(args.w, int(args.Hext_val), seed)
        os.makedirs(save_path, exist_ok=True)

        spin = initial_spin_prepare(args.w, args.layers, seed)
        if args.mask == 'True':
            mask = create_random_mask((args.w, args.w), np.random.randint(2, args.w), random.choice([True, False]))
            spin = film.SpinInit(spin * mask)
        else:
            spin = film.SpinInit(spin)

        Spins_list, Hds_list, error_list = simulate_spins(film, spin, Hext, args)
        save_simulation_data(Spins_list, Hds_list, args, save_path, error_list, Hext)

def simulate_spins(film, spin, Hext, args):
    Spins_list, Hds_list, error_list = [], [], []
    itern = 0
    error_ini = 1

    Spininit = np.reshape(spin[:,:,:,:], (args.w, args.w, args.layers*3))
    Spins_list.append(Spininit)

    while error_ini > args.error_min and itern < args.max_iter:
        error = film.SpinLLG_RK4(Hext=Hext, dtime=args.dtime, damping=args.damping)
        error_ini = error
        error_list.append(error)
        itern += 1

        Spins_list.append(np.reshape(film.Spin.cpu(), (args.w, args.w, args.layers*3)))
        Hds_list.append(np.reshape(film.Hd.cpu(), (args.w, args.w, args.layers*3)))

    Spins_list.pop(-1)
    return Spins_list, Hds_list, error_list

def save_simulation_data(Spins_list, Hds_list, args, save_path, error_list, Hext):
    random_indices = sorted(random.sample(range(1, len(Hds_list)-1), args.sav_samples))
    Spins_random_list = [Spins_list[i] for i in random_indices]
    Hds_random_list = [Hds_list[i] for i in random_indices]

    Spins_random_list.append(Spins_list[-1])
    Hds_random_list.append(Hds_list[-1])

    np.save(os.path.join(save_path, 'Spins.npy'), np.stack(Spins_random_list, axis=0))
    np.save(os.path.join(save_path, 'Hds.npy'), np.stack(Hds_random_list, axis=0))

    error_plot(error_list, os.path.join(save_path, 'iterns{:.1e}_errors_{:.1e}'.format(len(error_list), error_list[-1])),
               str('[{:.2f}, {:.2f}, {:.2f}]'.format(Hext[0], Hext[1], Hext[2])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--gpu',        type=int,    default=0,         help='GPU ID (default: 0)')
    parser.add_argument('--w',          type=int,    default=32,        help='MAG model size (default: 32)')
    parser.add_argument('--layers',     type=int,    default=2,         help='MAG model layers (default: 1)')
    
    parser.add_argument('--Ms',         type=float,  default=1000,      help='MAG model Ms (default: 1000)')
    parser.add_argument('--Ax',         type=float,  default=0.5e-6,    help='MAG model Ax (default: 0.5e-6)')
    parser.add_argument('--Ku',         type=float,  default=0.0,       help='MAG model Ku (default: 0.0)')
    parser.add_argument('--Kvec',       type=Culist, default=(0,0,1),   help='MAG model Kvec (default: (0,0,1))')
    parser.add_argument('--damping',    type=float,  default=0.1,       help='MAG model damping (default: 0.1)')
    parser.add_argument('--Hext_val',   type=float,  default=0,         help='external field value (default: 0.0)')
    parser.add_argument('--Hext_vec',   type=Culist, default=(1,1,0),   help='external field vector (default:(1,1,0))')

    parser.add_argument('--dtime',      type=float,  default=1.0e-13,   help='real time step (default: 1.0e-13)')
    parser.add_argument('--error_min',  type=float,  default=1.0e-6,    help='min error (default: 1.0e-6)')
    parser.add_argument('--max_iter',   type=int,    default=50000,     help='max iteration number (default: 50000)')
    parser.add_argument('--sav_samples',type=int,    default=500,       help='save samples (default: 500)')
    parser.add_argument('--mask',       type=str,    default='False',   help='mask (default: False)')
    parser.add_argument('--nseeds',     type=int,    default=100,       help='number of seeds (default: 100)')
    args = parser.parse_args() 

    device = torch.device("cuda:{}".format(args.gpu))
    
    #Prepare MAG model: film
    film = prepare_model(args)

    #Generate spin and Hd pairs data
    generate_data(args, film)
