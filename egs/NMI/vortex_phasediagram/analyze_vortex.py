# -*- coding: utf-8 -*-
"""
Created on Tue May 30 21:00:00 2023

#########################################
#                                       #
#  Get vortex counting from spin data   #
#                                       #
#########################################

"""
import os
import glob
import argparse
import numpy as np

from util.plot import *
from util.vortex_utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unet speed test method: LLG_RK4')    
    parser.add_argument('--w',           type=int,    default=32,        help='MAG model size (default: 32)')
    parser.add_argument('--pre_core',    type=int,    default=0,         help='MAG model pre_core (default: 0)')
    parser.add_argument('--split',       type=int,    default=1,         help='MAG model split (default: 1)')
    parser.add_argument('--method',      type=str,    default="fft",     help='calculation method (default: fft)')
    parser.add_argument('--errorfilter', type=float,  default=1.0e-5,    help='error_convergence (default: 1.0e-5)')
    args = parser.parse_args()

    #dir 
    path0 = "./{}/size{}/".format(args.method, args.w)+"pre_core{}/".format(args.pre_core)


    converge_sample_count = 0
    sample_count = 0
    single_domain = 0
    single_vortex = 0
    multi_vortex = 0
    for subpath in os.listdir(path0):
        if r"split" + str(args.split) in subpath:
            
            files_dir = glob.glob(os.path.join(path0 + subpath, '*'))
            matching_files = [f for f in files_dir if 'Spin_{}_converge.npy'.format(args.method) in f]
            if len(matching_files) > 0 :
                sample_count += 1
            else:
                print('\n',subpath, 'not complete')
                continue

            error_list = np.load(path0 + subpath + '/Dspin_{}_max.npy'.format(args.method))
            if error_list[1][-1] <= args.errorfilter :
                spin = np.load(path0 + subpath + '/Spin_{}_converge.npy'.format(args.method))

                model = np.load(path0 + 'model.npy')

                converge_sample_count += 1
        
                vortex, antivtx, \
                pp_vortex, pp_antivtx, \
                np_vortex, np_antivtx, \
                cw_vortex, ccw_vortex   = analyze_winding(spin, model)
                
            
                if vortex == 0 and antivtx == 0:
                    single_domain += 1
                elif vortex == 1 and antivtx == 0:
                    single_vortex += 1
                elif vortex == 0 and antivtx == 1:
                    single_vortex += 1
                else:
                    multi_vortex += 1

            else:
                print('\n',subpath, 'not converge')


    content = '''
    Summary-- Statistic_samples [{}/{}] method {} mmPre_core {} size {} split {},  
              single domain: {:.3f}, single vortex: {:.3f}, multi-vortices: {:.3f}
    '''.format(converge_sample_count, sample_count, args.method, args.pre_core, args.w, args.split, 
               single_domain/(converge_sample_count+1e-8), 
               single_vortex/(converge_sample_count+1e-8), 
               multi_vortex/(converge_sample_count+1e-8)
               )

    print(content)

 

    path_save="./analyze_vortex/{}/size{}/".format(args.method, args.w)
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    with open(path_save+"summary.txt", "w") as file:
        file.write(content)
