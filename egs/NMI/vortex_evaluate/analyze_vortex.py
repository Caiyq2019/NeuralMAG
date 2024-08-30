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
import matplotlib.pyplot as plt

from util.plot import *
from util.vortex_utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unet speed test method: LLG_RK4')    
    parser.add_argument('--w',           type=int,    default=32,        help='MAG model size (default: 32)')
    parser.add_argument('--krn',         type=int,    default=16,        help='MAG model kernel (default: 3)')
    parser.add_argument('--layers',      type=int,    default=2,         help='MAG model layers (default: 1)')
    parser.add_argument('--split',       type=int,    default=32,        help='MAG model split (default: 2)')
    parser.add_argument('--InitCore',    type=int,    default=0,         help='MAG model InitCore (default: 0)')
    parser.add_argument('--errorfilter', type=float,  default=1e-5,      help='MAG model errorlimit (default: 0.1)')
    args = parser.parse_args()

    #dir 
    path0 = "./k{}/size{}/".format(args.krn, args.w)+"InitCore{}/".format(args.InitCore)


    converge_sample_count = 0
    sample_count = 0
    single_vortex_2305 = 0
    single_vortex_unet = 0
    single_antivtx_2305 = 0
    single_antivtx_unet = 0
    multi_2305 = 0
    multi_unet = 0
    none_2305 = 0
    none_unet = 0
    core_match   = 0
    exact_match  = 0
    steps_mm = []
    steps_un = []
    steps_un_start=[]
    for subpath in os.listdir(path0):
        if r"split" + str(args.split) in subpath:
            
            files_dir = glob.glob(os.path.join(path0 + subpath, '*'))
            matching_files1 = [f for f in files_dir if 'Spin_2305_converge' in f]
            matching_files2 = [f for f in files_dir if 'Spin_unet_converge' in f]
            if len(matching_files1) > 0 or len(matching_files2) > 0:
                sample_count += 1
            else:
                print('\n',subpath, 'not complete')
                continue

            errormm = np.load(path0 + subpath + '/Dspin_2305_max.npy')
            errorun = np.load(path0 + subpath + '/Dspin_unet_max.npy')
            indexmm = np.where(errormm[1] < args.errorfilter)
            indexun = np.where(errorun[1] < args.errorfilter)

            if len(indexmm[0]) > 0 and len(indexun[0]) > 0:
                steps_mm.append(errormm[0][indexmm[0][0]])
                steps_un.append(errorun[0][indexun[0][0]])
                steps_un_start.append(errorun[0][0])

                spin_2305 = np.load(path0 + subpath + '/Spin_2305_converge.npy')
                spin_unet = np.load(path0 + subpath + '/Spin_unet_converge.npy')

                model = np.load(path0 + 'model.npy')

                converge_sample_count += 1
        
                vortex_2305, antivtx_2305, \
                pp_vortex_2305, pp_antivtx_2305, \
                np_vortex_2305, np_antivtx_2305, \
                cw_vortex_2305, ccw_vortex_2305   = analyze_winding(spin_2305, model)
                
                vortex_unet, antivtx_unet, \
                pp_vortex_unet, pp_antivtx_unet, \
                np_vortex_unet, np_antivtx_unet, \
                cw_vortex_unet, ccw_vortex_unet   = analyze_winding(spin_unet, model)
            
            
                if vortex_2305 == 0 and antivtx_2305 == 0:
                    none_2305 += 1
                elif vortex_2305 == 1 and antivtx_2305 == 0:
                    single_vortex_2305 += 1
                elif vortex_2305 == 0 and antivtx_2305 == 1:
                    single_antivtx_2305 += 1
                else:
                    multi_2305 += 1
                
                if vortex_unet == 0 and antivtx_unet == 0:
                    none_unet += 1
                elif vortex_unet == 1 and antivtx_unet == 0:
                    single_vortex_unet += 1
                elif vortex_unet == 0 and antivtx_unet == 1:
                    single_antivtx_unet += 1
                else:
                    multi_unet += 1
        
        
                if vortex_2305 == vortex_unet and antivtx_2305 == antivtx_unet:
                    core_match += 1
                    
                    if  pp_vortex_2305  == pp_vortex_unet   and \
                        pp_antivtx_2305 == pp_antivtx_unet  and \
                        np_vortex_2305  == np_vortex_unet   and \
                        np_antivtx_2305 == np_antivtx_unet  and \
                        cw_vortex_2305  == cw_vortex_unet   and \
                        ccw_vortex_2305 == ccw_vortex_unet :
                        exact_match += 1

                    
                    else:
                        print("\n"+subpath)
                        print("-------            2305  Unet")
                        print("Vortex    cores   : {}   {}".format(vortex_2305, vortex_unet))
                        print("Antivtx   cores   : {}   {}".format(antivtx_2305, antivtx_unet))
                        print("-------")
                        print("Positive  vortex  : {}   {}".format(pp_vortex_2305,  pp_vortex_unet))
                        print("Positive  antivtx : {}   {}".format(pp_antivtx_2305, pp_antivtx_unet))
                        print("Negative  vortex  : {}   {}".format(np_vortex_2305,  np_vortex_unet))
                        print("Negative  antivtx : {}   {}".format(np_antivtx_2305, np_antivtx_unet))
                        print("-------")
                        print("Clockwise vortex  : {}   {}".format(cw_vortex_2305,   cw_vortex_unet))
                        print("CounterCW vortex  : {}   {}".format(ccw_vortex_2305,  ccw_vortex_unet))
                else:
                    print("\n"+subpath)
                    print("-------            2305  Unet")
                    print("Vortex    cores   : {}   {}".format(vortex_2305, vortex_unet))
                    print("Antivtx   cores   : {}   {}".format(antivtx_2305, antivtx_unet))
                    print("-------")
                    print("Positive  vortex  : {}   {}".format(pp_vortex_2305,  pp_vortex_unet))
                    print("Positive  antivtx : {}   {}".format(pp_antivtx_2305, pp_antivtx_unet))
                    print("Negative  vortex  : {}   {}".format(np_vortex_2305,  np_vortex_unet))
                    print("Negative  antivtx : {}   {}".format(np_antivtx_2305, np_antivtx_unet))
                    print("-------")
                    print("Clockwise vortex  : {}   {}".format(cw_vortex_2305,   cw_vortex_unet))
                    print("CounterCW vortex  : {}   {}".format(ccw_vortex_2305,  ccw_vortex_unet))

            else:
                print('\n',subpath, 'not converge')


    content = '''
    Summary-- Statistic_samples [{}/{}] Unet k{} mm {} size {} split {} 
              Unet_core_precision: [ core_number {:.4f} / core_property {:.4f} ]\n
              errorfilter: {:.1e}, avg  mm_steps: {}, un_steps: [{} / {}] = {:.1f} %,  
    '''.format(converge_sample_count, sample_count, args.krn, args., args.w, args.split, 
               core_match/(converge_sample_count+1e-8), 
               exact_match/(converge_sample_count+1e-8),
               args.errorfilter, int(np.array(steps_mm).mean()), int(np.array(steps_un_start).mean()), int(np.array(steps_un).mean()),
                np.array(steps_un_start).mean()*100/np.array(steps_un).mean()
               )

    print(content)

 

    path_save="./analyze_vortex/k{}/size{}/".format(args.krn, args.w)
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    with open(path_save+"summary_size{}.txt".format(args.w), "a") as file:
        file.write(content)



    parameters = {'axes.labelsize' : 14,
                'axes.titlesize' : 14,
                'xtick.labelsize': 11,
                'ytick.labelsize': 11,
                'legend.fontsize': 11,
                'figure.dpi'     : 150
                }
    plt.rcParams.update(parameters)

    fig = plt.figure(figsize=(5,4))
    gs  = fig.add_gridspec(10,10)
    ax  = fig.add_subplot(gs[1:9, 1:10])
    counts1 = [none_2305, single_vortex_2305, single_antivtx_2305, multi_2305]
    counts2 = [none_unet, single_vortex_unet, single_antivtx_unet, multi_unet]

    width = 0.4
    xticks = ["single domain", "single vortex", "single anti", "multiple"]
    ax.bar( x=np.arange(len(counts1)), height=counts1, 
            width=width, color='green',  alpha=0.6, label='2305') 
    ax.bar( x=np.arange(len(counts2))+width, height=counts2, 
            width=width, color='orange', alpha=0.6, label='unet') 
    ax.set_xticks( np.arange(len(counts1)) + width/2, xticks, rotation=15 )
    ax.set_ylabel("state counts")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.set_title("Model mm{} size {} split {} \n corenumber{:.4f}_coreproperty{:.4f}".format(args., args.w, args.split, core_match/(converge_sample_count+1e-8), exact_match/(converge_sample_count+1e-8)), fontsize=10)
    plt.savefig(path_save+"mm{}_size{}_split{}_corenumber{:.4f}_coreproperty{:.4f}_error{:.1e}.png".format(
                    args.InitCore, args.w, args.split, 
                    core_match/(converge_sample_count+1e-8), exact_match/(converge_sample_count+1e-8),
                    args.errorfilter)
               )
    plt.close()

