# 2023.06.10

import numpy as np


def numpy_roll(arr, shift, axis, pbc):
    """
    Re-defined numpy.roll(), including pbc judgement
    
    Arguments
    ---------
    arr      : Numpy Float(...)
               Array to be rolled
    shift    : Int
               Roll with how many steps
    axis     : Int
               Roll along which axis
    pbc      : Int or Bool
               Periodic condition for rolling; 1: pbc, 0: non-pbc
    
    Returns
    -------
    arr_roll : Numpy Float(...)
               arr after rolling
    """
    arr_roll = np.roll(arr, shift=shift, axis=axis)
    
    if not pbc:
        if axis == 0:
            if shift > 0:
                arr_roll[:shift, ...] = 0.0
            elif shift < 0:
                arr_roll[shift:, ...] = 0.0
            
        elif axis == 1:
            if shift > 0:
                arr_roll[:, :shift, ...] = 0.0
            elif shift < 0:
                arr_roll[:, shift:, ...] = 0.0
    
    return arr_roll


def get_winding(spin, model):
    spin_xp = np.roll(spin, shift=-1, axis=0)
    spin_xm = np.roll(spin, shift= 1 ,axis=0)
    spin_yp = np.roll(spin, shift=-1, axis=1)
    spin_ym = np.roll(spin, shift= 1, axis=1)
    
    # Ignore boundary
    model_bd = (model > 0) & \
               ( (numpy_roll(model, shift=-1, axis=0, pbc=False) <=0 ) |
                 (numpy_roll(model, shift= 1, axis=0, pbc=False) <=0 ) |
                 (numpy_roll(model, shift=-1, axis=1, pbc=False) <=0 ) |
                 (numpy_roll(model, shift= 1, axis=1, pbc=False) <=0 ) )
    
    winding_density = (spin_xp[:,:,0] - spin_xm[:,:,0])/2 \
                      * (spin_yp[:,:,1] - spin_ym[:,:,1])/2 \
                    - (spin_xp[:,:,1] - spin_xm[:,:,1])/2 \
                      * (spin_yp[:,:,0] - spin_ym[:,:,0])/2
    winding_density[ model<=0 ] = 0
    winding_density[ model_bd ] = 0
    
    winding_density = winding_density / np.pi
    
    winding_abs = np.abs(winding_density).sum()
    winding_sum = winding_density.sum()
    
    return winding_density, winding_abs, winding_sum


def get_curl(spin, model):
    spin_xp = np.roll(spin, shift=-1, axis=0)
    spin_xm = np.roll(spin, shift= 1 ,axis=0)
    spin_yp = np.roll(spin, shift=-1, axis=1)
    spin_ym = np.roll(spin, shift= 1, axis=1)
    
    # Ignore boundary
    model_bd = (model > 0) & \
               ( (numpy_roll(model, shift=-1, axis=0, pbc=False) <=0 ) |
                 (numpy_roll(model, shift= 1, axis=0, pbc=False) <=0 ) |
                 (numpy_roll(model, shift=-1, axis=1, pbc=False) <=0 ) |
                 (numpy_roll(model, shift= 1, axis=1, pbc=False) <=0 ) )

    # Get vortex core chirality (clockwise, counterclockwise)
    curl = (model > 0) * (( spin_xp[:,:,1] - spin_xm[:,:,1] )/2
                       -  ( spin_yp[:,:,0] - spin_ym[:,:,0] )/2 )

    curl[model_bd] = 0

    return curl


def analyze_winding(spin, model):
    spin_xp = np.roll(spin, shift=-1, axis=0)
    spin_xm = np.roll(spin, shift= 1 ,axis=0)
    spin_yp = np.roll(spin, shift=-1, axis=1)
    spin_ym = np.roll(spin, shift= 1, axis=1)
    
    # Ignore boundary
    model_bd = (model > 0) & \
               ( (numpy_roll(model, shift=-1, axis=0, pbc=False) <=0 ) |
                 (numpy_roll(model, shift= 1, axis=0, pbc=False) <=0 ) |
                 (numpy_roll(model, shift=-1, axis=1, pbc=False) <=0 ) |
                 (numpy_roll(model, shift= 1, axis=1, pbc=False) <=0 ) )
    
    # Get vortex core number
    winding_density = (spin_xp[:,:,0] - spin_xm[:,:,0])/2 \
                      * (spin_yp[:,:,1] - spin_ym[:,:,1])/2 \
                    - (spin_xp[:,:,1] - spin_xm[:,:,1])/2 \
                      * (spin_yp[:,:,0] - spin_ym[:,:,0])/2
    winding_density[ model<=0 ] = 0
    winding_density[ model_bd ] = 0
    
    winding_density = winding_density / np.pi
    
    winding_abs = np.abs(winding_density).sum()
    winding_sum = winding_density.sum()
    
    vortex_cores = (winding_abs + winding_sum) /2
    antivortex_cores = (winding_abs - winding_sum) /2
    
    # Get vortex core polarity
    positive_abs = np.abs(winding_density[spin[:,:,2]>0]).sum()
    negative_abs = np.abs(winding_density[spin[:,:,2]<0]).sum()
    positive_sum = winding_density[spin[:,:,2]>0].sum()
    negative_sum = winding_density[spin[:,:,2]<0].sum()
    
    positive_vortices = (positive_abs + positive_sum) /2
    positive_antivortices = (positive_abs - positive_sum) /2
    negative_vortices = (negative_abs + negative_sum) /2
    negative_antivortices = (negative_abs - negative_sum) /2
    
    # Get vortex core chirality (clockwise, counterclockwise)
    curl = (model > 0) * (( spin_xp[:,:,1] - spin_xm[:,:,1] )/2
                       -  ( spin_yp[:,:,0] - spin_ym[:,:,0] )/2 )
    
    cw_vortices  = (winding_density * (winding_density>0))[curl<0].sum()
    ccw_vortices = (winding_density * (winding_density>0))[curl>0].sum()
    
    return np.around(vortex_cores), np.around(antivortex_cores), \
           np.around(positive_vortices), np.around(positive_antivortices), \
           np.around(negative_vortices), np.around(negative_antivortices), \
           np.around(cw_vortices),  np.around(ccw_vortices)
