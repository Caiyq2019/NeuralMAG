import numpy as np
import matplotlib.pyplot as plt
import cv2
import re, sys
import torch
import argparse
import torch_tensorrt

def initial_spin_prepare(width, layers, rand_seed):
    """
    Prepare the initial spin configuration.
    
    Args:
    - width (int): Width of the spin matrix.
    - layers (int): Number layers of the films.
    - seed (int): Seed for random number generation.
    
    Returns:
    - numpy.ndarray: Initialized spin matrix.
    """
    np.random.seed(rand_seed)
    matrix = np.random.randn(width, width, 2)
    matrix /= np.linalg.norm(matrix, axis=2, keepdims=True)
    matrix = np.expand_dims(matrix, axis=2)
    z_axis = np.zeros((width, width, 1, 1))
    spin = np.concatenate([matrix, z_axis], axis=3)
    spin = np.tile(spin, (1, 1, layers, 1))
    
    return spin


def initial_spin_split(spin_split, film, rand_seed):
    # Prapare spin orientation selections
    spin_cases = []
    spin_cases.append([ 1.0, 0.0, 0.0])   # +x
    spin_cases.append([-1.0, 0.0, 0.0])   # -x
    spin_cases.append([ 0.0, 1.0, 0.0])   # +y
    spin_cases.append([ 0.0,-1.0, 0.0])   # -y
    spin_cases.append([ 1.0, 1.0, 0.0])   # +x+y
    spin_cases.append([ 1.0,-1.0, 0.0])   # +x-y
    spin_cases.append([-1.0, 1.0, 0.0])   # -x+y
    spin_cases.append([-1.0,-1.0, 0.0])   # -x-y

    # Initialize spin state
    spin = np.empty( tuple(film.size) + (3,) )
    np.random.seed(rand_seed)

    xsplit = film.size[0] / spin_split  # x length of each split area
    ysplit = film.size[1] / spin_split  # y length of each split area

    for nx in range(spin_split):
        for ny in range(spin_split):
            
            xlow_bound = int(nx * xsplit)
            xhigh_bound = int((nx+1) * xsplit) if nx + 1 < spin_split \
                                            else film.size[0]
            
            ylow_bound = int(ny * ysplit)
            yhigh_bound = int((ny+1) * ysplit) if ny + 1 < spin_split \
                                            else film.size[1]
            
            spin_selected = spin_cases[np.random.randint(len(spin_cases))]
            spin[xlow_bound:xhigh_bound, ylow_bound:yhigh_bound, :] = spin_selected
    
    return spin


def spin_prepare(spin_split, film, rand_seed, fixshape=True, num_points=30,  mask=False, inverse=False):
    spin = initial_spin_split(spin_split, film, rand_seed)

    if mask==True:
        shape=(film.size[0], film.size[1])
        if num_points == None:
            num_points=np.random.randint(low=2, high=shape[0])
        spin_mask=create_random_mask(shape, num_points, fixshape=fixshape, inverse=inverse)
        
    elif type(mask) == str:
        shape=(film.size[0],film.size[1])
        spin_mask=create_regular_mask(shape, mask)

    else:
        spin_mask=1
    
    return spin * spin_mask
        

def create_random_mask(shape, num_points, fixshape, seed=42, inverse=False):
    """
    Create a random mask with a given shape.
    
    Args:
    - shape (tuple): Shape of the mask.
    - num_points (int): Number of points to generate the convex hull.
    - inverse (bool): Whether to invert the binary mask.
    
    Returns:
    - numpy.ndarray: Generated mask.
    """
    size=shape[0]
    if fixshape:
        np.random.seed(seed)
        points = np.random.randint(0, 32, size=(num_points, 2))
        hull = cv2.convexHull(points)
        scale_x = shape[0] / hull[:,:,0].max()
        scale_y = shape[1] / hull[:,:,1].max()
        hull = (hull * np.array([scale_x, scale_y])).astype(int)
    else:
        points = np.random.randint(0, size, size=(num_points, 2))
        hull = cv2.convexHull(points)

    image = np.zeros((size, size), dtype=np.uint8)
    cv2.drawContours(image, [hull], 0, 255, -1)

    _, binary_image = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY)
    if inverse:
        binary_image = 1 - binary_image
    
    new_shape = shape + (1, 3)
    new_arr = np.zeros(new_shape, dtype=binary_image.dtype)
    new_arr[..., :] = binary_image[..., np.newaxis, np.newaxis]

    return new_arr


def create_regular_mask(shape, mask_type='square'):
    size = shape[0]
    mask = np.ones([size, size])
    if re.match('square', mask_type, re.IGNORECASE) is not None:
        pass
    elif re.match('triangle', mask_type, re.IGNORECASE) is not None:
        mx, my = np.meshgrid(np.arange(size), np.arange(size))
        mask[ mx >  0.5814 * my + 0.5 * size -0.5 ] = 0
        mask[ mx < -0.5814 * my + 0.5 * size -0.5 ] = 0
        mask[ my > 0.86 * size ] = 0
    elif re.match('hole', mask_type, re.IGNORECASE) is not None:
        cc = (size/2 -0.5, size/2 -0.5)
        mx, my = np.meshgrid(np.arange(size), np.arange(size))
        rr = (size//4)**2
        mask[ (mx - cc[0])**2 + (my - cc[1])**2 < rr ] = 0
        mask[ (mx - cc[0])**2 + (my - cc[1])**2 < rr ] = 0
    else:
        print('Unknown mask type "{}"! Please use one of the folowing:'.format(mask_type))
        print(' Square  |  Triangle | Hole\n')
        sys.exit(0)

    # 定义目标数组形状
    new_shape = shape + (1, 3)
    # 将原始数组赋值给目标数组的三个通道
    new_arr = np.zeros(new_shape, dtype=mask.dtype)
    new_arr[..., :] = mask[..., np.newaxis, np.newaxis]

    return new_arr


def create_model(size=32, shape='square'):
    size = int(size)
    model = np.ones([size, size, 1])

    if re.match('square', shape, re.IGNORECASE) is not None:
        print("Create [square] model\n")

    elif re.match('circle', shape, re.IGNORECASE) is not None:
        cc = (size/2 -0.5, size/2 -0.5)
        mx, my = np.meshgrid(np.arange(size), np.arange(size))
        rr = (size//2)**2
        model[ (mx - cc[0])**2 + (my - cc[1])**2 > rr ] = 0
        model[ (mx - cc[0])**2 + (my - cc[1])**2 > rr ] = 0
        print("Create [circle] model\n")

    elif re.match('triangle', shape, re.IGNORECASE) is not None:
        mx, my = np.meshgrid(np.arange(size), np.arange(size))
        model[ mx >  0.5814 * my + 0.5 * size -0.5 ] = 0
        model[ mx < -0.5814 * my + 0.5 * size -0.5 ] = 0
        model[ my > 0.86 * size ] = 0
        print("Create [triangle] model\n")

    else:
        print('Unknown model shape "{}"! Please use one of the folowing:'.format(shape))
        print(' Square  |  Circle  |  Triangle\n')
        sys.exit(0)
    return model


def error_plot(data, save_path, Hext):
    """
    Plot the error data and save the plot to a file.
    
    Args:
    - data (list): List of error data points.
    - save_path (str): Path to save the plot.
    - Hext (float): External magnetic field value.
    """
    plt.plot(data)
    plt.yscale('log')
    plt.title(f'Spin update errors\n Hext(oe):{Hext}')
    plt.xlabel('Iterations')
    plt.ylabel('log_error')
    plt.savefig(f'{save_path}.png')
    plt.close()

def data_rotate(Spins, Hds, symtype=None):
    """
    Rotate and modify spin data based on symmetry type.
    
    Args:
    - Spins (numpy.ndarray): Spin data.
    - Hds (numpy.ndarray): Hd data.
    - symtype (str): Symmetry type ('RX', 'RY', 'R90', 'R180', 'R270').
    
    Returns:
    - tuple: Rotated Spins and Hds.
    """
    if symtype == 'RX': # X image
        transformation = np.array([-1, 1, 1, -1, 1, 1], dtype=np.float16)
        Spins_rotate = Spins[:, ::-1] * transformation
        Hds_rotate = Hds[:, ::-1] * transformation

    elif symtype == 'RY': # Y image
        transformation = np.array([1,-1,1,1,-1,1], dtype=np.float16)
        Spins_rotate = Spins[:,:,::-1] * transformation
        Hds_rotate = Hds[:,:,::-1] * transformation

    elif symtype == 'R90': # Rotate 90
        transformation = np.array([-1,1,1,-1,1,1], dtype=np.float16)
        Spins_rotate = Spins.transpose(0,2,1,3)[:,::-1].copy()
        spin_tmp = Spins_rotate.copy()
        Spins_rotate[...,0], Spins_rotate[...,1] = spin_tmp[...,1], spin_tmp[...,0] 
        Spins_rotate[...,3], Spins_rotate[...,4] = spin_tmp[...,4], spin_tmp[...,3] 
        Spins_rotate = Spins_rotate * transformation
        Hds_rotate = Hds.transpose(0,2,1,3)[:,::-1].copy()
        Hds_tmp = Hds_rotate.copy()
        Hds_rotate[...,0], Hds_rotate[...,1] = Hds_tmp[...,1], Hds_tmp[...,0] 
        Hds_rotate[...,3], Hds_rotate[...,4] = Hds_tmp[...,4], Hds_tmp[...,3] 
        Hds_rotate = Hds_rotate * transformation

    elif symtype == 'R180': # Rotate 180
        transformation = np.array([-1,-1,1,-1,-1,1], dtype=np.float16)
        Spins_rotate = Spins[:,::-1,::-1] * transformation
        Hds_rotate = Hds[:,::-1,::-1] * transformation

    elif symtype == 'R270': # Rotate 270
        transformation = np.array([1,-1,1,1,-1,1], dtype=np.float16)
        # Rotate 270
        Spins_rotate = Spins.transpose(0,2,1,3)[:,:,::-1].copy()
        spin_tmp = Spins_rotate.copy()
        Spins_rotate[...,0], Spins_rotate[...,1] = spin_tmp[...,1], spin_tmp[...,0] 
        Spins_rotate[...,3], Spins_rotate[...,4] = spin_tmp[...,4], spin_tmp[...,3] 
        Spins_rotate = Spins_rotate * transformation
        Hds_rotate = Hds.transpose(0,2,1,3)[:,:,::-1].copy()
        Hds_tmp = Hds_rotate.copy()
        Hds_rotate[...,0], Hds_rotate[...,1] = Hds_tmp[...,1], Hds_tmp[...,0] 
        Hds_rotate[...,3], Hds_rotate[...,4] = Hds_tmp[...,4], Hds_tmp[...,3] 
        Hds_rotate = Hds_rotate * transformation

    return Spins_rotate, Hds_rotate



def winding_density(spin_batch):
    """
    用于计算batch数据的winding density
    Args:
    spin_batch: torch.tensor
                形状为(batch_size, 3, 32, 32)的tensor，表示包含batch_size个样本的spin数据
    Returns:
    winding_density_batch: torch.tensor
                形状为(batch_size, 32, 32)的tensor，表示batch数据的winding density
    """
    # 调整spin的维度顺序为[batch_size, 32, 32, 1, 3]
    spin = spin_batch.permute(0, 2, 3, 1).unsqueeze(-2)
    spin_xp = torch.roll(spin, shifts=-1, dims=1)
    spin_xm = torch.roll(spin, shifts=1, dims=1)
    spin_yp = torch.roll(spin, shifts=-1, dims=2)
    spin_ym = torch.roll(spin, shifts=1, dims=2)
    spin_xp[:, -1, :, :, :] = spin[:, -1, :, :, :]
    spin_xm[:, 0, :, :, :]  = spin[:, 0, :, :, :]
    spin_yp[:, :, -1, :, :] = spin[:, :, -1, :, :]
    spin_ym[:, :, 0, :, :]  = spin[:, :, 0, :, :]
    winding_density = (spin_xp[:,:,:, 0, 0] - spin_xm[:,:,:, 0, 0]) / 2 * (spin_yp[:,:,:, 0, 1] - spin_ym[:,:,:, 0, 1]) / 2 \
                    - (spin_xp[:,:,:, 0, 1] - spin_xm[:,:,:, 0, 1]) / 2 * (spin_yp[:,:,:, 0, 0] - spin_ym[:,:,:, 0, 0]) / 2
    
    winding_density = winding_density / np.pi
    #winding_abs = torch.abs(winding_density).sum()
    winding_abs = torch.round(
                             torch.abs(winding_density).sum(dim=(1,2))
                             )
    winding_sum = torch.round(
                             winding_density.sum(dim=(1,2))
                             )

    return winding_density.squeeze(), winding_abs.item(), winding_sum.item()


def create_trt_model(model, inch, w, dtype_item, device):
    print('create trt-model size {}'.format(w))
    trt_model = torch_tensorrt.compile(
        model, 
        inputs=[torch_tensorrt.Input((1,inch,w,w), dtype=torch.float32)],
        enabled_precisions = {dtype_item},
        device=device
        )
    return trt_model



# Custom function to parse a list of floats
def Culist(string):
    try:
        float_values = [float(value) for value in string.split(',')]
        if len(float_values) != 3:
            raise argparse.ArgumentTypeError("List must contain exactly three elements")
        return float_values
    except ValueError:
        raise argparse.ArgumentTypeError("List must contain valid floats")


def MaskTp(string):
    if string == "False":
        return False
    elif string == "True":
        return True
    else:
        return string