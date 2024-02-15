import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torchvision.utils as vutils
import torchvision.transforms.functional as Func


class AverageMeter(object):
    """Computes and stores the average and current value.
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count





def tensor2rgb(tensor1, tensor2, tensor3, save_path):
    tensor1 = tensor1.to(torch.float)
    tensor2 = tensor2.to(torch.float)
    tensor3 = tensor3.to(torch.float)
    nrow = tensor1.shape[0]

    tensor1 = (tensor1 - tensor1.min()) / (tensor1.max() - tensor1.min())
    tensor2 = (tensor2 - tensor2.min()) / (tensor2.max() - tensor2.min())
    tensor3 = (tensor3 - tensor3.min()) / (tensor3.max() - tensor3.min())

    # 将张量转换为 RGB 图像，并将其拼接在一起
    combined_tensor = torch.cat((tensor1, tensor2, tensor3), dim=0)
    combined_image = vutils.make_grid(combined_tensor, nrow=nrow, padding=0)

    # 保存为 png 文件
    vutils.save_image(combined_image, save_path, normalize=True)



def vectorgraph(t1, t2, t3, save_path):
    mask = torch.where(t1 != 0, torch.ones_like(t1), torch.tensor(0.0, device=t1.device)).cpu().numpy()
    a1, a2, a3 = t1.cpu().numpy(), t2.cpu().numpy(), t3.cpu().numpy()
    fig, axs = plt.subplots(4, 10, figsize=(15, 7), dpi=1000)
    for i in range(len(a1)):
        arr = a1[i].transpose(1, 2, 0)
        axs[0, i].quiver(np.arange(arr.shape[0]), np.arange(arr.shape[1]), arr[:,:,0].T, arr[:,:,1].T, arr[:,:,2].T, clim=[-0.5, 0.5])
        axs[0, i].set_title('Input {}'.format(i))
        
        arr = a2[i].transpose(1, 2, 0)
        axs[1, i].quiver(np.arange(arr.shape[0]), np.arange(arr.shape[1]), arr[:,:,0].T, arr[:,:,1].T, arr[:,:,2].T, clim=[-0.5, 0.5])
        axs[1, i].set_title('Output {}'.format(i))
        
        arr = a3[i].transpose(1, 2, 0)
        axs[2, i].quiver(np.arange(arr.shape[0]), np.arange(arr.shape[1]), arr[:,:,0].T, arr[:,:,1].T, arr[:,:,2].T, clim=[-0.5, 0.5])
        axs[2, i].set_title('Label {}'.format(i))

        # Calculate MSE between a2 and a3 for this index
        mse = np.square(a2[i] - a3[i]).sum(axis=0)*(mask[i][0])
        mse = np.transpose(mse)
        axs[3, i].imshow(mse, cmap='hot', origin="lower")
        axs[3, i].set_title('MSE {:.1e} \n ({:.1e}, {:.1e})'.format(mse.mean(), mse.min(), mse.max()), fontsize=5)
        
    plt.tight_layout()
    fig.savefig(os.path.join(save_path))
    plt.close()






def create_mask(tensor):
    with torch.no_grad():
        device = tensor.device  # 获取张量所在的设备
        mask = torch.where(tensor != 0, torch.ones_like(tensor), torch.tensor(0.1, device=device))
    return mask

def mse(x, y):
    mse_tensor = torch.square(x-y)
    return mse_tensor

def SLA(x):
    return torch.where(x >= 0, torch.log(x+1), -torch.log(-x+1))

def ISLA(x):
    return torch.where(x >= 0, torch.exp(x)-1, -torch.exp(-x)+1)


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
    spin = torch.tensor(spin_batch).permute(0, 2, 3, 1).unsqueeze(-2)
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
    winding_abs = torch.abs(winding_density).sum(dim=(1,2))

    return winding_density, torch.round(winding_abs).cpu().numpy()



def tensor_rotate(tensor, symtype=None):
    #spins(bsz,w,h,channel)
    tensor=tensor.permute(0,2,3,1)

    if symtype=='RX': #x mirror 
        X_mirrored = torch.flip(tensor, [1])
        X_mirrored[:, :, :, 0] = -X_mirrored[:, :, :, 0]
        return X_mirrored.permute(0,3,1,2)
    
    elif symtype=='RY': #y mirror
        Y_mirrored = torch.flip(tensor, [2])
        Y_mirrored[:, :, :, 1] = -Y_mirrored[:, :, :, 1]
        return Y_mirrored.permute(0,3,1,2)
    
    elif symtype == 'R90': # 逆时针旋转90度
        spinrt90 = torch.rot90(tensor, k=1, dims=(1, 2))
        spinrt90[:, :, :, [0, 1]] = spinrt90[:, :, :, [1, 0]]
        spinrt90[:, :, :, 0] = -spinrt90[:, :, :, 0]
        return spinrt90.permute(0,3,1,2)
    
    elif symtype == 'R180': # 逆时针旋转180度
        spinrt180 = tensor.flip(1).flip(2)
        spinrt180[:, :, :, [0, 1]] = -spinrt180[:, :, :, [0, 1]]
        return spinrt180.permute(0,3,1,2)
    
    elif symtype == 'R270': # 逆时针旋转270度
        spinrt270 = torch.rot90(tensor, k=3, dims=(1, 2))
        spinrt270[:, :, :, [0, 1]] = spinrt270[:, :, :, [1, 0]]
        spinrt270[:, :, :, 1] = -spinrt270[:, :, :, 1]
        return spinrt270.permute(0,3,1,2)
     


def dataug(x,y):
    symtype=['R90', 'R180', 'R270', 'RX', 'RY']    
    selected_symmetry = random.choice(symtype)
    n=x.shape[0]//10
    xr_L1 = tensor_rotate(x[:n, :3, :, :], symtype=selected_symmetry)
    yr_L1 = tensor_rotate(y[:n, :3, :, :], symtype=selected_symmetry)
    xr_L2 = tensor_rotate(x[:n, 3:, :, :], symtype=selected_symmetry)
    yr_L2 = tensor_rotate(y[:n, 3:, :, :], symtype=selected_symmetry)
    xr = torch.cat((xr_L1, xr_L2), dim=1)
    yr = torch.cat((yr_L1, yr_L2), dim=1)
    x_combine = torch.cat((x, xr), dim=0)
    y_combine = torch.cat((y, yr), dim=0)
    return x_combine, y_combine



def visualize(mode, epoch, ex_path, x, y, ISLA_y, size):
    # Validate mode
    if mode not in ['eval', 'train']:
        raise ValueError("Mode must be 'eval' or 'train'")

    # Create necessary directories
    directories = [f'{size}rgb_{mode}', f'{size}vectgraph_{mode}']
    for dir_name in directories:
        os.makedirs(os.path.join(ex_path, dir_name), exist_ok=True)

    # Function to handle visualization for each layer
    def visualize_layer(ch_index, layer_name):
        tensor1 = x[:10, ch_index:ch_index+3, :, :].detach()
        tensor2 = ISLA_y[:10, ch_index:ch_index+3, :, :].detach()
        tensor3 = y[:10, ch_index:ch_index+3, :, :].detach()

        tensor2rgb(tensor1, tensor2, tensor3, f'{ex_path}/{size}rgb_{mode}/epoch{epoch}_{layer_name}.png')
        vectorgraph(tensor1, tensor2, tensor3, f'{ex_path}/{size}vectgraph_{mode}/epoch{epoch}_{layer_name}.png')

    # Visualize for each layer
    visualize_layer(0, 'L1')
    visualize_layer(3, 'L2')
