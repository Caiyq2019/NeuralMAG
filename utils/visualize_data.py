import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def np2rgb(data, save_path, sample_indices, dpi=600):
    """
    Convert numpy array to RGB and save as an image.
    """
    fig, axs = plt.subplots(2, 10, dpi=dpi, figsize=(20, 5))
    for i, sample_idx in enumerate(sample_indices):
        if sample_idx >= data.shape[0]:
            continue

        for layer_idx in range(2):
            layer = data[sample_idx, :, :, layer_idx * 3:(layer_idx + 1) * 3]
            normalized_layer = (layer - layer.min()) / (layer.max() - layer.min())
            axs[layer_idx, i].imshow(normalized_layer)
            axs[layer_idx, i].axis('off')
            axs[layer_idx, i].set_title(f'Sample {sample_idx}')

    plt.tight_layout()
    plt.savefig(f'{save_path}.png', dpi=dpi)
    plt.close()

def vectorgraph(data, save_path, sample_indices, dpi=600):
    """
    Create a vector field graph from the data and save as an image.
    """
    fig, axs = plt.subplots(2, 10, dpi=dpi, figsize=(20, 5))
    for i, sample_idx in enumerate(sample_indices):
        if sample_idx >= data.shape[0]:
            continue

        for layer_idx in range(2):
            layer = data[sample_idx, :, :, layer_idx * 3:(layer_idx + 1) * 3]
            axs[layer_idx, i].quiver(np.arange(layer.shape[0]), np.arange(layer.shape[1]),
                                     layer[:, :, 0].T, layer[:, :, 1].T, layer[:, :, 2].T, clim=[-0.5, 0.5])
            axs[layer_idx, i].axis('off')
            axs[layer_idx, i].set_title(f'Sample {sample_idx}')

    plt.tight_layout()
    plt.savefig(f'{save_path}.png', dpi=dpi)
    plt.close()

def plot_histograms(data, save_path, dpi=600):
    """
    Plot histograms for each channel in the data.
    """
    num_channels = data.shape[-1]
    fig, axs = plt.subplots(2, 3, figsize=(16, 8), dpi=dpi)

    for i in range(num_channels):
        channel_data = data[..., i].flatten()
        axs[i // 3, i % 3].hist(channel_data, bins=50, color='blue', alpha=0.7)
        axs[i // 3, i % 3].set_title(f'Channel {i+1} Histogram')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()

def load_data(file_path):
    """
    Safely load numpy data from file.
    """
    try:
        return np.load(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def visualize_data(folder_path, sample_indices):
    """
    Visualize and save different data representations.
    """
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    
    for path in tqdm(subfolders):
        # Process Spin data
        spin_data = load_data(os.path.join(path, 'Spins.npy'))
        if spin_data is not None:
            np2rgb(spin_data, os.path.join(path, 'Spins_rgb'), sample_indices)
            vectorgraph(spin_data, os.path.join(path, 'Spins_vector'), sample_indices)
            plot_histograms(spin_data, os.path.join(path, 'Spins_hist'))
        
        # Process Hd data
        hd_data = load_data(os.path.join(path, 'Hds.npy'))
        if hd_data is not None:
            np2rgb(hd_data, os.path.join(path, 'Hds_rgb'), sample_indices)
            vectorgraph(hd_data, os.path.join(path, 'Hds_vector'), sample_indices)
            plot_histograms(hd_data, os.path.join(path, 'Hd_hist'))

if __name__ == "__main__":
    folder_path = "./Dataset/data_Hd32_Hext0/"
    sample_indices = [0, 5, 10, 30, 50, 100, 200, 300, 400, 500]
    visualize_data(folder_path, sample_indices)
