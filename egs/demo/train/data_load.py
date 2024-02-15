import os
import glob
import torch
import random
import numpy as np
from tqdm import tqdm
from utils import *

def get_case_paths(paths):
    spin_paths = []
    Hd_paths = []
    for path in glob.glob(os.path.join(paths, '*')):
        try:
            if os.path.exists((path+f'/Spins.npy')) and os.path.exists((path+f'/Hds.npy')):
                spin_paths.append((path+f'/Spins.npy'))
                Hd_paths.append((path+f'/Hds.npy'))
        except ValueError:
            print(f"Invalid path: {path}")
    return spin_paths, Hd_paths

def path_split(paths, ntest, n128, ntrain, mode=None):
    random.seed(123)
    if mode=='eval128':
        spin_paths = []
        Hd_paths = []
        for path in paths:
            spin_path, Hd_path = get_case_paths(path)
            spin_paths.extend(spin_path)
            Hd_paths.extend(Hd_path)
        paired_paths = list(zip(spin_paths, Hd_paths))
        random.shuffle(paired_paths)
        test_data_paths, test_target_paths = zip(*paired_paths[:n128])
        print("test seed number:", len(test_data_paths), len(test_target_paths))
        return test_data_paths, test_target_paths
    else:
        spin_paths = []
        Hd_paths = []
        for path in paths:
            spin_path, Hd_path = get_case_paths(path)
            spin_paths.extend(spin_path)
            Hd_paths.extend(Hd_path)
        paired_paths = list(zip(spin_paths, Hd_paths))
        random.shuffle(paired_paths)
        train_data_paths, train_target_paths = zip(*paired_paths[ntest:ntrain])
        test_data_paths, test_target_paths = zip(*paired_paths[:ntest])
        print("train seed number:", len(train_data_paths))
        print("test seed number:", len(test_data_paths))
        return train_data_paths, train_target_paths, test_data_paths, test_target_paths


def combinfilter(x_paths, y_paths, cn):
    np.random.seed(123)
    selected_x = []
    selected_y = []
    a=0
    for x_path, y_path in zip(x_paths, y_paths):
        x_array = np.load(x_path).transpose((0, 3, 1, 2))
        y_array = np.load(y_path).transpose((0, 3, 1, 2))
        
        # randomly select half of the data
        indices = np.random.choice(x_array.shape[0], 500, replace=False)
        x_array = x_array[indices]
        y_array = y_array[indices]
        
        if cn <1000:
            _, winding_abs = winding_density(x_array)
            indices = np.where((0 <= winding_abs) & (winding_abs <= cn))
            selected_x.append(x_array[indices])
            selected_y.append(y_array[indices])
        else:
            selected_x.append(x_array)
            selected_y.append(y_array)
        a += x_array.shape[0]
    selected_x = np.concatenate(selected_x, axis=0)
    selected_y = np.concatenate(selected_y, axis=0)

    print('0<= core number <={}, selected_percent: {:.2f}'.format(cn, selected_x.shape[0]/a))
    print('selected x y shape: ', selected_x.shape, selected_y.shape, '\n')
    
    return selected_x, selected_y


def getdata(paths, ntest, n128, ntrain, cn, mode=None):
    if mode=='eval128':
        x_test_paths, y_test_paths = path_split(paths, ntest, n128, ntrain, mode)
        x_test, y_test = combinfilter(x_test_paths, y_test_paths, cn)
        return x_test, y_test
    else:
        x_train_paths, y_train_paths, x_test_paths, y_test_paths = path_split(paths, ntest, n128, ntrain)
        x_train, y_train = combinfilter(x_train_paths, y_train_paths, cn)
        x_test, y_test = combinfilter(x_test_paths, y_test_paths, cn)
        return x_train, y_train, x_test, y_test


def dataset_prepare(data_paths, ntest, n128, ntrain, cn, mode=None):
    print('loading data from: ', data_paths)
    if mode=='eval128':
        X_test, Y_test = getdata(data_paths, ntest, n128, ntrain, cn, mode)
        #prepare test set
        X_test_tensor = torch.from_numpy(X_test).float()
        y_test_tensor = torch.from_numpy(Y_test).float()
        test_dataset  = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        return test_dataset
    else:
        X_train, Y_train, X_test, Y_test = getdata(data_paths, ntest, n128, ntrain, cn)
        #prepare training set
        X_train_tensor = torch.from_numpy(X_train).float()
        y_train_tensor = torch.from_numpy(Y_train).float()
        train_dataset  = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        #prepare test set
        X_test_tensor = torch.from_numpy(X_test).float()
        y_test_tensor = torch.from_numpy(Y_test).float()
        test_dataset  = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        return train_dataset, test_dataset

