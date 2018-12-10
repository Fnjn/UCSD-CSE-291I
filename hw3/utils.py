import h5py
import os
import numpy as np

"""
Helper functions to implement PointNet
"""
#MODELNET40_PATH = "modelnet40_ply_hdf5_2048/"
MODELNET40_PATH = "/datasets/home/03/803/cs291ebj/modelnet40_ply_hdf5_2048/"

def get_dataset_paths(filename):
    filename = os.path.join(MODELNET40_PATH, filename)
    with open(filename, 'r') as f:
        data_paths = f.read().splitlines()
    return data_paths

def load_h5(h5_filename):
    """
    Data loader function.
    Input: The path of h5 filename
    Output: A tuple of (data,label)
    """
    h5_filename = os.path.join(MODELNET40_PATH, h5_filename)
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def get_category_names():
    """
    Function to list out all the categories in MODELNET40
    """
    shape_names_file = os.path.join(MODELNET40_PATH, 'shape_names.txt')
    shape_names = [line.rstrip() for line in open(shape_names_file)]
    return shape_names

def evaluate(true_labels,predicted_labels):
    """
    Function to calculate the total accuracy.
    Input: The ground truth labels and the predicted labels
    Output: The accuracy of the model
    """
    return np.mean(true_labels == predicted_labels)


from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def create_batch(X, Y, batch_size):
    m = X.shape[0]
    n_batch = int(m / batch_size)

    X_batches = []
    Y_batches = []

    perm = np.random.permutation(m)
    X_shuffle = X[perm, ...]
    Y_shuffle = Y[perm, ...]

    for i in range(n_batch):
        X_batch = X_shuffle[i * batch_size: (i+1) * batch_size, ...]
        Y_batch = Y_shuffle[i * batch_size: (i+1) * batch_size, ...]
        X_batches.append(X_batch)
        Y_batches.append(Y_batch.reshape(batch_size))

    return np.stack(X_batches, axis=0), np.stack(Y_batches, axis=0), n_batch

def create_val(datapath, val_ratio=0.05):
    X, Y = load_h5(datapath)
    m = X.shape[0]
    n_val = int(m * val_ratio)

    perm = np.random.permutation(m)
    val_X = X[perm[:n_val], ...]
    val_Y = Y[perm[:n_val], ...]
    return val_X, val_Y
