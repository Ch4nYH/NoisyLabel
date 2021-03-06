import numpy as np
from sklearn.datasets import fetch_openml
import torch
from torch.utils.data import Dataset
import logging
import os
import os.path as osp
import pickle
import copy
import pickle

#imports data
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_mnist():
    mnist_data = fetch_openml("mnist_784")
    x = mnist_data["data"]
    y = mnist_data["target"]
    # reshape to (#data, #channel, width, height)
    x = np.reshape(x, (x.shape[0], 1, 28, 28)) / 255.
    x_tr = np.asarray(x[:60000], dtype=np.float32)
    y_tr = np.asarray(y[:60000], dtype=np.int32)
    x_te = np.asarray(x[60000:], dtype=np.float32)
    y_te = np.asarray(y[60000:], dtype=np.int32)
    return (x_tr, y_tr), (x_te, y_te)

def _load_datafile(filename):
    with open(filename, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
        #print(data_dict.keys())
        assert data_dict[b'data'].dtype == np.uint8
        image_data = data_dict[b'data']
        image_data = image_data.reshape((image_data.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
        return image_data, np.array(data_dict[b'labels'])

def get_cifar():
    train_filenames = ['data_batch_{}'.format(ii + 1) for ii in range(5)]
    eval_filename = 'test_batch'

    x_tr = np.zeros((50000, 32, 32, 3), dtype='uint8')
    y_tr = np.zeros(50000, dtype='int32')

    for ii, fname in enumerate(train_filenames):
        cur_images, cur_labels = _load_datafile(osp.join('cifar', fname))
        x_tr[ii * 10000 : (ii+1) * 10000, ...] = cur_images
        y_tr[ii * 10000 : (ii+1) * 10000, ...] = cur_labels

    x_te, y_te = _load_datafile(osp.join('cifar', eval_filename))
    return (x_tr, y_tr), (x_te, y_te)

def make_symmetric_random_labels(labels, seed = 1, num_classes = 10, percent = 0.5):

    noisy_labels = np.zeros_like(labels)

    for i in range(num_classes):
        np.random.seed(seed + i)
        all_length = np.sum(labels == i)
        noisy_length = int(num_classes * percent / (num_classes - 1) * all_length)
        new_label = [i] * (all_length - noisy_length) + list(np.random.randint(0, 9, size = noisy_length))
        np.random.seed(seed + i)
        noisy_labels[labels == i] = np.random.permutation(np.array(new_label))

    return noisy_labels

class BaseDataset(Dataset):
    def __init__(self):

        super(BaseDataset, self).__init__()
        self.x = []
        self.y = []
        self.t = []
        self.transform = None

    def __getitem__(self, idx):

        if self.transform is not None:
            return self.transform(self.x[idx]), self.y[idx], self.t[idx]
        else:
            return self.x[idx], self.y[idx], self.t[idx]

    def __len__(self):

        return len(self.y)

class MNISTDataset(BaseDataset):
    def __init__(self, split = 'train', seed = 1, transform = None):
        super(BaseDataset, self).__init__()

        (self.x_tr, self.y_tr), (self.x_te, self.y_te) = get_mnist() 
        self.transform = transform

        num_classes = len(set(list(self.y_tr)))      
        noisy_tr = make_symmetric_random_labels(self.y_tr, num_classes = num_classes, seed = seed)

        if (split == 'train'):
            self.x = self.x_tr
            self.y = noisy_tr
            self.t = self.y_tr
        else:
            self.x = self.x_te
            self.y = self.y_te
            self.t = self.y_te

class CIFARDataset(BaseDataset):
    def __init__(self, split = 'train', seed = 1, transform = None, percent = 0.5):
        super(BaseDataset, self).__init__()

        (self.x_tr, self.y_tr), (self.x_te, self.y_te) = get_cifar() 
        self.transform = transform

        num_classes = len(set(list(self.y_tr)))      
        noisy_tr = make_symmetric_random_labels(self.y_tr, num_classes = num_classes, seed = seed, percent = percent)

        if (split == 'train'):
            self.x = self.x_tr
            self.y = noisy_tr
            self.t = self.y_tr
        else:
            self.x = self.x_te
            self.y = self.y_te
            self.t = self.y_te
            
class CIFAR100Dataset(BaseDataset):
    def __init__(self, split = 'train', seed = 1, transform = None, percent = 0.5):
        super(BaseDataset, self).__init__()

        if split == 'train':
            d = unpickle("cifar-100-python/train")
            self.x_tr = d[b'data']
            self.y_tr = d[b'fine_labels']
            self.x_tr = np.array(self.x_tr)
            self.x_tr= self.x_tr.reshape((self.x_tr.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
            self.y_tr = np.array(self.y_tr)
            num_classes = 100    
            noisy_tr = make_symmetric_random_labels(self.y_tr, num_classes = num_classes, seed = seed, percent = percent)
        else:
            d = unpickle("cifar-100-python/test")
            self.x_te = d[b'data']
            self.y_te = d[b'fine_labels']
            self.x_te = np.array(self.x_te)
            self.x_te= self.x_te.reshape((self.x_te.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
            self.y_te = np.array(self.y_te)

        self.transform = transform

        
        if (split == 'train'):
            self.x = self.x_tr
            self.y = noisy_tr
            self.t = self.y_tr
        else:
            self.x = self.x_te
            self.y = self.y_te
            self.t = self.y_te




