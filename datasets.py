import numpy as np
from sklearn.datasets import fetch_openml
import torch
from torch.utils.data import Dataset
import logging
import os
import copy


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

def make_random_labels(labels, seed = 1, num_classes = 10):

    noisy_labels = np.zeros_like(labels)

    for i in range(num_classes):
        np.random.seed(seed + i)
        all_length = np.sum(labels == i)
        noisy_length = int(4.0 / 9 * all_length)
        new_label = [i] * (all_length - noisy_length) + list(np.random.randint(0, 9, size = noisy_length))
        np.random.seed(seed + i)
        noisy_labels[labels == i] = np.random.permutation(np.array(new_label))

    return noisy_labels

class BaseDataset(Dataset):
    def __init__(self):

        super(BaseDataset, self).__init__()
        self.x = []
        self.y = []
        self.transform = None

    def __getitem__(self, idx):

        if self.transform is not None:
            return self.transform(self.x[idx]), self.y[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):

        return len(self.y)

class MNISTDataset(BaseDataset):
    def __init__(self, split = 'train', seed = 1, transform = None):
        super(BaseDataset, self).__init__()

        (self.x_tr, self.y_tr), (self.x_te, self.y_te) = get_mnist() 
        self.transform = transform

        num_classes = len(set(list(self.y_tr)))      
        noisy_tr = make_random_labels(self.y_tr, num_classes = num_classes, seed = seed)

        if (split == 'train'):
            self.x = self.x_tr
            self.y = noisy_tr
        else:
            self.x = self.x_te
            self.y = self.y_te


if __name__ == '__main__':
    (x_tr, y_tr), (x_te, y_te) = get_mnist()
    noisy_labels = make_random_labels(y_tr, seed = 1)
    print(noisy_labels)




