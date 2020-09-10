from __future__ import print_function
from __future__ import division

import copy
import glob
from mnist import MNIST
import os
from os import listdir,mkdir,rmdir
from os.path import join,isdir,isfile
import pickle
from PIL import Image
import time

import cv2
import numpy as np
from scipy.ndimage.morphology import binary_dilation,binary_erosion
from skimage import exposure

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import datasets, models, transforms,utils
from torchvision.transforms import functional as func

from data.transforms import *



def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res

def load_CIFAR100_batch(filename=None):
    data = unpickle(filename)
    Y = data[b'fine_labels']
    X = data[b'data'].reshape(-1,3,32,32)
    return X,Y

def load_CIFAR100_data(path_data=None):
    meta = unpickle(join(path_data,'meta'))
    fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]

    X_tr,Y_tr = load_CIFAR100_batch(join(path_data,'train'))
    X_te,Y_te = load_CIFAR100_batch(join(path_data,'test'))
    return X_tr,Y_tr,X_te,Y_te,fine_label_names
