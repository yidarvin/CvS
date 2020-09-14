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

from .transforms import *
from ..utils.tools import *


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












def preproc_CIFAR100_img(img,resize=128):
    img = cv2.resize(img.astype(np.float32), (resize,resize))
    img = img.astype(np.float32) / 128.0 - 1.0
    img = img.transpose(2,0,1)
    return img

def sample_extend_data_CIFAR100(images,segs,labels,num_examples=9999999999,dataset_size=1):
    images_tr = []
    segs_tr = []
    labels_tr = []
    for ii in range(100):
        counter = 0
        for jj in range(len(images)):
            if labels[jj] == ii:
                counter += 1
                images_tr.append(images[jj])
                segs_tr.append(segs[jj])
                labels_tr.append(labels[jj])
                if counter >= num_examples:
                    break
    images = []
    segs = []
    labels = []
    for ii in range(int(np.ceil(dataset_size / num_examples))):
        images += images_tr
        segs += segs_tr
        labels += labels_tr
    return images,segs,labels

class cifar100SegmentationDataset(Dataset):
    def __init__(self, path_data, resize=128, num_examples=9999999999,dataset_size=1,transform=None):
        self.path_data = path_data
        self.path_imgs = []
        self.path_segs = []
        self.labels = []
        for ii,name_class in enumerate(sorted(listdir(path_data))):
            for name_img in sorted(listdir(join(path_data, name_class))):
                if name_img[-5] == 'g':
                    continue
                path_img = join(path_data, name_class, name_img)
                path_seg = path_img[:-4] + 'seg' + '.png'
                self.path_imgs.append(path_img)
                self.path_segs.append(path_seg)
                self.labels.append(ii)
        self.path_imgs,self.path_segs,self.labels = sample_extend_data_CIFAR100(self.path_imgs,self.path_segs,self.labels,num_examples,dataset_size)
        self.resize = resize
        self.transform = transform
    def __len__(self):
        return len(self.path_imgs)
    def __getitem__(self,idx):
        img = cv2.imread(self.path_imgs[idx])
        path_seg = self.path_segs[idx]
        lab = self.labels[idx]
        seg = cv2.imread(path_seg)[:,:,0]

        img = prep_torchvision(img, self.resize)

        seg = cv2.resize(seg.astype(np.float32), (self.resize,self.resize))
        seg = (seg > 10) * (lab + 1)
        seg = seg.astype(np.int16)

        sample = {'X': img, 'Y': seg}
        if self.transform:
            sample = self.transform(sample)

        return sample

class cifar100ValidationDataset(Dataset):
    def __init__(self, X,Y, resize=128, transform=None):
        self.X = X
        self.Y = Y
        self.resize = resize
        self.transform = transform
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self,idx):
        img = self.X[idx,:,:,:].transpose(1,2,0)
        img = prep_torchvision(img, self.resize)

        seg = (self.Y[idx] + 1) * np.ones((self.resize,self.resize))
        seg = seg.astype(np.int16)

        sample = {'X': img, 'Y': seg}
        if self.transform:
            sample = self.transform(sample)

        return sample

def create_dataloaders_cifar100(path_data='/home/Data/CIFAR100/cifar-100-smalldata-seg10',
                                path_val='/home/Data/CIFAR100/cifar-100-python',
                                batch_size=48,img_size=128,num_examples=10,dataset_size=100,validation=True):
    data_tr = cifar100SegmentationDataset(path_data, img_size, num_examples,dataset_size,
                                          transform=transforms.Compose([RandomFlip(),RandomShift(),AddNoise(),ToTensor()]))
    loader_tr = DataLoader(data_tr, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloaders = {}
    dataloaders['train'] = loader_tr
    if validation:
        _,_,X_te,Y_te,_ = load_CIFAR100_data(path_val)
        data_va = cifar100ValidationDataset(X_te, Y_te,img_size,
                                           transform=transforms.Compose([ToTensor()]))

        loader_va = DataLoader(data_va, batch_size=batch_size, shuffle=False, num_workers=0)
        dataloaders['val'] = loader_va
    return dataloaders
