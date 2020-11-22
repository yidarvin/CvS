from __future__ import print_function
from __future__ import division

import copy
import glob
from mnist import MNIST
import os
from os import listdir,mkdir,rmdir
from os.path import join,isdir,isfile
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

HAND_SELECTED_INDS = (51,8,171,50,131,173,62,42,46,43)


def load_mnist_data(path_data=None):
    mndata = MNIST(path_data)
    img_tr,lab_tr = mndata.load_training()
    img_te,lab_te = mndata.load_testing()
    return img_tr,lab_tr,img_te,lab_te

class mnistSegmentationDataset(Dataset):
    def __init__(self, images, labels, resize=112, transform=None):
        self.images = images
        self.labels = labels
        self.resize = resize
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self,idx):
        img = np.array(self.images[idx]).reshape([28,28])# + np.random.randn(28,28)
        img = cv2.resize(img.astype(np.float32), (self.resize, self.resize))

        seg = (img > 0) * (self.labels[idx] + 1)
        seg = seg.astype(np.int16)

        img = img.astype(np.float32) / 255.0 #+ np.random.randn(self.resize,self.resize)/10
        img = img.reshape([1,img.shape[0],img.shape[1]])
        sample = {'X': img, 'Y': seg}
        if self.transform:
            sample = self.transform(sample)
        return sample

def sample_extend_data(images,labels,num_examples=9999999999,dataset_size=1):
    images_tr = []
    labels_tr = []
    for ii in range(10):
        counter = 0
        for jj in range(len(images)):
            if labels[jj] == ii:
                counter += 1
                images_tr.append(images[jj])
                labels_tr.append(labels[jj])
                if counter >= num_examples:
                    break
    images = []
    labels = []
    for ii in range(int(np.ceil(dataset_size / num_examples))):
        images += images_tr
        labels += labels_tr
    return images,labels

def create_dataloaders_mnist(path_data='/home/darvin/Data/mnist',batch_size=48,img_size=112,num_examples=10,dataset_size=100,validation=True):
    img_tr,lab_tr,img_te,lab_te = load_mnist_data(path_data)
    img_tr,lab_tr = sample_extend_data(img_tr,lab_tr,num_examples,dataset_size)
    data_tr = mnistSegmentationDataset(img_tr, lab_tr, resize=img_size,
                                       transform=transforms.Compose([RandomTurn(),RandomShiftZoom(int(img_size/4)),AddNoise(),ToTensor()]))
    data_va = mnistSegmentationDataset(img_te, lab_te,resize=img_size,
                                       transform=transforms.Compose([ToTensor()]))
    loader_tr = DataLoader(data_tr, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_va = DataLoader(data_va, batch_size=batch_size, shuffle=False, num_workers=0)
    dataloaders = {}
    dataloaders['train'] = loader_tr
    if validation:
        dataloaders['val'] = loader_va
    return dataloaders
