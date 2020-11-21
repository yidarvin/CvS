from __future__ import print_function
from __future__ import division

import copy
import glob
import os
from os import listdir,mkdir,rmdir
from os.path import join,isdir,isfile
import pickle
import platform
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
from utils.tools import *


def preproc_avdr_seg(seg,resize=304):
    seg = seg.astype(np.float32)
    if resize < seg.shape[0] and resize < seg.shape[1]:
        m,n,_ = seg.shape
        ii = np.random.choice(m-resize)
        jj = np.random.choice(n-resize)
        seg = seg[ii:(ii+resize),jj:(jj+resize),:]
    elif resize != seg.shape[0] and resize != seg.shape[1]:
        seg = cv2.resize(img.astype(np.float32), (resize,resize))
    new_seg = np.zeros((seg.shape[0], seg.shape[1]))
    new_seg[seg[:,:,0] > 200] = 1
    new_seg[seg[:,:,2] > 200] = 2
    return new_seg

def preproc_avdr_img(img,resize=304):
    img = img.astype(np.float32)
    if len(img.shape) == 3:
        img = np.max(img, axis=2)
    if resize < img.shape[0] and resize < img.shape[1]:
        m,n = img.shape
        ii = np.random.choice(m-resize)
        jj = np.random.choice(n-resize)
        img = img[ii:(ii+resize),jj:(jj+resize)]
    elif resize != img.shape[0] and resize != img.shape[1]:
        img = cv2.resize(img.astype(np.float32), (resize,resize))
    img -= img.min()
    img /= (img.max()+1e-6)
    
    img = img.reshape([1, img.shape[0], img.shape[1]])
    return img
    
def sample_extend_data_dd(images,labels,segs,dataset_size=1):
    images_tr = []
    labels_tr = []
    segs_tr = []
    for ii in range(int(np.ceil(dataset_size / len(images)))):
        images_tr += images
        labels_tr += labels
        segs_tr += segs
    return images_tr,labels_tr,segs_tr

class AVDRDataset(Dataset):
    def __init__(self, path_data, resize=1080, train=True, dataset_size=1, transform=None):
        self.path_data = path_data
        self.path_imgs = []
        self.labels    = []
        self.path_segs = []
        if train:
            folders = ['wt_train', 'mt_train']
        else:
            folders = ['wt_val', 'mt_val']
        for ii,name_class in enumerate(listdir(path_data)):#['AV_map', 'OCT', 'OCTA']
            for jj,name_img in enumerate(sorted(listdir(join(path_data, name_class, 'AV_map')))):
                if jj > 10 and train:
                    continue
                elif jj <= 10 and not train:
                    continue
                self.path_segs.append(join(path_data, name_class, 'AV_map', name_img))
                self.path_imgs.append((join(path_data, name_class, 'OCT', name_img),
                                       join(path_data, name_class, 'OCTA', name_img)))
                self.labels.append(ii)
        self.train = train
        self.path_imgs,self.labels,self.path_segs, = sample_extend_data_dd(self.path_imgs,self.labels,self.path_segs,dataset_size)
        self.resize = resize
        self.transform = transform
    def __len__(self):
        return len(self.path_imgs)
    def __getitem__(self,idx):
        seg = cv2.imread(self.path_segs[idx])
        im1 = cv2.imread(self.path_imgs[idx][0])
        im2 = cv2.imread(self.path_imgs[idx][1])
        lab = self.labels[idx]
        if self.train:
            resize = self.resize
        else:
            resize = max(im1.shape[0],im1.shape[1])
        seg = preproc_avdr_seg(seg,resize)
        im1 = preproc_avdr_img(im1,resize)
        im2 = preproc_avdr_img(im2,resize)
        img = np.concatenate([im1,im2],axis=0)
        #img = im1
        #seg[seg > 0] = 4*seg[seg > 0] - float(lab)
        seg = (seg > 0) * (float(lab) + 1)
        #seg = binary_dilation(seg>0, iterations=2) * (float(lab) + 1)
        sample = {'X': img, 'Y': seg}
        if self.transform:
            sample = self.transform(sample)

        return sample
        

def create_dataloaders_AVDR(path_data='/home/Data/AVDRDataset',
                            batch_size=4,img_size=304,dataset_size=1,validation=True):
    data_tr = AVDRDataset(path_data, img_size, True, dataset_size,
                          transform=transforms.Compose([RandomFlip(),RandomShift(),AddNoise(),ToTensor()]))
    loader_tr = DataLoader(data_tr, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    dataloaders = {}
    dataloaders['train'] = loader_tr
    if validation:
        data_va = AVDRDataset(path_data, img_size, False,
                              transform=transforms.Compose([ToTensor()]))
        loader_va = DataLoader(data_va, batch_size=1, shuffle=False, num_workers=0)
        dataloaders['val'] = loader_va
    return dataloaders


class drugDiscoveryInferenceDataset(Dataset):
    def __init__(self, path_data, transform=None):
        self.path_data = path_data
        self.path_imgs = [join(path_data, name_img) for name_img in listdir(path_data)]
        self.transform = transform
    def __len__(self):
        return len(self.path_imgs)
    def __getitem__(self,idx):
        img = cv2.imread(self.path_imgs[idx])
        img,seg = preproc_dd_img(img,resize=img.shape[0])
        
        sample = {'X': img, 'Y': seg, 'lab': self.path_imgs[idx]}
        if self.transform:
            sample = self.transform(sample)

        return sample
    
def create_dataloaders_dd_inference(path_data='/home/Data/drugDiscovery/exp000', batch_size=4):
    data = drugDiscoveryInferenceDataset(path_data, transform=transforms.Compose([ToTensor()]))
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False)
    return loader
