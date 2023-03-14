from __future__ import print_function
from __future__ import division

import copy
import glob
#from mnist import MNIST
import os
from os import listdir,mkdir,rmdir
from os.path import join,isdir,isfile
import pickle
from PIL import Image
import platform
from six.moves import cPickle as pickle
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


def pad_mura(img):
    m,n = img.shape
    if m >= n:
        img = np.pad(img, ((0,0),(int((m-n)/2),int((m-n)/2))))
    else:
        img = np.pad(img, ((int((n-m)/2),int((n-m)/2)), (0,0)))
    return img

def segment_mura(image):
    img = image
    image = cv2.equalizeHist(image)
    img_vals = np.unique(image.reshape([-1]))
    img_vals = sorted(img_vals)
    if len(img_vals) > 20:
    	image[image < img_vals[10]] = img_vals[10]
    	image[image > img_vals[-10]] = img_vals[-10]
    r,th = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    r2,th2 = cv2.threshold(image[th>0].reshape([-1,1]),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th = np.bitwise_and(img < r2, th>0).astype(np.uint8)
    
    #nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(th, connectivity=4)
    #sizes = stats[:, -1]
    #max_label = 1
    #max_size = sizes[1]
    #for i in range(2, nb_components):
    #    if sizes[i] > max_size:
    #        max_label = i
    #        max_size = sizes[i]
    #img2 = np.zeros(output.shape)
    #img2[output == max_label] = 1
    return th

def load_mura_data(path_data=None):
    path_root = '/'.join(path_data.split('/')[:-1])
    #print(path_data)
    #print(path_root)
    dict_path2lab = {}
    with open(join(path_data, 'train_labeled_studies.csv')) as f:
        line = f.readline()
        while(line):
            path_folder,lab_orig = line.split(',')
            lab = int(lab_orig[0]) * 7
            for body_part in ['ELBOW', 'FINGER', 'FOREARM', 'HAND', 'HUMERUS', 'SHOULDER', 'WRIST']:
                if body_part in path_folder:
                    break
                lab += 1
            path_imgs = join(path_root,path_folder)
            for name_img in sorted(listdir(path_imgs)):
                if name_img[0] == '.':
                    continue
                path_img = join(path_imgs, name_img)
                dict_path2lab[path_img] = 1+int(lab_orig[0])#lab
            line = f.readline()
    dict_path2lab_val = {}
    with open(join(path_data, 'valid_labeled_studies.csv')) as f:
        line = f.readline()
        while(line):
            path_folder,lab_orig = line.split(',')
            lab = int(lab_orig[0]) * 7
            for body_part in ['ELBOW', 'FINGER', 'FOREARM', 'HAND', 'HUMERUS', 'SHOULDER', 'WRIST']:
                if body_part in path_folder:
                    break
                lab += 1
            path_imgs = join(path_root,path_folder)
            for name_img in sorted(listdir(path_imgs)):
                if name_img[0] == '.':
                    continue
                path_img = join(path_imgs, name_img)
                dict_path2lab_val[path_img] = 1+int(lab_orig[0])#lab
            line = f.readline()
    return dict_path2lab,dict_path2lab_val


class muraSegmentationDataset(Dataset):
    def __init__(self, dictionary, resize=256, num_exp=99999999, dataset_size=1, transform=None):
        self.dict = dictionary
        self.resize = resize
        self.transform = transform
        self.num_exp = num_exp
        self.dataset_size=dataset_size
    def __len__(self):
        return len(self.dict.keys())
    def __getitem__(self,idx):
        key = list(self.dict.keys())[idx]
        val = self.dict[key]
        
        img = cv2.imread(key)[:,:,0]
        if np.mean(img.astype(np.float32)) > 128:
            img = 255 - img
        img = pad_mura(img)
        img = cv2.resize(img.astype(np.float32), (self.resize,self.resize))
        seg = segment_mura(img.astype(np.uint8)) * val
        img = img.astype(np.float32) / 255
        img = img.reshape([1,self.resize,self.resize])
        
        sample = {'X': img, 'Y': seg}
        if self.transform:
            sample = self.transform(sample)
        
        return sample





def create_dataloaders_mura(path_data='/home/Data/MURA-v1.1/',
                             batch_size=48,img_size=128,num_examples=10,dataset_size=100,validation=True):
    shift = int(img_size / 8)
    dict_tr,dict_va = load_mura_data(path_data)
    data_tr = muraSegmentationDataset(dict_tr, img_size, num_examples,dataset_size,
                                       transform=transforms.Compose([RandomTurn(),RandomFlip(),RandomRotate(),RandomShiftZoom(max_shift=shift),ToTensor()]))
    loader_tr = DataLoader(data_tr, batch_size=batch_size, shuffle=True, num_workers=8)
    dataloaders = {}
    dataloaders['train'] = loader_tr
    if validation:
        data_va = muraSegmentationDataset(dict_va,img_size,
                                           transform=transforms.Compose([ToTensor()]))

        loader_va = DataLoader(data_va, batch_size=batch_size, shuffle=False, num_workers=8)
        dataloaders['val'] = loader_va
    return dataloaders
