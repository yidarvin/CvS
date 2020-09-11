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

from data.mnist import *
from data.transforms import *
from engine.architectures import *
from engine.trainer import *

# EXPERIMENT SPECIFIC (SHOULD NOT CHANGE)
in_chan = 1
out_chan = 11
name_exp = 'MNIST-1e2'

# COMPUTER SPECIFIC
path_data = '/home/Data/MNIST'
path_save = '/home/Models'

# HYPER PARAMETERS
num_examples = 100
img_size     = 112
batch_size   = int(np.clip(num_examples*2,4,128))
dataset_size = 100
validation   = True
learning_rate = 3e-5
num_epochs = 100

if not isdir(path_save):
    mkdir(path_save)

# Create the Dataloaders
dataloaders = create_dataloaders_mnist(path_data,batch_size,img_size,num_examples,dataset_size,validation)

# Create the model
model = densenet101(in_chan, out_chan, pretrained=False)

# Do the training
model = trainer_CvS(model, dataloaders, path_save, name_exp, learning_rate, num_epochs)
