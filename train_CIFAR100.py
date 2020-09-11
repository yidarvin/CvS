from __future__ import print_function
from __future__ import division

import copy
import glob
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

from data.cifar10 import *
from data.transforms import *
from engine.architectures import *
from engine.trainer import *

def main(args):
    """
    Main function to parse arguments.
    """
    # Reading command line arguments into parser.
    parser = argparse.ArgumentParser(description = "Train On CIFAR10 data.")

    # Filepaths
    parser.add_argument("--pData", dest="path_data", type=str, default='/home/Data/CIFAR10/cifar-100-smalldata-seg10')
    parser.add_argument("--pVal", dest="path_val", type=str, default='/home/Data/CIFAR10/cifar-100-python')
    parser.add_argument("--pModel", dest="path_model", type=str, default='/home/Models')
    parser.add_argument("--name", dest="name", type=str, default='default')
    parser.add_argument("--numex", dest="num_examples", type=int, default=10)
    parser.add_argument("--lr", dest="lr", type=np.float32, default=3e-5)
    parser.add_argument("--epoch", dest="num_epochs", type=int, default=100)

    # Creating Parser Object
    opts = parser.parse_args(args[1:])

    if not isdir(opts.path_model):
        mkdir(opts.path_model)

    in_chan = 3
    out_chan = 101
    img_size = 128
    dataset_size = 100
    validation   = True
    batch_size   = int(np.clip(opts.num_examples,4,128))
    name_exp = opts.path_data.split('\')[-1] + '_' + str(opts.num_examples) + '_' + opts.name

    # Create the Dataloaders
    dataloaders = create_dataloaders_cifar10(opts.path_data,opts.path_val,
                                             batch_size,img_size,opts.num_examples,
                                             dataset_size,validation)

    # Create the model
    model = densenet101(in_chan, out_chan, pretrained=False)

    # Do the training
    model = trainer_CvS(model, dataloaders, opts.path_model, name_exp, opts.lr, opts.num_epochs)


if __name__ == "__main__":
    main(sys.argv)
