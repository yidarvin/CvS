from __future__ import print_function
from __future__ import division

import argparse
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

from data.avdr import *
from data.transforms import *
from engine.architectures import *
from engine.trainer import *



def main(args):
    """
    Main function to parse arguments.
    """
    # Reading command line arguments into parser.
    parser = argparse.ArgumentParser(description = "AV DR Experiments.")

    # Filepaths
    parser.add_argument("--pData", dest="path_data", type=str, default='/home/Data/AVDRDataset')
    parser.add_argument("--pModel", dest="path_model", type=str, default='/home/Models')
    parser.add_argument("--name", dest="name", type=str, default='default')
    parser.add_argument("--lr", dest="lr", type=np.float32, default=3e-5)
    parser.add_argument("--epoch", dest="num_epochs", type=int, default=100)

    # Creating Parser Object
    opts = parser.parse_args(args[1:])

    if not isdir(opts.path_model):
        mkdir(opts.path_model)

    in_chan = 2
    out_chan = 5
    img_size = 304
    validation   = True
    batch_size   = 15
    dataset_size = 1000
    name_exp = 'AVDR_' + opts.name

    # Create the Dataloaders
    dataloaders = create_dataloaders_AVDR(opts.path_data,batch_size,img_size,dataset_size,validation)

    # Create the model
    model = densenet50(in_chan, out_chan, pretrained=True)

    # Do the training
    model = trainer_CvS(model, dataloaders, opts.path_model, name_exp, opts.lr, opts.num_epochs)


if __name__ == "__main__":
    main(sys.argv)
