from __future__ import print_function
from __future__ import division

import copy
import glob
import os
from os import listdir,mkdir,rmdir
from os.path import join,isdir,isfile

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
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead

def main(args):
    """
    Main function to parse arguments.
    """
    # Reading command line arguments into parser.
    parser = argparse.ArgumentParser(description = "Prepare CIFAR10 data.")

    # Filepaths
    parser.add_argument("--pData", dest="path_data", type=str, default=None)
    parser.add_argument("--pSave", dest="path_save", type=str, default=None)
    parser.add_argument("--pModel", dest="path_model", type=str, default=None)

    # Creating Parser Object
    opts = parser.parse_args(args[1:])

    if not isdir(opts.path_save):
        mkdir(opts.path_save)



    urls = [('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', 'cifar-10-python.tar.gz'),
            ('https://uofi.box.com/shared/static/8sw0gj6d35zgw1z0isi6jy816r6x0g95.zip', 'cifar-10-smalldata-manualseg.zip')]
    #for url,filename in urls:
    #    wget.download(url, join(opts.path_data,filename))
    for url,filename in urls:
        if filename[-1] == 'z':
            with tarfile.open(join(opts.path_data, filename),'r:gz') as f:
                f.extractall(opts.path_data)
        else:
            with zipfile.ZipFile(join(opts.path_data, filename),'r') as f:
                f.extractall(opts.path_data)


if __name__ == "__main__":
    main(sys.argv)
